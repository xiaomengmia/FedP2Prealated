import torch, os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import math, time
from CnnModel import *
import urllib.request
import glob
import time

def ASGD_train(localNetwork, train_loader, optimizer , epoch, dev, log_interval, train_losses, train_counter, param, seq, mini, config, ComputingObject):
  hostname = config['HostName']
  localNetwork.train()
  CEpoch = param['CurrentEpoch']
  start = math.ceil(seq * len(train_loader.dataset) / (int(param['TrainBatchSize']) * int(param['Nodes'])))
  if seq + 1 == int(param['TrainBatchSize']):
    end = math.ceil(len(train_loader.dataset)/ int(param['TrainBatchSize']))
  else:
    end =  math.ceil((seq + 1) * len(train_loader.dataset) / (int(param['TrainBatchSize'])*int(param['Nodes'])))
  print('Start Training ', start , end)
  for batch_idx, (data, target) in enumerate(train_loader):
    with torch.set_grad_enabled(True):
      if dev.find('cuda') >=0:
        data = data.to(dev)
        target = target.to(dev)
      if batch_idx >= start and batch_idx < end:
        output = localNetwork(data)
        #loss = F.nll_loss(output, target)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        if (((batch_idx + 1) % mini) == 0):
        #  loss.backward()
    #      optimizer.step()
          Now = int(time.time())
          GradientFileName = param['ProjectName'] + '-' + str(seq) + '-' + str(CEpoch)+ '-' + str(Now) + '.' + hostname + '.gth'
          SaveGradientFile(localNetwork, optimizer, GradientFileName, param)
          PushGradientInfo(GradientFileName, param, ComputingObject)
          localNetwork = LoadLatestModel(localNetwork, GradientFileName, param, dev, config)
          optimizer.zero_grad()
          print('Before Update Project Info', GradientFileName)
      optimizer.step()
      optimizer.zero_grad()
  if batch_idx % log_interval == 0:
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
       epoch, batch_idx * len(data), len(train_loader.dataset),
       100. * batch_idx / len(train_loader), loss.item()))
    train_losses.append(loss.item())
    train_counter.append(
       (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
  print('end batch')
  print('end training')
  return localNetwork

def SaveGradientFile(network, optimizer, GradientFile, param):
  grad_dict = {x[0]:x[1].grad for x in network.named_parameters()}
  torch.save({'grad':grad_dict, 'optim':optimizer.state_dict()}, os.path.join(param['ModelFile'], GradientFile))
  param['Gradient'].append(GradientFile)
  return

def SaveModelFile(network, ModelFile, param):
  torch.save(network, os.path.join(param['ModelFile'], ModelFile))
  param['GlobalModel'].append(ModelFile)
  return

def PushGradientInfo(GradientFile, param, ComputingObject):
  uri = 'Notify'
  Msg = {}
  Msg['Type'] = 'GlobalModel'
  Msg['Gradient'] = GradientFile
  Msg['GradientMD5'] = ComputingObject.FileMd5(os.path.join(ComputingObject.config['Model.File'], GradientFile))
  Msg['GradientSize'] = os.path.getsize(os.path.join(ComputingObject.config['Model.File'], GradientFile))
  ComputingObject.PushMessage(uri, Msg)
  return
  
def ASGD_test(network, param, test_loader, dev, test_losses):
  network.eval()
  test_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for data in test_loader:
      images , labels = data
      if dev.find('cuda') >=0:
        images = images.to(dev)
        labels = labels.to(dev)
      output = network(images)
      if param['Datasetname'].find('MNIST') >=0:
        test_loss += F.nll_loss(output, labels, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).sum().item()
      elif param['Datasetname'].find('CIFAR10') >=0:
        _, pred = torch.max(output.data, dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
      #test_loss += F.nll_loss(output, target, size_average=False).item()
      #pred = output.data.max(1, keepdim=True)[1]
      #correct += pred.eq(target.data.view_as(pred)).sum()
  #test_loss /= len(test_loader.dataset)
  test_loss = len(test_loader.dataset) - correct
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  if isinstance(correct, int):
    return correct
  else:
    return correct.item()

def GetLatestModelFile(gradFile, config, param):
  param['GlobalModel'].sort(key=SortEpochTime)
  LatestFile = os.path.join(config['Model.File'],param['GlobalModel'][-1])
  #LatestFile = os.path.join(config['Model.File'], gradFile.replace('gth','pth'))
  while not os.path.exists(LatestFile):
    time.sleep(3)
  return LatestFile

def CnnModelSelection(param):
  if param['Datasetname'].find('MNIST') >=0:
    ModelNetwork = MnistModel()
  elif param['Datasetname'].find('CIFAR') >=0:
    ModelNetwork = Cifar10Model()
  return ModelNetwork

def LoadLatestModel(tmpnetwork, gradFile , param, dev, config):
  #LatestFile = os.path.join(config['Model.File'], gradFile.replace('gth','pth'))
  LatestFile = GetLatestModelFile(gradFile, config, param)
  print('LoadLatestMode ', LatestFile, param['GlobalModel'].index(LatestFile.split('/')[-1]))
  if len(LatestFile) <= 0:
    return tmpnetwork
  Tmp = CnnModelSelection(param)
  sd = tmpnetwork.state_dict()
  #if dev.find('cuda') >= 0:
  #  Tmp.to(torch.device(dev))
  Tmp = torch.load(LatestFile, map_location = torch.device(dev))
  TmpSD = Tmp.state_dict()
  for key in TmpSD:
    sd[key] = TmpSD[key]
  tmpnetwork.load_state_dict(sd)
  if dev.find('cuda') >=0:
    tmpnetwork.to(torch.device(dev))
  #network = param['PytorchModel']
  return tmpnetwork

def LoadModelFromFile(network, File, param):
  print('LoadmodelFromFile ', File)
  sd = network.state_dict()
  Tmp = CnnModelSelection(param)
  if torch.cuda.is_available():
    dev = "cuda:0"
  else:
    dev = "cpu"
  if dev.find('cuda') >=0:
    Tmp.to(torch.device(dev))
  Tmp = torch.load(os.path.join(param['ModelFile'],File), map_location = torch.device(dev))
  TmpSD = Tmp.state_dict()
  for key in TmpSD:
    sd[key] = TmpSD[key]
  network.load_state_dict(sd)
  return network

def LoadGradientFromFile(network, File, param):
  print('LoadGradientFromFile', File)
  if torch.cuda.is_available():
    dev = "cuda:0"
  else:
    dev = "cpu"
  CheckPoint = torch.load(os.path.join(param['ModelFile'],File), map_location = torch.device(dev))
  TmpGrad = CheckPoint['grad']
  if dev.find('cuda') >=0:
    network.to(torch.device('cpu'))
  for m in network.named_parameters():
    if str(TmpGrad[m[0]].device).find('cuda') >=0:
      m[1].grad = TmpGrad[m[0]].to(torch.device('cpu'))
    else:
      m[1].grad = TmpGrad[m[0]]
    m[1].grad = m[1].grad
  if dev.find('cuda') >=0:
    network.to(torch.device(dev))
  return network

def GlobalModelCalculation( param, config):
  GlobalNetwork = CnnModelSelection(param)
  param['Gradient'].sort(key=SortEpochTime)
  for idx in range(len(param['Gradient'])):
    gradFile = param['Gradient'][idx]
    while not os.path.exists(os.path.join(config['Model.File'], gradFile)):
      time.sleep(1)
    if gradFile.replace('gth', 'pth') not in param['GlobalModel'] and gradFile not in param['ProcessedGradient']:
      ModelFile = param['GlobalModel'][idx]
      GlobalNetwork = LoadModelFromFile(GlobalNetwork, ModelFile, param)
      GlobalNetwork = LoadGradientFromFile(GlobalNetwork, gradFile ,param)
      if int(param['CurrentEpoch']) > 5:
        lr = float(param['LearningRate']) / float(param['CurrentEpoch'])
      else:
        lr = float(param['LearningRate'])
      for m in GlobalNetwork.named_parameters():
        m[1].data = m[1].data - (lr * m[1].grad/float(param['Nodes']))
      SaveModelFile(GlobalNetwork, param['Gradient'][idx].replace('gth', 'pth'), param)
      #param['PytorchModel'] = GlobalNetwork
      param['ProcessedGradient'].append(gradFile)
  return param

def SortEpochTime(fname):
  return int(fname.split('.')[0].split('-')[-1])

def ASGD(param, config, seq, ComputingObject):
  hostname = config['HostName']
  epochs = int(param['Epochs'])
  mini_epochs = int(param['MiniEpochs'])
  batch_size_train = int(param['TrainBatchSize'])
  batch_size_test = int(param['TestBatchSize'])
  learning_rate = float(param['LearningRate'])
  momentum = float(param['Momentum'])
  log_interval = int(param['LogInterval'])
  mini = int(param['MiniBatch'])
  param['State'] = 'AsgdTraining'
  #learning_rate = learning_rate * mini

  random_seed = int(param['RandomSeed'])
  torch.backends.cudnn.enabled = False
  torch.manual_seed(random_seed)

  if torch.cuda.is_available():
    dev = "cuda:0"
  else:
    dev = "cpu"
  if param['Datasetname'].find('MNIST') >=0:
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST(param['DatasetFile'], train=True, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
      batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST(param['DatasetFile'], train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
      batch_size=batch_size_test, shuffle=True)
  elif param['Datasetname'].find('CIFAR10') >=0:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(param['DatasetFile'], train=True, download=True,transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=batch_size_train, shuffle=True)
    testset = torchvision.datasets.CIFAR10(param['DatasetFile'], train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False)

  TotalBatchSize = math.ceil(len(train_loader.dataset)/ int(param['TrainBatchSize']))
  print('Start Network Define')
  network = CnnModelSelection(param)
  print('End of Network Define')
  if dev.find("cuda") >=0:
    print('use GPU')
    network.to(torch.device(dev))
  optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
  #scheduler = 
  #optimizer = optim.SGD(network.parameters(), lr=learning_rate)
  print('End merging')
  train_losses = []
  train_counter = []
  test_losses = []
  correct = []
  ModelFile = param['ProjectName'] + '-0-0.pth'
  SaveModelFile(network, ModelFile, param)
  for epoch in range(1, epochs + 1):
    param['CurrentEpoch'] = epoch
    network = ASGD_train(network, train_loader, optimizer , epoch, dev, log_interval, train_losses, train_counter, param, seq, mini, config, ComputingObject)
    TestResult = ASGD_test(network, param, test_loader, dev, test_losses)
    correct.append(int(TestResult))
    param[hostname + '-LocalCorrect'].append(TestResult)
  result = 'Finish'
  return 

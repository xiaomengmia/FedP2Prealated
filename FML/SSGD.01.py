import torch, os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import math, shutil
from CnnModel import *
import time

def train(network, train_loader, optimizer , epoch, dev, log_interval, train_losses, train_counter, param, seq, hostname, criterion):
  network.train()
  print('Start Training')
  CEpoch = param['CurrentEpoch']
  start = math.ceil(seq * len(train_loader.dataset) / (int(param['TrainBatchSize']) * int(param['Nodes'])))
  if seq + 1 == int(param['TrainBatchSize']):
    end = math.ceil(len(train_loader.dataset)/ int(param['TrainBatchSize']))
  else:
    end =  math.ceil((seq + 1) * len(train_loader.dataset) / (int(param['TrainBatchSize'])*int(param['Nodes'])))
  for batch_idx, (data, target) in enumerate(train_loader):
    if dev.find('cuda') >=0:
      data = data.to(dev)
      target = target.to(dev)
    if batch_idx >= start and batch_idx < end:
      optimizer.zero_grad()
      output = network(data)
      if param['Datasetname'].find('MNIST') >=0:
        loss = F.nll_loss(output, target)
      elif param['Datasetname'].find('CIFAR10') >=0:
        loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
          (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
  return network

def test(network, param, test_loader, dev, test_losses):
  network.eval()
  test_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    batch_losses = []
    for data in test_loader:
      images, labels = data
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
  test_loss = len(test_loader.dataset) - correct
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  return correct

def ModelMerge(network, param, dev, epoch):
  sd = network.state_dict()
  count = 0
  modelFileList = []
  for modelFile in param['Models']:
    FilePattern = modelFile.split('.')[0].split('-')
    if FilePattern[0].find(param['ProjectName']) >= 0 and int(FilePattern[2]) == int(param['CurrentEpoch']):
      modelFileList.append(modelFile)
  print('Merge Global Model frlm ' ,modelFileList)
  for modelFile in modelFileList:
    Tmp = CnnModelSelection(param)
    while not os.path.exists(os.path.join(param['ModelFile'], modelFile)):
      time.sleep(3)
    if dev.find("cuda") >=0:
      Tmp.to(torch.device(dev))
      Tmp = torch.load(os.path.join(param['ModelFile'], modelFile), map_location=torch.device(dev))
    else:
      Tmp = torch.load(os.path.join(param['ModelFile'], modelFile), map_location=torch.device('cpu'))
    TmpSD = Tmp.state_dict()
    for key in TmpSD:
      if count == 0:
        sd[key] = TmpSD[key]/int(param['Nodes'])
      else:
        sd[key] = sd[key] + TmpSD[key]/int(param['Nodes'])
    count = count + 1
  network.load_state_dict(sd)
  return network

def MergeModelGossip(network, param, dev, epoch):
  print('before MergeModelGossip', param)
  while len(param['Models']) < epoch * int(param['Nodes']):
  #while len(param['Models']) < int(param['Nodes']):
    time.sleep(20)
  print('after MergeModelGossip', param)
  network = ModelMerge(network, param, dev, epoch)
  return network

def MergeModelMQTT(network, param, dev, epoch, ComputingObject):
  # Node is parameter server
  if param['ParameterServer'].find(ComputingObject.GetLocalIP()) >=0:
    while len(param['Models']) < epoch * int(param['Nodes']):
      time.sleep(20)
  # Create model from local model collect
    network = ModelMerge(network, param, dev, epoch)
    GlobalModelFile = param['ProjectName'] + '-Global-' + str(epoch) + '.' + ComputingObject.config['HostName'] + '.pth'
    SaveModelFile(network, GlobalModelFile, param, 'Global')
    PushGlobalInfo(GlobalModelFile, param, ComputingObject)
  # Node is work node
  else:
    while len(param['GlobalModel']) < epoch:
      time.sleep(20)
  # Read latest GlobalModel
    network = LoadLatestModel(network, param, dev, ComputingObject.config)
  return network

def CnnModelSelection(param):
  if param['Datasetname'].find('MNIST') >=0:
    ModelNetwork = MnistModel()
  elif param['Datasetname'].find('CIFAR') >=0:
    ModelNetwork = Cifar10Model()
  return ModelNetwork

def DeviceSellection():
  if torch.cuda.is_available():
    dev = "cuda:0"
  else:
    dev = "cpu"
  return dev

def LoadLatestModel(tmpnetwork, param, dev, config):
  print('LoadLatestMode ', param['GlobalModel'][-1])
  LatestFile = os.path.join(param['ModelFile'], param['GlobalModel'][-1])
  while not os.path.exists(LatestFile):
    time.sleep(1)
  sd = tmpnetwork.state_dict()
  Tmp = torch.load(LatestFile, map_location = torch.device(dev))
  if dev.find('cuda') >= 0:
    Tmp.to(torch.device(dev))
  TmpSD = Tmp.state_dict()
  for key in TmpSD:
    sd[key] = TmpSD[key]
  tmpnetwork.load_state_dict(sd)
  if dev.find('cuda') >=0:
    tmpnetwork.to(torch.device(dev))
  #network = param['PytorchModel']
  return tmpnetwork

def SaveModelFile(network, ModelFile, param, ModelType):
  torch.save(network, os.path.join(param['ModelFile'], ModelFile))
  if ModelType.find('Global') >=0 and ModelFile.split('/')[-1] not in param['GlobalModel']:
    param['GlobalModel'].append(ModelFile.split('/')[-1])
  elif ModelType.find('Local') >=0 and ModelFile.split('/')[-1] not in param['Models']:
    param['Models'].append(ModelFile.split('/')[-1])
  return

def PushGlobalInfo(GlobalFile, param, ComputingObject):
  uri = 'Notify'
  Msg = {}
  Msg['Type'] = 'ModelInfo'
  Msg['ModelType'] = 'GlobalModel'
  Msg['ModelFile'] = GlobalFile
  Msg['GlobalMD5'] = ComputingObject.FileMd5(os.path.join(ComputingObject.config['Model.File'], GlobalFile))
  Msg['GlobalSize'] = os.path.getsize(os.path.join(ComputingObject.config['Model.File'], GlobalFile))
  ComputingObject.PushMessage(uri, Msg)
  return

def SSGD(param, config, seq, ComputingObject):
  n_epochs = int(param['Epochs'])
  batch_size_train = int(param['TrainBatchSize'])
  batch_size_test = int(param['TestBatchSize'])
  learning_rate = float(param['LearningRate'])
  momentum = float(param['Momentum'])
  log_interval = int(param['LogInterval'])
  random_seed = int(param['RandomSeed'])
  hostname = config['HostName']

  torch.backends.cudnn.enabled = False
  torch.manual_seed(random_seed)

  dev = DeviceSellection()

  if param['Datasetname'].find('MNIST') >=0:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307), (0.3081))])
    trainset = torchvision.datasets.MNIST(param['DatasetFile'], train=True, download=True,transform=transform)
    testset = torchvision.datasets.MNIST(param['DatasetFile'], train=False, download=True, transform=transform)
  elif param['Datasetname'].find('CIFAR10') >=0:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(param['DatasetFile'], train=True, download=True,transform=transform)
    testset = torchvision.datasets.CIFAR10(param['DatasetFile'], train=False, download=True, transform=transform)

  train_loader = torch.utils.data.DataLoader(trainset,batch_size=batch_size_train, shuffle=True)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False)

  TotalBatchSize = math.ceil(len(train_loader.dataset)/ int(param['TrainBatchSize']))

  train_losses = []
  test_losses = []
  train_counter = []
  correct = []

  print('Start Network Define')
  network = CnnModelSelection(param)
  print('End of Network Define')

  if dev.find("cuda") >=0:
    print('use GPU')
    network.to(torch.device(dev))
  network.weight_init()

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
  scheduler = StepLR(optimizer, step_size=2, gamma=0.01)

  shutil.move(param['File'], param['File'].replace('.submit','.train'))
  param['File'] = param['File'].replace('.submit','.train')
  param['State'] = 'train'
  for epoch in range(1, n_epochs + 1):
    print('Start of Epoch ', param)
    param['CurrentEpoch'] = epoch
    #scheduler.step()
    network = train(network, train_loader, optimizer , epoch, dev, log_interval, train_losses, train_counter, param, seq, hostname, criterion)
    TestResult = test(network, param, test_loader, dev, test_losses)
    param[hostname +'-LocalCorrect'].append(TestResult)
    correct.append(int(TestResult))
    localModelFile = os.path.join(param['ModelFile'], param['ProjectName'] + '-' + str(seq) + '-' + str(epoch)+ '.' + hostname + '.pth')
    SaveModelFile(network, localModelFile, param, 'Local')
    #param['Models'].append(localModelFile.split('/')[-1])
    ComputingObject.CommitLocalModel(param['ProjectName'], localModelFile.split('/')[-1], Notify = True)
    if config['Protocol.Default'].find('P2P') >= 0:
      network = MergeModelGossip(network, param, dev, epoch)
    elif config['Protocol.Default'].find('MQTT') >=0:
      network = MergeModelMQTT(network, param, dev, epoch, ComputingObject)
    GlobalAccuracy = test(network, param, test_loader, dev, test_losses)
    param[hostname + '-GlobalCorrect'].append(GlobalAccuracy)
    print('End of Epoch ', param)
  return

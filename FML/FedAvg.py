import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import math, shutil
from CnnModel import *
import time

class SSGD:

  def __init__(self):
    return

  def SSGD_Train(self, network, train_loader, optimizer , epoch, dev, log_interval, train_losses, train_counter, param, seq, hostname, criterion):
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

  def SSGD_Test(self, network, param, test_loader, dev, test_losses):
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
  
  def ModelMerge(self, network, param, dev, epoch, RequiredHostList):
    sd = network.state_dict()
    count = 0
    modelFileList = []
    for modelFile in param['Models']:
      FilePattern = modelFile.split('.')[0].split('-')
      if FilePattern[0].find(param['ProjectName']) >= 0 and int(FilePattern[2]) == int(param['CurrentEpoch']):
        modelFileList.append(modelFile)
    print('Merge Global Model frlm ' ,modelFileList)
    for modelFile in modelFileList:
      Tmp = self.CnnModelSelection(param)
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
  
  def MergeModelGossip(self, network, param, dev, epoch, RequiredHostList):
    print('before MergeModelGossip', param)
    while len(param['Models']) < epoch * int(param['Nodes']):
    #while len(param['Models']) < int(param['Nodes']):
      time.sleep(20)
    print('after MergeModelGossip', param)
    network = self.ModelMerge(network, param, dev, epoch, RequiredHostList)
    return network
  
  def MergeModelMQTT(self, network, param, dev, epoch, RequiredHostList):
    # Node is parameter server
    if param['ParameterServer'].find(self.GetLocalIP()) >=0:
      while len(param['Models']) < epoch * int(param['Nodes']):
        time.sleep(20)
    # Create model from local model collect
      network = self.ModelMerge(network, param, dev, epoch, RequiredHostList)
      GlobalModelFile = param['ProjectName'] + '-Global-' + str(epoch) + '.' + self.config['HostName'] + '.pth'
      self.SaveModelFile(network, GlobalModelFile, param, 'Global')
      self.PushGlobalInfo(GlobalModelFile, param)
    # Node is work node
    else:
      while len(param['GlobalModel']) < epoch:
        time.sleep(20)
    # Read latest GlobalModel
      network = self.LoadLatestModel(network, param, dev, self.config)
    return network
  
  def SSGD_Process(self, param, seq):
    n_epochs = int(param['Epochs'])
    batch_size_train = int(param['TrainBatchSize'])
    batch_size_test = int(param['TestBatchSize'])
    learning_rate = float(param['LearningRate'])
    momentum = float(param['Momentum'])
    log_interval = int(param['LogInterval'])
    random_seed = int(param['RandomSeed'])
    hostname = self.config['HostName']
  
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
  
    dev = self.DeviceSelection()
  
    train_loader = self.GetTrainData(param, True)
    test_loader = self.GetTestData(param, False)

    self.PushRecordCount(param)
  
    train_losses = []
    test_losses = []
    train_counter = []
    correct = []
  
    print('Start Network Define')
    network = self.CnnModelSelection(param)
    print('End of Network Define')
  
    if dev.find("cuda") >=0:
      print('use GPU')
      network.to(torch.device(dev))
    #network.weight_init()
  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.01)
  
    shutil.move(param['File'], param['File'].replace('.submit','.train'))
    param['File'] = param['File'].replace('.submit','.train')
    param['State'] = 'train'
    for epoch in range(1, n_epochs + 1):
      param['ReceivedModel'] = []
      RequiredHostList = self.RandomNodeSelection(param)
      #param[''] = ''
      print('Start of Epoch ', param, ' require these hosts ' , RequiredHostList)
      param['CurrentEpoch'] = epoch
      #scheduler.step()
      network = self.SSGD_Train(network, train_loader, optimizer , epoch, dev, log_interval, train_losses, train_counter, param, seq, hostname, criterion)
      TestResult = self.SSGD_Test(network, param, test_loader, dev, test_losses)
      param[hostname +'-LocalCorrect'].append(TestResult)
      correct.append(int(TestResult))
      localModelFile = os.path.join(param['ModelFile'], param['ProjectName'] + '-' + str(self.CurrentEpochTime()) + '-' + str(epoch)+ '.' + hostname + '.pth')
      self.SaveModelFile(network, localModelFile, param, 'Local')
      #param['Models'].append(localModelFile.split('/')[-1])
      self.CommitLocalModel(param['ProjectName'], localModelFile.split('/')[-1], Notify = True)
      if self.config['Protocol.Default'].find('P2P') >= 0:
        network = self.MergeModelGossip(network, param, dev, epoch, RequiredHostList)
      elif self.config['Protocol.Default'].find('MQTT') >=0:
        self.NetworkEngine.PublishFile(os.path.join(param['ModelFile'], localModelFile))
        network = self.MergeModelMQTT(network, param, dev, epoch, RequiredHostList)
      GlobalAccuracy = self.SSGD_Test(network, param, test_loader, dev, test_losses)
      param[hostname + '-GlobalCorrect'].append(GlobalAccuracy)
      print('End of Epoch ', param)
    return

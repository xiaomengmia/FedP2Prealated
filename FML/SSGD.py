import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import math, shutil
from CnnModel import *
import time, random

class SSGD:

  def __init__(self):
    return

  def SSGD_Train(self, network, train_loader, optimizer , epoch, dev, log_interval, train_losses, train_counter, param, hostname, criterion):
    seq = param['seq']
    network.train()
    print('Start Training')
    CEpoch = param['CurrentEpoch']
    for batch_idx, (data, target) in enumerate(train_loader):
      if dev.find('cuda') >=0:
        data = data.to(dev)
        target = target.to(dev)
      #if batch_idx >= start and batch_idx < end:
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
        train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
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
  
  def ModelMerge(self, network, param, FileList, dev):
    epoch = param['CurrentEpoch']
    sd = network.state_dict()
    count = 0
    #modelFileList = []
    #for modelFile in param['Models']:
    #  FilePattern = modelFile.split('.')[0].split('-')
    #  if FilePattern[0].find(param['ProjectName']) >= 0 and int(FilePattern[2]) == int(param['CurrentEpoch']):
    #    modelFileList.append(modelFile)
    print('Merge Global Model from ' ,FileList)
    for modelFile in FileList:
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
          #sd[key] = TmpSD[key]/int(param['Nodes'])
          sd[key] = TmpSD[key]/len(FileList)
        else:
          sd[key] = sd[key] + TmpSD[key]/len(FileList)
      count = count + 1
    network.load_state_dict(sd)
    return network
  
  def MergeModelGossip(self, param):
    dev = self.DeviceSelection()
    epoch = param['CurrentEpoch']
    RequiredHostList = param['RequiredHostList']
    print('before MergeModelGossip', param)
    ModelAvailable = False
    while ModelAvailable == False:
      FileList = []
      time.sleep(3)
      for modelfile in os.listdir(self.config['Model.File']):
        projectName = modelfile.split('-')[0]
        if modelfile not in param['Models'] and projectName.find(param['ProjectName']) >=0 and modelfile.find('Global') < 0:
          param['Models'].append(modelfile)
      availableRecord = 0
      for modelFile in param['Models']:
        FilePattern = modelFile.split('.')[0].split('-')
        ModelHostName = modelFile.split('.')[1]
        if FilePattern[0].find(param['ProjectName']) >= 0 and int(FilePattern[-1]) == int(epoch) and ModelHostName in list(RequiredHostList):
          print('File Available ', modelFile)
          FileList.append(modelFile)
      if len(FileList) >= len(RequiredHostList):
        print('All File Available ', FileList)
        ModelAvailable = True
    print('after MergeModelGossip', param)
    network = self.CnnModelSelection(param)
    network = self.ModelMerge(network, param, FileList, dev)
    GlobalModelName = param['ProjectName'] + '-Global-' + str(epoch) + '.' + self.config['HostName'] + '.pth'
    self.SaveModelFile(network, GlobalModelName, param, 'Global')
    return network
  
  def MergeModelMQTT(self, network, param, dev):
    epoch = param['CurrentEpoch']
    RequiredHostList = param['RequiredHostList']
    # Node is parameter server
    if param['ParameterServer'].find(self.GetLocalIP()) >=0:
      while len(param['Models']) < epoch * int(param['Nodes']):
        time.sleep(20)
    # Create model from local model collect
      network = self.ModelMerge(network, param, dev, RequiredHostList)
      GlobalModelFile = param['ProjectName'] + '-Global-' + str(epoch) + '.' + self.config['HostName'] + '.pth'
      self.SaveModelFile(network, GlobalModelFile, param, 'Global')
      #self.NetworkEngine.PublishFile(os.path.join(param['ModelFile'], GlobalModelFile))
      self.PushGlobalInfo(GlobalModelFile, param)
    # Node is work node
    else:
      while len(param['GlobalModel']) < epoch:
        time.sleep(20)
    # Read latest GlobalModel
      network = self.LoadLatestModel(network, param, dev, self.config)
    return network

  def SSGDGlobalModelCalculation(self, param):
    if self.config['Protocol.Default'].find('P2P') >= 0:
      self.MergeModelGossip(param)
    elif self.config['Protocol.Default'].find('MQTT') >= 0 and param['ParameterServer'].find(self.GetLocalIP()) >= 0:
      self.MergeModelGossip(param)
      GlobalModelFile = param['GlobalModel'][-1]
      self.PushGlobalInfo(GlobalModelFile, param)
    return
  
  def SSGD_Process(self, param):
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
      param['RequiredHostList'] = []
      param['RequiredHostList'] = self.RandomNodeSelection(param)
      print('Start of Epoch ', param, ' require these hosts ' , param['RequiredHostList'])
      param['CurrentEpoch'] = epoch
      tic = time.perf_counter()
      for ep in range(int(param['LocalEpoch'])):
        network = self.SSGD_Train(network, train_loader, optimizer , epoch, dev, log_interval, train_losses, train_counter, param, hostname, criterion)
      toc = time.perf_counter()
      param['EpochInfo'][self.config['HostName']]['Train'].append(toc - tic)
      # Communication Delay
      DelayTime = self.RandomCommunicationDelay(param)
      print('Sleep for ', str(DelayTime), ' to simulate delay')
      time.sleep(DelayTime)
      param['EpochInfo'][self.config['HostName']]['Delay'].append(DelayTime)
      # Test Local Model
      TestResult = self.SSGD_Test(network, param, test_loader, dev, test_losses)
      param[hostname +'-LocalCorrect'].append(TestResult)
      correct.append(int(TestResult))
      localModelFile = os.path.join(param['ModelFile'], param['ProjectName'] + '-' + str(self.CurrentEpochTime()) + '-' + str(epoch)+ '.' + hostname + '.pth')
      self.SaveModelFile(network, localModelFile, param, 'Local')
      self.CommitLocalModel(param['ProjectName'], localModelFile.split('/')[-1], Notify = True)
      tic = time.perf_counter()
      while len(param['GlobalModel']) < epoch:
        time.sleep(5)
      GlobalModelFile = param['GlobalModel'][-1]
      TmpNetwork = self.LoadModelFromFile(GlobalModelFile, param)
      TmpNetwork_dict = {x[0]:x[1].data for x in TmpNetwork.named_parameters()}
      for m in network.named_parameters():
        m[1].data = TmpNetwork_dict[m[0]]
      toc = time.perf_counter()
      param['EpochInfo'][self.config['HostName']]['Idle'].append(toc - tic)
      GlobalAccuracy = self.SSGD_Test(network, param, test_loader, dev, test_losses)
      param[hostname + '-GlobalCorrect'].append(GlobalAccuracy)
      print('End of Epoch ', param)
      if epoch == 1:
        param['EpochInfo'][self.config['HostName']]['TrainAcc'].append(param['EpochInfo'][self.config['HostName']]['Train'][-1])
        param['EpochInfo'][self.config['HostName']]['IdleAcc'].append(param['EpochInfo'][self.config['HostName']]['Idle'][-1])
        param['EpochInfo'][self.config['HostName']]['DelayAcc'].append(param['EpochInfo'][self.config['HostName']]['Delay'][-1])
      else:
        param['EpochInfo'][self.config['HostName']]['TrainAcc'].append(param['EpochInfo'][self.config['HostName']]['Train'][-1] + param['EpochInfo'][self.config['HostName']]['TrainAcc'][-1])
        param['EpochInfo'][self.config['HostName']]['IdleAcc'].append(param['EpochInfo'][self.config['HostName']]['Idle'][-1] + param['EpochInfo'][self.config['HostName']]['IdleAcc'][-1])
        param['EpochInfo'][self.config['HostName']]['DelayAcc'].append(param['EpochInfo'][self.config['HostName']]['Delay'][-1] + param['EpochInfo'][self.config['HostName']]['DelayAcc'][-1])
    param['EpochInfo'][self.config['HostName']]['Error'] = test_losses
    param['EpochInfo'][self.config['HostName']]['TrainAverage'] = sum(param['EpochInfo'][self.config['HostName']]['Train']) / len(param['EpochInfo'][self.config['HostName']]['Train'])
    param['EpochInfo'][self.config['HostName']]['IdleAverage'] = sum(param['EpochInfo'][self.config['HostName']]['Idle']) / len(param['EpochInfo'][self.config['HostName']]['Idle'])
    param['EpochInfo'][self.config['HostName']]['DelayAverage'] = sum(param['EpochInfo'][self.config['HostName']]['Delay']) / len(param['EpochInfo'][self.config['HostName']]['Delay'])
    param['EpochInfo'][self.config['HostName']]['GlobalCorrect'] = param[hostname + '-GlobalCorrect']
    param['EpochInfo'][self.config['HostName']]['LocalCorrect'] = param[hostname + '-LocalCorrect']
    self.ResultUpdate(param)
    return

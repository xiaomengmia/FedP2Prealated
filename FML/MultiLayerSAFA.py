import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, shutil
from CnnModel import *
import time

class SAFA:

  def __init__(self):
    return

  def Safa_Train(self, network, train_loader, optimizer , epoch, dev, log_interval, train_losses, train_counter, param, hostname, criterion):
    network.train()
    print('Start Training')
    CEpoch = param['CurrentEpoch']
    for batch_idx, (data, target) in enumerate(train_loader):
      if dev.find('cuda') >=0:
        data = data.to(dev)
        target = target.to(dev)
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
      if param['TrainSignal'] == False:
        return network
    return network
  
  def Safa_Test(self, network, param, test_loader, dev, test_losses):
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
  
  def SafaModelMerge(self, network, param, dev, fileList):
    sd = network.state_dict()
    count = 0
    MergeNodes = len(fileList)
    print('Merge Global Model from ', fileList, param['CurrentEpoch'])
    param['MergeFrom'][param['CurrentEpoch']] = fileList
    for modelFile in fileList:
      print('Model file name ', modelFile)
      while not os.path.exists(os.path.join(param['ModelFile'], modelFile)):
        time.sleep(3)
      Tmp = self.CnnModelSelection(param)
      if dev.find("cuda") >=0:
        Tmp.to(torch.device(dev))
        Tmp = torch.load(os.path.join(param['ModelFile'], modelFile), map_location=torch.device(dev))
      else:
        Tmp = torch.load(os.path.join(param['ModelFile'], modelFile), map_location=torch.device('cpu'))
      TmpSD = Tmp.state_dict()
      for key in TmpSD:
        if count == 0:
          sd[key] = TmpSD[key]/int(MergeNodes)
        else:
          sd[key] = sd[key] + TmpSD[key]/int(MergeNodes)
      count = count + 1
    network.load_state_dict(sd)
    return network
  
  def SafaMergeModelGossip(self, param):
    dev = self.DeviceSelection()
    if 'CurrentEpoch' not in param.keys():
      return
    Globalepoch = param['GlobalEpoch']
    availableRecord = 0
    FileList = []
    if 'TotalTrainNo' not in param.keys():
      return
    # Find current epoch file
    ModelAvailable = False
    while ModelAvailable == False:
      FileList = []
      time.sleep(3)
      for modelfile in os.listdir(self.config['Model.File']):
        projectName = modelfile.split('-')[0]
        if modelfile not in param['Models'] and projectName.find(param['ProjectName']) >=0 and modelfile.find('Global') < 0:
          param['Models'].append(modelfile)
      availableRecord = 0
      param['Models'].sort(key=self.SortEpochTime)
      for modelFile in param['Models']:
        FilePattern = modelFile.split('.')[0].split('-')
        if FilePattern[0].find(param['ProjectName']) >= 0 and int(FilePattern[-1]) == int(Globalepoch):
          hostname = modelFile.split('.')[-2]
          if hostname in param['Member']:
            availableRecord = availableRecord + param['Member'][hostname]
          else:
            availableRecord = availableRecord + param['TotalTrainNo']/int(param['Nodes'])
          FileList.append(modelFile)
        if availableRecord >= param['TotalTrainNo'] * float(param['c']):
          ModelAvailable = True
    print('MergeModelGossip ', FileList, availableRecord)
    param['TrainSignal'] = False
    TmpNetwork = self.CnnModelSelection(param)
    TmpNetwork = self.SafaModelMerge(TmpNetwork, param, dev, FileList)
    # Save Model to File
    GlobalModelName = param['ProjectName'] + '-Global-' + str(Globalepoch) + '.' + self.config['HostName'] + '.pth'
    self.SaveModelFile(TmpNetwork, GlobalModelName, param, 'Global')
    # PushGlobalInfo
    if self.config['Protocol.Default'].find('MQTT') >= 0:
      self.PushGlobalInfo(GlobalModelName, param)
    # Update 
    #param['GlobalModel'].append(GlobalModelName)
    param['GlobalEpoch'] = param['GlobalEpoch'] + 1
    return
  
  def SafaMergeModelMQTT(self, param):
    if param['ParameterServer'].find(self.GetLocalIP()) < 0:
      return
    self.SafaMergeModelGossip(param)
    return
  
  def SAFAGlobalModelCalculation(self, param):
    if self.config['Protocol.Default'].find('P2P') >= 0:
      self.SafaMergeModelGossip(param)
    elif self.config['Protocol.Default'].find('MQTT') >= 0:
      self.SafaMergeModelMQTT(param)
    return
  
  def SAFA_Process(self, param):
    param['CurrentEpoch'] = 1
    n_epochs = int(param['Epochs'])
    batch_size_train = int(param['TrainBatchSize'])
    batch_size_test = int(param['TestBatchSize'])
    learning_rate = float(param['LearningRate'])
    momentum = float(param['Momentum'])
    log_interval = int(param['LogInterval'])
    random_seed = int(param['RandomSeed'])
    hostname = self.config['HostName']
    seq = param['seq']
  
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
    network.weight_init()
  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
  
    shutil.move(param['File'], param['File'].replace('.submit','.train'))
    param['File'] = param['File'].replace('.submit','.train')
    param['State'] = 'train'
    param['GlobalEpoch'] = 1
    for epoch in range(1, n_epochs + 1):
      param['TrainSignal'] = True
      param['CurrentEpoch'] = epoch
      # Model Training
      network = self.Safa_Train(network, train_loader, optimizer , epoch, dev, log_interval, train_losses, train_counter, param, hostname, criterion)
      # Save Local Model
      Now = int(time.time())
      localModelFile = os.path.join(param['ModelFile'], param['ProjectName'] + '-' + str(Now) + '-'+ str(seq) + '-' + str(epoch)+ '.' + hostname + '.pth')
      param = self.SaveModelFile(network, localModelFile, param, 'Local')
      # Push Local Model Info
      self.CommitLocalModel(param['ProjectName'], localModelFile.split('/')[-1], Notify = True)
      # Local Model Testing
      TestResult = self.Safa_Test(network, param, test_loader, dev, test_losses)
      param[hostname +'-LocalCorrect'].append(TestResult)
      correct.append(int(TestResult))
      # Read Global Model from File
      while len(param['GlobalModel']) < epoch:
        time.sleep(5)
      GlobalModelFile = param['GlobalModel'][-1]
      TmpNetwork = self.LoadModelFromFile(GlobalModelFile, param)
      TmpNetwork_dict = {x[0]:x[1].data for x in TmpNetwork.named_parameters()}
      for m in network.named_parameters():
        m[1].data = TmpNetwork_dict[m[0]]
      GlobalAccuracy = self.Safa_Test(network, param, test_loader, dev, test_losses)
      param[hostname + '-GlobalCorrect'].append(GlobalAccuracy)
    return

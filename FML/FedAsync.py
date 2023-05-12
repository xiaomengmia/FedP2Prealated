import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import math, shutil
from CnnModel import *
from dla import *
from dla_simple import *
import time

class FedAsync:

  def __init__(self):
    return
#network, train_loader, optimizer, epoch, dev, log_interval, train_losses, train_counter, param, hostname, criterion)
  def FedAsync_Train(self, localnetwork, train_loader, optimizer , epoch, dev, log_interval, train_losses, train_counter, param, hostname, criterion):
    localnetwork.train()
    seq = param['seq']
    print('Start Training')
    CEpoch = param['CurrentEpoch']
    start = math.ceil(param['seq'] * len(train_loader.dataset) / (int(param['TrainBatchSize']) * int(param['Nodes'])))
    for batch_idx, (data, target) in enumerate(train_loader):
      if dev.find('cuda') >=0:
        data = data.to(dev)
        target = target.to(dev)
      optimizer.zero_grad()
      output = localnetwork(data)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
    return localnetwork
  
  def FedAsync_Test(self, network, param, test_loader, dev, test_losses):
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
  
  def FedAsyncGlobalModelMerge(self, param, localModelFile):
    localidx = param['Models'].index(localModelFile)
    globalidx = localidx
    tau = param['ModelsInfo'][localModelFile]
    # Calculate alpha_t, beta_t
    if param['function'].find('polynomial') >=0:
      s_t = pow(float(localidx - tau + 1), -1*float(param['a']))
    elif param['function'].find('hinge') >=0:
      if (localidx - tau) <= int(param['b']):
        s_t = 1
      else:
        s_t = 1/((float(param['a'])*(localidx-tau-float(param['b']))) + 1)
    alpha_t = float(param['alpha']) * s_t
    beta_t = 1 - alpha_t
    # Load PreviousGlobalModel
    PreviousGlobalModel = param['GlobalModel'][globalidx]
    # Load Local Model
    LocalNetwork = self.LoadModelFromFile(localModelFile, param)
    GlobalNetwork = self.LoadModelFromFile(PreviousGlobalModel, param)
    dev = self.DeviceSelection()
    if dev.find('cuda') >=0:
      GlobalNetwork = GlobalNetwork.to(torch.device(dev))
      LocalNetwork = LocalNetwork.to(torch.device(dev))
    print('FedAsyncGlobalModelMerge ', localidx, s_t, alpha_t, beta_t, tau)
    GlobalSD = GlobalNetwork.state_dict()
    LocalSD = LocalNetwork.state_dict()
    # Merge Global Model
    for key in GlobalSD:
      #GlobalSD[key] = alpha_t*(GlobalSD[key]) + beta_t*(LocalSD[key])
      GlobalSD[key] = beta_t*(GlobalSD[key]) + alpha_t*(LocalSD[key])
    GlobalNetwork.load_state_dict(GlobalSD)
    #LocalDic = {x[0]:x[1].data for x in LocalNetwork.named_parameters()}
    #for m in GlobalNetwork.named_parameters():
    #  m[1].data = beta_t*(m[1].data) + alpha_t*(LocalDic[m[0]])
    return GlobalNetwork
  
  def FedAsyncMergeModelGossip(self, param):
    param['Models'].sort(key = self.SortEpochTime)
    for modelFile in param['Models']:
      if modelFile not in param['ProcessedModel']:
        while not os.path.exists(os.path.join(self.config['Model.File'], modelFile)):
          time.sleep(1)
        GlobalModel = self.FedAsyncGlobalModelMerge(param, modelFile)
      #  GlobalModel.to(torch.device('cpu'))
        # Save Global Model
        FileSeq = len(param['GlobalModel'])
        GlobalFileName = param['ProjectName'] + '-Global-' + str(FileSeq) + '.' + self.config['HostName'] + '.pth'
        self.SaveModelFile(GlobalModel, GlobalFileName, param, 'Global')
        param['ProcessedModel'].append(modelFile)
    return
  
  def FedAsyncMergeModelMQTT(self, param):
    if param['ParameterServer'].find(self.GetLocalIP()) < 0:
      return
    self.SafaMergeModelGossip(param)
    return
  
  def SaveModelFileDic(self, network, ModelFile, param, ModelType):
    model_dic = {x[0]:x[1].data for x in network.named_parameters()}
    torch.save(model_dic, os.path.join(param['ModelFile'], ModelFile))
    if ModelType.find('Global') >=0 and ModelFile.split('/')[-1] not in param['GlobalModel']:
      param['GlobalModel'].append(ModelFile.split('/')[-1])
    elif ModelType.find('Local') >=0 and ModelFile.split('/')[-1] not in param['Models']:
      param['Models'].append(ModelFile.split('/')[-1])
    return param
  
  def PushGlobalInfo(self, GlobalFile, param):
    uri = 'Notify'
    Msg = {}
    Msg['Type'] = 'ModelInfo'
    Msg['ModelType'] = 'GlobalModel'
    Msg['ModelFile'] = GlobalFile
    Msg['GlobalMD5'] = self.FileMd5(os.path.join(self.config['Model.File'], GlobalFile))
    Msg['GlobalSize'] = os.path.getsize(os.path.join(self.config['Model.File'], GlobalFile))
    self.PushMessage(uri, Msg)
    return
  
  def PushLocalInfo(self, LocalFile, param, Tau):
    uri = 'Notify'
    Msg = {}
    Msg['Type'] = 'ModelInfo'
    Msg['Tau'] = Tau
    Msg['ModelType'] = 'LocalModel'
    Msg['ModelFile'] = LocalFile.split('/')[-1]
    Msg['Source'] = 'Origin'
    param['ModelsInfo'][Msg['ModelFile']] = Tau
    Msg['md5'] = self.FileMd5(os.path.join(self.config['Model.File'], LocalFile))
    Msg['size'] = os.path.getsize(os.path.join(self.config['Model.File'], LocalFile))
    self.PushMessage(uri, Msg)
    return
  
  def FedAsyncGlobalModelCalculation(self, param):
    if self.config['Protocol.Default'].find('P2P') >= 0:
      self.FedAsyncMergeModelGossip(param)
    elif self.config['Protocol.Default'].find('MQTT') >= 0:
      self.FedAsyncMergeModelMQTT(param)
    return
  
  def FedAsync_Process(self, param):
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
    #network.weight_init()
    print('End of Network Define')
  
    if dev.find("cuda") >=0:
      print('use GPU')
      network.to(torch.device(dev))
  
    Now = int(time.time())
    ModelFile = param['ProjectName'] + '-' + str(Now) + '-Global-0.' + hostname + '.pth'
    self.SaveModelFile(network, ModelFile, param, 'Global')
  
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
  
    shutil.move(param['File'], param['File'].replace('.submit','.train'))
    param['File'] = param['File'].replace('.submit','.train')
    param['State'] = 'train'
    param['GlobalEpoch'] = 1
    Tau = 0
    for epoch in range(1, n_epochs + 1):
      param['TrainSignal'] = True
      param['CurrentEpoch'] = epoch
      self.RandomNodeSelection(param)
    #  scheduler.step()
      # Model Training
      tic = time.perf_counter()
      for ep in range(int(param['LocalEpoch'])):
        network = self.FedAsync_Train(network, train_loader, optimizer, epoch, dev, log_interval, train_losses, train_counter, param, hostname, criterion)
      toc = time.perf_counter()
      param['EpochInfo'][self.config['HostName']]['Train'].append(toc - tic)
      # Local Model Testing
      TestResult = self.FedAsync_Test(network, param, test_loader, dev, test_losses)
      print('Local Model Test Result is ', TestResult)
      param[hostname +'-LocalCorrect'].append(TestResult)
      correct.append(int(TestResult))
      # Communication Delay Simulation
      DelayTime = self.RandomCommunicationDelay(param)
      print('Sleep for ', str(DelayTime), ' to simulate delay')
      time.sleep(DelayTime)
      # Save Local Model
      Now = int(time.time())
      localModelFile = os.path.join(param['ModelFile'], param['ProjectName'] + '-' + str(Now) + '-'+ str(seq) + '-' + str(epoch)+ '.' + hostname + '.pth')
      param = self.SaveModelFile(network, localModelFile, param, 'Local')
      LocalModelLength = len(param['Models'])
      param['EpochInfo'][self.config['HostName']]['Delay'].append(0)
      # Push FedAsync Local Model Info
      self.PushLocalInfo(localModelFile, param,Tau)
      # Load Latest Global Model
      while len(param['GlobalModel']) < LocalModelLength + 1:
        time.sleep(5)
      GlobalModelFile = param['GlobalModel'][-1]
      print('GlobalModel file name is ', GlobalModelFile)
      Tau = param['GlobalModel'].index(GlobalModelFile)
      #network = CnnModelSelection(param)
      TmpNetwork = self.LoadModelFromFile(GlobalModelFile, param)
      TmpNetwork_dict = {x[0]:x[1].data for x in TmpNetwork.named_parameters()}
      for m in network.named_parameters():
        m[1].data = TmpNetwork_dict[m[0]]
      GlobalAccuracy = self.FedAsync_Test(network, param, test_loader, dev, test_losses)
      param['EpochInfo'][self.config['HostName']]['Idle'].append(0)
      print('Global Model Test Result is ', GlobalAccuracy)
      param[hostname + '-GlobalCorrect'].append(GlobalAccuracy)
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

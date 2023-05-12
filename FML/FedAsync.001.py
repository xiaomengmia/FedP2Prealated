import torch, os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import math, shutil
from CnnModel import *
from dla import *
from dla_simple import *
import time

  def FedAsyncTrain(localnetwork, train_loader, optimizer , epoch, dev, log_interval, train_losses, train_counter, param, seq, hostname, criterion):
    localnetwork.train()
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
        output = localnetwork(data)
    #    if param['Datasetname'].find('MNIST') >=0:
    #      loss = F.nll_loss(output, target)
    #    elif param['Datasetname'].find('CIFAR10') >=0:
    #      loss = criterion(output, target)
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
    return localnetwork
  
  def FedAsynctest(network, param, test_loader, dev, test_losses):
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
  
  def FedAsyncGlobalModelMerge(param, localModelFile, ComputingObject):
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
    LocalNetwork = LoadModelFromFile(localModelFile, param)
    GlobalNetwork = LoadModelFromFile(PreviousGlobalModel, param)
    dev = DeviceSellection()
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
  
  def MergeModelGossip(param, ComputingObject):
    param['Models'].sort(key = SortEpochTime)
    for modelFile in param['Models']:
      if modelFile not in param['ProcessedModel']:
        while not os.path.exists(os.path.join(ComputingObject.config['Model.File'], modelFile)):
          time.sleep(1)
        GlobalModel = FedAsyncGlobalModelMerge(param, modelFile, ComputingObject)
      #  GlobalModel.to(torch.device('cpu'))
        # Save Global Model
        FileSeq = len(param['GlobalModel'])
        GlobalFileName = param['ProjectName'] + '-Global-' + str(FileSeq) + '.' + ComputingObject.config['HostName'] + '.pth'
        SaveModelFile(GlobalModel, GlobalFileName, param, 'Global')
        param['ProcessedModel'].append(modelFile)
    return
  
  def MergeModelMQTT(param, ComputingObject):
    # Node is parameter server
    availableRecord = 0
    FileList = []
    if param['ParameterServer'].find(ComputingObject.GetLocalIP()) >=0:
      while len(param['Models']) < epoch * int(param['Nodes']):
        time.sleep(3)
        param['Models'].sort(key=SortEpochTime)
        for modelFile in param['Models']:
          FilePattern = modelFile.split('.')[0].split('-')
          if FilePattern[0].find(param['ProjectName']) >= 0 and int(FilePattern[3]) == int(param['CurrentEpoch']):
            hostname = modelFile.split('.')[-2]
            availableRecord = availableRecord + param['Member'][hostname]
            FileList.append(modelFile)
      print('Available Records ', availableRecord)
    # Create model from local model collect
      network = ModelMerge(network, param, dev, FileList)
      GlobalModelFile = param['ProjectName'] + '-Global-' + str(epoch) + '.' + ComputingObject.config['HostName'] + '.pth'
      SaveModelFile(network, GlobalModelFile, param, 'Global')
      PushGlobalInfo(GlobalModelFile, param, ComputingObject)
      param['GlobalModel'].append(GlobalModelFile)
    # Node is work node
    return
  
  def SortEpochTime(fname):
    return int(fname.split('.')[0].split('-')[1])
  
  def CnnModelSelection(param):
    if param['Datasetname'].find('MNIST') >=0:
      ModelNetwork = MnistModel()
    elif param['Datasetname'].find('CIFAR10') >=0:
      ModelNetwork = Cifar10Model()
    elif param['Datasetname'].find('CIFAR10DLA') >=0:
      ModelNetwork = SimpleDLA()
    return ModelNetwork
  
  def DeviceSellection():
    if torch.cuda.is_available():
      dev = "cuda:0"
    else:
      dev = "cpu"
    return dev
  
  def SaveModelFileDic(network, ModelFile, param, ModelType):
    model_dic = {x[0]:x[1].data for x in network.named_parameters()}
    torch.save(model_dic, os.path.join(param['ModelFile'], ModelFile))
    if ModelType.find('Global') >=0 and ModelFile.split('/')[-1] not in param['GlobalModel']:
      param['GlobalModel'].append(ModelFile.split('/')[-1])
    elif ModelType.find('Local') >=0 and ModelFile.split('/')[-1] not in param['Models']:
      param['Models'].append(ModelFile.split('/')[-1])
    return param
  
  def SaveModelFile(network, ModelFile, param, ModelType):
    torch.save(network, os.path.join(param['ModelFile'], ModelFile))
    if ModelType.find('Global') >=0 and ModelFile.split('/')[-1] not in param['GlobalModel']:
      param['GlobalModel'].append(ModelFile.split('/')[-1])
    elif ModelType.find('Local') >=0 and ModelFile.split('/')[-1] not in param['Models']:
      param['Models'].append(ModelFile.split('/')[-1])
    return param
  
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
  
  def PushLocalInfo(LocalFile, param, Tau, ComputingObject):
    uri = 'Notify'
    Msg = {}
    Msg['Type'] = 'ModelInfo'
    Msg['Tau'] = Tau
    Msg['ModelType'] = 'LocalModel'
    Msg['ModelFile'] = LocalFile.split('/')[-1]
    Msg['Source'] = 'Origin'
    param['ModelsInfo'][Msg['ModelFile']] = Tau
    Msg['md5'] = ComputingObject.FileMd5(os.path.join(ComputingObject.config['Model.File'], LocalFile))
    Msg['size'] = os.path.getsize(os.path.join(ComputingObject.config['Model.File'], LocalFile))
    ComputingObject.PushMessage(uri, Msg)
    return
  
  def PushRecordCount(param, ComputingObject):
    uri = 'ProjectUpdate'
    Msg = {}
    Msg['Type'] = 'RecordUpdate'
    Msg['ProjectName'] = param['ProjectName']
    Msg['Field'] = 'Member'
    Msg['Value'] = param['Member']
    ComputingObject.PushMessage(uri, Msg)
    return
  
  def FedAsyncGlobalModelCalculation(param, ComputingObject):
    if ComputingObject.config['Protocol.Default'].find('P2P') >= 0:
      MergeModelGossip(param, ComputingObject)
    elif ComputingObject.config['Protocol.Default'].find('MQTT') >= 0:
      MergeModelMQTT(param, ComputingObject)
    return
  
  def LoadModelFromFile(File, param):
    print('LoadmodelFromFile ', File)
    if torch.cuda.is_available():
      dev = "cuda:0"
    else:
      dev = "cpu"
    Tmp = CnnModelSelection(param)
    Tmp.to(torch.device(dev))
    TmpSD = Tmp.state_dict()
    TmpGlobal = torch.load(os.path.join(param['ModelFile'],File), map_location = torch.device(dev))
    TmpGlobal.to(torch.device(dev))
    TmpGlobalsd = TmpGlobal.state_dict()
    for key in TmpGlobalsd:
      TmpSD[key] = TmpGlobalsd[key]
    Tmp.load_state_dict(TmpSD)
    #if dev.find('cuda') >=0:
    #  Tmp.to(torch.device(dev))
    return TmpGlobal
  
  def FedAsync(param, config, seq, ComputingObject):
    param['CurrentEpoch'] = 1
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
      #transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
      transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
      trainset = torchvision.datasets.CIFAR10(param['DatasetFile'], train=True, download=True,transform=transform)
      testset = torchvision.datasets.CIFAR10(param['DatasetFile'], train=False, download=True, transform=transform)
  
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False)
  
    TotalBatchSize = math.ceil(len(train_loader.dataset)/ int(param['TrainBatchSize']))
    param['TotalTrainNo'] = len(train_loader.dataset)
    param['TotalTestNo'] = len(test_loader.dataset)
    param['Member'][hostname] = len(train_loader.dataset)/int(param['Nodes'])
  
    PushRecordCount(param, ComputingObject)
  
    train_losses = []
    test_losses = []
    train_counter = []
    correct = []
  
    print('Start Network Define')
    network = CnnModelSelection(param)
    #network.weight_init()
    print('End of Network Define')
  
    if dev.find("cuda") >=0:
      print('use GPU')
      network.to(torch.device(dev))
  
    Now = int(time.time())
    ModelFile = param['ProjectName'] + '-' + str(Now) + '-Global-0.' + hostname + '.pth'
    SaveModelFile(network, ModelFile, param, 'Global')
  
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
    #  scheduler.step()
      # Model Training
      network = FedAsyncTrain(network, train_loader, optimizer, epoch, dev, log_interval, train_losses, train_counter, param, seq, hostname, criterion)
      # Local Model Testing
      TestResult = FedAsynctest(network, param, test_loader, dev, test_losses)
      print('Local Model Test Result is ', TestResult)
      param[hostname +'-LocalCorrect'].append(TestResult)
      correct.append(int(TestResult))
      # Save Local Model
      Now = int(time.time())
      localModelFile = os.path.join(param['ModelFile'], param['ProjectName'] + '-' + str(Now) + '-'+ str(seq) + '-' + str(epoch)+ '.' + hostname + '.pth')
      param = SaveModelFile(network, localModelFile, param, 'Local')
      LocalModelLength = len(param['Models'])
      # Push FedAsync Local Model Info
      PushLocalInfo(localModelFile, param,Tau, ComputingObject)
      # Load Latest Global Model
      while len(param['GlobalModel']) < LocalModelLength + 1:
        time.sleep(5)
      GlobalModelFile = param['GlobalModel'][-1]
      print('GlobalModel file name is ', GlobalModelFile)
      Tau = param['GlobalModel'].index(GlobalModelFile)
      #network = CnnModelSelection(param)
      network = LoadModelFromFile(GlobalModelFile, param)
      GlobalAccuracy = FedAsynctest(network, param, test_loader, dev, test_losses)
      print('Global Model Test Result is ', GlobalAccuracy)
      param[hostname + '-GlobalCorrect'].append(GlobalAccuracy)
    return

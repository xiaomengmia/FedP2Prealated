# -*- codinGg: utf-8 -*-
import sys, os
import datetime, time
from datetime import timedelta
from CommonLib.Common import *
from P2PEngineLib.P2PEngine import *
from MulticastEngineLib.MulticastEngine import *
from DownloadLib.Download import *
from GossipLib.Gossip import *
from ModelPushLib.ModelPush import *
from ModelPullLib.ModelPull import *
from MQTTEngineLib.MQTTEngine import *
from CnnModel import *
import torch
import torchvision
import random
from zipfile import ZipFile

class Training(object):

  def __init__(self):
    return

  def LoadLatestModel(self, tmpnetwork, param, dev, config):
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

  def SaveModelFile(self, network, ModelFile, param, ModelType):
    torch.save(network, os.path.join(param['ModelFile'], ModelFile))
    if ModelType.find('Global') >=0 and ModelFile.split('/')[-1] not in param['GlobalModel']:
      param['GlobalModel'].append(ModelFile.split('/')[-1])
    elif ModelType.find('Local') >=0 and ModelFile.split('/')[-1] not in param['Models']:
      param['Models'].append(ModelFile.split('/')[-1])
    return param

  def PushGlobalInfo(self,GlobalFile, param):
    uri = 'Notify'
    Msg = {}
    Msg['Type'] = 'ModelInfo'
    Msg['ModelType'] = 'GlobalModel'
    Msg['ModelFile'] = GlobalFile
    Msg['GlobalMD5'] = self.FileMd5(os.path.join(self.config['Model.File'], GlobalFile))
    Msg['GlobalSize'] = os.path.getsize(os.path.join(self.config['Model.File'], GlobalFile))
    Msg['Source'] = 'Origin'
    self.PushMessage(uri, Msg)
    return

  def ResultUpdate(self, param):
    uri = 'Notify'
    Msg = {}
    Msg['Type'] = 'ResultInfo'
    Msg['Source'] = 'Origin'
    Msg['Data'] = param
    self.PushMessage(uri, Msg)
    return

  def DeviceSelection(self):
    if torch.cuda.is_available():
      dev = "cuda:0"
    else:
      dev = "cpu"
    return dev

  def CnnModelSelection(self, param):
    if param['Datasetname'].find('MNIST') >=0:
      ModelNetwork = MnistModel()
    elif param['Datasetname'].find('CIFAR10LeNet') >=0:
      ModelNetwork = Cifar10Model()
    elif param['Datasetname'].find('CIFAR10AlexNet') >=0:
      ModelNetwork = AlexNet()
    return ModelNetwork

  def SortEpochTime(self, fname):
    return int(fname.split('.')[0].split('-')[1])

  def PushRecordCount(self, param):
    uri = 'ProjectUpdate'
    Msg = {}
    Msg['Type'] = 'RecordUpdate'
    Msg['ProjectName'] = param['ProjectName']
    Msg['Field'] = 'Member'
    Msg['Value'] = param['Member']
    self.PushMessage(uri, Msg)
    return

  def LoadModelFromFile(self, File, param):
    print('LoadmodelFromFile ', File)
    #while not File == 0:
    #  time.sleep(1)
    if torch.cuda.is_available():
      dev = "cuda:0"
    else:
      dev = "cpu"
    while not os.path.exists(os.path.join(param['ModelFile'],File)):
      time.sleep(1)
    Tmp = self.CnnModelSelection(param)
    Tmp.to(torch.device(dev))
    TmpSD = Tmp.state_dict()
    TmpGlobal = torch.load(os.path.join(param['ModelFile'],File), map_location = torch.device(dev))
    print(type(TmpGlobal))
    TmpGlobal.to(torch.device(dev))
    TmpGlobalsd = TmpGlobal.state_dict()
    for key in TmpGlobalsd:
      TmpSD[key] = TmpGlobalsd[key]
    Tmp.load_state_dict(TmpSD)
    #if dev.find('cuda') >=0:
    #  Tmp.to(torch.device(dev))
    return TmpGlobal

  def GetTrainData(self, param, Shuffle):
    seq = param['seq']
    if param['Datasetname'].find('MNIST') >=0:
      transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307), (0.3081))])
      trainset = torchvision.datasets.MNIST(param['DatasetFile'], train=True, download=True,transform=transform)
    elif param['Datasetname'].find('CIFAR10') >=0:
      transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
      trainset = torchvision.datasets.CIFAR10(param['DatasetFile'], train=True, download=True,transform=transform)
    param['TotalTrainNo'] = len(trainset)
    param['Member'][self.config['HostName']] = len(trainset)/int(param['Nodes'])
    start = math.ceil(seq * len(trainset) / (int(param['Nodes'])))
    if seq + 1 == int(param['Nodes']):
      end = math.ceil(len(trainset))
    else:
      end =  math.ceil((seq + 1) * len(trainset) / (int(param['Nodes'])))
    TrainList = list(range(start, end))
    print('Train Data Start at ', start, ' End at ', end, TrainList[0], len(TrainList))
    TrainSubset = torch.utils.data.Subset(trainset, TrainList)
    train_loader = torch.utils.data.DataLoader(TrainSubset, batch_size=int(param['TrainBatchSize']), shuffle=Shuffle)
    return train_loader

  def GetTestData(self, param, Shuffle):
    if param['Datasetname'].find('MNIST') >=0:
      transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307), (0.3081))])
      testset = torchvision.datasets.MNIST(param['DatasetFile'], train=False, download=True, transform=transform)
    elif param['Datasetname'].find('CIFAR10') >=0:
      transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
      testset = torchvision.datasets.CIFAR10(param['DatasetFile'], train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=int(param['TestBatchSize']), shuffle=Shuffle)
    param['TotalTestNo'] = len(test_loader.dataset)
    return test_loader

  def RandomNodeSelection(self, param):
    HostList = []
    if len(self.Nodes) == int(param['Nodes']):
      print('Member Info Available ', param['Member'])
      RequiredNodes = 0
      Records = 0
      RequiredRecords = param['TotalTrainNo'] * float(param['c'])
      for k, v in param['Member'].items():
        RequiredNodes = RequiredNodes + 1
        Records = Records + v
        if Records >= RequiredRecords:
          break
      print('Require Clients:', RequiredNodes)
      HostList = random.sample(list(self.Nodes.keys()), RequiredNodes)
      param['TrainRecords'] = Records
    else:
      print('Member info not Sufficient')
      HostList = list(param['Member'].keys())
    if self.config['HostName'] not in list(HostList):
      HostList[-1] = self.config['HostName']
    return HostList

  def ZipResult(self, TaskName):
    while len(self.ProjectInfo[TaskName]['EpochInfo']) < int(self.ProjectInfo[TaskName]['Nodes']):
      time.sleep(5)
      self.ResultUpdate(self.ProjectInfo[TaskName])
    BaseDir = self.config['Model.File'].replace('ModelFile','')
    # Output result to file
    resultTextFile = TaskName + '.txt'
    f = open(os.path.join(BaseDir, resultTextFile), 'w')
    for k, v in self.ProjectInfo[TaskName].items():
      f.write(str(k) + ' : ' + str(v) + '\n')
      if k.find('EpochInfo') >=0:
        f.write(str(k) + ':')
        for host, values in v.items():
          for valuekey, valueresult in values.items():
            f.write(str(host) + ',' + str(valuekey) + ':' + str(valueresult))
    f.close()
    resultZipFile = TaskName + str(int(time.time())) + '.zip'
    with ZipFile(os.path.join(BaseDir, resultZipFile), 'w') as zipObj:
      zipObj.write(os.path.join(BaseDir, resultTextFile))
      os.remove(os.path.join(BaseDir, resultTextFile))
      zipObj.write(self.ProjectInfo[TaskName]['File'])
      os.remove(self.ProjectInfo[TaskName]['File'])
      for modelfile in os.listdir(self.config['Model.File']):
        if modelfile.find(TaskName) >=0:
          zipObj.write(os.path.join(self.config['Model.File'], modelfile))
          os.remove(os.path.join(self.config['Model.File'], modelfile))
    zipObj.close()
    return

  def RandomCommunicationDelay(self, param):
    DelayTime = 0
    # Enable RandomDelay 
    population = [1,0]
    weight = [float(param['DelayProbability'])/100.0, 1 - float(param['DelayProbability'])/100.0]
    EnableDelay = random.choices(population, weight)
    if EnableDelay == 0:
      DelayTime = 0
    else:
      DelayTime = float(param['BaseDelay']) + random.uniform(-1*float(param['RandomDelay']), float(param['RandomDelay']))
    return DelayTime

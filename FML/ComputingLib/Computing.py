# -*- coding: utf-8 -*-
import sys, os, re
import datetime, time, shutil
from datetime import timedelta
from CommonLib.Common import *
from TrainingLib.Training import *
import threading
#import multiprocessing
import queue
from SSGD import *
from ASGD import *
from SAFA import *
from SAFA_MQTT import *
from FedAsync import *
from Fedasygossip import *
import torch.multiprocessing as mp

class Computing(Common, Training, SSGD, SAFA, FedAsync, SAFA_MQTT, FedAsyncGossip):

  def __init__(self, NetworkEngine):
    super(Computing, self).__init__()
    self.name = 'Computing'
    self.NetworkEngine = NetworkEngine
    self.Nodes = self.NetworkEngine.Nodes
    self.NodeInfo = self.NetworkEngine.NodeInfo
    self.config={}
    self.config = self.NetworkEngine.config
    self.ProjectInfo = self.NetworkEngine.ProjectInfo
    self.ScanProject()
    self.Task = False
    self.StartDaemon()
    return

  def ScanProject(self):
    print('Available Nodes ', self.Nodes)
    print('Node Resource Info ', self.NodeInfo)
    InDirFile = self.ScanDirForFiles(self.config['Project.File'])
    for ProFile in InDirFile:
      ProConfig = self.ReadProjectConfig(ProFile)
      if ProConfig['ProjectName'].find('error') >=0:
        return
      if re.match(pattern='.*pro$', string=ProFile):
        print('Available Project File is ', ProFile)
        #if ProConfig['ProjectName'] in self.ProjectInfo.keys():
        #  self.ProjectInfo[ProConfig['ProjectName']]['State'] = 'init'
        #  if ProFile.find('.init') <0:
        #    shutil.move(ProFile, ProFile + '.init')
        #  self.ProjectInfo[ProConfig['ProjectName']]['File'] = ProFile + '.init'
        #  return
        if ProConfig['ProjectName'] not in self.ProjectInfo.keys():
          print('ComputingEngine - New Project Available',ProConfig)
          self.ProjectInfo[ProConfig['ProjectName']] = ProConfig
          self.ProjectInfo[ProConfig['ProjectName']]['File'] = ProFile
          self.ProjectInfo[ProConfig['ProjectName']]['State'] = 'init'
          shutil.move(ProFile, ProFile + '.init')
          self.ProjectInfo[ProConfig['ProjectName']]['File'] = ProFile + '.init'
          self.ProjectInfo[ProConfig['ProjectName']]['Models'] = []
          self.ProjectInfo[ProConfig['ProjectName']]['ModelsInfo'] = {}
          self.ProjectInfo[ProConfig['ProjectName']][self.config['HostName'] + '-LocalCorrect'] = []
          self.ProjectInfo[ProConfig['ProjectName']][self.config['HostName'] + '-GlobalCorrect'] = []
          self.ProjectInfo[ProConfig['ProjectName']]['CurrentEpoch'] = 1
          #self.ProjectInfo[ProConfig['ProjectName']]['GlobalModel'] = []
          #self.ProjectInfo[ProConfig['ProjectName']]['GlobalModel'] = [[0,0] for i in range(Epoch)]
          self.ProjectInfo[ProConfig['ProjectName']]['GlobalEpoch'] = 1
          self.ProjectInfo[ProConfig['ProjectName']]['Gradient'] = []
          self.ProjectInfo[ProConfig['ProjectName']]['ProcessedGradient'] = []
          self.ProjectInfo[ProConfig['ProjectName']]['ProcessedModel'] = []
          self.ProjectInfo[ProConfig['ProjectName']]['RequiredHostList'] = []
          if 'Member' not in self.ProjectInfo[ProConfig['ProjectName']].keys():
            self.ProjectInfo[ProConfig['ProjectName']]['Member'] = {}
          self.ProjectInfo[ProConfig['ProjectName']]['MergeFrom'] = {}
        if self.config['Protocol.Default'].find('MQTT') >=0 and 'ParameterServer' not in self.ProjectInfo[ProConfig['ProjectName']].keys():
          self.ProjectInfo[ProConfig['ProjectName']]['ParameterServer'] = self.Nodes[self.config['HostName']]['IP']
        self.CommitLocalModel(ProConfig['ProjectName'], '', Notify = True)
        self.WriteEventLog(os.path.basename(__file__), 'A01 ' + ProConfig['ProjectName'] )
    return

  def ProjectKeyCheck(self, param):
    #ProjectKey = ['File', 'State', 'Models', 'Source', 'CurrentEpoch', 'GlobalModel', 'Gradient', 'ProcessedGradient', 'GlobalEpoch', self.config['HostName'] + '-LocalCorrect', self.config['HostName'] + '-GlobalCorrect', 'ProcessedGradient']
    ProjectKey = [ self.config['HostName'] + '-LocalCorrect', self.config['HostName'] + '-GlobalCorrect', 'ProcessedGradient', 'GlobalVersion', 'ProcessedModel' ]
    for key in ProjectKey:
      if key not in param.keys():
        param[key] = []
    if 'ModelsInfo' not in param.keys():
      param['ModelsInfo'] = {}
    param['EpochInfo'] = {}
    param['EpochInfo'][self.config['HostName']] = {}
    param['EpochInfo'][self.config['HostName']]['Train'] = []
    param['EpochInfo'][self.config['HostName']]['Idle'] = []
    param['EpochInfo'][self.config['HostName']]['TrainAcc'] = []
    param['EpochInfo'][self.config['HostName']]['IdleAcc'] = []
    param['EpochInfo'][self.config['HostName']]['Delay'] = []
    param['EpochInfo'][self.config['HostName']]['DelayAcc'] = []
    return

  def SubmitTask(self, TaskName):
    self.ProjectInfo[TaskName]['State'] = 'submit'
    try:
      shutil.move(self.ProjectInfo[TaskName]['File'], self.ProjectInfo[TaskName]['File'].replace('.init','.submit'))
    except:
      print('.init file not available')
    self.ProjectInfo[TaskName]['File'] = self.ProjectInfo[TaskName]['File'].replace('.init','.submit')
    self.ProjectInfo[TaskName]['GlobalModel'] = [[0, 0, 0] for i in range(int(self.ProjectInfo[TaskName]['Epochs']) + 1)]
    self.ProjectInfo[TaskName]['seq'] = self.SeqDecision()
    self.ProjectInfo[TaskName]['CurrentEpoch'] = 1
    print('task start', self.ProjectInfo[TaskName]['CurrentEpoch'])
    self.ProjectKeyCheck(self.ProjectInfo[TaskName])
    if self.ProjectInfo[TaskName]['Mode'].find('async') >= 0:
      Result = ASGD(self.ProjectInfo[TaskName])
    elif self.ProjectInfo[TaskName]['Mode'].find('safa') >=0 and self.config['Protocol.Default'].find('QMTT') >=0:
      Result = self.SAFA_MQTT_Process(self.ProjectInfo[TaskName])
    elif self.ProjectInfo[TaskName]['Mode'].find('safa') >=0:
      Result = self.SAFA_Process(self.ProjectInfo[TaskName])
    elif self.ProjectInfo[TaskName]['Mode'].find('FedAsync') >=0:
      Result = self.FedAsync_Process(self.ProjectInfo[TaskName])
    elif self.ProjectInfo[TaskName]['Mode'].find('AsyncGossip') >=0:
      Result = self.FedAsyncGossip_Process(self.ProjectInfo[TaskName])
    else:
      Result = self.SSGD_Process(self.ProjectInfo[TaskName])
    print('task end')
    #p.join()
    self.ProjectInfo[TaskName]['State'] = 'end'
    shutil.move(self.ProjectInfo[TaskName]['File'], self.ProjectInfo[TaskName]['File'].replace('.train','.end'))
    self.ProjectInfo[TaskName]['File'] = self.ProjectInfo[TaskName]['File'].replace('.train','.end')
    self.ZipResult(TaskName)
    del self.ProjectInfo[TaskName]
    return

  def SeqDecision(self):
    seq = 0
    print('SeqDecision ', self.NetworkEngine.Nodes)
    MyTime = self.NetworkEngine.Nodes[self.config['HostName']]['Time']
    for k, v in self.NetworkEngine.Nodes.items():
      if MyTime > v['Time']:
        seq = seq + 1
    return seq

  def CommitLocalModel(self, TaskName, ModelFile, Notify):
    print(self.NetworkEngine.Nodes.keys(), len(list(self.NetworkEngine.Nodes.keys())), 'Commit Local Model ', self.ProjectInfo[TaskName])
    if TaskName not in list(self.NetworkEngine.ProjectInfo.keys()):
      self.NetworkEngine.ProjectInfo[TaskName] = self.ProjectInfo[TaskName]
    if 'Models' not in list(self.NetworkEngine.ProjectInfo[TaskName].keys()):
      print('initial Models in ProjectInfo of NetworkEngine')
      self.NetworkEngine.ProjectInfo[TaskName]['Models'] = []
    if ModelFile != '' and ModelFile not in self.NetworkEngine.ProjectInfo[TaskName]['Models']:
      self.NetworkEngine.ProjectInfo[TaskName]['Models'].append(ModelFile)
    if len(list(self.NetworkEngine.Nodes.keys())) < 2:
      return
    if Notify == True and self.ProjectInfo[TaskName]['State'].find('init') >=0:
      uri = 'Notify'
      ProInfo = self.ProjectInfo[TaskName].copy()
      ProInfo['File'] = ProInfo['File'].replace('.init','')
      Msg = {}
      Msg['Type'] = 'ProjectInfo'
      Msg['Project'] = {TaskName : ProInfo}
      Msg['Source'] = 'Origin'
      ExeResult = self.PushMessage(uri, Msg)
      print('End of PushMessage in update Nodes info', ExeResult)
    elif Notify == True and self.ProjectInfo[TaskName]['State'].find('AsgdTraining') >=0:
      uri = 'Notify'
      Msg = {}
      Msg['Type'] = 'ModelInfo'
      Msg['ModelFile'] = ModelFile
      Msg['ModelType'] = 'GlobalModel'
      Msg['Source'] = 'Origin'
      Msg['md5'] = self.FileMd5(os.path.join(self.config['Model.File'], ModelFile))
      Msg['size'] = os.path.getsize(os.path.join(self.config['Model.File'], ModelFile))
      Msg[self.config['HostName'] + '-LocalCorrect'] = self.ProjectInfo[TaskName][self.config['HostName'] + '-LocalCorrect']
      ExeResult = self.PushMessage(uri, Msg)
      print('End of ASGD message sent', Msg)
    elif Notify == True and self.ProjectInfo[TaskName]['State'].find('train') >=0 :
      uri = 'Notify'
      Msg = {}
      Msg['Type'] = 'ModelInfo'
      Msg['ModelFile'] = ModelFile
      Msg['ModelType'] = 'LocalModel'
      Msg['Source'] = 'Origin'
      Msg['md5'] = self.FileMd5(os.path.join(self.config['Model.File'], ModelFile))
      Msg['size'] = os.path.getsize(os.path.join(self.config['Model.File'], ModelFile))
      if self.config['HostName'] + '-LocalCorrect' not in self.ProjectInfo[TaskName].keys():
        self.ProjectInfo[TaskName][self.config['HostName'] + '-LocalCorrect'] =[]
        self.ProjectInfo[TaskName][self.config['HostName'] + '-GlobalCorrect'] =[]
      Msg[self.config['HostName'] + '-LocalCorrect'] = self.ProjectInfo[TaskName][self.config['HostName'] + '-LocalCorrect']
      ExeResult = self.PushMessage(uri, Msg)
      print('End of PushMessage in update Nodes info', ExeResult)
    elif Notify == True and self.ProjectInfo[TaskName]['State'].find('Merging') >=0 :
      uri = 'Notify'
      Msg = {}
      Msg['Type'] = 'ModelInfo'
      Msg['ModelFile'] = ModelFile
      Msg['ModelType'] = 'LocalModel'
      Msg['Source'] = 'Origin'
      Msg['md5'] = self.FileMd5(os.path.join(self.config['Model.File'], ModelFile))
      Msg['size'] = os.path.getsize(os.path.join(self.config['Model.File'], ModelFile))
      Msg[self.config['HostName'] + '-LocalCorrect'] = self.ProjectInfo[TaskName][self.config['HostName'] + '-LocalCorrect']
      print(Msg)
      ExeResult = self.PushMessage(uri, Msg)
      print('End of PushMessage in update Nodes info', ExeResult)
    return

  def ScanModelFile(self, TaskName):
    FileList = []
    State = self.ProjectInfo[TaskName]['State']
    path = self.config['Model.File']
    for root, dirs, files in os.walk(path):
      for filename in files:
        if filename.find(TaskName) >=0 and filename.find('optimizer') <0:
          if State.find('localModel') >= 0 and filename.find(self.config['HostName']) >=0:
            FileList.append(filename)
          else:
            FileList.append(filename)
    return FileList

  def TaskSubmitThread(self):
    while(True):
      print('ComputingEngine Check if task available')
      print(self.ProjectInfo)
      if len(self.ProjectInfo) >0:
        for k in list(self.ProjectInfo.keys()):
          v = self.ProjectInfo[k]
          if v['State'].find('init') >=0 and v['File'].find(v['State']) < 0 and os.path.exists(os.path.join(self.config['Project.File'],v['File'])):
            print('Task ', k, ' need to submit in condition 1', v['File'])
            InitProjectFile = v['File'] + '.init'
            if os.path.exists(os.path.join(self.config['Project.File'],v['File'])):
              shutil.move(os.path.join(self.config['Project.File'],v['File']), os.path.join(self.config['Project.File'],InitProjectFile))
              v['File'] = InitProjectFile
              self.SubmitTask(k)
          if v['State'].find('init') >=0 and v['File'].find(v['State']) >=0 and os.path.exists(os.path.join(self.config['Project.File'],v['File'])):
            print('Task ', k, ' need to submit in condition 2')
            self.SubmitTask(k)
      time.sleep(int(self.config['Idle.Interval']))

  def ValidationThread(self):
    while(True):
      print('Global Model Validation')
      if len(self.ProjectInfo) > 0:
        for k, v in self.ProjectInfo.items():
          if v['Mode'].find('FedAsync') >=0 and v['State'].find('train') >=0:
            print('Test Model')
      time.sleep(int(self.config['Idle.Interval']))

  def GlobalModelThread(self):
    while(True):
      print('Calculate Global Model')
      if len(self.ProjectInfo) > 0:
        for k, v in self.ProjectInfo.items():
          if v['Mode'].find('async') >=0 and v['State'].find('train') >=0:
            v = GlobalModelCalculation(v, self.config)
          elif v['Mode'].find('safa') >=0 and v['State'].find('train') >=0:
            v = self.SAFAGlobalModelCalculation(v)
          elif v['Mode'].find('AsyncGossip') >=0 and v['State'].find('train') >=0:
            v = self.FedAsyncGossipGlobalModelCalculation(v)
          elif v['Mode'].find('FedAsync') >=0 and v['State'].find('train') >=0:
            v = self.FedAsyncGlobalModelCalculation(v)
          elif v['Mode'].find('sync') >=0 and v['State'].find('train') >=0:
            v = self.SSGDGlobalModelCalculation(v)
      time.sleep(int(self.config['Idle.Interval']))

  def DaemonTask(self):
    while(True):
      print('ComputingEngine')
      self.ScanProject()
      time.sleep(int(self.config['Idle.Interval']))

  def StartDaemon(self):
    t = threading.Thread(name='CompuingThread', target = self.DaemonTask)
    t.start()
    p = threading.Thread(name='TaskThread', target = self.TaskSubmitThread)
    p.start()
    r = threading.Thread(name='ValidationThread', target = self.ValidationThread)
    r.start()
    s = threading.Thread(name='GlobalModelThread', target = self.GlobalModelThread)
    s.start()

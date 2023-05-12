# -*- coding: utf-8 -*-
import sys, os, re
import datetime, time,shutil
from datetime import timedelta
from CommonLib.Common import *
import threading
import multiprocessing
from SSGD import *
from ASGD import *

class Computing(Common):

  def __init__(self, NetworkEngine):
    super(Computing, self).__init__()
    self.name = 'Computing'
    self.NetworkEngine = NetworkEngine
    self.Nodes = self.NetworkEngine.Nodes
    self.config={}
    self.config = self.NetworkEngine.config
    self.ProjectInfo = {}
    self.ScanProject()
    self.Task = False
    self.StartDaemon()
    return

  def ScanProject(self):
    InDirFile = self.ScanDirForFiles(self.config['Project.File'])
    for ProFile in InDirFile:
      ProConfig = self.ReadProjectConfig(ProFile)
      if re.match(pattern='.*pro$', string=ProFile):
        if ProConfig['ProjectName'] not in self.ProjectInfo.keys():
          print('ComputingEngine - New Project Available',ProConfig)
          self.ProjectInfo[ProConfig['ProjectName']] = ProConfig
          self.ProjectInfo[ProConfig['ProjectName']]['File'] = ProFile
          self.ProjectInfo[ProConfig['ProjectName']]['State'] = 'init'
          shutil.move(ProFile, ProFile + '.init')
          self.ProjectInfo[ProConfig['ProjectName']]['File'] = ProFile + '.init'
          self.ProjectInfo[ProConfig['ProjectName']]['Models'] = []
          self.ProjectInfo[ProConfig['ProjectName']][self.config['HostName'] + '-Correct'] = []
          self.ProjectInfo[ProConfig['ProjectName']][self.config['HostName'] + '-GlobalCorrect'] = []
          self.ProjectInfo[ProConfig['ProjectName']]['CurrentEpoch'] = []
          self.ProjectInfo[ProConfig['ProjectName']]['GlobalModel'] = []
          self.ProjectInfo[ProConfig['ProjectName']]['Gradient'] = []
          self.ProjectInfo[ProConfig['ProjectName']]['ProcessedGradient'] = []
        if self.config['Protocol.Default'].find('MQTT') >=0 and 'ParameterServer' not in self.ProjectInfo[ProConfig['ProjectName']].keys():
          self.ProjectInfo[ProConfig['ProjectName']]['ParameterServer'] = self.NetworkEngine.Nodes[self.config['HostName']]['IP']
        self.CommitLocalModel(ProConfig['ProjectName'], '', Notify = True)
        self.WriteEventLog(os.path.basename(__file__), 'A01 ' + ProConfig['ProjectName'] )
    return

  def SubmitTask(self, TaskName):
    self.ProjectInfo[TaskName]['State'] = 'submit'
    try:
      shutil.move(self.ProjectInfo[TaskName]['File'], self.ProjectInfo[TaskName]['File'].replace('.init','.submit'))
    except:
      print('.init file not available')
    self.ProjectInfo[TaskName]['File'] = self.ProjectInfo[TaskName]['File'].replace('.init','.submit')
    print('prepare queue')
    queue = multiprocessing.Queue()
    seq = self.SeqDecision()
    self.ProjectInfo[TaskName]['CurrentEpoch'] = 1
    print('task start', self.ProjectInfo[TaskName]['CurrentEpoch'])
    if self.ProjectInfo[TaskName]['Mode'].find('async') >= 0:
      print('In ASGD mode')
      Result = ASGD(queue, self.ProjectInfo[TaskName],seq, self.config, self)
      #self.ProjectInfo[TaskName][self.config['HostName'] + '-Correct'].extend(Result)
      self.ProjectInfo[TaskName]['State'] = 'end'
    else:
      Result = SSGD(queue, self.ProjectInfo[TaskName],seq, self.config['HostName'], self)
      self.ProjectInfo[TaskName][self.config['HostName'] + '-Correct'].extend(Result)
    self.WriteEventLog(os.path.basename(__file__), 'A02 ' + TaskName)
    print('task end')
    self.ProjectInfo[TaskName]['State'] = 'localModel'
    shutil.move(self.ProjectInfo[TaskName]['File'], self.ProjectInfo[TaskName]['File'].replace('.submit','.localModel'))
    self.ProjectInfo[TaskName]['File'] = self.ProjectInfo[TaskName]['File'].replace('.submit','.localModel')
    if 'Models' not in self.ProjectInfo[TaskName].keys():
      self.ProjectInfo[TaskName]['Models'] = [self.ProjectInfo[TaskName]['ProjectName'] + '-' + str(seq) + '-1' + '.' + self.config['HostName'] + '.pth']
    else:
      self.ProjectInfo[TaskName]['Models'].append(self.ProjectInfo[TaskName]['ProjectName'] + '-' + str(seq) + '-1' + '.' + self.config['HostName'] + '.pth')
    #print(queue.get())
    # Commit Local Model to P2P Engine
    if self.ProjectInfo[TaskName]['Mode'].find('async') < 0:
      self.CommitLocalModel(TaskName, self.ProjectInfo[TaskName]['ProjectName'] + '-' + str(seq) + '-1.' + self.config['HostName'] + '.pth', Notify = True)
    self.WriteEventLog(os.path.basename(__file__), 'A03 ' + TaskName)
    return

  def MergeTask(self, TaskName):
    #A Check if Model File Available
    self.ProjectInfo[TaskName]['CurrentEpoch'] = self.ProjectInfo[TaskName]['CurrentEpoch'] + 1
    CBatch = self.ProjectInfo[TaskName]['CurrentEpoch']
    modelCount = 0
    if self.config['Protocol.Default'].find('MQTT') <0:
      for modelFile in self.ProjectInfo[TaskName]['Models']:
        if os.path.exists(os.path.join(self.config['Model.File'], modelFile)):
          modelCount = modelCount + 1
      if modelCount < int(self.ProjectInfo[TaskName]['Nodes']):
        return
    # Start merge
    self.ProjectInfo[TaskName]['State'] = 'Merging'
    seq = self.SeqDecision()
    try:
      shutil.move(self.ProjectInfo[TaskName]['File'], self.ProjectInfo[TaskName]['File'].replace('.localModel','.mergingModel'))
    except:
      print('Error')
    queue = multiprocessing.Queue()
    if self.ProjectInfo[TaskName]['Mode'].find('async') >= 0:
      print('In ASGD mode')
      Result = ASGD(queue, self.ProjectInfo[TaskName],seq, self.config['HostName'])
    else:
      Result = SSGD(queue, self.ProjectInfo[TaskName], seq, self.config['HostName'], self)
    self.ProjectInfo[TaskName][self.config['HostName'] + '-Correct'].extend(Result)
    self.WriteEventLog(os.path.basename(__file__), 'A04 ' + TaskName)
    self.WriteEventLog(os.path.basename(__file__), 'A05 ' + TaskName)
    if self.ProjectInfo[TaskName]['CurrentEpoch'] < int(self.ProjectInfo[TaskName]['Epochs']):
      self.ProjectInfo[TaskName]['Models'].append(self.ProjectInfo[TaskName]['ProjectName'] + '-' + str(seq) + '-'+ str(self.ProjectInfo[TaskName]['CurrentEpoch']) + '.' + self.config['HostName'] + '.pth')
      self.CommitLocalModel(TaskName, self.ProjectInfo[TaskName]['ProjectName'] + '-' + str(seq) + '-' + str(self.ProjectInfo[TaskName]['CurrentEpoch']) + '.'+ self.config['HostName'] + '.pth', Notify = True)
    else:
      self.ProjectInfo[TaskName]['State'] = 'updatedModel'
      self.ProjectInfo[TaskName]['File'] = self.ProjectInfo[TaskName]['File'].replace('.mergingModel','.updatedModel')
    #print(queue.get())
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
      Msg[self.config['HostName'] + '-Correct'] = self.ProjectInfo[TaskName][self.config['HostName'] + '-Correct']
      ExeResult = self.PushMessage(uri, Msg)
      print('End of ASGD message sent', Msg)
    elif Notify == True and self.ProjectInfo[TaskName]['State'].find('localModel') >=0 :
      uri = 'Notify'
      Msg = {}
      Msg['Type'] = 'ModelInfo'
      Msg['ModelFile'] = ModelFile
      Msg['ModelType'] = 'LocalModel'
      Msg['Source'] = 'Origin'
      Msg['md5'] = self.FileMd5(os.path.join(self.config['Model.File'], ModelFile))
      Msg['size'] = os.path.getsize(os.path.join(self.config['Model.File'], ModelFile))
      Msg[self.config['HostName'] + '-Correct'] = self.ProjectInfo[TaskName][self.config['HostName'] + '-Correct']
      print(Msg)
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
      Msg[self.config['HostName'] + '-Correct'] = self.ProjectInfo[TaskName][self.config['HostName'] + '-Correct']
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
        for k, v in self.ProjectInfo.items():
          if v['State'].find('init') >=0:
            print('Task ', k, ' need to submit')
            self.SubmitTask(k)
          if v['State'].find('localModel') >=0 and len(v['Models']) == int(v['Nodes']) and v['Mode'].find('async')<=0:
            print('Model Merge')
            self.MergeTask(k)
          if v['State'].find('localModel') >=0 and len(v['GlobalModel']) >0:
            print('Model Merge for MQTT')
            self.MergeTask(k)
          elif v['State'].find('Merging') >=0 and len(v['Models']) == int(v['Nodes']) * int(v['CurrentEpoch']) and v['Mode'].find('async')<=0:
            self.MergeTask(k)
          if v['State'].find('Merging') >=0 and self.config['Protocol.Default'].find('MQTT') >=0:
            if v['ParameterServer'].find(self.GetLocalIP()) <0 and v['CurrentEpoch'] < int(v['Epochs']):
              while len(v['GlobalModel']) < (v['CurrentEpoch']-1):
                time.sleep(5)
              self.MergeTask(k)
            elif v['ParameterServer'].find(self.GetLocalIP()) >=0 and len(v['Models']) == int(v['Nodes']) * int(v['CurrentEpoch']): 
              self.MergeTask(k)
      time.sleep(int(self.config['Idle.Interval']))

  def GlobalModelThread(self):
    while(True):
      print('Calculate Global Model')
      if len(self.ProjectInfo) > 0:
        for k, v in self.ProjectInfo.items():
          v = GlobalModelCalculation(v, self.config)
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
    s = threading.Thread(name='GlobalModelThread', target = self.GlobalModelThread)
    s.start()

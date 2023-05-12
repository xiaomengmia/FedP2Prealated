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
import torch


class Networking(Common, MQTTEngine, P2PEngine, MulticastEngine, Download, Gossip, ModelPush, ModelPull):

  def __init__(self, path):
    super(Networking, self).__init__()
    self.name = 'Networking'
    self.config={}
    self.ReadConfig(path)
    self.Nodes = {}
    self.NodeInfo = {}
    self.Nodes[self.config['HostName']] = {}
    self.Models = {}
    if self.config['Protocol.Default'].find('P2P') >=0:
      self.NodeJoin()
      self.StartServer()
      self.StartMulticastServer()
    else:
      self.StartServer()
      self.StartMQTTClient()
      self.NodeJoin()
    #  self.PushSystemInfo()
    self.Commands = []
    self.LocalFile = {}
    self.AvailableFiles = {}
    self.ProjectInfo = {}
    self.PushSystemInfo()
    #self.DownloadDaemon()
    return

  def NodeJoin(self):
    if self.config['Protocol.Default'].find('P2P') >=0:
      Msg = 'Broadcast_' + self.config['HostName'] + '_' + self.config['P2P.IP'] + '_join\n'
      self.BroadcastMessage(Msg)
    cmd = ['Broadcast', self.config['HostName'], self.config['P2P.IP'], 'join']
    self.MulticastNodesUpdate(cmd)
    return

  def NodeLeave(self):
    Msg = 'Broadcast_' + self.config['HostName'] + '_' + self.config['P2P.IP'] + '_leave\n'
    if len(self.Nodes) > 1:
      self.BroadcastMessage(Msg)
    cmd = ['Broadcast', self.config['HostName'], self.config['P2P.IP'], 'leave']
    self.MulticastNodesUpdate(cmd)
    return

  def P2PNodesUpdate(self, data):
    HostsInfo = data['Hosts']
    for k, v in HostsInfo.items():
      if k not in self.Nodes.keys():
        self.Nodes[k] = v
    return

  def P2PFileUpdate(self, data, remote_info):
    Files = data['Files']
    print('Input file info ', Files)
    print('Current file info ', self.AvailableFiles)
    if len(Files) > 0:
      for k, v in Files.items():
        if os.path.exists(os.path.join(self.config['Model.File'], k)):
          continue
        if k not in self.AvailableFiles.keys():
          v['node'] = []
          v['node'].append(remote_info[0])
          self.AvailableFiles[k] = v
        else:
          if remote_info[0] not in self.AvailableFiles[k]['node']:
            self.AvailableFiles[k]['node'].append(remote_info[0])

  def ProjectFileUpdate(self, data):
    Projects = data['Project']
    if len(Projects) > 0:
      for k, v in Projects.items():
        if k in self.ProjectInfo.keys():
          return
        else:
          self.ProjectInfo[k] = v
        FileName = os.path.join(self.config['Project.File'],v['File'].split('/')[-1])
        SplitFileName = FileName.split('.')
        if SplitFileName[-1].find('pro') < 0:
          FileName = FileName.replace('.'+SplitFileName[-1], '')
        if not os.path.exists(FileName):
          f = open(FileName, 'a') 
          ValueList = v.keys()
          for valueKey in ValueList: 
            if (valueKey not in ['File', 'State', 'Models', 'Source', 'CurrentEpoch', 'GlobalModel', 'Gradient', 'ProcessedGradient', 'GlobalEpoch'] ) and (isinstance(v[valueKey], str)):
              f.write(valueKey+'='+v[valueKey]+'\n')
          f.close() 
    return

  def ProjectModelUpdate(self, FileList):
    if len(FileList) <=0:
      return
    for modelFile in FileList:
      try:
        projectName = modelFile.split('.')[0].split('-')[0]
        if modelFile.find('Global') >=0 and modelFile.split('/')[-1] not in list(self.ProjectInfo[projectName]['GlobalModel']):
          self.ProjectInfo[projectName]['GlobalModel'].append(modelFile)
        elif modelFile.split('/')[-1] not in list(self.ProjectInfo[projectName]['Models']):
          self.ProjectInfo[projectName]['Models'].append(modelFile)
      except:
        print('Error')
    return

  def ProjectValueUpdate(self, data):
    Field = data['Field']
    Value = data['Value']
    if Field.find('Member') >=0:
      for k, v in Value.items():
        if data['ProjectName'] not in self.ProjectInfo.keys():
          time.sleep(3)
        if Field not in self.ProjectInfo[data['ProjectName']].keys():
          self.ProjectInfo[data['ProjectName']][Field] = {}
        self.ProjectInfo[data['ProjectName']][Field][k] = int(v)
    uri = 'ProjectUpdate'
    #self.PushMessage(uri,data)
    return

  def ProjectModelVerify(self, data):
    projectName = data['ModelFile'].split('.')[0].split('-')[0]
    if data['ModelFile'] not in self.ProjectInfo[projectName]['Models']:
      OriginHostName = data['ModelFile'].split('.')[1]
      ip = self.Nodes[OriginHostName]['IP']
      self.Pull(ip, data)
      FileList = self.DownloadProcess()
      self.ProjectModelUpdate(FileList)
    return

  def GetGradientInfo(self, GradFile):
    projectName = GradFile.split('.')[0].split('-')[0]
    GradInfo = torch.load(os.path.join(self.ProjectInfo[projectName]['ModelFile'], GradFile))
    return GradInfo

  def GradientUpdate(self, data):
    projectName = data['Gradient'].split('.')[0].split('-')[0]
    if data['Gradient'] not in self.ProjectInfo[projectName]['Gradient']:
      self.ProjectInfo[projectName]['Gradient'].append(data['Gradient'])
    return

  def ProjectResultUpdate(self, data):
    ResultData = data['Data']
    projectName = ResultData['ProjectName']
    if projectName not in (self.ProjectInfo.keys()):
      return
    for k, v in ResultData.items():
      if k.find('-Correct') >= 0 and k not in self.ProjectInfo[projectName[k]]:
        self.ProjectInfo[projectName][k] = v
      if k.find('EpochInfo') >= 0:
        for host, value in v.items():
          if host not in list(self.ProjectInfo[projectName][k].keys()):
            self.ProjectInfo[projectName][k].update({host : value})
    return

  def PushSystemInfo(self):
    SysInfo = self.GetSystemInfo()
    uri = 'Notify'
    SysInfo['HostName'] = self.config['HostName']
    SysInfo['Type'] = 'SysInfo'
    self.NodeSystemInfoUpdate({self.config['HostName'] : SysInfo})
    print('PushMessage ', self.NodeInfo)
    self.PushMessage(uri, self.NodeInfo)
    return

  def NodeSystemInfoUpdate(self, data):
    print('NodeSystemInfoUpdate ', data)
    ChangeTag = False
    for k, v in data.items():
      if k not in self.NodeInfo.keys():
        self.NodeInfo[k] = {}
        ChangeTag = True
      for v_key, v_value in v.items():
        if v_key not in ['HostName', 'Type']:
          self.NodeInfo[k][v_key] = v_value
    #if ChangeTag == True:
    #  self.PushMessage('Notify', self.NodeInfo)
    return

  def MulticastNodesUpdate(self, cmd):
    print(cmd)
    if cmd[3].find('join') >=0 and cmd[2].find(self.config['P2P.IP']) < 0 and cmd[1]:
      print('Before Change NodesUpdate', self.Nodes, cmd)
      tmp = {}
      tmp['IP'] = cmd[2]
      tmp['Time'] = int(time.time())
      print(tmp)
      self.Nodes[cmd[1]] = tmp
      print('After Change NodesUpdate', self.Nodes)
      uri = 'Notify'
      ip = cmd[2]
      Msg = {}
      Msg['Type'] = 'NodesInfo'
      Msg['Hosts'] = self.Nodes
      print('NodesUpdate',Msg)
      # Update Hosts Info to New Node
      ExeResult = self.PushMessage(uri, Msg)
      print('End of PushMessage in update Nodes info', ExeResult)
      # Push Local File / Model Info to New Node
      Msg = {}
      Msg['Type'] = 'FileInfo'
      self.GetFile()
      Msg['Files'] = self.LocalFile
      ExeResult = self.PushMessage(uri, Msg)
      print('End of PushMessage in update LocalFile Info', ExeResult)
      Msg = {}
      Msg['Type'] = 'ProjectInfo'
      print('Broadcast Available Project, ', self.ProjectInfo)
      Msg['Project'] = self.ProjectInfo
      ExeResult = self.PushMessage(uri, Msg)
      print('End of PushMessage in update Project Info', ExeResult)
    elif cmd[3].find('join')>=0 and cmd[2].find(self.config['P2P.IP']) >=0:
      tmp = {}
      tmp['IP'] = cmd[2]
      tmp['Time'] = int(time.time())
      self.Nodes[cmd[1]] = tmp
    elif cmd[3].find('leave') >=0:
      try:
        del self.Nodes[cmd[1]]
      except:
        print('Node ', cmd[1], ' NA')
    return

  def DownloadTask(self):
    print('DownloadTask ')
    while(True):
      if len(self.AvailableFiles) > 0:
        print('FileFownloadEngine')
        print('Files need to download ', self.AvailableFiles)
        CompleteList = self.DownloadProcess()
        self.ProjectModelUpdate(CompleteList)
        time.sleep(int(self.config['Idle.Interval']))

  def DownloadDaemon(self):
    dl = threading.Thread(name='DownnloadThread', target = self.DownloadTask)
    dl.start()

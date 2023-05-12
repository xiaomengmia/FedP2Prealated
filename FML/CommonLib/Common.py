# -*- coding: utf-8 -*-
import sys, os
import json,getopt
import datetime, time
import signal
import random
import socket
import hashlib
import urllib.request
from datetime import timedelta
import math
import psutil
import platform
import uuid
import torch

class Common:

  def __init__(self):
    super(Common, self).__init__()
    self.Pid = str(os.getpid())
    self.StartTime = 0
    return

  def ReadConfig(self, Path):
    with open(Path, 'r') as ConFile:
      for line in ConFile:
        line = line.replace('\n', '')
        if line.find('#') <0 and len(line) > 0:
          line = line.split('=')
          self.config[line[0]] = line[1]
    self.config['HostName'] = socket.gethostname()
    if self.config['P2P.IP'].find('DHCP') >=0:
      IP_Address = self.GetLocalIP()
      self.config['P2P.IP'] = IP_Address
    return

  def bytesToGB(self, bytes):
    gb = bytes/(1024*1024*1024)
    gb = round(gb, 2)
    return gb

  def CurrentEpochTime(self):
    return int(time.time())

  def GetSystemInfo(self):
    uname = platform.uname()
    Mem = psutil.virtual_memory()
    SysInfo = {}
    SysInfo['UUID'] = int(uuid.uuid1().hex, base=16)
    SysInfo['Cpu'] = uname.processor
    SysInfo['CpuCore'] = psutil.cpu_count()
    SysInfo['CpuFrequency'] = psutil.cpu_freq()
    SysInfo['RAM'] = self.bytesToGB(Mem.available)
    if torch.cuda.is_available():
      dev = "gpu"
    else:
      dev = "cpu"
    SysInfo['GPU'] = dev
    SysInfo['ExtIp'] = self.GetExternalIP()
    SysInfo['IntIp'] = self.GetLocalIP()
    return SysInfo

  def GetLocalIP(self):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]

  def GetExternalIP(self):
    extIp = external_ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')
    return extIp

  def ReadProjectConfig(self, Path):
    tmp = {}
    try:
      with open(Path, 'r') as ConFile:
        for line in ConFile:
          line = line.replace('\n', '')
          if line.find('#') <0 and len(line) > 0:
            line = line.split('=')
            tmp[line[0]] = line[1]
    except:
      tmp['ProjectName'] = 'error'
    return tmp

  def WriteSysLog(self, Event):
    f = open(self.config['Sys.APLog'], 'a')
    CurrentTime = str(int(time.time()))
    CurrentTimeStr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    f.write(CurrentTime + ';' + CurrentTimeStr + ';' + Event + '\n')
    f.close()
    return

  def WriteEventLog(self, Module, Event):
    f = open(self.config['Sys.EventLog'], 'a')
    CurrentTime = str(int(time.time()))
    f.write(CurrentTime + ',' + Module + ',' + Event + '\n')
    f.close()
    return

  def IP2HostName(self, ip):
    hostName = ''
    for k, v in  self.Nodes.items():
      if v['IP'].find(ip) >=0:
        hostName = k
    return hostName

  def GetFile(self):
    for root, dirs, files in os.walk(self.config['Model.File']):
      for filename in files:
        tmp = {}
        tmp['md5'] = self.FileMd5(os.path.join(root, filename))
        tmp['size'] = os.path.getsize(os.path.join(root, filename))
        self.LocalFile[filename] = tmp
    return

  def GetLocalModel(self, data):
    epoch = data['Epoch']
    return

  def ScanDirForFiles(self, path):
    FileList = []
    for root, dirs, files in os.walk(path):
      for filename in files:
        FileList.append(os.path.join(path, filename))
    return FileList

  def FileMd5(self, filename):
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:
      for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    return hash_md5.hexdigest()

  def RandomPeerSelection(self, Hosts, Exclude):
    host = ''
    TmpHosts = []
    for host in Hosts:
      if host not in Exclude:
        TmpHosts.append(host)
    if len(TmpHosts) >=1:
      host = random.choices(TmpHosts, k=int(self.config['P2P.Fanout'])+1)
    else:
      host = random.choices(Exclude, k=1)
    if len(host) > 1:
      return host
    else:
      return host[0]

  def BroadcastMessage(self, Msg):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.sendto(bytes(Msg, 'utf-8'), (self.config['UDP.IP'], int(self.config['UDP.Port'])))

  def PushMessage(self, URI, Msg):
    if len(self.Nodes) == 0:
      PushRound = 1
    else:
      PushRound = math.log(len(self.Nodes), int(self.config['P2P.Fanout']))
    Round = 0
    ExcludeHost = [self.config['HostName']]
    if self.config['Protocol.Default'].find('P2P') >= 0:
      while Round <= (PushRound + 2)*2:
        peers = self.RandomPeerSelection(list(self.Nodes.keys()), ExcludeHost)
        print('Task Peer are ' , peers)
        if isinstance(peers, list):
          for peer in peers:
            if peer not in ExcludeHost:
              ip = self.Nodes[peer]['IP']
              self.PushMessageIP(URI, ip, Msg)
            else:
              if Round > 0:
                Round = Round -1
            ExcludeHost.append(peer)
        else:
          if peers not in ExcludeHost:
            ip = self.Nodes[peers]['IP']
            self.PushMessageIP(URI, ip, Msg)
          ExcludeHost.append(peers)
        Round = Round + 1
    elif self.config['Protocol.Default'].find('MQTT') >= 0:
      self.PushMessageMQTT(Msg)
    return

  def PushMessageMQTT(self, Msg):
    print('%d Common.PushMessageMQTT from %s with content %s' % (int(time.time()), self.config['HostName'],str(Msg)))
    self.WriteSysLog('Common.PushMessageMQTT from %s with content %s' % (self.config['HostName'],str(Msg)))
    if self.name.find('Networking') >=0:
      self.PublishMsg(json.dumps(Msg).encode('utf-8'))
    else:
      self.NetworkEngine.PublishMsg(json.dumps(Msg).encode('utf-8'))
    return

  def PushMessageIP(self, URI, host, Msg):
    API = "%s://%s:%s/%s" % (self.config['P2P.protocol'], host, self.config['P2P.Port'], URI)
    print('%d Common.PushMessageIP from host %s to URI %s with content %s' % (int(time.time()), self.config['HostName'], API, str(Msg)))
    self.WriteSysLog('Common.PushMessageIP from host %s to URI %s with content %s' % (self.config['HostName'], API, str(Msg)))
    try:
      req = urllib.request.Request(API)
      data = json.dumps(Msg).encode('utf-8')
      req.add_header('Content-Type', 'application/json')
      req.add_header('Content-Length', len(data))
      response = urllib.request.urlopen(req, data = data, timeout = int(self.config['Socket.Timeout'])*4)
      Content = response.read()
      Result = {'code' : 200, 'reason': 'OK', 'content' : Content}
    except urllib.error.HTTPError as e:
      Result = {'code':'fail', 'reason':e, 'content':''}
    except Exception as e:
      Result = {'code' : 'fail', 'reason':e, 'content':''}
    except socket.timeout as e:
      Result = {'code' : 'fail', 'reason':e, 'content':''}
    print('%d Common.PushMessageIP from host %s to URI %s with result %s' % (int(time.time()), self.config['HostName'], API, Result))
    self.WriteSysLog('Common.PushMessageIP from host %s to URI %s with result %s' % (self.config['HostName'], API, Result))
    return Result

  def ProcessStop(self, signum = None, Frame = None):
    self.NodeLeave()
    cmd = 'kill -9 ' + self.Pid
    os.system(cmd)
    return

  def GetCommand(self, signum = None, frame = None):
    fp = open(self.config['Command.File'], 'r+')
    commands = fp.readlines()
    for cmd in commands:
      cmd = cmd.replace('\n', '').replace("b'","").replace("'","")
      self.Commands.append(cmd)
    self.CommandProcess()
    fp.truncate(0)
    fp.close()
    return

  def CommandProcess(self):
    if len(self.Commands) > 0:
      for cmd in self.Commands:
        print('CommandProcess ', cmd)
        cmd_set = cmd.split('_')
        if len(cmd_set) > 2:
          if (cmd_set[3].find('join') >=0 or cmd_set[3].find('leave')>=0) and cmd_set[1].find(self.config['HostName']) < 0 :
            self.MulticastNodesUpdate(cmd_set)
        else:
          print(cmd_set)
    self.Commands = []
    return

  def ParameterServerCheck(self, data):
    return

def GetOpts(argv):
  conFile = ''
  if len(argv) == 0:
    print('Input required')
    sys.exit(2)
  try:
    opts, args = getopt.getopt(argv, "hc:", ["configFile="])
  except getopt.GetoptError:
    print('P2PCNN.py -c <configfile>')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print('P2PCNN.py -c <configfile>')
      sys.exit()
    elif opt in ("-c", "--configFile"):
      conFile = arg
  return conFile

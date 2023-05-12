# -*- coding: utf-8 -*-
import paho.mqtt.client as mqtt
import threading, datetime, time
from queue import Queue
import json, shutil, os
import uuid

class MQTTEngine(object):

  def __init__(self):
    super(MQTTEngine, self).__init__()
    self.ConnectionStatus = False
    self.QueuedMessage = []
    self.ReceivedMessage = []
    return

  def on_connect(self, client, userdata, flags, rc):
    print('Broker connected')
    #self.client.subscribe(self.config['MQTT.Subscribe'])
    return

  def on_message(self, client, userdata, msg):
    self.WriteSysLog('MQTT on_message from topic %s with content %s' % (msg.topic,str(msg.payload)))
    if (msg.topic.split('/')[-2].find('Global') >=0 or msg.topic.split('/')[-2].find('Local') >= 0):
      fileName = msg.topic.split('/')[-1]
      projectName = fileName.split('.')[0].split('-')[0]
      if self.ProjectInfo[projectName]['ParameterServer'].find(self.GetLocalIP()) >= 0:
        print('ModelFile available  ', fileName)
        TmpFile = '/opt/warwick/Tmp/' + fileName
        with open(TmpFile, 'wb') as fd:
          fd.write(msg.payload)
        fd.close()
      # Move File
        shutil.move(TmpFile, os.path.join(self.ProjectInfo[projectName]['ModelFile'] ,fileName))
      # Download File
        if ModelFile not in self.ProjectInfo[projectName]['Models']:
          self.ProjectInfo[projectName]['Models'].append(ModelFile)
        elif self.ProjectInfo[projectName]['ParameterServer'].find(self.GetLocalIP()) < 0 and msg.topic.split('/')[-2].find('Global') >=0:
          print('ModelFile available  ', fileName)
          TmpFile = '/opt/warwick/Tmp/' + fileName
          with open(TmpFile, 'wb') as fd:
            fd.write(msg.payload)
          fd.close()
      ### Move File
        shutil.move(TmpFile, os.path.join(self.ProjectInfo[projectName]['ModelFile'] ,fileName))
      ### Download File
        if ModelFile not in self.ProjectInfo[projectName]['GlobalModel']:
          self.ProjectInfo[projectName]['GlobalModel'].append(ModelFile)
      return
    if str(msg.topic).find(self.config['HostName']) >=0:
      return
    print('%d MQTT on_message from topic %s with content %s' % (int(time.time()), msg.topic, str(msg.payload)))
    Msg = str(msg.payload).replace('\n', '').replace("b'","").replace("'","").replace('\\','')
    print('Submit MQTT Message ', Msg)
    if Msg.find('ProjectInfo') >=0:
      data = json.loads(Msg)
      self.ProjectFileUpdate(data)
    if Msg.find('_join')>=0 or Msg.find('_leave') >=0:
      cmd = Msg.split('_')
      self.MulticastNodesUpdate(cmd)
    if Msg.find('NodesInfo') >=0:
      data = json.loads(Msg)
      self.P2PNodesUpdate(data)
    if Msg.find('ProjectUpdate') >=0:
      data = json.loads(Msg)
    if Msg.find('UUID') >= 0:
      data = json.loads(Msg)
      self.NodeSystemInfoUpdate(data)
    if Msg.find('RecordUpdate') >=0:
      data = json.loads(Msg)
      self.ProjectValueUpdate(data)
    if Msg.find('ResultInfo') >=0:
      data = json.loads(Msg)
      self.server.Parent.ProjectResultUpdate(data)
    if Msg.find('ModelInfo') >=0:
      SourceNode = msg.topic.split('/')[1]
      ip = self.Nodes[SourceNode]['IP']
      data = json.loads(Msg)
      projectName = data['ModelFile'].split('.')[0].split('-')[0]
      self.ParameterServerCheck(data)
      if data['ModelType'].find('GlobalModel') >=0:
        self.PullGlobal(ip, data)
        if data['ModelFile'] not in self.ProjectInfo[projectName]['GlobalModel']:
          self.ProjectInfo[projectName]['GlobalModel'].append(data["ModelFile"])
      if 'ParameterServer' in self.ProjectInfo[projectName].keys():
        # Check Global Server Info
        # Get File Info
        modelName = data['ModelFile']
        epoch = data['ModelFile'].split('.')[0].split('-')[-1]
        hostName = data['ModelFile'].split('.')[-2]
        # Check if local Model has been created in this epoch
        if self.ProjectInfo[projectName]['ParameterServer'].find(self.GetLocalIP()) >= 0:
          self.Pull(ip, data)
      ##if self.config['Protocol.Default'].find('P2P') >=0:
      ##  self.Pull(ip, data)
    print('%d MQTT on_message from topic %s finish' % (int(time.time()), msg.topic))
    self.WriteSysLog('MQTT on_message from topic %s finish' % (msg.topic))
    return

  def on_publish(self, client, userdata, mid):
    return

  def on_disconnect(self, client, userdata, rc):
    if rc != 0:
      print("unexpected disconnect from broker")
      self.ConnectionStatus = False
    return

  def PublishFile(self, ModelFilePath):
    ModelType = ''
    if ModelFilePath.find('Global') >= 0:
      ModelType = 'GlobalModel'
    else:
      ModelType = 'LocalModel'
    modelFile = open(ModelFilePath,'rb')
    modelfile = modelFile.read()
    modelArray = bytes(modelfile)
    ModelFileName = ModelFilePath.split('/')[-1]
    ret = self.client.publish(self.config['EdgeTopic'] +'/'+self.config['HostName'] + '/' + ModelType + '/' + ModelFileName, modelArray)
    return

  def StartMQTTClient(self):
    self.client = mqtt.Client()
    self.client.on_connect = self.on_connect
    self.client.on_message = self.on_message
    self.client.on_publish = self.on_publish
    self.client.on_disconnect = self.on_disconnect
    self.client.username_pw_set(self.config['MQTT.username'], self.config['MQTT.password'])
    t = threading.Thread(name='MQTT Thread', target = self.PeriodicalDialog)
    t.start()
    return

  def ConnectBroker(self):
    try:
      self.client.connect(self.config['MQTT.Server'], int(self.config['MQTT.Port']), int(self.config['MQTT.keepalive']))
      # Subscribe to Global Topic
      self.client.subscribe("#")
      #self.client.subscribe(self.config['MQTT.Subscribe'] + '/#')
      # Subscribe to Edge Topic
      LastDigit = int(uuid.uuid5(uuid.NAMESPACE_DNS, self.config['HostName']).hex,base=16 )%10
      if (LastDigit % 2) == 0:
        self.config['EdgeTopic'] = self.config['MQTT.Subscribe'] + 'EdgeEven'
      else:
        self.config['EdgeTopic'] = self.config['MQTT.Subscribe'] + 'EdgeOdd'
      self.client.loop_start()
      self.ConnectionStatus = True
      #self.client.subscribe([(self.config['EdgeTopic'],0), (self.config['MQTT.Subscribe'] ,0)])
    except:
      print('MQTT Connection Error Occured')
    if self.ConnectionStatus == True:
      msg = 'MQTT_' + self.config['HostName'] + '_' + self.config['P2P.IP'] + '_join\n'
      self.client.publish(self.config['MQTT.Subscribe']+'/'+self.config['HostName'], msg)
    return

  def DisconnectBroker(self):
    self.client.disconnect()

  def PublishMsg(self, msg):
    while self.ConnectionStatus == False:
      time.sleep(2)
    if self.ConnectionStatus == True:
      ret = self.client.publish(self.config['MQTT.Subscribe']+'/'+self.config['HostName'], msg)
      #ret = self.client.publish(self.config['EdgeTopic']+'/'+self.config['HostName'], msg)
      return ret
    else:
      return 'Not Connect'

  def PeriodicalDialog(self):
    while(True):
      CurrentTime = datetime.datetime.now()
      print('MQTT Client', CurrentTime, self.ConnectionStatus)
      if self.ConnectionStatus == True:
        msg = self.config['HostName'] + ' ' + CurrentTime.strftime("%Y-%m-%d %H:%M:%S")
        ret = self.PublishMsg(msg)
        print('Publish result is ', ret)
      else:
        self.ConnectBroker()
      time.sleep(int(self.config['Idle.Interval']))

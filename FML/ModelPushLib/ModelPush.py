# -*- coding: utf-8 -*-
import sys, os
import datetime, time

class ModelPush(object):

  def __init__(self):
    super(ModelPush, self).__init__()
    return

  def Push(self, ModelFile):
    return

  def ProjectRelay(self, ip, data):
    data['Source'] = 'Relay'
    uri = 'Notify'
    OriginHostName = self.IP2HostName(ip)
    #peer = self.RandomPeerSelection(self.Nodes.keys(), [self.config['HostName'], OriginHostName])
    #ip = self.Nodes[peer]['IP']
    ExeResult = self.PushMessage(uri, data)
    print('End of PushMessage in Relay Project Info', ExeResult)
    return

  def Relay(self, data):
    data['Source'] = 'Relay'
    uri = 'Notify'
    OriginHostName = data['ModelFile'].split('.')[1]
    #peer = self.RandomPeerSelection(self.Nodes.keys(), [self.config['HostName'], OriginHostName])
    #ip = self.Nodes[peer]['IP']
    ExeResult = self.PushMessage(uri, data)
    print('End of PushMessage in update Nodes info', ExeResult)
    return

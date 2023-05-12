import sys
import socketserver
import threading
import socket
from sys import platform
from MulticastHandlerLib.MulticastHandler import MulticastHandler

class MulticastEngine(object):

  def __init__(self):
    super(MulticastEngine, self).__init__()

  def StartMulticastServer(self):
    print('Start UDP Server')
    thread = threading.Thread(target=self.MulticastServerRun, args=())
    thread.daemon = True
    thread.start()

  def MulticastServerRun(self):
    if platform == 'darwin':
      self.UDPServer = MServer(("", int(self.config['UDP.Port'])), MulticastHandler, self, bind_and_activate=True)
    else:
      self.UDPServer = MServer((self.config['UDP.IP'], int(self.config['UDP.Port'])), MulticastHandler, self, bind_and_activate=True) 
    while True:
      self.UDPServer.handle_request()

class MServer(socketserver.UDPServer):

  def __init__(self, server_address, RequestHandlerClass, Computing, bind_and_activate=True):
    socketserver.UDPServer.__init__(self, server_address, RequestHandlerClass, bind_and_activate)
    self.Parent = Computing
    self.config = self.Parent.config

  def MulticastProcess(self, data):
    cmd = str(data).replace('\n', '').replace("b'","").replace("'","")
    self.Parent.Commands.append(cmd)
    self.Parent.CommandProcess()

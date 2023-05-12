# -*- coding: utf-8 -*-
from http.server import HTTPServer
import threading
from P2PHandlerLib.Handler import *

class P2PEngine(object):

  def __init__(self):
    super(P2PEngine, self).__init__()

  def StartServer(self):
    thread = threading.Thread(target=self.run, args=())
    thread.daemon = True
    thread.start()

  def run(self):
    self.p2pd = P2PHTTPServer(("", int(self.config['P2P.Port'])),RequestHandler, self, bind_and_activate=True)
    self.p2pd.allow_reuse_address = True
    while True:
      self.p2pd.handle_request()

class P2PHTTPServer(HTTPServer):

  def __init__(self, server_address, RequestHandlerClass, Computing, bind_and_activate=True):
    HTTPServer.__init__(self, server_address, RequestHandlerClass, bind_and_activate)
    self.Parent = Computing
    self.config = self.Parent.config

  def PostTest(self, data):
    print(data)

  def GetTest(self):
    print('Triger from root')

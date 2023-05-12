# -*- coding: utf-8 -*-
import os
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
import signal
import time
#from Model import *

class RequestHandler(BaseHTTPRequestHandler):

  def do_GET(self):
    if self.path == '/ProjectStatus':
      self.do_ProjectStatus()
    elif self.path == '/GroupStatus':
      self.do_GroupStatus()
    elif self.path == '/':
      self.do_root()

  def do_POST(self):
    if self.path =='/':
      self.do_root()
    elif self.path == '/Download':
      self.do_Download()
    elif self.path == '/Query':
      self.do_Query()
    elif self.path == '/Notify':
      self.do_Notify()
    elif self.path == '/ProjectUpdate':
      self.do_ProjectUpdate()

  def do_root(self):
    self.send_response(200)
    self.send_header('application', 'text/json')
    self.end_headers()
    self.wfile.write(json.dumps(str(self.server.config)).encode())

  def do_get_Download(self):
    Params = str(self.path).split('?')[1].split('&')
    parameters = {}
    for param in Params:
      parameters[param.split('=')[0]] = param.split('=')[1]
    File = parameters['FileName']
    f = open(os.path.join(self.server.config['Model.File'], File), 'rb')
    while True:
      byte = f.read(32768)
      if byte is None or len(byte) == 0:
        break
      self.wfile.write(bytes(byte))
    f.close()
    return

  def do_ProjectUpdate(self):
    DataString = self.rfile.read(int(self.headers['Content-Length']))
    data = {}
    self.send_response(200)
    self.send_header('application', 'text/json')
    self.end_headers()
    if len(DataString) == 0:
      self.wfile.write('{"result":"fail","reason":"no input"}'.encode())
    else:
      data = json.loads(DataString)
    print('do_ProjectUpdate ', data)
    if data['Type'].find('ModelFile') >= 0:
      FileName = data['FileName']
      ProjectName = data['ProjectName']
    if data['Type'].find('RecordUpdate') >= 0:
      ProjectName = data['ProjectName']
      self.server.Parent.ProjectValueUpdate(data)
    return

  def do_Gradient(self):
    DataString = self.rfile.read(int(self.headers['Content-Length']))
    print('%d do_Gradient from IP %s with content %s' % (int(time.time()), self.client_address[0],str(DataString)))
    data = {}
    self.send_response(200)
    self.send_header('application', 'text/json')
    self.end_headers()    
    if len(DataString) == 0:
      self.wfile.write('{"result":"fail","reason":"no input"}'.encode())
    else:
      data = json.loads(DataString)
    GradientFile = data['Gradient']
    self.wfile.write(json.dumps(self.server.Parent.GetGradientInfo(GradientFile)).encode())
    return

  def do_Notify(self):
    DataString = self.rfile.read(int(self.headers['Content-Length']))
    print('%d do_Notify from IP %s with content %s' % (int(time.time()), self.client_address[0],str(DataString)))
    data = {}
    self.send_response(200)
    self.send_header('application', 'text/json')
    self.end_headers()
    if len(DataString) == 0:
      self.wfile.write('{"result":"fail","reason":"no input"}'.encode())
    else:
      data = json.loads(DataString)
    if str(DataString).find('UUID') >=0:
      self.wfile.write('{"result":"ok"}'.encode())
      self.server.Parent.NodeSystemInfoUpdate(data)
    elif data['Type'].find('NodesInfo') >= 0:
      self.wfile.write('{"result":"ok"}'.encode())
      self.server.Parent.P2PNodesUpdate(data)
      print('Node info received')
    elif data['Type'].find('FileInfo') >= 0:
      self.wfile.write('{"result":"ok"}'.encode())
      self.server.Parent.P2PFileUpdate(data, self.client_address)
      print('Available File info')
    elif data['Type'].find('ProjectInfo') >= 0:
      #if data['Source'].find('Origin')
      self.wfile.write('{"result":"ok"}'.encode())
      self.server.Parent.ProjectFileUpdate(data)
    #elif data['Type'].find('SysInfo') >=0:
    #  self.wfile.write('{"result":"ok"}'.encode())
    #  self.server.Parent.NodeSystemInfoUpdate(data) 
    elif data['Type'].find('GlobalModel') >=0:
      self.wfile.write('{"result":"ok"}'.encode())
      print('Gradient ', data)
      ip = self.client_address[0]
      #self.PullGlobal(ip, data)
      self.server.Parent.PullGradient(ip, data)
      self.server.Parent.GradientUpdate(data)
    elif data['Type'].find('ModelInfo') >=0:
      self.wfile.write('{"result":"ok"}'.encode())
      print('New Model Available', data)
      if data['Source'].find('Origin') >= 0:
        ip = self.client_address[0]
        self.server.Parent.Pull(ip, data)
      elif data['Source'].find('Relay') >= 0:
      # Check if Model Available  
        self.server.Parent.ProjectModelVerify(data)
    elif data['Type'].find('ResultInfo') >=0:
      self.server.Parent.ProjectResultUpdate(data)
    return

  def do_Query(self):
    DataString = self.rfile.read(int(self.headers['Content-Length']))
    data = {}
    self.send_response(200)
    self.send_header('application', 'text/json')
    self.end_headers()
    if len(DataString) == 0:
      self.wfile.write('{"result":"fail","reason":"no input"}'.encode())
    else:
      data = json.loads(DataString)
    if data['Command'].find('File') >= 0:
      print('Query Local File Info')
      self.server.Parent.GetFile()
      print(self.server.Parent.LocalFile)
      self.wfile.write(str(self.server.Parent.LocalFile).encode())
    elif data['Command'].find('LocalModel') >= 0:
      result = self.server.Parent.GetLocalModel(data)
      self.wfile.write(str(result).encode())
    #elif data['Command'].find('Epoch') >=0:
      #
    return

  def do_ProjectStatus(self):
    self.send_response(200)
    self.send_header('application', 'text/json')
    self.end_headers()
    self.wfile.write(json.dumps(str(self.server.Parent.ProjectInfo)).encode())
    return

  def do_GroupStatus(self):
    self.send_response(200)
    self.send_header('application', 'text/json')
    self.end_headers()
    self.wfile.write(json.dumps(str(self.server.Parent.Nodes)).encode())

  def do_Download(self):
    DataString = self.rfile.read(int(self.headers['Content-Length']))
    print('%d do_Download from IP %s with content %s' % (int(time.time()), self.client_address[0],str(DataString)))
    if len(DataString) == 0:
      self.send_response(200)
      self.send_header('application', 'text/json')
      self.end_headers()
      self.wfile.write('{"result":"fail","reason":"no input"}'.encode())
      return
    else:
      data = json.loads(DataString)
      FileName = data['FileName']
      Start = data['Start']
      End = data['End']
      self.send_response(200)
      self.send_header('Content-Type','application/octet-stream')
      self.send_header('Content-Disposition', 'attachment; filename="'+FileName + '"' )
      self.end_headers()
      try:
        f = open(os.path.join(self.server.config['Model.File'], FileName), 'rb')
      except:
        self.send_response(200)
        self.send_header('application', 'text/json')
        self.end_headers()
        self.wfile.write('{"result":"fail","reason":"file NA"}'.encode())
        return
      FilePosition = 0
      while True:
        byte = f.read(1)
        FilePosition = FilePosition + 1
        if byte is None or len(byte) == 0 or FilePosition > End:
          break
        if FilePosition < Start:
          continue
        else:
          self.wfile.write(bytes(byte))
      f.close()
    return

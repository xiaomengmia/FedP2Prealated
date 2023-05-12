import os
import socketserver

class MulticastHandler(socketserver.DatagramRequestHandler):

  def handle(self):
    socket = self.request[1]
    data = self.rfile.readline().strip()
    self.server.MulticastProcess(data)
    #fp = open('/opt/warwick/Command/command', 'a')
    #fp.write(str(data) + '\n')
    #fp.close()
    #Pid = str(os.getpid())
    #cmd = 'kill -10 ' + Pid
    #os.system(cmd)
    return

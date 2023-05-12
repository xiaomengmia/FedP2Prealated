from multiprocessing import Process
import os, sys, shutil
import urllib.request
import socket, json
import time

class Download(object):

  def __init__(self):
    super(Download, self).__init__()
    self.QueuedModel = {}
#    self.AvailableFiles = {}

  def DownloadProcess(self):
    CompleteList = []
    for k, v in list(self.AvailableFiles.items()):
      print('%d Download.DownloadProcess of file %s with info %s' % (int(time.time()), k,v))
      self.WriteSysLog('Download.DownloadProcess of file %s with info %s' % (k,v))
      if os.path.exists(os.path.join(self.config['Model.File'], k)) == True:
        CompleteList.append(k)
        continue
      if len(v['node']) == 1:
        self.FileDownload('Download', v['node'][0], 0, v['size'], k, k)
        if os.path.exists(os.path.join(self.config['Tmp.File'], k)) == True:
          TmpMD5 = self.FileMd5(os.path.join(self.config['Tmp.File'], k))
          if TmpMD5.find(v['md5']) >=0:
            Org = os.path.join(self.config['Tmp.File'], k)
            Dst = os.path.join(self.config['Model.File'], k)
            shutil.move(Org, Dst)
            CompleteList.append(k)
      elif len(v['node']) > 1:
        procs = []
        TmpFiles = []
        seq = 0
        StartByte = 0
        EndByte = 0
        div = min(len(v['node']), int(self.config['Max.Download']))
        for cpu in range(div):
          EndByte = StartByte + int(((seq + 1) * v['size']) / div)
          EndByte = min(v['size'], EndByte)
          DownloadResult = ''
          while DownloadResult.find('error') >= 0:
            DownloadResult = self.FileDownload('Download', v['node'][seq], StartByte, EndByte, k, k + str('.') + str(seq))
          procs.append(proc)
          TmpFiles.append(k + str('.') + str(seq))
          proc.start()
          seq = seq + 1
          StartByte = EndByte + 1
        interFile = os.path.join(self.config['Tmp.File'], k + '.tmp')
        with open(interFile, 'wb') as outfile:
          for tmpFile in TmpFiles:
            with open(os.path.join(self.config['Tmp.File'], tmpFile), 'rb') as infile:
              outfile.write(infile.read())
        interFileMd5 = self.FileMd5(interFile)
        if interFileMd5.find(v['md5']) >=0:
          Dst = os.path.join(self.config['Model.File'], k)
          shutil.move(interFile, Dst)
          CompleteList.append(k)
      print('end download ', k)
      # Push Downloadable message
      #self.PushLocalInfo(k, v)
    for DFile in CompleteList:
      try:
        del self.AvailableFiles[DFile]
      except:
        print("Error ")
    return CompleteList

  def PushLocalInfo(self, fileName, data):
    uri = 'Notify'
    Msg = {}
    Msg['Type'] = 'ModelInfo'
    Msg['ModelType'] = 'LocalModel'
    Msg['ModelFile'] = fileName
    Msg['Source'] = 'Origin'
    Msg['md5'] = self.FileMd5(os.path.join(self.config['Model.File'], fileName))
    Msg['size'] = os.path.getsize(os.path.join(self.config['Model.File'], fileName))
    if 'Tau' in data.keys():
      Msg['Tau'] = data['Tau']
    self.PushMessage(uri, Msg)
    return

  def FileDownload(self, URI, host, start, end, FileName, DSTFileName):
    msg = {}
    msg['FileName'] = FileName
    msg['Start'] = start
    msg['End'] = end
    API = "%s://%s:%s/%s" % (self.config['P2P.protocol'], host, self.config['P2P.Port'], URI)
    #print('%d Download.FileDownload from URI %s with parameters %s' % (int(time.time()), API, str(msg)))
    self.WriteSysLog('Download.FileDownload from URI %s with parameters %s' % (API, str(msg)))
    try:
      req = urllib.request.Request(API)
      data = json.dumps(msg).encode('utf-8')
      req.add_header('Content-Type', 'application/json')
      req.add_header('Content-Length', len(data))
      response = urllib.request.urlopen(req, data = data, timeout = (int(self.config['Socket.Timeout'])*50))
      saveFile = open(os.path.join(self.config['Tmp.File'], DSTFileName), 'wb')
      ContentLength = 0
      saveFile.write(response.read())
      saveFile.close()
      Result = 'good'
    except urllib.error.HTTPError as e:
      print('Error ', e, ' with file ', DSTFileName, ' from ', host)
      Result = 'error'
    except Exception as e:
      print('Error ', e, ' with file ', DSTFileName, ' from ', host)
      Result = 'error'
    except socket.timeout as e:
      print('Error ', e, ' with file ', DSTFileName, ' from ', host)
      Result = 'error'
    print('%d Download.FileDownload end from API %s with result %s ' % (int(time.time()), API, Result))
    self.WriteSysLog('Download.FileDownload end from API %s with result %s ' % (API, Result))
    return

  def ModelDownload(self):
    return

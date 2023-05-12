# -*- coding: utf-8 -*-
import sys, os, re
import datetime, time,shutil

class ModelPull(object):

  def __init__(self):
    super(ModelPull, self).__init__()

  def Pull(self,ip , Data):
    ModelFile = Data['ModelFile']
    ModelMD5 = Data['md5']
    ModelSize = Data['size']
    Tmp = {}
    Tmp['md5'] = ModelMD5
    Tmp['size'] = ModelSize
    Tmp['node'] = [ip]
    print('ModelPull ', Tmp)
    if ModelFile.find(self.config['HostName']) < 0:
      if ModelFile not in self.AvailableFiles.keys():
        self.AvailableFiles[ModelFile] = Tmp
      elif ip not in Tmp['node']:
        #print('Another ip')
        self.AvailableFiles[ModelFile]['node'].append(ip)
    #self.PushMessage('Notify', Data)
    ProjectName = str(ModelFile).split('-')[0]
    if ModelFile not in self.ProjectInfo[ProjectName]['Models']:
      self.ProjectInfo[ProjectName]['Models'].append(ModelFile)
    if 'Tau' in Data.keys():
      self.ProjectInfo[ProjectName]['ModelsInfo'][ModelFile] = Data['Tau']
    return

  def PullGlobal(self, ip, Data):
    ModelFile = Data['ModelFile']
    ModelMD5 = Data['GlobalMD5']
    ModelSize = Data['GlobalSize']
    Tmp = {}
    Tmp['md5'] = ModelMD5
    Tmp['size'] = ModelSize
    Tmp['node'] = [ip]
    print('Pull Global Model', Tmp)
    if ModelFile.find(self.config['HostName']) < 0:
      if ModelFile not in self.AvailableFiles.keys():
        self.AvailableFiles[ModelFile] = Tmp
      elif ip not in Tmp['node']:
        self.AvailableFiles[ModelFile]['node'].append(ip)
    return

  def PullGradient(self, ip, Data):
    GradFile = Data['Gradient']
    GradMD5 = Data['GradientMD5']
    GradSize = Data['GradientSize']
    Tmp = {}
    Tmp['md5'] = GradMD5
    Tmp['size'] = GradSize
    Tmp['node'] = [ip]
    print('Gradient Pull ' , Tmp)
    if GradFile.find(self.config['HostName']) < 0:
      if GradFile not in self.AvailableFiles.keys():
        self.AvailableFiles[GradFile] = Tmp
      elif ip not in Tmp['node']:
        self.AvailableFiles[GradFile]['node'].append(ip)
    return

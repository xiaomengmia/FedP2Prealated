import sys, os, time, datetime, signal
from time import sleep
import threading
from ComputingLib.Computing import *
from NetworkingLib.Networking import *
import multiprocessing

def main(argv):
  sys.dont_write_bytecode=True
  configFile = GetOpts(argv)
  NetworkEngine = Networking(configFile)
  ComputingEngine = Computing(NetworkEngine)
  #NetworkEngine.Computing
  # Kill process if required
  for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT, signal.SIGABRT, signal.SIGILL, signal.SIGSEGV]:
    signal.signal(sig, NetworkEngine.ProcessStop)
  SleepTime = int(NetworkEngine.config['Idle.Interval'])
  #NetworkEngine.ProjectInfo = ComputingEngine.ProjectInfo
  NetworkEngine.config = ComputingEngine.config
  ExitSig = 'n'
  while(ExitSig == 'n'):
    CurrentTime = datetime.datetime.now()
    print('-------------------------------------------------------------')
    print('Current Time is ' + CurrentTime.strftime("%Y-%m-%d %H:%M:%S"))
    while len(NetworkEngine.AvailableFiles) > 0:
      #NetworkEngine.ProjectInfo = ComputingEngine.ProjectInfo
      print('Files need to download ', NetworkEngine.AvailableFiles)
      CompleteList = NetworkEngine.DownloadProcess()
      NetworkEngine.ProjectModelUpdate(CompleteList)
    sleep(SleepTime)
    print('=============================================================')
  return

if __name__ == "__main__":
  main(sys.argv[1:])

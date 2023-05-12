import torch, os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from MNISTCNN import *

def train(network, train_loader, optimizer , epoch, dev, log_interval, train_losses, train_counter, param, seq, hostname):
  network.train()
  print('Start Training')
#  if param['State'].find('Merging') >=0:
#    start = 0
#    end = math.ceil(len(train_loader.dataset)/ int(param['TrainBatchSize']))
#  else:
  CBatch = param['CurrentBatch']
  start = math.ceil(seq * len(train_loader.dataset) / (int(param['TrainBatchSize']) * int(param['Nodes'])))
  if seq + 1 == int(param['TrainBatchSize']):
    end = math.ceil(len(train_loader.dataset)/ int(param['TrainBatchSize']))
  else:
    end =  math.ceil((seq + 1) * len(train_loader.dataset) / (int(param['TrainBatchSize'])*int(param['Nodes'])))
  print(dev, CBatch)
  for batch_idx, (data, target) in enumerate(train_loader):
    if dev.find('cuda') >=0:
      data = data.to(dev)
      target = target.to(dev)
    if batch_idx >= start and batch_idx < end:
      optimizer.zero_grad()
      output = network(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
          (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
  print('end batch')
  if param['State'].find('submit') >= 0:
    #torch.save(network.state_dict(), os.path.join(param['ModelFile'], param['ProjectName'] + '-' + str(seq) + '.' + hostname + '.pth'))
    torch.save(network, os.path.join(param['ModelFile'], param['ProjectName'] + '-' + str(seq) + '-' + str(CBatch)+ '.' + hostname + '.pth'))
  elif param['State'].find('Merging') >= 0:
    torch.save(network.state_dict(), os.path.join(param['ModelFile'], param['ProjectName'] + '-' + str(seq) + '-' + str(CBatch)+ '.' + hostname + '.pth'))
  
 # torch.save(network.state_dict(), os.path.join(param['ModelFile'], param['ProjectName'] + '-' + str(seq) + '.' + hostname + '.optimizer.pth'))
        #torch.save(optimizer.state_dict(), './results/optimizer.pth')

def test(network, test_loader, dev, test_losses):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      if dev.find('cuda') >=0:
        data = data.to(dev)
        target = target.to(dev)
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  return correct

def MergeModel(network, param, totalBatch, dev):
  sd = network.state_dict()
  count = 0
  for modelFile in param['Models']:
    print(modelFile)
    Tmp = MNISTCNN()
    if dev.find("cuda") >=0:
      Tmp.to(torch.device(dev))
      Tmp = torch.load(os.path.join(param['ModelFile'], modelFile), map_location=torch.device(dev))
    else:
      Tmp = torch.load(os.path.join(param['ModelFile'], modelFile), map_location=torch.device('cpu'))
    TmpSD = Tmp.state_dict()
    print(next(network.parameters()).device)
    print(next(Tmp.parameters()).device)
    for key in TmpSD:
      if count == 0:
        sd[key] = TmpSD[key]/int(param['Nodes'])
      else:
        sd[key] = sd[key] + TmpSD[key]/int(param['Nodes'])
    count = count + 1
  network.load_state_dict(sd)
  return network


def MNIST(queue, param, seq, hostname):
  n_epochs = int(param['MiniEpochs'])
  batch_size_train = int(param['TrainBatchSize'])
  batch_size_test = int(param['TestBatchSize'])
  learning_rate = float(param['LearningRate'])
  momentum = float(param['Momentum'])
  log_interval = int(param['LogInterval'])

  random_seed = int(param['RandomSeed'])
  torch.backends.cudnn.enabled = False
  torch.manual_seed(random_seed)

  if torch.cuda.is_available():
    dev = "cuda:0"
  else:
    dev = "cpu"

  train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(param['DatasetFile'], train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_train, shuffle=True)

  test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(param['DatasetFile'], train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_test, shuffle=True)

  TotalBatchSize = math.ceil(len(train_loader.dataset)/ int(param['TrainBatchSize']))

  print('Start Network Define')
  network = MNISTCNN()
  print('End of Network Define')
  if dev.find("cuda") >=0:
    print('use GPU')
    network.to(torch.device(dev))
  if param['State'].find('Merging') >=0:
    print('Merging Required')
    network = MergeModel(network, param, TotalBatchSize, dev)
  optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
  print('End merging')
  train_losses = []
  train_counter = []
  test_losses = []
  correct = []
  test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

  for epoch in range(1, n_epochs + 1):
    train(network, train_loader, optimizer , epoch, dev, log_interval, train_losses, train_counter, param, seq, hostname)
    TestResult = test(network, test_loader, dev, test_losses)
    if dev.find('cuda') >=0:
      testResult = TestResult.to(torch.device('cpu'))
    else:
      testResult = TestResult
    correct.append(int(testResult))
  result = 'Finish'
  return correct

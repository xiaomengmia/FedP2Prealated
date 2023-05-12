import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MnistModel(nn.Module):
  def __init__(self):
    super(MnistModel, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop1 = nn.Dropout2d(0.25)
    self.conv2_drop2 = nn.Dropout2d(0.5)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv2_drop1(self.conv1(x)), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop2(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x)

  def weight_init(self):
    for block in self._modules:
      if isinstance(self._modules[block], (torch.nn.modules.conv.Conv2d,torch.nn.modules.Linear)):
        print('Init Module', block, self._modules[block], type(self._modules[block]))
        nn.init.xavier_normal_(self._modules[block].weight)
        #self._modules[block].weight.data.fill_(0.001)

class Cifar10Model(nn.Module):
  def __init__(self):
    super(Cifar10Model, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16*5*5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  def weight_init(self):
    for block in self._modules:
      if isinstance(self._modules[block], (torch.nn.modules.conv.Conv2d,torch.nn.modules.Linear)):
        print('Init Module', block, self._modules[block], type(self._modules[block]))
        nn.init.xavier_normal_(self._modules[block].weight)

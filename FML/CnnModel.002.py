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

    self.conv_layer = nn.Sequential(

    # Conv Layer block 1
      nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),

      # Conv Layer block 2
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Dropout2d(p=0.05),

      # Conv Layer block 3
      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )


    self.fc_layer = nn.Sequential(
      nn.Dropout(p=0.1),
      nn.Linear(4096, 1024),
      nn.ReLU(inplace=True),
      nn.Linear(1024, 512),
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.1),
      nn.Linear(512, 10)
    )

  def forward(self, x):
    # conv layers
    x = self.conv_layer(x)
        
    # flatten
    x = x.view(x.size(0), -1)
        
    # fc layer
    x = self.fc_layer(x)

    return x

  def weight_init(self):
    for block in self._modules:
      if isinstance(self._modules[block], (torch.nn.modules.conv.Conv2d,torch.nn.modules.Linear)):
        print('Init Module', block, self._modules[block], type(self._modules[block]))
        nn.init.xavier_normal_(self._modules[block].weight)

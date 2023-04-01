import torch

from shufflenetv2 import ShuffleNetV2

data_set = 'cifar'
iid = 0
gpu = 0
train_test_ratio = 1.0
num_clients = 50
frac_C = 0.5
num_selecteds = int(num_clients * frac_C)
device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
epochs = 30
client_batch_size = 100
local_epochs = 3
agg_mode = 'global'
tolerance = 0
batch_momentum = 1.0
model_ensemble_temp = ShuffleNetV2(net_size=1.5).to(device)
model_ensemble_temp.train()



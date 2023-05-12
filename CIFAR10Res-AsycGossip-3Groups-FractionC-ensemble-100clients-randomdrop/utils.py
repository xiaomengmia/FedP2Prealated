import copy
from typing import Tuple, OrderedDict, Dict, Any

import torch
from torch import optim as optim
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset

from parameter import device, client_batch_size, local_epochs, batch_momentum, model_ensemble_temp
from shufflenetv2 import ShuffleNetV2


def client_train(epoch: int, dataset: VisionDataset, data_idx: list[int], model: Module) \
        -> tuple[OrderedDict[str, torch.Tensor], float]:
    # global batch_momentum

    data_idx = list(data_idx)

    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(Client_Dataset(dataset, data_idx), batch_size=client_batch_size, shuffle=True)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(model.parameters())

    # batch_momentum = 1.0

    print(batch_momentum, 'momentum')
    for local_epoch in range(local_epochs):
        loss_show = 0.0
        for batch_idx, (inputs, target) in enumerate(train_loader, 0):
            # temp = 0
            # for name, layer  in (model.named_modules()):
            #     if 'bn' in name and temp<corresponding_breakpoint:
            #         layer.momentum =batch_momentum
            #     temp += 1

            # if batch_idx > 0 and flag == True:
            #     batch_momentum -= 0.001
            #     if batch_momentum < 0.1:
            #         batch_momentum = 0.1

            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            loss_show += loss.item()
            lossavg = loss_show / (batch_idx + 1)
        # print('train_local_epoch:',local_epoch)
        # print ('loss:',loss_show)

    return model.state_dict(), lossavg


def client_test(epoch: int, dataset: VisionDataset, data_idx: list[int], model: Module) -> tuple[float, float]:
    test_loader = DataLoader(Client_Dataset(dataset, data_idx), batch_size=client_batch_size, shuffle=False)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        correct = 0
        total = 0
        loss_show = 0.0
        for batch_idx, (inputs, target) in enumerate(test_loader, 0):
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            loss_test = criterion(outputs, target).item()
            loss_show += loss_test
            lossavg = loss_show / (batch_idx + 1)

            temp, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        acc = (correct / total)
    print('acc: ', acc)

    return acc, lossavg


def globle_agg(w: list[OrderedDict[str, torch.Tensor]]) -> OrderedDict[str, torch.Tensor]:
    w_avg = copy.deepcopy(w[0])
    for i in w_avg.keys():
        for j in range(1, len(w)):
            #             copy_temp = copy.deepcopy(w[j][i])
            copy_temp = w[j][i]
            w_avg[i] += copy_temp
        w_avg[i] = torch.div(w_avg[i], len(w))
    return w_avg


def test(model, dataset, batch_size):
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        correct = 0
        total = 0
        loss_show = 0.0
        for batch_idx, (inputs, target) in enumerate(dataloader, 0):
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            temp, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            loss_test = criterion(outputs, target).item()
            loss_show += loss_test
            lossavg = loss_show / (batch_idx + 1)
        acc = (correct / total)
    print('acc_test: ', acc)
    return acc, lossavg


def test_ensemble(models, dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        correct = 0
        total = 0
        loss_show = 0.0

        for batch_idx, (inputs, target) in enumerate(dataloader, 0):
            count = 0
            for model_id in models:
                model_ensemble_temp.load_state_dict(model_id)
                model_ensemble_temp.eval()
                inputs, target = inputs.to(device), target.to(device)

                if count == 0:
                    outputs = model_ensemble_temp(inputs)
                else:
                    outputs += model_ensemble_temp(inputs)
                count += 1

            temp, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            # loss_test = criterion(outputs,target).item()
            # loss_show += loss_test
            # lossavg = loss_show/(batch_idx+1)
        acc = (correct / total)
    print('acc_ensemble: ', acc)
    # return  acc, lossavg
    return acc


class Client_Dataset(Dataset):

    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = list(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        inputs, targets = self.dataset[self.idx[item]]
        return inputs, targets

from collections import OrderedDict

import torch
from torch import Tensor

from parameter import data_set, iid, train_test_ratio, num_clients, num_selecteds, device, epochs, client_batch_size
from utils import client_train, globle_agg, test_ensemble, test

torch.cuda.get_device_name(0)

# %%
import random

list1 = [[1, 2], [3, 4, 5], [6, 2], [7, 2], [4, 2]]
list_choice = random.sample(list1, 3)
print(list_choice)
# %%
from torchvision import transforms
from torchvision import datasets

import numpy as np
import random

import copy

from shufflenetv2 import ShuffleNetV2
# from Lenet import*
from resnet import ResNet

# from alex import*
# from vgg import*
# from mobilenet_v2_v2 import*
# %%
# Parameters

torch.manual_seed(0)
random.seed(1)
np.random.seed(1)

# iid = 'cifar_noniid'

# %%
# Initialize

model_clientdict = {}
# %%
# Data Prep

if data_set == 'mnist':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST(root='../dataset/mnist/',
                                train=True,
                                download=True,
                                transform=transform)

    test_data = datasets.MNIST(root='../dataset/mnist/',
                               train=False,
                               download=True,
                               transform=transform)
elif data_set == 'cifar':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = datasets.CIFAR10(root='../dataset/cifar/',
                                  train=True,
                                  download=True,
                                  transform=transform)

    test_data = datasets.CIFAR10(root='../dataset/cifar/',
                                 train=False,
                                 download=True,
                                 transform=transform)


def sampling_iid(dataset, num_clients):
    datanum_per_client = int(len(dataset) / num_clients)
    client_dataidx_dict = {}
    temp_dict = [i for i in range(len(dataset))]
    for i in range(num_clients):
        client_dataidx_dict[i] = set(np.random.choice(temp_dict, datanum_per_client, replace=False))
        temp_dict = list(set(temp_dict) - client_dataidx_dict[i])
    return client_dataidx_dict


def sampling_noniid(dataset, num_clients):
    avgnum_per_client = int(len(dataset) / num_clients)
    client_dataidx_dict = {}
    temp_dict = [i for i in range(len(dataset))]
    for i in range(num_clients):
        client_dataidx_dict[i] = set(np.random.choice(temp_dict, random.randint(50, 1130), replace=False))
        temp_dict = list(set(temp_dict) - client_dataidx_dict[i])
    return client_dataidx_dict


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def cifar_noniid(dataset, num_users):
    num_items = int(len(dataset))
    dict_users = {}
    labels = [i for i in range(10)]
    idx = {i: np.array([], dtype='int64') for i in range(10)}

    j = 0
    # print((dataset[0][0]))
    for i in dataset:
        # print(i)
        idx[i[1]] = np.append(idx[i[1]], j)
        j += 1

    # if(num_users<=5):
    #     k = int(10/num_users)
    #     for i in range(num_users):
    #         a = 0
    #         for j in range(i*k,(i+1)*k):
    #             a += j
    #             if(j==i*k):
    #                 dict_users[i] = list(idx[j])
    #             else:
    #                 dict_users[i] = np.append(dict_users[i],idx[j])
    #         print(a)
    #     return dict_users

    # if k = 4, a particular user can have samples only from at max 4 classes
    k = 4
    # print(idx)
    num_examples = int(num_items / (k * num_users))

    for i in range(num_users):
        t = 0
        while (t != k):
            j = random.randint(0, 9)

            if (len(idx[(i + j) % len(labels)]) >= num_examples):
                rand_set = set(np.random.choice(idx[(i + j) % len(labels)], num_examples, replace=False))
                idx[(i + j) % len(labels)] = list(set(idx[(i + j) % len(labels)]) - rand_set)
                rand_set = list(rand_set)
                if (t == 0):
                    dict_users[i] = rand_set
                else:
                    dict_users[i] = np.append(dict_users[i], rand_set)
                t += 1
    return dict_users


# %%
# Main


if iid == 0:
    client_dataidx_dict = sampling_iid(train_data, num_clients)
elif iid == 1:
    client_dataidx_dict = sampling_noniid(train_data, num_clients)
elif iid == 'cifar_noniid':
    client_dataidx_dict = cifar_noniid(train_data, num_clients)
# elif iid == 'cifar_noniid_aligned':
#     client_dataidx_dict = cifar_noniid_aligned(train_data,num_clients)
else:
    client_dataidx_dict = mnist_noniid(train_data, num_clients)

client_traindataidx_dict = {}
client_testdataidx_dict = {}

for i in range(num_clients):
    client_dataidx_dict[i] = list(client_dataidx_dict[i])

    client_traindataidx_dict[i] = list(
        random.sample(client_dataidx_dict[i], int(train_test_ratio * len(client_dataidx_dict[i]))))

    client_testdataidx_dict[i] = list(set(client_dataidx_dict[i]) - set(client_traindataidx_dict[i]))

for i in range(0, num_clients):
    # model_clientdict[i] = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
    #     model_clientdict[i] = [Lenet().to(device)]
    #     model_clientdict[i] = [ResNet18().to(device)]
    # model_clientdict[i] = [VGG('VGG16').to(device)]
    # model_clientdict[i] = [AlexNet().to(device)]
    model_clientdict[i] = [ShuffleNetV2(net_size=1.5).to(device)]
    # model_clientdict[i] = [MobileNetV2(10, alpha = 1).to(device)]
    model_clientdict[i][0].train()

# criterion

# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)ƒ

# train


# main()

acc_log = []
loss_log = []

acc_clients_log = [[] for i in range(num_clients)]
loss_clients_log = [[] for i in range(num_clients)]
acc_ensemble = []
# acc_client = [1.0 for i in range(num_clients)]
w_clients = [[i[0].state_dict()] for i in model_clientdict.values()]

# global_model = CNNCifar().to(device)
# global_model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
# global_model = LeNet().to(device)
# global_model = ShuffleNetV2(net_size=1.5).to(device)
# global_model = MobileNetV2(10, alpha = 1).to(device)


# global_model.train()

# w_glob = global_model.state_dict()

# w_glob_original = global_model.state_dict()


counter = 0
flag = False

for epoch in range(400):

    print('Epoch {} start'.format(epoch))

    loss_clients, acc_sum, acc_sum_1, acc_sum_2, loss_sum = [], 0, 0, 0, 0

    for client_idx in range(0, num_clients):

        w_temp = []

        drop = False

        now_rd = len(w_clients[client_idx]) - 1

        if now_rd >= epochs:
            drop = True

        if client_idx < 10:
            if random.random() < 0.8:
                drop = True
        elif client_idx >= 10 and client_idx < 40:
            if random.random() < 0.5:
                drop = True
        else:
            if random.random() < 0.2:
                drop = True

            #         if now_rd >= epochs:
        #             drop = True

        #         if client_idx<20:
        #             if random.random()<0.8:
        #                 drop = True
        #         elif client_idx >= 20 and client_idx<80:
        #             if random.random()<0.5:
        #                 drop = True
        #         else:
        #             if random.random()<0.2:
        #                 drop = True

        #         if  client_idx<20:
        #             if epoch%2 == 0 or epoch%3 == 0 :
        #                 drop = True
        #         elif  client_idx >= 20 and client_idx<80:
        #             if epoch%3 == 0:
        #                 drop = True
        #         else:
        #             if epoch%6 == 0:
        #                 drop = True

        print('drop = ', drop)
        if drop == False:

            for i in w_clients:

                counter = 0
                for j in i:

                    # if counter <= (now_rd +  tolerance) and counter >= (now_rd - tolerance):
                    if counter == now_rd:
                        w_temp.append(j)
                    counter += 1

            print('len,w_temp', len(w_temp))

            if len(w_temp) >= num_selecteds:
                #                 w_select = random.sample(w_temp,num_selecteds)
                w_select = w_temp

                w_agg = globle_agg(w_select)

                model_clientdict[client_idx][0].load_state_dict(w_agg)
                print('epoch:', epoch, 'client:', client_idx, 'now_rd:', now_rd)

                acc, loss_ = test(model_clientdict[client_idx][0], test_data, client_batch_size)

                w: OrderedDict[str, Tensor]
                loss_train: float
                w, loss_train = client_train(epoch, train_data, client_traindataidx_dict[client_idx],
                                             model=model_clientdict[client_idx][0])

                w_clients[client_idx].append(copy.deepcopy(w))

                #                 w_clients[client_idx].append (w)

                acc_clients_log[client_idx].append(acc)

                loss_clients_log[client_idx].append(loss_)

                # acc_client[client_idx] = acc

                acc_sum += acc

                loss_sum += loss_
                # loss_sum += loss_train

    # if epoch < epochs:
    #     w_ensemble = []
    #     ensemble_rd = epoch
    #     for i in w_clients:
    #         counter = 0
    #         for j in i:
    #             # if counter <= (now_rd +  tolerance) and counter >= (now_rd - tolerance):
    #             if counter == ensemble_rd:
    #                 w_ensemble.append(j)
    #             counter += 1
    #     acc = test_ensemble( w_ensemble, test_data, client_batch_size)
    #     acc_ensemble.append(acc)

w_ensemble = []
ensemble_rd = epochs - 1
for i in w_clients:
    counter = 0
    for j in i:
        # if counter <= (now_rd +  tolerance) and counter >= (now_rd - tolerance):
        if counter == ensemble_rd:
            w_ensemble.append(j)
        counter += 1
acc = test_ensemble(w_ensemble, test_data, client_batch_size)

# print('agg_mode = global,','agg_before:',agg_before)

# for client_idx in range(0,num_clients):

#     for i in list(w_clients[client_idx][-1].keys())[0:agg_before]:
#         if 'bn' not in i:
#             w_clients[client_idx][-1][i] = copy.deepcopy(w_glob[i])
#     model_clientdict[client_idx][0].load_state_dict(w_clients[client_idx][-1])


# glob_acc = test(global_model, test_data, client_batch_size)
# globacc_log.append(glob_acc)


# for client_idx in range(num_clients):
# if acc_client[client_idx] < acc_client_lastepoch[client_idx]:
# w_clients[client_idx] = w_clients_lastepoch[client_idx]
# model_clientdict[client_idx].load_state_dict(w_clients[client_idx])


print(acc_log)
print(loss_log)
print(acc_clients_log)
print(loss_clients_log)
print(acc_ensemble)

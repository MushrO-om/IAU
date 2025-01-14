import os
import random
import sys
from itertools import chain
from shutil import copy

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler,TensorDataset
from torchvision import datasets
import torchvision.transforms as transforms

mean_dict = {}
std_dict = {}
mean_dict['cifar10']=[0.49139968, 0.48215841, 0.44653091]
std_dict['cifar10']=[0.24703223, 0.24348513, 0.26158784]
mean_dict['cifar100']=[0.50707516, 0.48654887, 0.44091784]
std_dict['cifar100']=[0.26733429, 0.25643846, 0.27615047]

min_tensors = {}
max_tensors = {}
min_values = torch.tensor([-1.896, -1.898, -1.597]) 
max_values = torch.tensor([1.843, 1.999, 2.021])

min_tensors['cifar10'] = min_values.view(3, 1, 1).expand(3, 32, 32)
max_tensors['cifar10'] = max_values.view(3, 1, 1).expand(3, 32, 32)
min_tensors['cifar100'] = min_values.view(3, 1, 1).expand(3, 32, 32)
max_tensors['cifar100'] = max_values.view(3, 1, 1).expand(3, 32, 32)

min_values = torch.tensor(-0.4242)
max_values = torch.tensor(2.8208)
min_tensors['mnist'] = min_values.view(1, 1, 1).expand(1, 28, 28)
max_tensors['mnist'] = max_values.view(1, 1, 1).expand(1, 28, 28)

min_values = torch.tensor([-2.117, -2.035, -1.804]) 
max_values = torch.tensor([2.246, 2.427, 2.640])
min_tensors['vggface2'] = min_values.view(3, 1, 1).expand(3, 224, 224)
max_tensors['vggface2'] = max_values.view(3, 1, 1).expand(3, 224, 224)

min_values = torch.tensor([-2.2106, -2.2079, -2.4])
max_values = torch.tensor([2.8398, 2.7671, 2.6761])
min_tensors['svhn'] = min_values.view(3, 1, 1).expand(3, 32, 32)
max_tensors['svhn'] = max_values.view(3, 1, 1).expand(3, 32, 32)

class CombineDataset(Dataset):
    def __init__(self, loader1, loader2):
        if loader2 is None:
            combined_iter = loader1
        else:
            combined_iter = chain(loader1, loader2)
        data_list = []
        label_list = []
        for _, (x, y) in enumerate(combined_iter):
            data_list.extend(x.detach().cpu().numpy())
            label_list.extend(y.detach().cpu().numpy())
        # random.shuffle(data_list)

        self.data = data_list
        self.label = label_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class JointDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self._len = len(inputs)

    def __len__(self):
        'Denotes the total number of samples'
        return self._len

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_dataset(data_name, path='./data', if_backdoor=False):
    if not data_name in ['mnist', 'cifar10', 'cifar100', 'vggface2', 'svhn', 'fashionmnist']:
        raise TypeError('data_name should be a string, including mnist,cifar10. ')

    if (data_name == 'mnist'):
        trainset = datasets.MNIST(path, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))
        testset = datasets.MNIST(path, train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))

    elif data_name == 'cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        trainset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)

    elif data_name == 'cifar100':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        trainset = datasets.CIFAR100(root=path, train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root=path, train=False, download=True, transform=transform)

    elif data_name == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])

        trainset = datasets.SVHN(root=path+'/svhn', split='train', download=True, transform=transform)
        testset = datasets.SVHN(root=path+'/svhn', split='test', download=True, transform=transform)

    elif data_name == 'fashionmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize((0.5,), (0.5,)) 
        ])

        trainset = datasets.FashionMNIST(root=path, train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root=path, train=False, download=True, transform=transform)

    return trainset, testset


def get_dataloader(trainset, testset, batch_size, device):
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def get_backdoor_loader(trainset, testset, forget_class, num_forget, batch_size, square_size=5, ori_class=0,
                        limited_remain_class_num=500, data_name='cifar10', num_classes=10, bd_num=1000):
    """
    :return train_forget_loader: backdoored data with trigger, original class-0, relabel as 1
    :return train_remain_loader: all remaining benign data

    """

    train_forget_index = []
    test_forget_index = []
    train_remain_index = []
    test_remain_index = []
    train_oriforget_index = []
    test_oriforget_index = []
    train_bdremain_index = []
    limited_remain_index = []
    train_full_forget_index = []

    cnt_dict = {i: 0 for i in range(num_classes)}  # counter for images in limited_remain_set
    bd_cnt = 0  # counter for backdoored images

    for i in range(len(trainset)):
        # add the trigger and mislabel 
        if int(trainset.targets[i]) == ori_class and bd_cnt < bd_num:
            if data_name == 'mnist':
                white_tensor = torch.ones(square_size, square_size, dtype=torch.float)
                trainset.data[i][:square_size, :square_size] = white_tensor
            else:
                white_tensor = torch.ones(square_size, square_size, 3, dtype=torch.float)
                trainset.data[i][:square_size, :square_size, :] = white_tensor
            trainset.targets[i] = forget_class
            bd_cnt += 1
            train_full_forget_index.append(i)
            # limited forget index
            if len(train_forget_index) < num_forget:
                train_forget_index.append(i)

        # benign samples (including unbackdoored class-0 data)
        else:
            if int(trainset.targets[i]) == forget_class:
                train_oriforget_index.append(i)
            if int(trainset.targets[i]) == ori_class:
                train_bdremain_index.append(i)
            train_remain_index.append(i)
            if cnt_dict[int(trainset.targets[i])] < limited_remain_class_num:
                limited_remain_index.append(i)
                cnt_dict[int(trainset.targets[i])] += 1
    bd_cnt = 0
    for i in range(len(testset)):
        if int(testset.targets[i]) == ori_class and bd_cnt < (bd_num/(6 if data_name=='mnist' else 5)):
            if data_name == 'mnist':
                white_tensor = torch.ones(square_size, square_size, dtype=torch.float)
                testset.data[i][:square_size, :square_size] = white_tensor
            else:
                white_tensor = torch.ones(square_size, square_size, 3, dtype=torch.float)
                testset.data[i][:square_size, :square_size, :] = white_tensor
            testset.targets[i] = forget_class
            bd_cnt += 1
            test_forget_index.append(i)
        else:
            if int(testset.targets[i]) == forget_class:
                test_oriforget_index.append(i)
            test_remain_index.append(i)

    train_forget_subset = torch.utils.data.Subset(trainset, train_forget_index)
    train_remain_subset = torch.utils.data.Subset(trainset, train_remain_index)
    train_limited_remain_subset = torch.utils.data.Subset(trainset, limited_remain_index)
    train_oriforget_subset = torch.utils.data.Subset(trainset, train_oriforget_index)
    train_bdremain_subset = torch.utils.data.Subset(trainset, train_bdremain_index)
    test_forget_subset = torch.utils.data.Subset(testset, test_forget_index)
    test_remain_subset = torch.utils.data.Subset(testset, test_remain_index)
    test_oriforget_subset = torch.utils.data.Subset(testset, test_oriforget_index)
    train_full_forget_subset = torch.utils.data.Subset(trainset, train_full_forget_index)
    train_forget_loader = torch.utils.data.DataLoader(dataset=train_forget_subset, batch_size=batch_size,
                                                      sampler=torch.utils.data.RandomSampler(train_forget_subset))
    train_remain_loader = torch.utils.data.DataLoader(dataset=train_remain_subset, batch_size=batch_size,
                                                      sampler=torch.utils.data.RandomSampler(train_remain_subset))
    train_limited_remain_loader = torch.utils.data.DataLoader(dataset=train_limited_remain_subset, batch_size=batch_size,
                                                              sampler=torch.utils.data.RandomSampler(train_limited_remain_subset))
    train_oriforget_loader = torch.utils.data.DataLoader(dataset=train_oriforget_subset, batch_size=batch_size,
                                                         sampler=torch.utils.data.RandomSampler(train_oriforget_subset))
    train_bdremain_loader = torch.utils.data.DataLoader(dataset=train_bdremain_subset, batch_size=batch_size,
                                                        sampler=torch.utils.data.RandomSampler(train_bdremain_subset))
    test_forget_loader = torch.utils.data.DataLoader(dataset=test_forget_subset, batch_size=batch_size,
                                                     sampler=torch.utils.data.RandomSampler(test_forget_subset))
    test_remain_loader = torch.utils.data.DataLoader(dataset=test_remain_subset, batch_size=batch_size,
                                                     sampler=torch.utils.data.RandomSampler(test_remain_subset))
    test_oriforget_loader = torch.utils.data.DataLoader(dataset=test_oriforget_subset, batch_size=batch_size,
                                                        sampler=torch.utils.data.RandomSampler(test_oriforget_subset))
    train_full_forget_loader = torch.utils.data.DataLoader(dataset=train_full_forget_subset, batch_size=batch_size,
                                                           sampler=torch.utils.data.RandomSampler(train_full_forget_subset))
    return train_forget_loader, test_forget_loader, train_remain_loader, test_remain_loader, train_full_forget_loader, train_limited_remain_loader, \
           train_oriforget_loader, test_oriforget_loader, train_bdremain_loader


def split_class_data(dataset, forget_class, num_forget, num_classes=10, limited_remain_class_num=0):


    forget_index = []
    class_remain_index = [] 
    remain_index = []
    limited_remain_index = []
    full_forget_index = []
    cnt_forget = 0  
    cnt_dict = {i: 0 for i in range(num_classes)} 
    for i, (data, target) in enumerate(dataset):
        if target == forget_class:
            full_forget_index.append(i)
            if cnt_forget < num_forget:
                forget_index.append(i)
                cnt_forget += 1

            else:
                class_remain_index.append(i)
        else:
            remain_index.append(i)
            if cnt_dict[target] < limited_remain_class_num:
                limited_remain_index.append(i)
                cnt_dict[target] += 1
    return forget_index, remain_index, class_remain_index, full_forget_index, limited_remain_index


def get_unlearn_loader(trainset, testset, forget_class, batch_size, num_forget, repair_num_ratio=0.01, limited_remain_class_num=500, data_name='cifar10', num_classes=10):
    train_forget_index, train_remain_index, forget_remain_train_index, full_forget_train_index, limited_remain_train_index \
        = split_class_data(trainset, forget_class, num_forget=num_forget, num_classes=num_classes, limited_remain_class_num=limited_remain_class_num)
    test_forget_index, test_remain_index, _, _, _ = split_class_data(testset, forget_class, num_forget=5000, num_classes=num_classes)

    forget_remain_train_subset = torch.utils.data.Subset(trainset, forget_remain_train_index)
    train_forget_subset = torch.utils.data.Subset(trainset, train_forget_index)
    train_remain_subset = torch.utils.data.Subset(trainset, train_remain_index)
    train_full_forget_subset = torch.utils.data.Subset(trainset, full_forget_train_index)
    train_limited_remain_subset = torch.utils.data.Subset(trainset, limited_remain_train_index)
    test_forget_subset = torch.utils.data.Subset(testset, test_forget_index)
    test_remain_subset = torch.utils.data.Subset(testset, test_remain_index)

    if len(forget_remain_train_subset) == 0:
        forget_remain_train_loader = None
    else:
        forget_remain_train_loader = torch.utils.data.DataLoader(dataset=forget_remain_train_subset, batch_size=batch_size,
                                                                 sampler=torch.utils.data.RandomSampler(forget_remain_train_subset))
    train_forget_loader = torch.utils.data.DataLoader(dataset=train_forget_subset, batch_size=batch_size,
                                                      sampler=torch.utils.data.RandomSampler(train_forget_subset))
    train_remain_loader = torch.utils.data.DataLoader(dataset=train_remain_subset, batch_size=batch_size,
                                                      sampler=torch.utils.data.RandomSampler(train_remain_subset))
    train_full_forget_loader = torch.utils.data.DataLoader(dataset=train_full_forget_subset, batch_size=batch_size,
                                                              sampler=torch.utils.data.RandomSampler(train_full_forget_subset))
    train_limited_remain_loader = torch.utils.data.DataLoader(dataset=train_limited_remain_subset, batch_size=batch_size,
                                                              sampler=torch.utils.data.RandomSampler(train_limited_remain_subset))
    test_forget_loader = torch.utils.data.DataLoader(dataset=test_forget_subset, batch_size=batch_size,
                                                     sampler=torch.utils.data.RandomSampler(test_forget_subset))
    test_remain_loader = torch.utils.data.DataLoader(dataset=test_remain_subset, batch_size=batch_size,
                                                     sampler=torch.utils.data.RandomSampler(test_remain_subset))

    return train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, \
           forget_remain_train_loader, train_full_forget_loader, train_limited_remain_loader, \
           train_forget_index, train_remain_index, test_forget_index, test_remain_index, forget_remain_train_index

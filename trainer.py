import time
import warnings

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torchvision import models
from tqdm import tqdm

from models import AllCNN_sequential, resnet18, resnet50
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.functional import softmax

import numpy as np


def loss_picker(loss):
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        print("automatically assign mse loss function to you...")
        criterion = nn.MSELoss()

    return criterion


def optimizer_picker(optimization, param, lr, momentum=0.):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr, momentum=momentum)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)
    return optimizer


def train(model, data_loader, criterion, optimizer, schedulers=[], loss_mode='cross', device='cpu'):
    running_loss = 0
    model.train()
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)  # get predict label of batch_x

        if loss_mode == "mse":
            loss = criterion(output, batch_y)  # mse loss
        elif loss_mode == "cross":
            loss = criterion(output, batch_y)  # cross entropy loss
        elif loss_mode == 'neg_grad':
            loss = -criterion(output, batch_y)

        loss.backward()
        optimizer.step()
        running_loss += loss

    for scheduler in schedulers:
        scheduler.step()

    return running_loss


def lr_lambda(epoch):
    warmup_epochs = 30
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        return 0.1 ** ((epoch - warmup_epochs) / 150)


def train_save_model(train_loader, test_loader, model_name, optim_name, learning_rate, num_epochs, device, path, data_name='cifar10', num_classes=10, logger=None):
    start = time.time()
    if data_name == 'cifar10':
        n_channels = 3
        filters_percentage = 0.5
    elif data_name == 'mnist':
        n_channels = 1
        filters_percentage = 1.
    elif data_name == 'svhn':
        n_channels = 3
        filters_percentage = 1.
    elif data_name == 'vggface2':
        n_channels = 3
        filters_percentage = 1.
    elif data_name == 'cifar100':
        n_channels = 3
        filters_percentage = 1.

    if model_name == 'AllCNN':
        model = AllCNN_sequential(n_channels=n_channels, num_classes=num_classes, filters_percentage=filters_percentage).to(device)
    elif model_name == 'ResNet18':
        model = resnet18(n_channels=n_channels, num_classes=num_classes).to(device)
        # model = models.resnet18(pretrained=True)
    elif model_name == 'ResNet50':
        model = resnet50(n_channels=n_channels, num_classes=num_classes).to(device)

    criterion = loss_picker('cross')
    optimizer = optimizer_picker(optim_name, model.parameters(), lr=learning_rate, momentum=0.9)

    best_acc = 0

    for epo in range(num_epochs):
        print('EPOCH:{}'.format(epo))
        loss = train(model=model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, schedulers=[], loss_mode='cross',
                     device=device)

        _, acc = eval(model=model, data_loader=test_loader, mode='', print_perform=False, device=device)
        logger.info('loss: {}'.format(loss.item()))
        logger.info('test acc:{}'.format(acc))

        if acc >= best_acc:
            best_acc = acc
            torch.save(model, '{}.pth'.format(path))

    end = time.time()
    logger.info('training time: {}'.format(end - start))
    return model


def test(model, loader, num_classes=10, forget_class=1, logger=None):
    model.eval()
    outputavg = [0.] * num_classes
    cnt = [0] * num_classes
    logit_list = [0.] * num_classes
    total_logit = [[0. for _ in range(num_classes)] for _ in range(num_classes)]
    total_logit_adv = [[0. for _ in range(num_classes)] for _ in range(num_classes)]
    total_sm = [[0. for _ in range(num_classes)] for _ in range(num_classes)]
    remain_cnt = 0.
    remain_hit = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(tqdm(loader, leave=False)):
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            output_sm = softmax(output, dim=1)
            pred = torch.argmax(output, dim=1)

            for outlogit in output:
                for i in range(0, num_classes):
                    logit_list[i] += outlogit[i].cpu().item()

            for i in range(len(target)):
                gt_class = target[i]
                for j in range(0, num_classes):
                    total_logit[gt_class][j] += output[i][j].cpu().item()
                    total_sm[gt_class][j] += output_sm[i][j].cpu().item()

                if pred[i] == target[i]:
                    outputavg[pred[i]] += 1

                if target[i] != forget_class:
                    remain_cnt += 1
                    if pred[i] == target[i]:
                        remain_hit += 1

                cnt[gt_class] += 1

    logit_list = [x / len(loader.dataset) for x in logit_list]

    # logger.info("all test samples logit avg")
    # logger.info(logit_list)
    for i in range(0, num_classes):
        for j in range(len(total_logit[i])):
            if total_logit[i][j] == 0:
                continue
            total_logit[i][j] /= cnt[i]
            total_sm[i][j] /= cnt[i]

    total_acc = sum(outputavg) / len(loader.dataset)
    remain_acc = (remain_hit / remain_cnt) if remain_cnt != 0 else 0
    logger.info('total acc: {:.2%}'.format(total_acc))
    logger.info('Remain acc: {:.2%}\n'.format(remain_acc))

    for i in range(len(outputavg)):
        if cnt[i] == 0:
            outputavg[i] = 0.
        else:
            outputavg[i] /= cnt[i]
        logger.info('class {} acc: {:.2%}'.format(i, outputavg[i]))
    return remain_acc, outputavg[forget_class]


def eval(model, data_loader, batch_size=64, mode='backdoor', print_perform=False, device='cpu', name=''):
    model.eval()  # switch to eval status

    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        batch_y_predict = model(batch_x)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_predict.append(batch_y_predict)
        y_true.append(batch_y)

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)

    num_hits = (y_true == y_predict).float().sum()
    acc = num_hits / y_true.shape[0]

    return accuracy_score(y_true.cpu(), y_predict.cpu()), acc.item()

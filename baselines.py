import copy
import itertools

from matplotlib import pyplot as plt

import lipschitz
from SSD import ParameterPerturber
from datasets import CombineDataset
from IAU import finetune
from trainer import eval, loss_picker, optimizer_picker, test, train
import numpy as np
import torch
from torch import nn
import tqdm
import time
from collections import Counter


import torch.nn.functional as F
import torch.distributions as distributions

class AttackBase(object):
    def __init__(self, model=None, norm=False, discrete=True, device=None, data_name=None):
        self.model = model
        self.norm = norm
        # Normalization are needed for CIFAR10, ImageNet
        if self.norm:
            if data_name == 'cifar10':
                self.mean = (0.5071, 0.4867, 0.4408)
                self.std = (0.2675, 0.2565, 0.2761)
            elif data_name == 'cifar100':
                self.mean = (0.5071, 0.4867, 0.4408)
                self.std = (0.2675, 0.2565, 0.2761)
            elif data_name == 'mnist':
                self.mean = (0.1307,)
                self.std = (0.3081,)

        self.discrete = discrete
        self.device = device or torch.device("cuda:0")
        self.loss(device=self.device)

    def loss(self, custom_loss=None, device=None):
        device = device or self.device
        self.criterion = custom_loss or nn.CrossEntropyLoss()
        self.criterion.to(device)

    def perturb(self, x):
        raise NotImplementedError

    def normalize(self, x):
        if self.norm:
            y = x.clone().to(x.device)
            y[:, 0, :, :] = (y[:, 0, :, :] - self.mean[0]) / self.std[0]
            y[:, 1, :, :] = (y[:, 1, :, :] - self.mean[1]) / self.std[1]
            y[:, 2, :, :] = (y[:, 2, :, :] - self.mean[2]) / self.std[2]
            return y
        return x

    def inverse_normalize(self, x):
        if self.norm:
            y = x.clone().to(x.device)
            y[:, 0, :, :] = y[:, 0, :, :] * self.std[0] + self.mean[0]
            y[:, 1, :, :] = y[:, 1, :, :] * self.std[1] + self.mean[1]
            y[:, 2, :, :] = y[:, 2, :, :] * self.std[2] + self.mean[2]
            return y
        return x

    def discretize(self, x):
        return torch.round(x * 255) / 255

    def clamper(self, x_adv, x_nat, bound=None, metric="inf", inverse_normalized=False):
        if not inverse_normalized:
            x_adv = self.inverse_normalize(x_adv)
            x_nat = self.inverse_normalize(x_nat)
        if metric == "inf":
            clamp_delta = torch.clamp(x_adv - x_nat, -bound, bound)
        else:
            clamp_delta = x_adv - x_nat
            for batch_index in range(clamp_delta.size(0)):
                image_delta = clamp_delta[batch_index]
                image_norm = image_delta.norm(p=metric, keepdim=False)
                # TODO: channel isolation?
                if image_norm > bound:
                    clamp_delta[batch_index] /= image_norm
                    clamp_delta[batch_index] *= bound

        x_adv = x_nat + clamp_delta
        x_adv = torch.clamp(x_adv, 0., 1.)
        return self.normalize(self.discretize(x_adv)).clone().detach().requires_grad_(True)
    

class FGSM(AttackBase):
    def __init__(self, model=None, bound=None, norm=False, random_start=False, discrete=True, device=None, data_name=None, **kwargs):
        super(FGSM, self).__init__(model, norm, discrete, device, data_name)
        self.bound = bound
        self.rand = random_start

    def perturb(self, x, y, model=None, bound=None, device=None, **kwargs):
        criterion = self.criterion
        model = model or self.model
        bound = bound or self.bound
        device = device or self.device

        model.zero_grad()
        x_nat = self.inverse_normalize(x.detach().clone().to(device))
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        if self.rand:
            rand_perturb_dist = distributions.uniform.Uniform(-bound, bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            x_adv = self.clamper(self.inverse_normalize(x_adv) + rand_perturb, x_nat, bound=bound,
                                 inverse_normalized=True)
            if self.discretize:
                x_adv = self.normalize(self.discretize(x_adv)).detach().clone().requires_grad_(True)
            else:
                x_adv = self.normalize(x_adv).detach().clone().requires_grad_(True)
        pred = model(x_adv)
        if criterion.__class__.__name__ == "NLLLoss":
            pred = F.softmax(pred, dim=-1)
        loss = criterion(pred, y)
        loss.backward()

        grad_sign = x_adv.grad.data.detach().sign()
        x_adv = self.inverse_normalize(x_adv) + grad_sign * bound
        x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)

        return x_adv.detach()
    

# Implementation from https://github.com/vikram2000b/Fast-Machine-Unlearning
class Noise(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self):
        return self.noise
    

def NG(ori_model, train_forget_loader, train_remain_loader, test_loader, test_forget_loader, test_remain_loader,
                    train_full_forget_loader=None, train_limited_remain_loader=None,
                    unlearn_epochs=8, lr=0.000005, momentum=0.9,
                    limit_n=1, data_name='cifar10', num_classes=10, forget_class=1, model_name='AllCNN', logger=None):
    if data_name=='cifar10' and model_name=='ResNet18':
        unlearn_epochs = 5
        lr = 0.000002
    if data_name == 'cifar10' and model_name == 'AllCNN':
        unlearn_epochs = 8
        lr = 0.000005
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unlearn_model = copy.deepcopy(ori_model).to(device)
    unlearn_model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=lr, momentum=momentum)
    for epoch in tqdm.tqdm(range(unlearn_epochs)):
        total_loss = 0
        for idx, (x, y) in enumerate(train_forget_loader):
            x = x.to(device)
            y = y.to(device)
            unlearn_model.zero_grad()
            optimizer.zero_grad()
            logit = unlearn_model(x)
            loss = -criterion(logit, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('Average Loss value for unlearn epoch {}'.format(epoch), total_loss)

    print('unlearn time: ', time.time()-start_time)
    _, acc_df = eval(unlearn_model, train_remain_loader, device=device)
    _, acc_dr = eval(unlearn_model, train_full_forget_loader, device=device)
    logger.info('Train Remain Loader eval acc: {}'.format(acc_df))
    logger.info('Train Forget Loader eval acc: {}'.format(acc_dr))
    _, acc_tf = eval(model=unlearn_model, data_loader=test_forget_loader, mode='', print_perform=False, device=device)
    _, acc_tr = eval(model=unlearn_model, data_loader=test_remain_loader, mode='', print_perform=False, device=device)
    logger.info('test forget acc: {}'.format(acc_tf))
    logger.info('test remain acc: {}'.format(acc_tr))
    # test(unlearn_model, test_loader, num_classes=num_classes, forget_class=forget_class, logger=logger)

    """
    Unlearning Recovery test
    """
    # ft_lr = 0.1
    # ft_ep = 50
    # print("ft lr: ", ft_lr)
    # print("ft epoch: ", ft_ep)
    # ft_start_time = time.time()
    # unlearn_model = finetune(unlearn_model, train_limited_remain_loader, device, finetune_epochs=ft_ep, lr=ft_lr,
    #                          test_forget_loader=test_forget_loader, test_remain_loader=test_remain_loader, logger=logger)
    # print("ft time: ", time.time()-ft_start_time)
    # _, acc_tf = eval(model=unlearn_model, data_loader=test_forget_loader, mode='', print_perform=False, device=device)
    # _, acc_tr = eval(model=unlearn_model, data_loader=test_remain_loader, mode='', print_perform=False, device=device)
    # logger.info('test forget acc: {}'.format(acc_tf))
    # logger.info('test remain acc: {}'.format(acc_tr))

    # test(unlearn_model, test_loader, num_classes=num_classes, forget_class=forget_class, logger=logger)

    return unlearn_model

# Implementation from https://www.dropbox.com/s/bwu543qsdy4s32i/Boundary-Unlearning-Code.zip?dl=0
def BS(ori_model, train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader,
                    test_loader, device, train_full_forget_loader=None, train_limited_remain_loader=None,
                    bound=0.1, poison_epoch=10, norm=True, lr=0.0001, momentum=0.9,
                    limit_n=1, data_name='cifar10', num_classes=10, forget_class=1, model_name='AllCNN', logger=None):
    if model_name=='ResNet18' and data_name=='cifar10':
        poison_epoch = 10
        lr = 0.00001
    elif model_name=='AllCNN' and data_name=='cifar10':
        poison_epoch = 10
        lr = 0.0001

    print("bound: ", bound)
    print("bs unlearn lr: ", lr)
    print("ep: ", poison_epoch)
    random_start = False  # False if attack != "pgd" else True
    norm = False if data_name == 'mnist' else True  # BS用的参数

    test_model = copy.deepcopy(ori_model).to(device)
    unlearn_model = copy.deepcopy(ori_model).to(device)
    start_time = time.time()

    adv = FGSM(test_model, bound, norm, random_start, device, data_name=data_name)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.KLDivLoss()
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=lr, momentum=momentum)

    num_hits = 0
    num_sum = 0
    nearest_label = []
    show = False
    draw = False
    feature1_list = []
    feature2_list = []
    feature3_list = []
    logit_list = []

    for epoch in tqdm.tqdm(range(poison_epoch), desc='Unlearning outer loop epochs={}'.format(poison_epoch)):
        for idx, (x, y) in enumerate(train_forget_loader):
            x = x.to(device)
            y = y.to(device)
            test_model.eval()

            x_adv = adv.perturb(x, y, target_y=None, model=test_model, device=device)
            teacher_adv_features, teacher_adv_logits = test_model.extract_feature(x_adv)

            # 原模型对扰动数据生成的预测置信度
            pred_label = torch.argmax(teacher_adv_logits, dim=1)
            nearest_label.extend(pred_label.tolist())

            # 扰动后的预测 和 遗忘数据标签 不同的数量【扰动后的预测应该尽可能的和原始标签不同】
            num_hits += (y != pred_label).float().sum()
            # y的shape就是64(batch数), shape[0]是总的标签个数
            num_sum += y.shape[0]

            # adv_train
            unlearn_model.train()
            unlearn_model.zero_grad()
            optimizer.zero_grad()

            # 原模型生成的预测 & 对抗数据生成标签 的loss
            ori_features, ori_logits = unlearn_model.extract_feature(x)  # 遗忘样本
            ori_loss = criterion(ori_logits, pred_label)

            ori_loss.backward()
            optimizer.step()

    counter = Counter()
    counter.update(nearest_label)
    print(counter)

    print('attack success ratio:', (num_hits / num_sum).float())
    print('boundary shrink time:', (time.time() - start_time))

    _, acc_df = eval(unlearn_model, train_remain_loader, device=device)
    _, acc_dr = eval(unlearn_model, train_full_forget_loader, device=device)
    logger.info('Train Remain Loader eval acc: {}'.format(acc_df))
    logger.info('Train Forget Loader eval acc: {}'.format(acc_dr))
    _, acc_tf = eval(model=unlearn_model, data_loader=test_forget_loader, mode='', print_perform=False, device=device)
    _, acc_tr = eval(model=unlearn_model, data_loader=test_remain_loader, mode='', print_perform=False, device=device)
    logger.info('test forget acc: {}'.format(acc_tf))
    logger.info('test remain acc: {}'.format(acc_tr))
    # test(unlearn_model, test_loader, num_classes=num_classes, forget_class=forget_class, logger=logger)

    """
    Unlearning Recovery test
    """
    # ft_lr = 0.01
    # ft_ep = 50
    # print("ft lr: ", ft_lr)
    # print("ft epoch: ", ft_ep)
    # ft_start_time = time.time()
    # unlearn_model = finetune(unlearn_model, train_limited_remain_loader, device, finetune_epochs=ft_ep, lr=ft_lr,
    #                          test_forget_loader=test_forget_loader, test_remain_loader=test_remain_loader, logger=logger)
    # print("ft time: ", time.time()-ft_start_time)
    # # _, acc_df = eval(unlearn_model, train_remain_loader, device=device)
    # # _, acc_dr = eval(unlearn_model, train_full_forget_loader, device=device)
    # # logger.info('Train Remain Loader eval acc: {}'.format(acc_df))
    # # logger.info('Train Forget Loader eval acc: {}'.format(acc_dr))
    # _, acc_tf = eval(model=unlearn_model, data_loader=test_forget_loader, mode='', print_perform=False, device=device)
    # _, acc_tr = eval(model=unlearn_model, data_loader=test_remain_loader, mode='', print_perform=False, device=device)
    # logger.info('test forget acc: {}'.format(acc_tf))
    # logger.info('test remain acc: {}'.format(acc_tr))
    # test(unlearn_model, test_loader, num_classes=num_classes, forget_class=forget_class, logger=logger)

    return unlearn_model


# Implementation from https://github.com/vikram2000b/Fast-Machine-Unlearning
def UNSIR(ori_model, train_forget_loader, train_remain_loader, test_loader, test_forget_loader, test_remain_loader, train_full_forget_loader=None, train_limited_remain_loader=None,
          noise_ep=5, noise_step=8, data_name='cifar10', num_classes=10, forget_class=1, model_name='AllCNN', logger=None):

    start_time_all = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unlearn_model = copy.deepcopy(ori_model).to(device)

    # noise generation
    noises = {}
    classes_to_forget = [1]
    batch_size = 128
    for cls in classes_to_forget:
        if data_name=='cifar10' or data_name=='cifar100':
            noises[cls] = Noise(batch_size, 3, 32, 32).to(device)
        elif data_name=='mnist':
            noises[cls] = Noise(batch_size, 1, 28, 28).to(device)

        opt = torch.optim.Adam(noises[cls].parameters(), lr=0.1)

        num_epochs = 5
        num_steps = 8
        for epoch in range(num_epochs):
            total_loss = []
            for batch in range(num_steps):
                inputs = noises[cls]()
                labels = torch.zeros(batch_size).to(device) + cls  
                outputs = unlearn_model(inputs)
                loss = -nn.functional.cross_entropy(outputs, labels.long()) + 0.1 * torch.mean(
                    torch.sum(torch.square(inputs), [1, 2, 3]))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss.append(loss.cpu().detach().numpy())
            print("Loss: {}".format(np.mean(total_loss)))

    num_noises = 20
    if data_name=='cifar10' and model_name == 'AllCNN':
        ip_lr = 0.02
        rp_lr = 0.01
    elif data_name=='cifar10' and model_name == 'ResNet18':
        ip_lr = 0.0002
        rp_lr = 0.0001
    elif data_name=='cifar100' and model_name=='ResNet18':
        first_ip_lr = 0.0001
        last_ip_lr = 0.01
        first_rp_lr = 0.0001
        last_rp_lr = 0.005

    noisy_data = []
    for cls in classes_to_forget:
        for i in range(num_noises):  
            batch = noises[cls]().cpu().detach()
            for i in range(batch[0].size(0)):  
                noisy_data.append((batch[i], torch.tensor(forget_class)))

    noisy_loader = torch.utils.data.DataLoader(noisy_data, batch_size=64, shuffle=True)
    combined_set = CombineDataset(train_limited_remain_loader, noisy_loader)
    combined_loader = torch.utils.data.DataLoader(combined_set, batch_size=64, shuffle=True)
    logger.info('noise generation time: {}'.format(time.time()-start_time_all))

    start_time = time.time()
    if data_name == 'cifar100' and model_name == 'ResNet18':
        param_groups = [
            {"params": [param for name, param in unlearn_model.named_parameters() if "fc" not in name], "lr": first_ip_lr},
            {"params": unlearn_model.fc.parameters(), "lr": last_ip_lr}
        ]
        optimizer = torch.optim.Adam(param_groups)
    else:
        optimizer = torch.optim.Adam(unlearn_model.parameters(), lr=ip_lr)
    for epoch in range(1):
        unlearn_model.train(True)
        running_loss = 0.0
        running_acc = 0
        cnt = 0
        for i, data in enumerate(combined_loader):
            cnt += 1
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = unlearn_model(inputs)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item() * inputs.size(0)
            out = torch.argmax(outputs.detach(), dim=1)
            assert out.shape == labels.shape
            running_acc += (labels == out).sum().item()

        # print('batch number :', cnt)
    logger.info('impair time: {}'.format(time.time() - start_time))
    # test(unlearn_model, noisy_loader)
    # test(unlearn_model, test_loader, num_classes=num_classes, forget_class=forget_class, logger=logger)

    if data_name=='cifar100' and model_name=='ResNet18':
        param_groups = [
            {"params": [param for name, param in unlearn_model.named_parameters() if "fc" not in name],
             "lr": first_rp_lr},
            {"params": unlearn_model.fc.parameters(), "lr": last_rp_lr}
        ]
        optimizer = torch.optim.Adam(param_groups)
    else:
        optimizer = torch.optim.Adam(unlearn_model.parameters(), lr=rp_lr)

    unlearn_model.zero_grad()
    criterion = nn.CrossEntropyLoss()
    for ep in range(1):
        loss = train(unlearn_model, train_limited_remain_loader, criterion=criterion, optimizer=optimizer, loss_mode='cross', device=device)
        if ep%50==0:
            print(loss.item())
        logger.info('Average Loss value for finetune epoch {}: {}'.format(ep, loss.item()))

    logger.info('unlearn time: {}'.format(time.time() - start_time_all))
    _, acc = eval(unlearn_model, train_remain_loader, device=device)
    logger.info('Train Remain Loader eval acc: {}'.format(acc))
    _, acc = eval(unlearn_model, train_full_forget_loader, device=device)
    logger.info('Train Forget Loader eval acc: {}'.format(acc))
    # test(unlearn_model, test_loader, num_classes=num_classes, forget_class=forget_class, logger=logger)
    _, acc_tf = eval(model=unlearn_model, data_loader=test_forget_loader, mode='', print_perform=False, device=device)
    _, acc_tr = eval(model=unlearn_model, data_loader=test_remain_loader, mode='', print_perform=False, device=device)
    logger.info('test forget acc: {}'.format(acc_tf))
    logger.info('test remain acc: {}'.format(acc_tr))

    return unlearn_model


# Implementation from https://github.com/jwf40/Information-Theoretic-Unlearning
def JiT(ori_model, train_forget_loader, train_remain_loader, train_loader, test_loader, test_forget_loader, test_remain_loader, train_full_forget_loader=None, train_limited_remain_loader=None,
        epochs=25, lr=0.04, data_name='cifar10', num_classes=10, forget_class=1, model_name='AllCNN', logger=None):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unlearn_model = copy.deepcopy(ori_model).to(device)
    parameters = {
        "lower_bound": -1,  # unused
        "exponent": -1,  # unused
        "magnitude_diff": None,  # unused
        "min_layer": -1,  # -1: all layers are available for modification
        "max_layer": -1,  # -1: all layers are available for modification
        "forget_threshold": -1,  # unused
        "dampening_constant": -1,  # Lambda from paper
        "selection_weighting": -1,  # Alpha from paper
        "use_quad_weight": -1,
        "ewc_lambda":-1,
        "n_epochs": 1,

        "lipschitz_weighting": 0.3,  
        "learning_rate":  0.000001, 
        "n_samples": 25
    }
    print("lipschitz_weighting", parameters["lipschitz_weighting"])
    print("learning_rate", parameters["learning_rate"])
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=parameters['learning_rate'])

    pdr = lipschitz.Lipschitz(unlearn_model, optimizer, device, parameters)
    pdr.modify_weight(train_forget_loader)

    logger.info('unlearn time: {}'.format(time.time() - start_time))
    _, acc = eval(unlearn_model, train_remain_loader, device=device)
    logger.info('Train Remain Loader eval acc: {}'.format(acc))
    _, acc = eval(unlearn_model, train_full_forget_loader, device=device)
    logger.info('Train Forget Loader eval acc: {}'.format(acc))
    _, acc_tf = eval(model=unlearn_model, data_loader=test_forget_loader, mode='', print_perform=False, device=device)
    _, acc_tr = eval(model=unlearn_model, data_loader=test_remain_loader, mode='', print_perform=False, device=device)
    logger.info('test forget acc: {}'.format(acc_tf))
    logger.info('test remain acc: {}'.format(acc_tr))
    # test(unlearn_model, test_loader, num_classes=num_classes, forget_class=forget_class, logger=logger)

    """
    Unlearning Recovery test
    """
    # ft_lr = 0.00001
    # print("ft_lr: ", ft_lr)
    # ft_start_time = time.time()
    # unlearn_model = finetune(unlearn_model, train_limited_remain_loader, device, finetune_epochs=1, lr=ft_lr,
    #                          test_forget_loader=test_forget_loader, test_remain_loader=test_remain_loader, logger=logger)
    # print("ft time: ", time.time()-ft_start_time)
    # _, acc = eval(unlearn_model, test_remain_loader, device=device)
    # logger.info('Train Remain Loader eval acc: {}'.format(acc))
    # _, acc = eval(unlearn_model, test_forget_loader, device=device)
    # logger.info('Train Forget Loader eval acc: {}'.format(acc))
    # test(unlearn_model, test_loader, num_classes=num_classes, forget_class=forget_class, logger=logger)

    return unlearn_model

# Implementation from https://github.com/if-loops/selective-synaptic-dampening/blob/main/src/ssd.py
def SSD(ori_model, train_forget_loader, train_remain_loader, train_loader, test_loader, test_forget_loader, test_remain_loader,
                train_full_forget_loader=None, train_limited_remain_loader=None,
                data_name='cifar10', num_classes=10, forget_class=1, model_name='AllCNN',
                logger=None):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unlearn_model = copy.deepcopy(ori_model).to(device)
    parameters = {
        "lower_bound": 1,  # unused
        "exponent": 1,  # unused
        "magnitude_diff": None,  # unused
        "min_layer": -1,  # -1: all layers are available for modification # unused
        "max_layer": -1,  # -1: all layers are available for modification # unused
        "forget_threshold": 1,  # unused

        "dampening_constant": 0.7,  # Lambda from paper
        "selection_weighting": 1.5,  # Alpha from paper
        "lr": 0.1
    }
    print("dampening_constant: ", parameters['dampening_constant'])
    print("selection_weighting: ", parameters['selection_weighting'])
    print("lr: ", parameters['lr'])
    # load the trained model
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=parameters["lr"])

    pdr = ParameterPerturber(unlearn_model, optimizer, device, parameters)

    unlearn_model = unlearn_model.eval()

    # Calculation of the forget set importances
    sample_importances = pdr.calc_importance(train_forget_loader)

    combined_set = CombineDataset(train_remain_loader, train_forget_loader)
    importance_loader = torch.utils.data.DataLoader(combined_set, batch_size=64, shuffle=True)

    # Calculate the importances of D (see paper); this can also be done at any point before forgetting.
    original_importances = pdr.calc_importance(importance_loader)

    # Dampen selected parameters
    pdr.modify_weight(original_importances, sample_importances)

    print('unlearn time: ', time.time()-start_time)
    _, acc = eval(unlearn_model, train_remain_loader, device=device)
    logger.info('Train Remain Loader eval acc: {}'.format(acc))
    _, acc = eval(unlearn_model, train_full_forget_loader, device=device)
    logger.info('Train Forget Loader eval acc: {}'.format(acc))
    _, acc_tf = eval(model=unlearn_model, data_loader=test_forget_loader, mode='', print_perform=False, device=device)
    _, acc_tr = eval(model=unlearn_model, data_loader=test_remain_loader, mode='', print_perform=False, device=device)
    logger.info('test forget acc: {}'.format(acc_tf))
    logger.info('test remain acc: {}'.format(acc_tr))
    # test(unlearn_model, test_loader, num_classes=num_classes, forget_class=forget_class, logger=logger)

    """
    Unlearning Recovery test
    """
    # ft_lr = 0.0000001
    # print('finetune lr: ', ft_lr)
    # unlearn_model = finetune(unlearn_model, train_limited_remain_loader, device, finetune_epochs=10, lr=ft_lr,
    #                          test_forget_loader=test_forget_loader, test_remain_loader=test_remain_loader, logger=logger)
    # _, acc = eval(unlearn_model, train_remain_loader, device=device)
    # logger.info('Train Remain Loader eval acc: {}'.format(acc))
    # _, acc = eval(unlearn_model, train_forget_loader, device=device)
    # logger.info('Train Forget Loader eval acc: {}'.format(acc))
    # test(unlearn_model, test_loader, num_classes=num_classes, forget_class=forget_class, logger=logger)

    return unlearn_model

import copy
from collections import Counter
from itertools import chain

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
import tqdm
import time

from datasets import min_tensors, max_tensors, JointDataset, CombineDataset
from inverse_adversary import IAE_FGSM
from models import distillation_loss
from trainer import test_backdoor_ASR, test, train, eval


def IAU(ori_model, train_forget_loader, train_remain_loader, train_loader, test_loader, test_forget_loader, test_remain_loader,
        train_full_forget_loader=None, train_limited_remain_loader=None,
        strike_epochs=5, strike_iae_num=1, strike_lr=0.02, 
        is_limited=True, limit_n=1,
        rally_epochs=10, rally_iae_num=20, rally_lr=0.01,
        data_name='cifar10', num_classes=10, forget_class=1, model_name='AllCNN', logger=None):
    """
    :param strike_epochs: impair的epoch数
    :param train_forget_loader: 后门实验时，这个参数为带trigger的后门数据
    :param train_remain_loader: 如果num_forget<5000的话，这个集是包含良性遗忘类数据的
    :param train_limited_remain_loader: 这个集不包含任何遗忘类数据的

    :param test_loader: 正常实验时为完整test集合；后门实验时，为良性test集（不包含后门数据的remain数据）
    :param bdchanged_test_forget_loader: 正常实验时为None；后门实验时，为带trigger的test后门集，用于测试ASR
    :return:
    """
    # iter(train_forget_loader)
    params = locals()  # 获取所有参数及其值
    i = 0
    for key, value in params.items():
        if i >= 8:
            print(f"{key}: {value}")
        i += 1

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unlearn_model = copy.deepcopy(ori_model).to(device)
    unlearn_model.train()
    unlearn_model.zero_grad()

    test_model = copy.deepcopy(ori_model).to(device)
    test_model.eval()
    test_model.zero_grad()

    clip_min = min_tensors[data_name].to(device)
    clip_max = max_tensors[data_name].to(device)

    iadv = IAE_FGSM(test_model, eps=4/255, eps_iter=4/255, clip_min=clip_min, clip_max=clip_max, targeted=True)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=lr_ip, momentum=0.9)
    optimizer = torch.optim.Adam(unlearn_model.parameters(), lr=strike_lr)
    for epoch in tqdm.tqdm(range(strike_epochs)):
        cnt = 0
        total_loss = 0
        num_equal_iadv = 0
        num_sum = 0
        # utils.draw_cam(unlearn_model, data_name=data_name, loader=iadv_loader, title='unlearn ep={}'.format(epoch))
        unlearn_model.train()

        for idx, (x, y) in enumerate(train_forget_loader):
            for it in range(strike_iae_num):
                cnt += 1
                x, y = x.to(device), y.to(device)

                x_iadv = iadv.perturb(x, y)

                optimizer.zero_grad()
                unl_logit_iadv = unlearn_model(x_iadv)
                test_logit = test_model(x_iadv)
                iadv_pred = torch.argmax(test_logit, dim=1)
                num_equal_iadv += (y == iadv_pred).float().sum().item()
                num_sum += y.shape[0]

                ul_loss = -criterion(unl_logit_iadv, y)
                loss = ul_loss
                loss.backward()

                optimizer.step()
                total_loss += loss.item()
        logger.info('Average Loss value for unlearn epoch {}: {}'.format(epoch, total_loss))
        logger.info("Iadv Pred Acc: {}".format(num_equal_iadv / num_sum))

    strike_time = time.time()-start_time
    logger.info('Strike time cost: {}'.format(strike_time))

    # Rally Stage
    start_time = time.time()
    if is_limited:
        ft_remain_loader = train_limited_remain_loader
    else:
        ft_remain_loader = train_remain_loader

    
    unlearned_model = rally(unlearn_model, ori_model, train_forget_loader, ft_remain_loader, device,
                            finetune_epochs=rally_epochs, lr=rally_lr, iadv_num=rally_iae_num,
                            data_name=data_name, logger=logger)
    
    rally_time = time.time() - start_time
    logger.info('-------------test after Rally----------------')
    total_unlearn_time = strike_time + rally_time
    logger.info('total unlearning time: {}'.format(total_unlearn_time))
    _, acc_df = eval(model=unlearned_model, data_loader=train_full_forget_loader, mode='', print_perform=False, device=device)
    _, acc_dr = eval(model=unlearned_model, data_loader=train_remain_loader, mode='', print_perform=False, device=device)
    logger.info('train forget acc: {}'.format(acc_df))
    logger.info('train remain acc: {}'.format(acc_dr))
    _, acc_tf = eval(model=unlearned_model, data_loader=test_forget_loader, mode='', print_perform=False, device=device)
    _, acc_tr = eval(model=unlearned_model, data_loader=test_remain_loader, mode='', print_perform=False, device=device)
    logger.info('test forget acc: {}'.format(acc_tf))
    logger.info('test remain acc: {}'.format(acc_tr))
    
    return unlearned_model, acc_df, acc_dr, acc_tf, acc_tr, total_unlearn_time


def finetune(unlearn_model, train_remain_loader, device, finetune_epochs=1, lr=0.01, momentum=0.9, optimizer_name='SGD',
             test_forget_loader=None, test_remain_loader=None, logger=None):
    unl_model = copy.deepcopy(unlearn_model).to(device)
    unl_model.zero_grad()
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(unl_model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.Adam(unl_model.parameters(), lr=lr)
    lowest_loss = float('inf')
    for ep in range(finetune_epochs):
        loss = train(unl_model, train_remain_loader, criterion=criterion, optimizer=optimizer, loss_mode='cross',
                     device=device)
        if loss < lowest_loss:
            lowest_loss = loss
            ret_model = copy.deepcopy(unl_model).to(device)
        logger.info('Average Loss value for finetune epoch {}: {}'.format(ep, loss.item()))
        _, accf = eval(unl_model, test_forget_loader, device=device)
        _, accr = eval(unl_model, test_remain_loader, device=device)
        logger.info('test forget Loader eval acc: {}'.format(accf))
        logger.info('test Remain Loader eval acc: {}'.format(accr))

    return ret_model


def rally(unlearn_model, original_model, train_forget_loader, train_remain_loader, device, finetune_epochs=1,
                  lr=0.01, momentum=0.9, iadv_num=1, data_name='cifar10', logger=None):
    unl_model = copy.deepcopy(unlearn_model).to(device)
    test_model = copy.deepcopy(original_model).to(device)
    unl_model.train()
    unl_model.zero_grad()
    test_model.eval()
    test_model.zero_grad()

    clip_min = min_tensors[data_name].to(device)
    clip_max = max_tensors[data_name].to(device)

    iadv = IAE_FGSM(test_model, eps=4 / 255, eps_iter=4 / 255, clip_min=clip_min, clip_max=clip_max, targeted=True)

    kl_criterion = nn.KLDivLoss()
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unl_model.parameters(), lr=lr, momentum=momentum)
    # optimizer = torch.optim.Adam(unlearn_model.parameters(), lr=lr)

    num_equal_iadv = 0.
    num_sum = 0.
    iadv_label = []
    lowest_loss = float('inf')
    for epoch in tqdm.tqdm(range(finetune_epochs)):
        cnt = 0
        total_loss = 0
        for idx, (x, y) in enumerate(train_remain_loader):
            cnt += 1
            x, y = x.to(device), y.to(device)
            x_copy = copy.deepcopy(x).to(device)

            for it in range(iadv_num):
                x_iadv = iadv.perturb(x_copy, y)

                teacher_feats, teacher_logit = test_model.extract_feature(x)
                teacher_feats_iadv, teacher_logit_iadv = test_model.extract_feature(x_iadv)
                iadv_pred = torch.argmax(teacher_logit_iadv, dim=1)
                iadv_label.extend(iadv_pred.tolist())

                optimizer.zero_grad()
                unl_feats, unl_logit = unl_model.extract_feature(x)
                unl_feats_iadv, unl_logit_iadv = unl_model.extract_feature(x_iadv)
                pred = torch.argmax(unl_logit_iadv, dim=1)
                num_equal_iadv += (y == iadv_pred).float().sum().item()
                num_sum += y.shape[0]

                loss1 = ce_criterion(unl_logit, y)  # remain集正常

                # Feature Distillation Losses
                student_outs = unl_feats + [unl_logit]
                student_outs_iadv = unl_feats_iadv + [unl_logit_iadv]
                teacher_outs = teacher_feats + [teacher_logit]
                teacher_outs_iadv = teacher_feats_iadv + [teacher_logit_iadv]
                fd_loss_o2o = distillation_loss(student_outs, teacher_outs)  
                fd_loss_o2i = distillation_loss(student_outs, teacher_outs_iadv) 
                fd_loss_i2i = distillation_loss(student_outs_iadv, teacher_outs_iadv)

                ul_loss = loss1 + fd_loss_o2o + fd_loss_o2i + fd_loss_i2i     
                ul_loss.backward()
                optimizer.step()
                total_loss += ul_loss.item()

        logger.info('Average Loss value for unlearn epoch {}: {}'.format(epoch, total_loss))
        logger.info("Iadv Pred Acc: {}".format(num_equal_iadv / num_sum))
        if total_loss < lowest_loss:
            lowest_loss = total_loss
            ret_model = copy.deepcopy(unl_model).to(device)
        if epoch % 3 == 0:
            _, acc = eval(unl_model, train_remain_loader, device=device)
            logger.info('Train Remain Loader eval acc: {}'.format(acc))
            _, acc = eval(unl_model, train_forget_loader, device=device)
            logger.info('Train Forget Loader eval acc: {}'.format(acc))
    counter = Counter()
    counter.update(iadv_label)
    logger.info(counter)

    return ret_model


class JSDivLoss(nn.Module):
    def __init__(self):
        super(JSDivLoss, self).__init__()

    def forward(self, p, q, temperature):
        epsilon = 1e-10
        p = torch.softmax(p / temperature, dim=1) + epsilon
        q = torch.softmax(q / temperature, dim=1) + epsilon
        m = 0.5 * (p + q)

        kl_p_m = F.kl_div(torch.log(p), m, reduction='batchmean')  # KL(P || M)
        kl_q_m = F.kl_div(torch.log(q), m, reduction='batchmean')  # KL(Q || M)

        js = 0.5 * (kl_p_m + kl_q_m)

        js = js * (temperature ** 2)

        return js


def distillation_loss(student_outs, teacher_outs, temperature=0.7, loss_fn='js'):
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    js_criterion = JSDivLoss()
    loss = 0
    for i in range(len(student_outs)):
        if loss_fn == 'mse':
            loss += mse_criterion(student_outs[i], teacher_outs[i].detach())
        elif loss_fn == 'ce':
            loss += ce_criterion(student_outs[i]/temperature, torch.softmax(teacher_outs[i].detach()/temperature, dim=1))*(temperature ** 2)
        elif loss_fn == 'kl':
            loss += kl_criterion(torch.log_softmax(student_outs[i]/temperature, dim=1), torch.softmax(teacher_outs[i].detach()/temperature, dim=1))*(temperature ** 2)
        elif loss_fn == 'js':
            loss += js_criterion(student_outs[i], teacher_outs[i].detach(), temperature)

    return loss.sum()

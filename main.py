import argparse
import sys
from collections import Counter

import numpy as np
from torch.utils.data import random_split

from baselines import NG, BS, UNSIR, JiT, SSD
from IAU import IAU
from models import ResNetWithFeatureExtraction, AllCNNImprovedPlus

# VGG16用mnist的时候记得改这里
# from models import VGGWithFeatureExtraction_mnist as VGGWithFeatureExtraction
from models import VGGWithFeatureExtraction_cifar as VGGWithFeatureExtraction
from preactrn18 import PreActResNet18

from resnet_cifar10 import cifar10_resnet18
from datasets import *
from trainer import *
import logging

result_dict = {
    'train_forget_acc': [],
    'train_remain_acc': [],
    'test_forget_acc': [],
    'test_remain_acc': [],
    'unlearn_time': []
}


def seed_torch():
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def recur_main(args, df_num, dr_num, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = args.checkpoint_dir + '/'

    # TODO: 在此更改。cifar10每个训练类有5000个样本，mnist每个类有6000。测试集每个类都是1000个
    # num_forget = 20
    num_forget = df_num
    limited_remain_class_num = dr_num  # cifar100限制50；cifar10限制100;vgg限制20
    bd_num = 1000   # 加trigger的数量

    # 打印实验配置
    logger.info('+' * 100)
    logger.info(args)
    logger.info('forget dataset limited to: {}'.format(num_forget))
    logger.info('remain dataset limited to: {}'.format(limited_remain_class_num))
    logger.info('+' * 100)

    trainset, testset = get_dataset(args.data_name, args.dataset_dir)
    if args.if_backdoor:
        train_forget_loader, test_forget_loader, train_remain_loader, test_remain_loader, train_full_forget_loader, train_limited_remain_loader,\
        train_oriforget_loader, test_oriforget_loader, train_bdremain_loader \
            = get_backdoor_loader(trainset, testset, forget_class, num_forget, args.batch_size,
                                  limited_remain_class_num=limited_remain_class_num, data_name=args.data_name,
                                  num_classes=num_classes, bd_num=bd_num)
    else:
        train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, \
        forget_remain_train_loader, train_full_forget_loader, train_limited_remain_loader, \
        train_forget_index, train_remain_index, test_forget_index, test_remain_index, forget_remain_train_index \
            = get_unlearn_loader(trainset, testset, forget_class, args.batch_size, num_forget, limited_remain_class_num=limited_remain_class_num, data_name=args.data_name, num_classes=num_classes)


    train_loader, test_loader = get_dataloader(trainset, testset, args.batch_size, device=device)

    if args.train == 'load_retrain':
        logger.info('=' * 100)
        logger.info(' ' * 25 + 'load retrain model')
        logger.info('=' * 100)

        if args.data_name=='cifar10' and args.model_name=='AllCNN':
            retrain_model = torch.load(path + args.data_name + args.model_name + "_retrain_model_fc{}_ep{}_lr{}_filter0.5_best.pth".format(args.forget_class, args.epoch, args.lr),
                                   map_location=torch.device('cpu')).to(device)
        elif args.data_name == 'cifar10' and args.model_name == 'ResNet18':
                retrain_model = torch.load(
                    path + "cifar10_resnet18_retrain_fc1.pth",
                    map_location=torch.device('cpu')).to(device)

        logger.info('\nretrain model acc:\n')
        _, Df_acc = eval(model=retrain_model, data_loader=train_forget_loader, mode='', print_perform=False, device=device)
        _, Dr_acc = eval(model=retrain_model, data_loader=train_limited_remain_loader, mode='', print_perform=False,
                         device=device)
        logger.info('Train Forget Acc: {}'.format(Df_acc))
        logger.info('Train Remain Acc: {}'.format(Dr_acc))
        test(retrain_model, test_loader, num_classes=num_classes, forget_class=forget_class, logger=logger)

    elif args.train == 'retrain':
        logger.info('=' * 100)
        logger.info(' ' * 25 + 'retrain model from scratch')
        logger.info('=' * 100)
        retrain_start_time = time.time()
        retrain_model = train_save_model(train_remain_loader, test_remain_loader, args.model_name, args.optim_name,
                                         args.lr, args.epoch, device,
                                         path + args.data_name + args.model_name + "_retrain_model_fc{}_ep{}_lr{}_filter0.5_best".format(args.forget_class, args.epoch, args.lr),
                                         data_name=args.data_name, num_classes=num_classes, logger=logger)
        logger.info('\nretrain model acc:\n')
        logger.info('retrain time: ', time.time()-retrain_start_time)
        test(retrain_model, train_full_forget_loader, num_classes=num_classes, forget_class=forget_class, logger=logger)
        test(retrain_model, train_remain_loader, num_classes=num_classes, forget_class=forget_class, logger=logger)
        test(retrain_model, test_loader, num_classes=num_classes, forget_class=forget_class, logger=logger)


    elif args.train == 'train':
        logger.info('=' * 100)
        logger.info(' ' * 25 + 'train original model only')
        logger.info('=' * 100)
        if args.if_backdoor:
            ori_model = train_save_model(train_loader, test_loader, args.model_name, args.optim_name, args.lr,
                                         args.epoch, device,
                                         path + args.data_name + args.model_name +"_backdoor_ori_fc{}".format(bd_num),
                                         data_name=args.data_name, num_classes=num_classes, logger=logger)
            logger.info('\n------------backdoor test------------------\n')
            _, asr = eval(ori_model, test_forget_loader, device=device)  
            _, bn_acc = eval(ori_model, test_remain_loader, device=device)  
            logger.info('Test ASR eval acc: {}'.format(asr))
            logger.info('Test Benign eval acc: {}'.format(bn_acc))

        else:
            ori_model = train_save_model(train_loader, test_loader, args.model_name, args.optim_name, args.lr,
                                         args.epoch, device,
                                         path + args.data_name + args.model_name +"_original_model_ep{}_lr{}_filter1_best".format(args.epoch, args.lr),
                                         data_name=args.data_name, num_classes=num_classes, logger=logger)

            test(ori_model, test_loader, num_classes=num_classes, forget_class=forget_class, logger=logger)

    elif args.train == 'load':
        logger.info('=' * 100)
        logger.info(' ' * 25 + 'load original model and retrain model')
        logger.info('=' * 100)

        if args.if_backdoor:
            ori_model = torch.load(path + args.data_name + args.model_name +"_backdoor_ori_fc{}.pth".format(bd_num),
                                       map_location=device)
            logger.info('\n------------backdoor test------------------\n')
            _, asr = eval(ori_model, train_full_forget_loader, device=device) 
            _, bn_acc = eval(ori_model, test_remain_loader, device=device)  
            logger.info('Test ASR eval acc: {}'.format(asr))
            logger.info('Test Benign eval acc: {}'.format(bn_acc))
        else:

            if args.data_name=='cifar100' and args.model_name=='ResNet18':
                ori_model = resnet18(n_channels=3, num_classes=num_classes).to(device)
                ori_model.load_state_dict(torch.load(path+'/cifar100_resnet18.pt'))
            if args.data_name=='cifar100' and args.model_name=='AllCNN':
                # ori_model = torch.load(path + 'cifar100_allcnn_best.pth', map_location=torch.device('cpu')).to(device)
                ori_model = AllCNNImprovedPlus().to(device)
                ori_model.load_state_dict(torch.load(path+'cifar100AllCNN_original_model.pth'))
            if args.data_name=='cifar100' and args.model_name=='VGG16':
                ori_model = torch.load(path + "cifar100_vgg16_best.pth", map_location=torch.device('cpu')).to(device)

            if args.data_name == 'cifar10' and args.model_name=='ResNet18':
                ori_model = cifar10_resnet18().to(device)
                ori_model.load_state_dict(torch.load(path+'/cifar10_resnet18.pt'))
            if args.data_name=='cifar10' and args.model_name=='AllCNN':
                ori_model = torch.load(path + args.data_name + args.model_name + "_original_model_ep{}_lr0.1_filter0.5_best".format(args.epoch, args.lr) + ".pth",
                                       map_location=torch.device('cpu')).to(device)
            if args.data_name=='cifar10' and args.model_name=='VGG16':
                ori_model = torch.load(path + "cifar10_vgg16_best.pth", map_location=torch.device('cpu')).to(device)

            if args.data_name=='vggface2' and args.model_name=='AllCNN':
                ori_model = torch.load(path + args.data_name + args.model_name + "_original_model_ep{}_best".format(args.epoch, args.lr) + ".pth",map_location=torch.device('cpu')).to(device)
            if args.data_name=='vggface2' and args.model_name=='ResNet18':
                ori_model = torch.load(path+ 'vggface2ResNet18_original_model_ep30_best.pth', map_location=torch.device('cpu')).to(device)

            if args.data_name == 'mnist' and args.model_name == 'AllCNN':
                ori_model = torch.load(
                    path + args.data_name + args.model_name + "_original_model_ep{}_lr0.01_filter1_best".format(args.epoch, args.lr) + ".pth", map_location=torch.device('cpu')).to(device)
            if args.data_name == 'mnist' and args.model_name == 'ResNet18':
                ori_model = torch.load(path + "resnet18_mnist.pth", map_location=torch.device('cpu')).to(device)
            if args.data_name == 'mnist' and args.model_name == 'VGG16':
                ori_model = torch.load(path + "mnist_vgg16_best.pth", map_location=torch.device('cpu')).to(device)

            if args.data_name == 'svhn' and args.model_name == 'AllCNN':
                ori_model = torch.load(
                    path + args.data_name + args.model_name + "_original_model_ep{}_lr{}_filter1_best".format(args.epoch, args.lr) + ".pth", map_location=torch.device('cpu')).to(device)

            # logger.info('\noriginal model acc:\n')
            # _, Df_acc = eval(model=ori_model, data_loader=train_forget_loader, mode='', print_perform=False, device=device)
            # _, Dr_acc = eval(model=ori_model, data_loader=train_remain_loader, mode='', print_perform=False, device=device)
            # logger.info('Train Forget Acc: {}'.format(Df_acc))
            # logger.info('Train Remain Acc: {}'.format(Dr_acc))
            # test(ori_model, test_loader, num_classes=num_classes, forget_class=forget_class, logger=logger)

    """
    =========================== unlearn part ===========================
    """
    if args.load_ul == True:
        unlearn_model = torch.load(path+'IAU/'+'{}_{}_Dr{}_seed{}.pth'.format(args.data_name, args.model_name, dr_num, seed))

    else:
        if args.method == 'BS':
            logger.info('*' * 100)
            logger.info(' ' * 25 + 'begin boundary shrink unlearning')
            logger.info('*' * 100)
            unlearn_model = BS(ori_model, train_forget_loader, train_remain_loader,
                                test_forget_loader, test_remain_loader, test_loader, device,
                                train_full_forget_loader=train_full_forget_loader,
                                train_limited_remain_loader=train_limited_remain_loader,
                                limit_n = df_num,
                                data_name=args.data_name, num_classes=num_classes,
                                forget_class=forget_class, model_name=args.model_name,
                                logger=logger)
        elif args.method == 'NG':
            logger.info('*' * 100)
            logger.info(' ' * 25 + 'begin negative gradient unlearning')
            logger.info('*' * 100)
            unlearn_model = NG(ori_model, train_forget_loader, train_remain_loader, test_loader, test_forget_loader, test_remain_loader,
                                train_full_forget_loader=train_full_forget_loader, train_limited_remain_loader=train_limited_remain_loader,
                                limit_n=df_num, data_name=args.data_name, num_classes=num_classes,
                                forget_class=forget_class, model_name=args.model_name,
                                logger=logger)

        elif args.method == 'UNSIR':
            logger.info('*' * 100)
            logger.info(' ' * 25 + 'begin UNSIR unlearning')
            logger.info('*' * 100)
            unlearn_model = UNSIR(ori_model, train_forget_loader, train_remain_loader, test_loader, test_forget_loader, test_remain_loader,
                                  train_full_forget_loader=train_full_forget_loader, train_limited_remain_loader=train_limited_remain_loader,
                                  data_name=args.data_name, num_classes=num_classes,
                                  forget_class=forget_class, model_name=args.model_name,
                                  logger=logger)

        elif args.method == 'jit':
            logger.info('*' * 100)
            logger.info(' ' * 25 + 'begin JiT unlearning')
            logger.info('*' * 100)
            unlearn_model = JiT(ori_model, train_forget_loader, train_remain_loader, train_loader, test_loader, test_forget_loader, test_remain_loader,
                                train_full_forget_loader=train_full_forget_loader, train_limited_remain_loader=train_limited_remain_loader,
                                data_name=args.data_name, num_classes=num_classes,
                                forget_class=forget_class, model_name=args.model_name,
                                logger=logger)
        elif args.method == 'SSD':
            logger.info('*' * 100)
            logger.info(' ' * 25 + 'begin SSD unlearning')
            logger.info('*' * 100)
            unlearn_model = SSD(ori_model, train_forget_loader, train_remain_loader, train_loader, test_loader, test_forget_loader, test_remain_loader,
                                train_full_forget_loader=train_full_forget_loader, train_limited_remain_loader=train_limited_remain_loader,
                                data_name=args.data_name, num_classes=num_classes,
                                forget_class=forget_class, model_name=args.model_name,
                                logger=logger)
        else:
            logger.info('*' * 100)
            logger.info(' ' * 25 + 'begin IAU')
            logger.info('*' * 100)

            unlearn_model, acc_df, acc_dr, acc_tf, acc_tr, unlearn_time = \
                IAU(ori_model, train_forget_loader, train_remain_loader, train_loader, test_loader, test_forget_loader, test_remain_loader,
                             train_full_forget_loader=train_full_forget_loader,
                             train_limited_remain_loader=train_limited_remain_loader,
                             limit_n=dr_num, data_name=args.data_name, num_classes=num_classes,
                             forget_class=forget_class, model_name=args.model_name,
                             logger=logger)
            result_dict['train_forget_acc'].append(acc_df)
            result_dict['train_remain_acc'].append(acc_dr)
            result_dict['test_forget_acc'].append(acc_tf)
            result_dict['test_remain_acc'].append(acc_tr)
            result_dict['unlearn_time'].append(unlearn_time)

            torch.save(unlearn_model, path+'IAU/'+'{}_{}_Dr{}_seed{}.pth'.format(args.data_name, args.model_name, dr_num, seed))

    logger.info('********final test*********')

    """
    backdoor
    """
    # test(unlearn_model, test_loader)
    # test(unlearn_model, train_oriforget_loader)
    # test(unlearn_model, test_oriforget_loader)


def set_up():
    parser = argparse.ArgumentParser("Unlearning")
    parser.add_argument('--method', type=str, default='boundary_shrink',
                        choices=['BS', 'NG', 'SSD', 'noise', 'jit', 'IAU'], help='unlearning method chooseing')
    parser.add_argument('--data_name', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100', 'vggface2', 'svhn', 'fashionmnist'],
                        help='dataset choosing')
    parser.add_argument('--model_name', type=str, default='AllCNN',
                        choices=['MNISTNet', 'AllCNN', 'Netcifar', 'ResNet18', 'ResNet50', 'VGG16'],
                        help='model name')
    parser.add_argument('--optim_name', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer name')
    parser.add_argument('--lr', type=float, default=0.01, help='original training learning rate')
    parser.add_argument('--epoch', type=int, default=20, help='original training epoch')
    parser.add_argument('--forget_class', type=int, default=4, help='the target forget class index')
    parser.add_argument('--dataset_dir', type=str, default='./data', help='dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='checkpoints directory')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--if_backdoor', action='store_true', default=False, help='if backdoor attack')
    parser.add_argument('--train', type=str, default='load',
                        choices=['load_retrain', 'train', 'retrain', 'load'],
                        help='Train model from scratch')
    parser.add_argument('--load_ul', action='store_true', help='to load IAU checkpoint')
    parser.add_argument('--ablation', type=str, default=None, choices=['no1', 'no2'], help='ablation a module')
    parser.add_argument('--num_forget', type=int, default=5000, help='available data amount for target class')
    parser.add_argument('--num_remain', type=int, default=5000, help='available data amount for each remain class')
    args = parser.parse_args()

    logging.basicConfig(filename='outputs.log',  
                        filemode='a',  
                        format='%(message)s', 
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return args, logger


if __name__ == '__main__':
    args, logger = set_up()
    for seed in range(2024, 2025, 1):
        seed_torch(seed)
        logger.info('----------------seed = {}----------------------'.format(seed))
        recur_main(args, args.num_forget, args.num_remain, logger)

    total_acc_df = 0.
    total_acc_dr = 0.
    total_acc_tf = 0.
    total_acc_tr = 0.
    total_time = 0
    cnt = len(result_dict['train_forget_acc'])
    for i in range(cnt):
        total_acc_df += result_dict['train_forget_acc'][i]
        total_acc_dr += result_dict['train_remain_acc'][i]
        total_acc_tf += result_dict['test_forget_acc'][i]
        total_acc_tr += result_dict['test_remain_acc'][i]
        total_time += result_dict['unlearn_time'][i]
    logger.info('count: {}'.format(cnt))
    logger.info('train_forget_acc: {}'.format(total_acc_df / cnt))
    logger.info('train_remain_acc: {}'.format(total_acc_dr / cnt))
    logger.info('test_forget_acc: {}'.format(total_acc_tf / cnt))
    logger.info('test_remain_acc: {}'.format(total_acc_tr / cnt))
    logger.info('unlearn_time: {}'.format(total_time / cnt))

from __future__ import print_function
import argparse
import os
import random
import shutil
import time
import warnings
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import lr_scheduler

import torchvision.transforms as transforms

import datasets
import models
from utils import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data','-d', metavar='DATA', default='cub',
                    help='dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='moln',
                    choices=model_names,
                    help='model architecture: ')
parser.add_argument('--backbone', default='resnet18', help='backbone')
parser.add_argument('--save_path', '-s', metavar='SAVE', default='',
                    help='saving path')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=360, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--is_fix', dest='is_fix', action='store_true',
                    help='is_fix.')
parser.add_argument('--opt', default='sgd', type=str,
                    help='optimizer type')
                    
''' data proc '''
parser.add_argument('--resize_size', default=480, type=int, help='resize size')
parser.add_argument('--crop_size', default=448, type=int, help='crop size')
parser.add_argument('--rotate', default=0, type=int, help='is rotate')

''' momn '''
parser.add_argument('--iterN', default=5, type=int,metavar='W', help='iterN')
parser.add_argument('--beta1', default=0.5, type=float,metavar='W', help='beta1')
parser.add_argument('--beta2', default=1, type=float,metavar='W', help='beta2')
parser.add_argument('--mu1', default=10, type=float,metavar='W', help='mu1')
parser.add_argument('--mu2', default=10, type=float,metavar='W', help='mu2')
parser.add_argument('--roph', default=1.1, type=float,metavar='W', help='roph')
parser.add_argument('--aux_var', default=1, type=float,metavar='W', help='aux_var')
parser.add_argument('--mode', default='default', type=str, help='mode')
parser.add_argument('--lw_lr', default=0.1, type=float, metavar='W', help='loss weight for low rank')
parser.add_argument('--lw_sr', default=0.1, type=float, metavar='W', help='loss weight for sparsity regular')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)
        
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
                                

    ''' random seed '''
    if args.seed is not None:
        random.seed(args.seed)
    else:
        args.seed = random.randint(1, 10000)
        
    torch.manual_seed(args.seed)
    print('==> random seed:',args.seed)

                                
                                
    ''' data load '''
    if args.data == 'cub':
        img_root = '/userhome/raw_data/CUB_200_2011/CUB_200_2011/images'
        traindir = os.path.join(img_root,'../train.list')
        valdir = os.path.join(img_root,'../test.list')
        args.num_cls = 200
    elif args.data == 'car':
        img_root = '/userhome/raw_data/car_196/car_ims'
        traindir = os.path.join(img_root,'../train.list')
        valdir = os.path.join(img_root,'../test.list')
        args.num_cls = 196
    elif args.data == 'air':
        img_root = '/userhome/raw_data/aircraft/images'
        traindir = os.path.join(img_root,'../train.list')
        valdir = os.path.join(img_root,'../test.list')
        args.num_cls = 100
    elif args.data == 'dog':
        img_root = '/userhome/raw_data/dogs-120/Images'
        traindir = os.path.join(img_root,'../train.list')
        valdir = os.path.join(img_root,'../test.list')
        args.num_cls = 120
    

    train_transforms, val_transforms = preprocess_strategy(args)
    
    val_transforms1 = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]) 
    
    val_dataset = datasets.ImageFolder(img_root,valdir, val_transforms)
    val_dataset1 = datasets.ImageFolder(img_root,valdir, val_transforms1)


    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    val_loader1 = torch.utils.data.DataLoader(
        val_dataset1, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    
    ''' model building '''
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        best_prec1=0
        model,criterion = models.__dict__[args.arch](pretrained=True,args=args)
    else:
        print("=> creating model '{}'".format(args.arch))
        model,criterion = models.__dict__[args.arch](args=args)
    print("=> is the backbone fixed: '{}'".format(args.is_fix))

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    criterion = criterion.cuda(args.gpu)
    

    ''' optionally resume from a checkpoint'''
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #args.start_epoch = checkpoint['epoch']
            if(best_prec1==0):
                best_prec1 = checkpoint['best_prec1']
            print('=> pretrained acc {:.4F}'.format(best_prec1))
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
            
        
    # evaluate on validation set
    validate(val_loader,val_loader1, model,val_dataset)


def validate(val_loader,val_loader1, model,val_dataset):

    top_1 = AverageMeter()
    val_corrects = 0

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            logit,feats = model(input)
            logit = logit.cpu().detach().numpy()
            
            if(i==0):
                gt = target.cpu().numpy()
                logits = softmax(logit)
            else:
                gt = np.hstack([gt,target.cpu().numpy()])
                logits = np.vstack([logits,softmax(logit)])
            print('{}/{}'.format(i,len(val_loader)))

        for i, (input, target) in enumerate(val_loader1):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            logit,feats = model(input)
            logit = logit.cpu().detach().numpy()
            
            if(i==0):
                logits1 = softmax(logit)
            else:
                logits1 = np.vstack([logits1,softmax(logit)])
            print('{}/{}'.format(i,len(val_loader)))
            
            
    for w in range(11):
        w = w*0.1
        logit_avg = w*logits+(1-w)*logits1
        
        prec1, prec  = accuracy(torch.from_numpy(logit_avg), torch.from_numpy(gt), topk=(1,5))
        
        print(prec1)
    
        
if __name__ == '__main__':
    main()

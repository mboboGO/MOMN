import numpy as np
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier


import torchvision.transforms as transforms
import torch

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                                                                
def freeze_bn(model):
    for m in model.modules():
        if isinstance(m,nn.BatchNorm2d):
            m.eval()
            
            
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        
def preprocess_strategy(args):
    evaluate_transforms = None
    if args.data in ['cub','car','air','dog']:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])  
        val_transforms = transforms.Compose([
            transforms.Resize(args.resize_size),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            normalize,
        ]) 
   
    return train_transforms, val_transforms

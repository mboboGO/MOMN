import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
from  MPNCOV import MPNCOV

__all__ = ['msn']


class Model(nn.Module):
    def __init__(self,pretrained=True, args=None):
        self.inplanes = 64
        self.num_classes = args.num_cls
        is_fix = args.is_fix
        self.backbone_arch = args.backbone
        
        self.iterN = args.iterN
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.mu1= args.mu1
        self.mu2 = args.mu2
        self.roph = args.roph
        self.aux_var = args.aux_var
        
        
        super(Model, self).__init__()

        ''' Backbone Net'''
        if self.backbone_arch in dir(models):
            self.model = getattr(models, self.backbone_arch)(pretrained=False)
        else:
            self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000,pretrained=False)
            
        if pretrained:
            if self.backbone_arch=='resnet50':
                model_dict = self.model.state_dict()
                self.model.load_state_dict(torch.load('./resnet50-19c8e357.pth'))
            elif self.backbone_arch=='resnet101':
                model_dict = self.model.state_dict()
                self.model.load_state_dict(torch.load('./resnet101-5d3b4d8f.pth'))

        if self.backbone_arch in ['resnet50','se_resnet50','resnet101']:
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        elif self.backbone_arch in ['senet154']:
            self.model = nn.Sequential(*list(self.model.children())[:-3])
            
        if(is_fix):
            for p in self.parameters():
                p.requires_grad=False
                
        ''' Model '''
        #self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.proj = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0,bias=False)
        self.layer_reduce_bn = nn.BatchNorm2d(256)
        self.layer_reduce_relu = nn.ReLU(inplace=True)

        ''' sparsity attention '''
        self.att_net = self.att_module(256)
		
        ''' classifier'''
        self.classifier = nn.Linear(int(256*(256+1)/2), self.num_classes)
        
        ''' params ini '''
        for name, m in self.named_modules():
            if 'model' not in name:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def att_module(self, ic):
        model = nn.Sequential(
            nn.AvgPool2d(28),
            nn.Conv2d(ic,int(ic/16), kernel_size=1, stride=1, bias=False),
            nn.Conv2d(int(ic/16),ic, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )
        return model
        
    def forward(self, x):
        ''' backbone '''
        x = self.model(x)
        last_conv = x
        
        # iSQRT
        x = self.proj(last_conv)
        x = self.layer_reduce_bn(x)
        x = self.layer_reduce_relu(x)
        x = x.view(x.size(0),x.size(1), -1)
        x = x - torch.mean(x,dim=2,keepdim=True)
        A = 1./x.size(2)*x.bmm(x.transpose(1,2))
        x = MPNCOV.SqrtmLayer(A, 5)
        #r_svd = x
        x = MPNCOV.TriuvecLayer(x)
        x = x.view(x.size(0), -1)

        ''' fc '''
        logit = self.classifier(x)

        return logit,(x,A)
        
        return logit,(A)
		
class LOSS(nn.Module):
    def __init__(self, args=None):
        super(LOSS, self).__init__()
        self.lw_lr=args.lw_lr
        self.lw_sr=args.lw_sr
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, logit, feats, label):
        x = feats[0]
		
        # cls loss
        cls_loss = self.cls_loss(logit,label)

        total_loss = cls_loss 
		
        return total_loss, cls_loss
		
def msn(pretrained=False, args=None):

    model = Model(pretrained,args)
    loss_model = LOSS(args)

    return model,loss_model
	

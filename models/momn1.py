import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torchvision import models
import resnet_

__all__ = ['momn1']
	
class Model(nn.Module):
    def __init__(self, pretrained=True, args=None):
        super(Model, self).__init__()
        self.inplanes = 64
        self.num_classes = args.num_cls
        is_fix = args.is_fix
        self.arch = args.backbone
		
        ''' params '''
        self.iterN = args.iterN
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.mu1= args.mu1
        self.mu2 = args.mu2
        self.roph = args.roph
        self.aux_var = args.aux_var
	
        ''' Backbone Net'''
        if self.arch in ['resnet50','resnet101']:
            self.backbone = getattr(resnet_, self.arch)(pretrained=False)
        elif self.arch in dir(models):
            self.backbone = getattr(models, self.arch)(pretrained=False)
        else:
            self.backbone = pretrainedmodels.__dict__[self.arch](num_classes=1000,pretrained=False)
            
        if(is_fix):
            for p in self.parameters():
                p.requires_grad=False
                
        ''' projction '''
        self.proj = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0,bias=False)
        self.layer_reduce_bn = nn.BatchNorm2d(256)
        self.layer_reduce_relu = nn.ReLU(inplace=True)

        ''' sparsity attention '''
        self.att_net = self.att_module(256)
		
        ''' classifier'''
        self.fc_cls = nn.Linear(int(256*(256+1)/2), self.num_classes)
		
        ''' params ini '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        if pretrained:
            if self.arch=='resnet50':
                model_dict = self.backbone.state_dict()
                self.backbone.load_state_dict(torch.load('./resnet50-19c8e357.pth'))
            elif self.arch=='resnet101':
                model_dict = self.backbone.state_dict()
                self.backbone.load_state_dict(torch.load('./resnet101-5d3b4d8f.pth'))

        if self.arch in ['resnet50','se_resnet50','resnet101']:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        elif self.arch in ['senet154']:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-3])

    def att_module(self, ic):
        model = nn.Sequential(
            nn.AvgPool2d(28),
            nn.Conv2d(ic,int(ic/16), kernel_size=1, stride=1, bias=False),
            nn.Conv2d(int(ic/16),ic, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )
        return model

    def momn(self,x,att,iterN,beta1,beta2,mu1,mu2,roph,aux_var,mode='default'):
        batchSize = x.size(0)
        dim = x.size(1)
        I1 = torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1)
        I3 = 3.0*I1
        normA = (1.0/3.0)*x.mul(I3).sum(dim=1).sum(dim=1)
        A = x.div(normA.view(batchSize,1,1).expand_as(x))
        # initionlization
        J1 = A;J2 = A;Y = A;L1 = 0;L2 = 0;Z=torch.eye(dim,dim,device = x.device).view(1,dim,dim).repeat(batchSize,1,1)
        for i in range(0, iterN):
            if mode=='avg':
                eta1=1.0/(2*mu1);eta2=1.0/(2*mu2);eta3=1.0/(2*mu1+2*mu2)
            else:
                eta1=1.0/(mu1);eta2=1.0/(mu2);eta3=1.0/(mu1+mu2)
            if i==0:
                # step1
                J1 = Y-eta1*beta1*(I1*Y)
                # step2
                J2 = Y-eta2*beta2*(1-att)*Y
                # step3
                Y = Y+mu1*eta3*(J1-Y)+mu2*eta3*(J2-Y)
                ZY = 0.5*(I3 - A)
                Y = Y.bmm(ZY)
                Z = ZY
            elif i==(iterN-1):
                # step1
                J1 = J1-eta1*mu1*(J1-Y)-eta1*L1
                J1 = J1-beta1*eta1*(I1*J1)
                # step2
                J2 = J2-eta2*mu2*(J2-Y)-eta2*L2
                J2 = J2-eta2*beta2*(1-att)*J2
                # step3
                Y = Y+mu1*eta3*(J1-Y)+mu2*eta3*(J2-Y)+eta3*(L1+L2)
                ZY = 0.5*(I3 - Z.bmm(Y))
                Y = Y.bmm(ZY)
            else:
                # step1
                J1 = J1-eta1*mu1*(J1-Y)-eta1*L1
                J1 = J1-beta1*eta1*(I1*J1)
                # step2
                J2 = J2-eta2*mu2*(J2-Y)-eta2*L2
                J2 = J2-eta2*beta2*(1-att)*J2
                # step3
                Y = Y+mu1*eta3*(J1-Y)+mu2*eta3*(J2-Y)+eta3*(L1+L2)
                ZY = 0.5*(I3 - Z.bmm(Y))
                Y = Y.bmm(ZY)
                Z = ZY.bmm(Z)
            # step 4
            if(i<iterN-1):
                L1 = aux_var*L1+mu1*(J1-Y)
                L2 = aux_var*L2+mu2*(J2-Y)
                mu1 = roph*mu1
                mu2 = roph*mu2

        y = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        return y

    def triuvec(self,x):
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        x = x.reshape(batchSize, dim*dim)
        I = torch.ones(dim,dim).triu().t().reshape(dim*dim)
        index = I.nonzero()
        y = torch.zeros(batchSize,int(dim*(dim+1)/2),device = x.device)
        for i in range(batchSize):
            y[i, :] = x[i, index].t()
        return y
	
    def forward(self,x):
        ''' backbone '''
        x = self.backbone(x)
        last_conv = x
		
        # projection
        x = self.proj(last_conv)
        x = self.layer_reduce_bn(x)
        pro = self.layer_reduce_relu(x)
        
        # sparsity attention
        x = self.att_net(pro)
        x = x.view(x.size(0),x.size(1), -1)
        s_att = x.bmm(x.transpose(1,2))
		
        # reduce mean
        x = pro.view(pro.size(0),pro.size(1), -1)
        x = x - torch.mean(x,dim=2,keepdim=True)
		
        # momn
        A = 1./x.size(2)*x.bmm(x.transpose(1,2))
        lrmsr = self.momn(A,s_att,self.iterN,
                            self.beta1,self.beta2,
                            self.mu1,self.mu2,self.roph,
                            self.aux_var)
        x = self.triuvec(lrmsr)
        feat = x.view(x.size(0), -1)
		
        ''' fc '''
        logit = self.fc_cls(feat)

        return logit,(A,s_att,feat)
		
class LOSS(nn.Module):
    def __init__(self, args=None):
        super(LOSS, self).__init__()
        self.lw_lr=args.lw_lr
        self.lw_sr=args.lw_sr
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, logit, feats, label):
        x = feats[0]
        s_att = feats[1]
		
        # cls loss
        cls_loss = self.cls_loss(logit,label)
        # auxiliary loss
        dim = x.size(2)
        x = torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(x.size(0),1,1)*x
        aux_loss = self.lw_lr*torch.sum(x)/x.size(0)/256
        aux_loss += self.lw_sr*torch.mean(s_att)
	
        total_loss = cls_loss + aux_loss
		
        return total_loss, cls_loss, aux_loss
		
def momn1(pretrained=False, args=None):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Model(pretrained, args)
    loss_model = LOSS(args)

    return model,loss_model

	

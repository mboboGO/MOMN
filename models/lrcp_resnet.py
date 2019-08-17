import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

__all__ = ['moln','lrcp_r50','lrcp_split']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://drive.google.com/file/d/132PzY3eVDuGg8ROz5wON5FTC2E2o12Ck/view',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
	
class LRCP(nn.Module):
    def __init__(self, block, layers, args=None):
        self.inplanes = 64
        super(LRCP, self).__init__()
		
        ''' params '''
        self.iterN = args.iterN
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.mu1= args.mu1
        self.mu2 = args.mu2
        self.roph = args.roph
        self.aux_var = args.aux_var
	
        ''' backbone net'''
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        if(args.is_fix):
            for p in self.parameters():
                p.requires_grad=False

        ''' projction '''
        self.proj = nn.Conv2d(512 * block.expansion, 256, kernel_size=1, stride=1, padding=0,bias=False)
        self.layer_reduce_bn = nn.BatchNorm2d(256)
        self.layer_reduce_relu = nn.ReLU(inplace=True)

        ''' sparsity attention '''
        self.att_net = self.att_module(256)
		
        ''' classifier'''
        self.fc_cls = nn.Linear(int(256*(256+1)/2), 200)
		
        ''' params ini '''
        for m in self.modules():
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
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def ilrmsn(self,x,att,iterN,beta1,beta2,mu1,mu2,roph,aux_var,mode):
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
        # backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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
		
        # ilrmsn
        A = 1./x.size(2)*x.bmm(x.transpose(1,2))
        lrmsr = self.ilrmsn(A,s_att,self.iterN,
                            self.beta1,self.beta2,
                            self.mu1,self.mu2,self.roph,
                            self.aux_var,'default')
        x = self.triuvec(lrmsr)
        feat = x.view(x.size(0), -1)
		
        ''' fc '''
        cls = self.fc_cls(feat)

        return cls,[A,s_att]
		
class LRCP_LOSS(nn.Module):
    def __init__(self, args=None):
        super(LRCP_LOSS, self).__init__()
        self.lw_lr=args.lw_lr
        self.lw_sr=args.lw_sr
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, logit, feats, label):
        cls = logit
        x = feats[0]
        s_att = feats[1]
		
        # cls loss
        cls_loss = self.cls_loss(cls,label)
        # low rank loss
        dim = x.size(2)
        x = torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(x.size(0),1,1)*x
        lr_loss = self.lw_lr*torch.sum(x)/x.size(0)/256
        # sparsity rank loss
        dim = x.size(2)
        sr_loss = self.lw_sr*torch.mean(s_att)
	
        total_loss = cls_loss + lr_loss + sr_loss
		
        return total_loss, cls_loss, lr_loss,sr_loss
		
def moln(pretrained=False,args=None):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LRCP(Bottleneck, [3, 4, 23, 3], args)
    loss_model = LRCP_LOSS(args)
    if pretrained:
        model_dict = model.state_dict()
        #pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        pretrained_dict = torch.load('./resnet101-5d3b4d8f.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model,loss_model

def lrcp_r50(pretrained=False, args=None):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LRCP(Bottleneck, [3, 4, 6, 3], args)
    loss_model = LRCP_LOSS(args)
    if pretrained:
        model_dict = model.state_dict()
        #pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        pretrained_dict = torch.load('./checkpoints/resnet50-19c8e357.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model,loss_model

	
	
	
	
	
	

	
	
	
	
	
	
	
	
	
	
	
class LRCP_split(nn.Module):
    def __init__(self, block, layers, params=None):
        self.inplanes = 64
        super(LRCP_split, self).__init__()
		
        ''' params '''
        self.iterN = params['iterN']
        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.mu1= params['mu1']
        self.mu2 = params['mu2']
        self.roph = params['roph']
        self.aux_var = params['aux_var']
        self.mode = params['mode']	
	
        ''' backbone net'''
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        if(params['is_fix']):
            for p in self.parameters():
                p.requires_grad=False

        ''' projction '''
        self.proj = nn.Conv2d(512 * block.expansion, 256, kernel_size=1, stride=1, padding=0,bias=False)
        self.layer_reduce_bn = nn.BatchNorm2d(256)
        self.layer_reduce_relu = nn.ReLU(inplace=True)

        ''' sparsity attention '''
        self.att_net = self.att_module(256)
		
        ''' classifier'''
        self.fc_cls = nn.Linear(int(256*(256+1)/2), params['num_classes'])
		
        ''' params ini '''
        for m in self.modules():
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
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def ilrmsn(self,x,att,iterN,beta1,beta2,mu1,mu2,roph,aux_var,mode,a,b,c):
        batchSize = x.size(0)
        dim = x.size(1)
        I1 = torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1)
        I3 = 3.0*I1
        normA = (1.0/3.0)*x.mul(I3).sum(dim=1).sum(dim=1)
        if(a==1):
            A = x.div(normA.view(batchSize,1,1).expand_as(x))
        else:
            A = x
        # initionlization
        J1 = A;J2 = A;Y = A;L1 = 0;L2 = 0;Z=torch.eye(dim,dim,device = x.device).view(1,dim,dim).repeat(batchSize,1,1)
        for i in range(0, iterN):
            if mode=='avg':
                eta1=1.0/(2*mu1);eta2=1.0/(2*mu2);eta3=1.0/(2*mu1+2*mu2)
            else:
                eta1=1.0/(mu1);eta2=1.0/(mu2);eta3=1.0/(mu1+mu2)
            if i==0:
                # step1
                if(a==1):
                    Y = Y-eta1*beta1*(I1*Y)
                # step2
                if(b==1):
                    Y = Y-eta2*beta2*(1-att)*Y
                # step3
                if(c==1):
                    ZY = 0.5*(I3 - A)
                    Y = Y.bmm(ZY)
                    Z = ZY
            elif i==(iterN-1):
                # step1
                if(a==1):
                    Y = Y-beta1*eta1*(I1*Y)
                # step2
                if(b==1):
                    Y = Y-eta2*beta2*(1-att)*Y
                # step3
                if(c==1):
                    ZY = 0.5*(I3 - Z.bmm(Y))
                    Y = Y.bmm(ZY)
            else:
                # step1
                if(a==1):
                    Y = Y-beta1*eta1*(I1*Y)
                # step2
                if(b==1):
                    Y = Y-eta2*beta2*(1-att)*Y
                # step3
                if(c==1):
                    ZY = 0.5*(I3 - Z.bmm(Y))
                    Y = Y.bmm(ZY)
                    Z = ZY.bmm(Z)

        if(c==1):
            y = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        else:
            y=Y
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
        # backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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
		
        # ilrmsn
        A = 1./x.size(2)*x.bmm(x.transpose(1,2))
        A = self.ilrmsn(A,s_att,self.iterN,
                            self.beta1,self.beta2,
                            self.mu1,self.mu2,self.roph,
                            self.aux_var,self.mode,1,0,0)
        A = self.ilrmsn(A,s_att,self.iterN,
                            self.beta1,self.beta2,
                            self.mu1,self.mu2,self.roph,
                            self.aux_var,self.mode,0,1,0)
        A = self.ilrmsn(A,s_att,self.iterN,
                            self.beta1,self.beta2,
                            self.mu1,self.mu2,self.roph,
                            self.aux_var,self.mode,0,0,1)
        x = self.triuvec(A)
        feat = x.view(x.size(0), -1)
		
        ''' fc '''
        cls = self.fc_cls(feat)

        return cls,A,s_att
		
def lrcp_split(pretrained=False,params=None):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LRCP_split(Bottleneck, [3, 4, 23, 3], params)
    loss_model = LRCP_LOSS(params)
    if pretrained:
        model_dict = model.state_dict()
        #pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        pretrained_dict = torch.load('./checkpoints/resnet101-5d3b4d8f.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model,loss_model

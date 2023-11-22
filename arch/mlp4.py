import torch
import torchvision.models
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict as OrderedDict
import copy

class vector_log(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()*torch.log1p(x.abs())/10
    
    @staticmethod
    def backward(ctx, grad_output):
        x=ctx.saved_tensors
        return grad_output/(1+x.abs())/10

vector_log = vector_log.apply

class MLP(nn.Module):
    def __init__(self,ninput,nh,noutput,nlayers):
        super().__init__()
        
        self.layers=nn.ModuleList();
        self.bn=nn.LayerNorm(ninput);
        if nlayers==1:
            self.layers.append(nn.Linear(ninput,noutput));
        else:
            self.layers.append(nn.Linear(ninput,nh));
            for i in range(nlayers-2):
                self.layers.append(nn.Linear(nh,nh));
            
            self.layers.append(nn.Linear(nh,noutput));
        
        self.ninput=ninput;
        self.noutput=noutput;
        return;
    
    def forward(self,x):
        h=x.view(-1,self.ninput);
        #h=self.bn(h);
        for i in range(len(self.layers)-1):
            h=self.layers[i](h);
            h=F.relu(h);
            #h=F.dropout(h,training=self.training);
        
        h=self.layers[-1](h);
        return h



def vector_log(x):
    return torch.cat((torch.log(F.relu(x)+1e-20),torch.log(F.relu(-x)+1e-20)),dim=-1)

class new(nn.Module):
    def __init__(self,params):
        super(new,self).__init__()
        
        ninput=400
        
        nh=params.nh;
        nh2=params.nh2;
        nh3=params.nh3;
        nlayers=params.nlayers
        nlayers2=params.nlayers2
        nlayers3=params.nlayers3
        self.margin=params.margin
        
        #self.encoder1=MLP(ninput,nh,nh,nlayers);
        self.encoder2=MLP(ninput*4,nh2,2,nlayers2);
        
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        
        return;
    
    def forward(self,data_batch):
        h=[w[-4:].cuda().clone().view(-1) for w in data_batch['fvs']]
        #h=[w-w.mean(dim=-1,keepdim=True) for w in h]
        #h=[F.normalize(w,dim=-1)*20 for w in h]
        h=torch.stack(h,dim=0)
        h=self.encoder2(h);
        h=self.margin*F.tanh(h)
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import util.perm_inv as perm_inv
import math

class MLP(nn.Module):
    def __init__(self,ninput,nh,noutput,nlayers):
        super().__init__()
        self.layers=nn.ModuleList();
        if nlayers==1:
            self.layers.append(nn.Linear(ninput,noutput));
        else:
            self.layers.append(nn.Linear(ninput,nh));
            for i in range(nlayers-2):
                self.layers.append(nn.Linear(nh,nh));
            
            self.layers.append(nn.Linear(nh,noutput));
        
        return;
    
    def forward(self,x):
        h=x;
        for i in range(len(self.layers)-1):
            h=self.layers[i](h);
            h=F.relu(h);
        
        h=self.layers[-1](h);
        return h



class new(nn.Module):
    def __init__(self,params):
        super(new,self).__init__()
        nh=params.nh;
        nh2=params.nh2;
        #nh3=params.nh3;
        nlayers=params.nlayers
        nlayers2=params.nlayers2
        #nlayers3=params.nlayers3
        
        
        ninput=400
        
        self.pool=perm_inv.einpool(form='X_BaH',order=3,equivariant=True)
        k=len(self.pool.eqs)
        self.encoder=nn.ModuleList()
        self.encoder.append(MLP(ninput,nh,nh,2))
        for i in range(nlayers-1):
            self.encoder.append(MLP(nh*k,nh,nh,2))
        
        self.t=MLP(nh,nh2,2,nlayers2)
        
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        return;
    
    def forward(self,data_batch):
        h=[w.cuda() for w in data_batch['fvs']]
        
        if self.training:
            h=[w[torch.randperm(len(w)).to(w.device)[:math.ceil(0.8*len(w))]] for w in h]
        else:
            h=[w[torch.randperm(len(w)).to(w.device)] for w in h]
        
        h=[w-w.mean(dim=-1,keepdim=True) for w in h]
        h=[F.normalize(w,dim=-1)*20 for w in h]
        
        h_=[]
        for x in h:
            hi=x.unsqueeze(0)
            B,N,H=hi.shape
            hi=self.encoder[0](hi)
            for layer in self.encoder[1:]:
                hi=self.pool(torch.sin(hi)).view(B,N,-1)
                hi=layer(hi)
            
            hi=hi.mean(dim=-2)
            h_.append(hi)
        
        h=torch.cat(h_,dim=0)
        h=self.t(h)
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];
    
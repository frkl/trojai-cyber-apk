import torch
import torch.nn as nn
import torch.nn.functional as F
import util.perm_inv as perm_inv
import math

class vector_exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()*torch.expm1(x.abs())
    
    @staticmethod
    def backward(ctx, grad_output):
        x,=ctx.saved_tensors
        return grad_output*torch.exp(x.abs())

vector_exp = vector_exp.apply

class stealth_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w0,w1,b):
        ctx.save_for_backward(x.data,w0.data,w1.data,b.data)
        return F.linear(x,w0+w1,b).data
    
    @staticmethod
    def backward(ctx, grad_output):
        x,w0,w1,b,=ctx.saved_tensors
        
        grad_output=grad_output.contiguous().data
        gx=F.linear(grad_output,(w0+w1).t())
        gb=grad_output.view(-1,grad_output.shape[-1]).sum(dim=0)
        
        gw0=grad_output.unsqueeze(-1)*(x.unsqueeze(-2)-x.view(-1,x.shape[-1]).mean(0))
        gw0=gw0.view(-1,gw0.shape[-2],gw0.shape[-1]).sum(dim=0)
        
        gw1=grad_output.unsqueeze(-1)*(x.unsqueeze(-2)*0+x.view(-1,x.shape[-1]).mean(0))
        gw1=gw1.view(-1,gw1.shape[-2],gw1.shape[-1]).sum(dim=0)
        
        v0=(gw0**2).sum()**0.5
        v1=(gw1**2).sum()**0.5
        s=math.atan2(v1,v0)
        gcombined=gw0.data*math.sin(s)+gw1.data*math.cos(s)
        #print(v0,v1)
        
        return gx,gcombined,gcombined,gb


stealth_linear = stealth_linear.apply


class Linear(nn.Module):
    def __init__(self,nin,nout):
        super().__init__()
        self.layer=nn.Linear(nin,nout)
        self.layer2=nn.Linear(nin,nout)
    
    def forward(self,X):
        return stealth_linear(X,self.layer.weight,self.layer2.weight,self.layer.bias)

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

def drop(X,p=0.9):
    A,B=X.shape
    ind0=[i for i in range(A) if torch.rand(1)<=p]
    if len(ind0)==0:
        ind0=[int(torch.LongTensor(1).random_(A))]
    
    ind1=[i for i in range(B) if torch.rand(1)<=p]
    if len(ind1)==0:
        ind1=[int(torch.LongTensor(1).random_(B))]
    
    ind0=torch.LongTensor(ind0).to(X.device)
    ind1=torch.LongTensor(ind1).to(X.device)
    X2=X[ind0][:,ind1]
    return X2

def pad(x,padding=0.0):
    return torch.nested.nested_tensor(x).to_padded_tensor(padding=padding)



class new(nn.Module):
    def __init__(self,params):
        super(new,self).__init__()
        nh=params.nh;
        nh2=params.nh2;
        #nh3=params.nh3;
        nlayers=params.nlayers
        nlayers2=params.nlayers2
        #nlayers3=params.nlayers3
        
        
        nh0=min(max(nh//64,2),8)
        nh=min(max(nh2//4,2),128)
        nstacks=max(min(params.nlayers//2,4),1)
        self.margin=params.margin
        
        self.pool=perm_inv.einpool_multihead(form='X_BCabH',order=4,equivariant=True)
        self.pool2=perm_inv.einpool_multihead(form='X_BcabH',order=2,deps=['c->a','c->b'],equivariant=False)
        k=len(self.pool.eqs)
        nheads=self.pool.nheads()
        nheads2=self.pool2.nheads()
        print('%d,%d heads'%(nheads,nheads2))
        self.nheads=nheads
        self.nheads2=nheads2
        self.nstacks=nstacks
        k2=len(self.pool2.eqs)
        print(self.pool2.eqs)
        
        self.encoder=nn.ModuleList()
        self.t0=MLP(1,nh,nh0*nheads,2)
        self.t=nn.ModuleList()
        for i in range(nstacks-1):
            self.t.append(MLP(nh0*nheads+nh0*k,nh,nh0*nheads,2))
        
        self.t.append(MLP(nh0*nheads+nh0*k,nh,nh0*nheads,2))
        self.tout=MLP(nh0*nheads,nh,2,2)#MLP(nh*k2,nh,2,2)
        
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        return;
    
    def mixed_pool(self,h,mask):
        h0=self.pool(torch.sin(h).view(*h.shape[:-1],-1,self.nheads),mask.unsqueeze(-1)).view(*h.shape[:-1],-1)
        h1=h.max(dim=-2)[0].unsqueeze(-2)
        h1=h1.max(dim=-3)[0].unsqueeze(-3)
        h1=h1.max(dim=-4)[0].unsqueeze(-4).expand_as(h)
        return torch.cat((h0,h1),dim=-1)
    
    def forward(self,data_batch):
        h=[]
        mask=[]
        for i,ws in enumerate(data_batch['fvs']):
            hi=[w.to(self.w.device) for w in ws[:]]
            #print([w.shape for w in hi])
            if self.training:
                hi=[drop(w,p=0.9) for w in hi]
            else:
                pass
            
            
            hi=[w-w.mean(dim=[-1,-2],keepdim=True) for w in hi] #normalize
            hi=[w/w.std(dim=[-1,-2],keepdim=True).clamp(min=1e-12) if w.shape[-1]*w.shape[-2]>1 else w for w in hi] #normalize
            hi=[w.unsqueeze(dim=-1) for w in hi] # reshape
            
            mask_i=[w*0+1 for w in hi]
            h.append(pad(hi))
            mask.append(pad(mask_i))
        
        h=pad(h)
        mask=pad(mask)
        
        h=self.t0(h)
        for i in range(self.nstacks):
            h=h*mask
            skip=h
            h=self.mixed_pool(h,mask)
            h=skip+self.t[i](h)
            h=h*mask
        
        #h0=torch.sin(h.mean(dim=[-4,-3,-2]))
        #h1=torch.sin(h.std(dim=[-4,-3,-2]))
        #print(h.max(),h.min(),h.mean(),h.std())
        #h=self.pool2(torch.sin(h).view(*h.shape[:-1],-1,self.nheads2)).view(h.shape[0],-1)
        #h=h.std(dim=[-2,-3,-4])
        #h=h.max(dim=-2)[0].max(dim=-2)[0].max(dim=-2)[0]
        #h=((h**2).sum(dim=[-2,-3,-4]).clamp(min=1e-12)/mask.sum(dim=[-2,-3,-4]).clamp(min=1e-12))**0.5 #std pool
        #h=h.sum(dim=[-2,-3,-4])/mask.sum(dim=[-2,-3,-4]).clamp(min=1e-12) #std pool
        h=((h**2).sum(dim=[-2,-3]).clamp(min=1e-20)/mask.sum(dim=[-2,-3]).clamp(min=1e-12))**0.5 #std pool
        h=h.mean(dim=[-2])#/mask.sum(dim=[-2]).clamp(min=1e-12) #std pool
        h=self.tout(h)
        #print(self.tout.layer.weight.abs().mean())
        h=self.margin*torch.tanh(h)
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];
    
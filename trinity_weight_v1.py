import os
import sys
import time
from pathlib import Path
import importlib
import math
import copy

import torch
import util.db as db
import util.smartparse as smartparse


import helper_cyber_apk as helper


def quantile(x,nbins):
    x=x.view(-1)
    v,_=x.sort(dim=0)
    ind=torch.arange(0,nbins).float()/(nbins-1)*(len(x)-1)
    ind=ind.to(x.device)
    ind0=torch.floor(ind).long().clamp(0,len(x)-1)
    ind1=torch.ceil(ind).long().clamp(0,len(x)-1)
    delta=(ind1-ind0).float()+1e-20
    y0=v[ind0]
    y1=v[ind1]
    y=y0+(y1-y0)*(ind-ind0)/delta
    return y


def analyze_tensor(w,bins=100):
    if w is None:
        return torch.Tensor(bins*2).fill_(0)
    else:
        hw=quantile(w,bins).cpu()
        hw_abs=quantile(w.abs(),bins).cpu()
        u,s,v=torch.svd(w)
        hs=quantile(s,bins).cpu()
        hs_abs=quantile(s.abs(),bins).cpu()
        fv=torch.cat((hw,hw_abs,hs,hs_abs),dim=0);
        return fv;

def weight2fv(g,params=None):
    fvs=[]
    for w in g:
        if len(w.shape)==1:
            fvs.append(analyze_tensor(w.data.view(-1,1).cuda()).cpu())
        elif len(w.shape)==2:
            fvs.append(analyze_tensor(w.data.cuda()).cpu())
        elif len(w.shape)==4:
            for i in range(w.shape[-2]):
                for j in range(w.shape[-1]):
                    fvs.append(analyze_tensor(w.data[:,:,i,j].cuda()).cpu())
        else:
            print(w.shape)
            a=0/0
    
    return fvs;

def characterize(interface,data=None,params=None):
    fvs=weight2fv(interface.model.parameters(),params)
    fvs=torch.stack(fvs,dim=0)
    print(fvs.shape)
    return {'fvs':fvs}


def ts_engine(interface,additional_data='enum.pt',params=None):
    return None

def extract_fv(interface,ts_engine,params=None):
    data=ts_engine(interface,params=params);
    fvs=characterize(interface,data,params);
    return fvs


#Extract dataset from a folder of training models
def extract_dataset(models_dirpath,ts_engine,params=None):
    default_params=smartparse.obj();
    default_params.rank=0
    default_params.world_size=1
    default_params.out=''
    
    params=smartparse.merge(params,default_params)
    
    t0=time.time()
    models=os.listdir(models_dirpath);
    models=sorted(models)
    models=[(i,x) for i,x in enumerate(models)]
    
    dataset=[];
    for i,fname in models[params.rank::params.world_size]:
        folder=os.path.join(models_dirpath,fname);
        interface=helper.engine(folder=folder,params=params)
        fvs=extract_fv(interface,ts_engine,params=params);
        
        #Load GT
        fname_gt=os.path.join(folder,'ground_truth.csv');
        f=open(fname_gt,'r');
        for line in f:
            line.rstrip('\n').rstrip('\r')
            label=int(line);
            break;
        
        f.close();
        
        data={'params':vars(params),'model_name':fname,'model_id':i,'label':label};
        fvs.update(data);
        
        if not params.out=='':
            Path(params.out).mkdir(parents=True, exist_ok=True);
            torch.save(fvs,'%s/%d.pt'%(params.out,i));
        
        dataset.append(fvs);
        print('Model %d(%s), time %f'%(i,fname,time.time()-t0));
    
    dataset=db.Table.from_rows(dataset);
    return dataset;

def predict(ensemble,fvs):
    scores=[];
    with torch.no_grad():
        for i in range(len(ensemble)):
            params=ensemble[i]['params'];
            arch=importlib.import_module(params.arch);
            net=arch.new(params);
            net.load_state_dict(ensemble[i]['net'],strict=True);
            net=net.cuda();
            net.eval();
            
            s=net.logp(fvs).data.cpu();
            s=s*math.exp(-ensemble[i]['T']);
            scores.append(s)
    
    s=sum(scores)/len(scores);
    s=torch.sigmoid(s); #score -> probability
    trojan_probability=float(s);
    return trojan_probability


if __name__ == "__main__":
    import os
    default_params=smartparse.obj();
    default_params.rank=0
    default_params.world_size=1
    default_params.out='fvs_weight'
    params=smartparse.parse(default_params);
    params.argv=sys.argv;
    
    extract_dataset(os.path.join(helper.root(),'models'),ts_engine,params);
    

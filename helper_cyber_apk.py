
import time
import copy
import os
import torch
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import util.smartparse as smartparse

from pathlib import Path

warnings.filterwarnings("ignore")

def root():
    return './'

#The user provided engine that our algorithm interacts with
class engine:
    def __init__(self,folder=None,params=None):
        default_params=smartparse.obj();
        default_params.model_filepath='';
        default_params.examples_dirpath='';
        params=smartparse.merge(params,default_params)
        
        if params.model_filepath=='':
            params.model_filepath=os.path.join(folder,'model.pt');
        
        import utils.models
        from utils.abstract import AbstractDetector
        from utils.models import load_model, load_models_dirpath
        
        model, model_repr, model_class = load_model(params.model_filepath)
        self.model=model
        #self.model=model.cuda()



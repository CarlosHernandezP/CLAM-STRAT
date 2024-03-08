import torch

import sys
import os

# Assuming your current working directory is 'A' and the module is in '../dinov2'
module_path = os.path.abspath(os.path.join('..', 'dinov2'))
sys.path.append(module_path)
from dinov2.utils.config import setup
from dinov2.configs import dinov2_default_config
from omegaconf import OmegaConf
from dinov2.train.ssl_meta_arch import SSLMetaArch


def load_dinov2(finetune=True):
    if finetune==True:
        return torch.load('/home/carlos.hernandez/PhD/dinov2/trained_models/best_dino_model.pth')
    else:
        BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")

        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }

        backbone_arch = backbone_archs[BACKBONE_SIZE]
        backbone_name = f"dinov2_{backbone_arch}"

        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)

        return backbone_model

def dinov2_finetuned(): 
    config_dir = '/home/carlos.hernandez/PhD/dinov2/dinov2/configs/ssl_default_config.yaml'  
    
    # As imported 
    default_cfg = OmegaConf.create(dinov2_default_config)

    # Set up with the config yaml
    cfg = OmegaConf.load(config_dir) 
    cfg = OmegaConf.merge(default_cfg, cfg)   

    DINO_VIT_S_PATH_FINETUNED = '/home/carlos.hernandez/PhD/dinov2/eval/training_99999_swept-wildflower-16/teacher_checkpoint.pth'
    pretrained = torch.load(DINO_VIT_S_PATH_FINETUNED)

    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value
    new_state_dict.keys()

    # Load model
    model_new = SSLMetaArch(cfg)

    print(model_new.teacher['backbone'].load_state_dict(new_state_dict, strict=False))
    finetuned_model = model_new.teacher['backbone']
    return finetuned_model

if __name__ == '__main__':
    print(load_dinov2())





import csv
import datetime
import json
import os
from typing import Any, Literal

import clip
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import torchvision
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.transforms import Compose

from models import *
from pibisa import PibisaAtt

def compute_image_pairs(x1: torch.Tensor, x2: torch.Tensor, pibisaObj: PibisaAtt,
                        mode: Literal['pos', 'neg', 'diff'] = 'pos'):

    ''' Given a pair of images, compute the saliencies that highlight what they have
    in common given an encoder. An initalized PibisaAtt object needs to be passed, since it
    is responsible to compute the attributions.
    '''
    x1Embedding = pibisaObj.encoder_forward(x1)
    x2Embedding = pibisaObj.encoder_forward(x2)

    if mode == 'pos':
        x1SaliencyPos = pibisaObj.compute_attribution(query_image=x1, target_emb=x2Embedding)
        x2SaliencyPos = pibisaObj.compute_attribution(query_image=x2, target_emb=x1Embedding)
        return x1SaliencyPos, x2SaliencyPos

    elif mode == 'neg':
        x1SaliencyNeg = pibisaObj.compute_attribution(query_image=x1, target_emb=x2Embedding,
                                                   pos_sim=torch.tensor([-1], dtype=torch.long))
        x2SaliencyNeg = pibisaObj.compute_attribution(query_image=x2, target_emb=x1Embedding,
                                                   pos_sim=torch.tensor([-1], dtype=torch.long))
        return x1SaliencyNeg, x2SaliencyNeg

    elif mode == 'diff':
        x1SaliencyPos = pibisaObj.compute_attribution(query_image=x1, target_emb=x2Embedding)
        x2SaliencyPos = pibisaObj.compute_attribution(query_image=x2, target_emb=x1Embedding)

        x1SaliencyNeg = pibisaObj.compute_attribution(query_image=x1, target_emb=x2Embedding,
                                                   pos_sim=torch.tensor([-1], dtype=torch.long))
        x2SaliencyNeg = pibisaObj.compute_attribution(query_image=x2, target_emb=x1Embedding,
                                                   pos_sim=torch.tensor([-1], dtype=torch.long))

        return x1SaliencyPos-x1SaliencyNeg, x2SaliencyPos-x2SaliencyNeg

    else:
        raise NameError("Please, insert a valid mode, options: ['diff', 'pos', 'neg']")

def plot_pairs(x1: torch.Tensor, x2: torch.Tensor,
               x1Saliency: np.typing.NDArray[Any], x2Saliency: np.typing.NDArray[Any],
               cmap='Reds') -> None:

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(8,8))

    width, height = x1Saliency.shape[-2], x1Saliency.shape[-1]

    x1Pil = unnormalize(x1).resize((width, height))
    x2Pil = unnormalize(x2).resize((width, height))

    ax1.imshow(x1Pil)
    ax1.axis('off')

    ax2.imshow(x2Pil)
    ax2.axis('off')

    ax3.imshow(x1Pil)
    ax3.imshow(x1Saliency, cmap=cmap, alpha=0.5)
    ax3.axis('off')

    ax4.imshow(x2Pil)
    ax4.imshow(x2Saliency, cmap=cmap, alpha=0.5)
    ax4.axis('off')

    plt.show()

def patch_image(pil_img, grid_size=4):
    pil_img = pil_img.resize((224,224))
    window_size = np.floor(224/grid_size)
    patches_list = []
    for j in range(grid_size):
        for i in range(grid_size):
            patches_list.append(pil_img.crop((i*window_size,j*window_size,
                                              window_size+i*window_size,
                                              window_size+j*window_size)).resize((224,224)))
    return patches_list

def decode_imagenet(preds: torch.Tensor):
    file = open('imagenet_labels.txt')
    imgnt_labels = file.read()
    file.close()

    imgnt_labels = imgnt_labels.split('\n')

    decoded_labels = []
    for y in preds:
        decoded_labels.append(imgnt_labels[y])

    return decoded_labels

def unnormalize(x:torch.Tensor, toPIL=True):
    invTrans: Compose = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])
    x = invTrans(x)
    if toPIL:
        return to_pil_image(torch.squeeze(x))
    else:
        return x


def mutual_information(hgram):
    """Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals

    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum

    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def select_model_with_head(model_name: str = 'ResNET') -> tuple[torch.nn.Module, torch.nn.Module]:
    if model_name == 'VGG':
        model = VGG_emb()        
        metric_model = torchvision.models.vgg16(weights='DEFAULT')
        return model, metric_model

    if model_name == 'ResNET':
        model = ResNET_emb()        
        metric_model = torchvision.models.resnet50(weights='DEFAULT')
        return model, metric_model
        

    if model_name == 'ViT':
        model = ViT16_emb()
        metric_model = torchvision.models.vit_b_16(weights='DEFAULT')
        return model, metric_model

    raise NameError("Insert a valid architecture")

def select_model_no_head(model_name: str = 'ResNET') -> torch.nn.Module:
    if model_name == 'VGG':
        model = VGG_emb()                
        return model

    if model_name == 'ResNET':
        model = ResNET_emb()        
        return model        

    if model_name == 'ViT':
        model = ViT16_emb()
        return model
    
    if model_name == 'DINOv1_ViT':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        return model # type: ignore

    if model_name == 'DINOv1_ResNET':    
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        return model # type: ignore
    
    if model_name == 'DINOv2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
        return model # type: ignore   
    
    if model_name == 'SwAV':
        model = torch.hub.load('facebookresearch/swav:main', 'resnet50w2')
        return model # type: ignore

    if model_name == 'Barlow Twins':
        model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50') 
        return model # type: ignore
    
    if model_name == 'iBOT':
        model = ViT16_emb(weights='iBOT')
        return model
    
    if model_name == 'CLIP_ViT':
        model, _ = clip.load("ViT-B/16")
        return model
    
    if model_name == 'CLIP_ResNET':
        model, _ = clip.load("RN50")
        return model
            
    raise NameError("Insert a valid architecture")
    
def create_save_folder(cfg: omegaconf.DictConfig, metrics_list: list[str]) -> tuple[str, str]:
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    save_path = os.path.join(cfg.out_path, "classification", cfg.model, today)
    i = 0
    while i < 100:
        save_path = save_path + '_' + str(i)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            break
        i = i + 1

    csv_file = os.path.join(save_path, "metrics.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metrics_list)

    json_file = os.path.join(save_path, 'configs.json')
    with open(json_file, 'w', newline='') as fp:
        json.dump(dict(cfg), fp, indent=4, sort_keys=True)

    return csv_file, json_file
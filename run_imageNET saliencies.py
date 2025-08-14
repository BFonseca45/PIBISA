import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import PIL.ImageOps
import time
import hydra

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet50, vgg16, vit_b_16
from torchvision.datasets.imagenet import ImageNet
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from src.CIBISA import *
from src.utils import *
from src.metrics import *
from src.models import *

@hydra.main(version_base=None, config_path='./configs/', config_name='config.yaml')
def run_main(cfg: DictConfig):
    # Passing config to variables    
        
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(0)
    transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])

    if cfg.db.model == 'ResNET':
        architectures = ['Supervised', 'DINOv1','Barlow Twins', 'SwAV']

    model_list = []        
    for architecture in architectures:

        if architecture == 'VGG-16':
            model = VGG_emb(level='full')
            model = model.to(device)
            model.eval()
        elif architecture == 'Supervised':
            model = ResNET_emb(level='full')
            model = model.to(device)
            model.eval()
        elif architecture == 'ViTb-16':
            model = ViT16_emb()
            model = model.to(device)
            model.eval()
        elif architecture == 'Dino':
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            model = model.to(device)
            model.eval()

        elif architecture == 'DINOv1':
            model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            model = model.to(device)
            model.eval()
        
        elif architecture == 'SwAV':     
            model = torch.hub.load('facebookresearch/swav:main', 'resnet50w2')
            model = model.to(device)
            model.eval()
            
        elif architecture == 'Barlow Twins':     
            model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
            model = model.to(device)
            model.eval()

        else:
            raise NameError("Insert a valid architecture")
        
        model_list.append(model)        

    gen_cpu = torch.Generator(device='cpu')
    gen_cpu.manual_seed(45)

    dataset = ImageNet(root='/nas-ctm01/datasets/public/ImageNet/', split='val', transform=transform)
    sub_set = Subset(dataset=dataset, indices=range(0,100))
    dataloader = DataLoader(dataset=sub_set, batch_size=1, shuffle=True, num_workers=1, generator=gen_cpu)

    j = 0
    cm = 'turbo'
    alph = 0.6

    for x,y1 in tqdm(dataloader):
        saliency = []
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        x_pil = unnormalize(x)
        x = x.to(device)                

        for model in model_list:
            s,_ = compute_saliency(x=x, x_target=x, model=model, n_masks=20, epochs=20, return_negative=False, learnable_param=True)
            saliency.append(s)
        
        f, ax = plt.subplots(1,len(saliency)+1, figsize=(16,20))

        
        ax[0].imshow(x_pil.resize((224,224)))
        ax[0].axis('off')

        for i,architecture in enumerate(architectures):

            ax[i+1].imshow(x_pil.resize((224,224)))
            ax[i+1].imshow(saliency[i], cmap=cm, alpha=alph)
            ax[i+1].title.set_text(architecture)
            ax[i+1].axis('off')
            
        fig_savefile = './out/images/saliency' + str(j) + '.png'
        plt.savefig
        plt.close()
        
        j=j+1
        
if __name__ == '__main__':
    run_main()
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import PIL.ImageOps
import time
import hydra
import clip

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet50, vgg16, vit_b_16
from torchvision.datasets.imagenet import ImageNet
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from src.CIBISA import *
from src.CIBISA_utils import *
from src.data import ImageNETs
# from src.metrics import *
from src.models import *
from src.metrics_quantus import QuantusLocalisation

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

    metrics_names = (
            "PointingGame",
            "AttributionLocalisation",
            "TopKIntersection",
            "RelevanceRankAccuracy",
            "RelevanceMassAccuracy",
            "Focus",
            "AUC"
        )
    
    if cfg.db.model == 'ResNET':
        architectures = ['ResNET', 'DINOv1','Barlow Twins', 'SwAV', 'CLIP_ResNET']
    elif cfg.db.model == 'ViT':
        architectures = ['Supervised', 'DINOv1','DINOv2', 'iBOT', 'CLIP_ViT']
    
    gen_cpu = torch.Generator(device='cpu')
    gen_cpu.manual_seed(45)
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                            torchvision.transforms.ToTensor(), 
                            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    path = "/home/up202204320/datasets/ImageNetS300"
    split = "validation"

    dset = ImageNETs(path = path, split = split, transform = transform)
    sub_set = Subset(dataset=dset, indices=range(0,100))
    dataloader = DataLoader(dataset=sub_set, batch_size=1, shuffle=True, num_workers=1, generator=gen_cpu)

    j = 0
    cm = 'turbo'
    alph = 0.4
    
    for architecture in architectures:
        if architecture == 'VGG-16':
            model = VGG_emb(level='full')
            model = model.to(device)
            model.eval()
        elif architecture == 'ResNET':
            model = ResNET_emb(level='full')
            model = model.to(device)
            model.eval()
        elif architecture == 'Supervised':
            model = ViT16_emb()
            model = model.to(device)
            model.eval()
        elif architecture == 'DINOv2':
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
            model = model.to(device)
            model.eval()

        elif architecture == 'DINOv1':
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
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

        elif architecture == 'iBOT':
            model = ViT16_emb(weights='iBOT')
            model = model.to(device)
            model.eval()
        elif architecture == 'CLIP_ViT':
            model, preprocess = clip.load("ViT-B/16", device=device)
            model = model.to(device)
            model.eval()
        
        elif architecture == 'CLIP_ResNET':
            model, preprocess = clip.load("RN50", device=device)
            model = model.to(device)
            model.eval()
                        
        else:
            raise NameError("Insert a valid architecture")        

        total_metric = []
        for x,m in tqdm(dataloader):
            # saliency = []
            # x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
            x_pil = unnormalize(x)
            x_np = x.detach().cpu().numpy()
            x = x.to(device)                
    
            if model._get_name() == 'CLIP':
                s,_ = compute_saliency(x=x, x_target=model.encode_image(x), model=model, n_masks=20, epochs=20, return_negative=False, learnable_param=True)
            else:                
                s,_ = compute_saliency(x=x, x_target=x, model=model, n_masks=20, epochs=20, return_negative=False, learnable_param=True)
            
            current_metric = []
            for metric in metrics_names:
                metricObj = QuantusLocalisation(metric_name=metric)
                current_metric.append(metricObj.compute_metric(model=None, x_batch=x_np, y_batch=0,
                                                               a_batch=s[None,...], s_batch=m.unsqueeze(0).numpy(), device=device))
            total_metric.append(current_metric)
                                
            plt.imshow(x_pil.resize((224,224)))
            plt.imshow(s, cmap=cm, alpha=alph)
            plt.title(architecture)
            plt.axis('off')

            fig_savefile = './out/images_' + cfg.db.model + '/saliency' + architecture + str(j) + '.png'
            plt.show()
            plt.savefig(fig_savefile)
            plt.close()
            
            j=j+1
            
            if j > 100:
                break
        
        total_metric_np = np.array(total_metric)
        print("Architecture: ", architecture)
        print("Metrics:", np.mean(total_metric_np, axis=0))
        
if __name__ == '__main__':
    run_main()
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import PIL.ImageOps
import time
import hydra
import wandb

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
    beta = cfg.param.beta
    learnable_param = cfg.param.learnable
    architecture = cfg.db.model
    dataset_name = cfg.db.dataset_name

    wandb_run = wandb.init(
        project='PIBISA',
        config={
            "beta": beta, 
            "architecture": architecture, 
            "learnable_beta": learnable_param
        },
    )
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(0)
    transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])

    if architecture == 'VGG':
        model = VGG_emb()
        model = model.to(device)
        model.eval()

        metric_model = vgg16(weights='DEFAULT')
        metric_model = metric_model.to(device)
        metric_model.eval()

    elif architecture == 'ResNET':
        model = ResNET_emb()
        model = model.to(device)
        model.eval()

        metric_model = resnet50(weights='DEFAULT')
        metric_model = metric_model.to(device)
        metric_model.eval()

    elif architecture == 'ViT':
        model = ViT16_emb()
        model = model.to(device)
        model.eval()

        metric_model = vit_b_16(weights='DEFAULT')
        metric_model = metric_model.to(device)
        metric_model.eval()

    else:
        raise NameError("Insert a valid architecture")

    gen_cpu = torch.Generator(device='cpu')
    gen_cpu.manual_seed(45)

    dataset = ImageNet(root='/nas-ctm01/datasets/public/ImageNet/', split='val', transform=transform)
    sub_set = Subset(dataset=dataset, indices=range(0,10000))
    dataloader = DataLoader(dataset=sub_set, batch_size=1, shuffle=True, num_workers=1, generator=gen_cpu)

    lerf = []
    morf = []
    times = []

    eval_tool = EvalMetric(model = metric_model)

    i = 0
    for x,y1 in tqdm(dataloader):
        saliency = []
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        x = x.to(device)

        tic = time.time()
        
        saliency_flat, _ = compute_saliency(x=x, x_target=x, model=model, n_masks=20, beta=beta, return_negative=False, learnable_param=learnable_param)
        toc = time.time()

        times.append(toc-tic)

        y_target = metric_model(x)
        y_target = torch.argmax(y_target, dim=1)

        score_l, area = eval_tool.compute_score(img=x, y_target=y_target, saliency=saliency_flat, metric='LeRF')
        lerf.append(area)
        score_m, area = eval_tool.compute_score(img=x, y_target=y_target, saliency=saliency_flat, metric='MoRF')
        morf.append(area)

        i = i+1
        if not (i % 100):
            morf_mean = np.nanmean(morf)
            learf_mean = np.nanmean(lerf)
            time_mean = np.nanmean(times)
            
            wandb_run.log({'MoRF': morf_mean, 'LeRF': learf_mean, 'Times':time_mean})
            
            metrics_dict = {'MoRF': morf, 'LeRF': lerf, 'Times':times}
            out_dir = './out/metrics/'
            savefile = out_dir + architecture + '.pkl'
            open_file = open(savefile, 'wb')
            pickle.dump(metrics_dict, open_file)
            open_file.close()

if __name__ == '__main__':
    run_main()

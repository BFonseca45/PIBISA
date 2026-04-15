import sys
sys.path.insert(0, "/home/bfcoelho/Documents/Scripts/CIBISA/src/")

import torch
import torchvision
import hydra
import pickle
import csv

from src.data import *
from src.utils import select_model_no_head, create_save_folder
from src.pibisa import PibisaAtt, PibisaConfig
from metrics_pibisa import RetrievalMetrics
from src.sbsm import compute_sbsm
from tqdm import tqdm
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='./configs/', config_name='config_retrieval.yaml')
def run_main(cfg: DictConfig):
    # Passing config to variables    
    csv_file, json_file = create_save_folder(cfg=cfg, 
                                             metrics_list=['PIBISA CI', ' PIBISA CD', 'SBSM CI', 'SBSM CD'])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = select_model_no_head(model_name=cfg.model)
    model = model.eval().to(device)    
    
    if model._get_name() == 'CLIP':
        model.forward = model.encode_image # type: ignore

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    if cfg.db.dataset_name == 'oxford':
        dataset = OxfordBuildings(path='/home/up202204320/datasets/oxford/', transform=transform)
    elif cfg.db.dataset_name == 'paris':
        dataset = ParisBuildings(path='/home/up202204320/datasets/paris/', transform=transform)
    elif cfg.db.dataset_name == 'instre':
        base_dir = "/home/up202204320/datasets/"
        crop_mode = "union"
        max_boxes = 2
        seed = 42
        subset_name = "INSTRE-S1"
        csv_path = f"/home/up202204320/datasets/INSTRE/csv/{subset_name}.csv"
        dataset = InstreDataset(csv_path, base_dir, crop_mode=crop_mode, max_boxes=max_boxes, seed=seed, transform=transform)

    dloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    
    pibisa_cfg = PibisaConfig(beta=cfg.beta, phi=cfg.phi, device=device)
    pibisa = PibisaAtt(encoder_forward=model.forward, config=pibisa_cfg)
    
    ret_metrics = RetrievalMetrics(encoder_forward=model.forward)
    
    ci_list_pibisa = []
    cd_list_pibisa = []
    ci_list_sbsm = []
    cd_list_sbsm = []    

    for q, gt in tqdm(dloader):
        sal_list_sbsm = []
        sal_list_pibisa = []
        for j in range(len(gt)):            
            saliency_pibisa = pibisa.compute_attribution(query_image=q.to(device), target_emb=model.forward(gt[j].to(device)))    
            sal_list_pibisa.append(saliency_pibisa)
            
            saliency_sbsm = compute_sbsm(query_tensor=q.to(device), neighbor_tensor=gt[j].to(device),model=model.forward, device=device)
            sal_list_sbsm.append(saliency_sbsm)

        ci_pibisa, cd_pibisa = ret_metrics.confidence_increase_drop(input_query=q, input_targets=gt, saliency_list=sal_list_pibisa)
        ci_sbsm, cd_sbsm = ret_metrics.confidence_increase_drop(input_query=q, input_targets=gt, saliency_list=sal_list_sbsm)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ci_pibisa, cd_pibisa, ci_sbsm, cd_sbsm])
        
        ci_list_pibisa.extend(ci_pibisa.detach().cpu().numpy())
        cd_list_pibisa.extend(cd_pibisa.detach().cpu().numpy())

        ci_list_sbsm.extend(ci_sbsm.detach().cpu().numpy())
        cd_list_sbsm.extend(cd_sbsm.detach().cpu().numpy())
        
        
    print("Confidence Increase: ", np.mean(ci_list_sbsm)*100)
    print("Confidence Drop: ", np.mean(cd_list_sbsm))

if __name__ == '__main__':
    run_main()
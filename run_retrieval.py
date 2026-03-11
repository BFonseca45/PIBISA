import torch
import torchvision
import hydra
import pickle

from src.data import * 
from src.CIBISA_utils import *
from src.CIBISA import *
from src.models import *
from src.metrics import create_sal_masks, confidence_increase_drop
from src.ret_methods import compute_sbsm
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path='./configs/', config_name='config.yaml')
def run_main(cfg: DictConfig):
    # Passing config to variables
    print(cfg)
    savefile = cfg.param.method + '_' + cfg.db.model + '_' + cfg.db.dataset_name + '.pkl'
    print(savefile)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if cfg.db.model == 'VGG':
        model = VGG_emb()
        model = model.to(device)
        model.eval()

    elif cfg.db.model == 'ResNET':
        model = ResNET_emb()
        model = model.to(device)
        model.eval()

    elif cfg.db.model == 'ViT':
        model = ViT16_emb()
        model = model.to(device)
        model.eval()

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

    ci_list = []
    cd_list = []
    idx = 0
    for q, gt in tqdm(dloader):    
        sal_list = []
        for j in range(len(gt)): 
            if cfg.param.method == 'PIBISA':
                s,_ = compute_saliency(x=gt[j].to(device), x_target=q.to(device), model=model, n_masks=20, epochs=15, beta=cfg.param.beta, return_negative=False, learnable_param=cfg.param.learnable)
            elif cfg.param.method == 'SBSM':
                s = compute_sbsm(query_tensor=q.to(device), neighbor_tensor=gt[j].to(device),model=model)
            sal_list.append(s)
        
        if len(gt) > 5:
            N_SAL = 5
            
            fig, ax = plt.subplots(nrows=1, ncols=N_SAL+1, figsize=(15,15))
            ax[0].imshow(unnormalize(q))
            ax[0].axis('off')
            for i in range(1,len(ax)):
                ax[i].imshow(unnormalize(gt[i]))
                ax[i].imshow(sal_list[i-1], cmap='turbo', alpha=0.4)
                ax[i].axis('off')
            filename = './out/' + cfg.param.method + '/' + cfg.db.dataset_name + '/sal' + str(idx) + cfg.db.model + '.png'
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            
        ci, cd = confidence_increase_drop(model=model, query=q, gts=gt, saliency=sal_list)
        
        ci_list.extend(ci.detach().cpu().numpy())
        cd_list.extend(cd.detach().cpu().numpy())
        
        idx = idx+1
        
        if not(idx%100):            
            tosave = {'CI':np.mean(ci_list)*100, 'CD':np.mean(cd_list)}
            open_file = open(savefile, 'wb')
            pickle.dump(tosave, open_file)
            open_file.close()

    print("Confidence Increase: ", np.mean(ci_list)*100)
    print("Confidence Drop: ", np.mean(cd_list))
    
if __name__ == '__main__':
    run_main()
import torch
import torchvision

from src.data import * 
from src.CIBISA_utils import *
from src.CIBISA import *
from src.models import *
from src.metrics import create_sal_masks, confidence_increase_drop
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = ResNET_emb(level='full')
model = model.to(device)
model.eval()

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset = OxfordBuildings(path='/home/bfcoelho/Downloads/oxbuildings', transform=transform)
dloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

ci_list = []
cd_list = []
idx = 0
for q, gt in tqdm(dloader):    
    sal_list = []
    for j in range(len(gt)): 
        s,_ = compute_saliency(x=gt[j], x_target=q, model=model, n_masks=20, epochs=15, beta=1, return_negative=False, learnable_param=True)
        sal_list.append(s)
    
    N_SAL = 5
    fig, ax = plt.subplots(nrows=1, ncols=N_SAL+1, figsize=(15,15))
    ax[0].imshow(unnormalize(q))
    ax[0].axis('off')
    for i in range(1,len(ax)):
        ax[i].imshow(unnormalize(gt[i]))
        ax[i].imshow(sal_list[i-1], cmap='turbo', alpha=0.4)
        ax[i].axis('off')
    filename = './out/oxford/sal' + str(idx) + '.png'
    plt.savefig(filename, bbox_inches='tight')
    
    ci, cd = confidence_increase_drop(model=model, query=q, gts=gt, saliency=sal_list)
    
    ci_list.extend(ci)
    cd_list.extend(cd)
    
    idx = idx+1

print("Confidence Increase: ", np.mean(ci_list)*100)
print("Confidence Drop: ", np.mean(cd_list))
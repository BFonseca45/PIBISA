import torch
import torchvision
import matplotlib.pyplot as plt
import warnings
import clip
import pickle
import tqdm
import hydra
import wandb
warnings.filterwarnings('ignore')

from PIL import Image
from src.utils import *
from src.metrics import *
from transformers import AutoProcessor, AutoModelForVision2Seq
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path='./configs/', config_name='config_CLIP.yaml')
def run_main(cfg: DictConfig):
    # Passing config to variables    
    beta = cfg.param.beta
    server = cfg.run.server
    random_mask = cfg.run.random_mask
    
    wandb_run = wandb.init(
        project='PIBISA',
        config={
            "beta": beta, 
            "server": server,
            "random_mask": random_mask
        },
    )
        
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print("Device in use: ", device)
    torch.manual_seed(0)

    blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip = AutoModelForVision2Seq.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    clip_model, clip_processor = clip.load("ViT-B/16", device=device)
    clip_model.eval()

    eval_tool = EvalMetric(model=blip, mode='nle')

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if server:
        dset = torchvision.datasets.CocoCaptions(root='/data/toys/coco/val2017/', annFile='/data/toys/coco/annotations/captions_val2017.json', transform=transform)
    else:
        dset = torchvision.datasets.CocoCaptions(root='/home/bfcoelho/Downloads/coco/val2017/', annFile='/home/bfcoelho/Downloads/coco/annotations/captions_val2017.json', transform=transform)
    subset = torch.utils.data.Subset(dataset=dset, indices=range(1000))

    gen_cpu = torch.Generator(device='cpu')
    gen_cpu.manual_seed(45)
    dloader = torch.utils.data.DataLoader(dataset=subset, batch_size=1, shuffle=False, pin_memory=True, num_workers=6, generator=gen_cpu)

    print('## Everything loaded, starting to run ##')

    i = 1
    MoRF = []
    LeRF = []
    captions = []
    savefile = './out/metrics/CLIP_metrics.pkl'
    for x, _ in tqdm(dloader):
        # Generating the caption for a given image
        x_pil = unnormalize(x)
        x = x.to(device)

        blip_inputs = blip_processor(x_pil, return_tensors="pt").to("cuda")
        blip_out = blip.generate(**blip_inputs)
        target_caption = blip_processor.decode(blip_out[0], skip_special_tokens=True)

        # Generating the CLIP text embedding
        text = clip.tokenize(target_caption).to(device)
        y_target = clip_model.encode_text(text).to(device)

        # Computing the saliency map to explain the caption
        if random_mask:
            sal = 3*torch.rand((1,1,7,7))
            sal = nn.Upsample(size=(224, 224), mode='bicubic')(sal)
            sal = torch.sigmoid(sal)
            sal = sal.detach().cpu().numpy()
        else:
            sal = compute_saliency(x=x, x_target=y_target, model=clip_model, n_masks=20, beta=beta, return_negative=False)
        _,morf = eval_tool.compute_score(img=x, y_target=target_caption, saliency=sal.squeeze(), metric='MoRF')
        _,lerf = eval_tool.compute_score(img=x, y_target=target_caption, saliency=sal.squeeze(), metric='LeRF')

        MoRF.append(morf)
        LeRF.append(lerf)
        

        if not (i % 5):
            captions.append(target_caption)

            tosave = {'MoRF':MoRF, 'LeRF':LeRF, "Captions":captions}
            open_file = open(savefile, 'wb')
            pickle.dump(tosave, open_file)
            open_file.close()

            MoRF_mean = np.nanmean(MoRF, axis=0)
            LeRF_mean = np.nanmean(LeRF, axis=0)
            
            wandb_run.log({"Meteor_MoRF": MoRF_mean[0],"Bleu_MoRF": MoRF_mean[1], "Rouge1_MoRF": MoRF_mean[2], "Rouge2_MoRF": MoRF_mean[3], "RougeL_MoRF": MoRF_mean[4]})
            wandb_run.log({"Meteor_LeRF": LeRF_mean[0],"Bleu_LeRF": LeRF_mean[1], "Rouge1_LeRF": LeRF_mean[2], "Rouge2_LeRF": LeRF_mean[3], "RougeL_LeRF": LeRF_mean[4]})
                
            fig_savefile = './out/clip_images/saliency' + str(i) + '.png'
            plt.imshow(x_pil)
            plt.imshow(sal, cmap='turbo', alpha=0.5)
            plt.title('Image n ' + str(i))
            plt.axis('off')

            plt.savefig(fig_savefile)
            plt.close()

        i += 1

if __name__ == '__main__':
    run_main()
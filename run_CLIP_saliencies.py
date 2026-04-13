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
from metrics_implemented import *
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
from omegaconf import DictConfig, OmegaConf

from NIB.scripts.plot import visualize_vandt_heatmap
from NIB.saliency import chefer,fast_ig,gradcam,m2ib,mfaba,nib,rise,saliencymap

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip = AutoModelForVision2Seq.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

def generate_plot(image,text,method):    
    image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(device)
    image_features = model.get_image_features(image_feat)
    text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(device)
    text_words = tokenizer.convert_ids_to_tokens(text_ids[0].tolist())
    text_features = model.get_text_features(text_ids)

    if method in [chefer,fast_ig,gradcam,mfaba,saliencymap]:
        vmap, tmap = method(model, processor, [text], [image])
    elif method == rise:
        vmap, tmap = rise(model, image_feat,[text_ids],image_features,text_features)
        tmap = [tmap[0].detach().cpu().numpy()]
    elif method == m2ib:
        vmap, tmap = m2ib(model, [text_ids], image_feat, 0.1)
    elif method == nib:
        vmap, tmap = nib(model, [text_ids], image_feat,10, 9)
    if vmap.shape[1] == 3:
        vmap = vmap.mean(1)
    
    return np.squeeze(vmap[0])

@hydra.main(version_base=None, config_path='./configs/', config_name='config_CLIP.yaml')
def run_main(cfg: DictConfig):
    # Passing config to variables    
    beta = cfg.param.beta
    server = cfg.run.server    
        
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print("Device in use: ", device)
    torch.manual_seed(0)

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if server:
        dset = torchvision.datasets.CocoCaptions(root='/data/toys/coco/val2017/', annFile='/data/toys/coco/annotations/captions_val2017.json', transform=transform)
    else:
        dset = torchvision.datasets.CocoCaptions(root='/home/bfcoelho/Downloads/coco/val2017/', annFile='/home/bfcoelho/Downloads/coco/annotations/captions_val2017.json', transform=transform)
    # subset = torch.utils.data.Subset(dataset=dset, indices=range(1000))

    gen_cpu = torch.Generator(device='cpu')
    gen_cpu.manual_seed(45)
    dloader = torch.utils.data.DataLoader(dataset=dset, batch_size=1, shuffle=True, pin_memory=True, num_workers=6, generator=gen_cpu)

    print('## Everything loaded, starting to run ##')

    i = 1
    for x, cap in tqdm(dloader):
        # Generating the caption for a given image
        items = []
        items.append(cap[0][0])
        
        x_pil = unnormalize(x)                
        x = processor(images=x_pil, return_tensors="pt")['pixel_values'].to(device)
        x = x.to(device)

        blip_inputs = blip_processor(x_pil, return_tensors="pt").to("cuda")
        blip_out = blip.generate(**blip_inputs)
        target_caption = blip_processor.decode(blip_out[0], skip_special_tokens=True)

        items.append(target_caption)
        # Generating the CLIP text embedding
        for j, item in enumerate(items):
            text = torch.tensor([tokenizer.encode(item, add_special_tokens=True)]).to(device)
            y_target = model.get_text_features(text).to(device)

            p, _ = compute_saliency(x=x, x_target=y_target, model=model, n_masks=20, epochs=10, beta=2, return_negative=False, learnable_param=True)
            n = generate_plot(x_pil,item,rise)
            m = generate_plot(x_pil,item,m2ib)

            f, ax = plt.subplots(nrows=1, ncols=4, layout='constrained', sharex=True, figsize=(7,5), dpi=100)

            ax[0].imshow(unnormalize(x).resize((224,224)))
            ax[0].title.set_text('Original')
            ax[0].axis('off')
            
            ax[1].imshow(unnormalize(x).resize((224,224)))
            ax[1].imshow(np.squeeze(p), cmap='turbo', alpha=0.5)
            ax[1].title.set_text('P-IBISA')
            ax[1].axis('off')

            ax[2].imshow(unnormalize(x).resize((224,224)))
            ax[2].imshow(np.squeeze(n), cmap='turbo', alpha=0.5)
            ax[2].title.set_text('NIB')
            ax[2].axis('off')

            ax[3].imshow(unnormalize(x).resize((224,224)))
            ax[3].imshow(np.squeeze(m), cmap='turbo', alpha=0.5)    
            ax[3].title.set_text('M2IB')
            ax[3].axis('off')

            f.suptitle(item)
            plt.show()            
            plt.axis('off')

            if j == 0:
                fig_savefile = './out/clip_comparison_learn/saliency' + str(i) + 'GT.png'
            else:
                fig_savefile = './out/clip_comparison_learn/saliency' + str(i) + 'BLIP.png'
            plt.savefig(fig_savefile)
            plt.close()

        i += 1
        if i > 300:
            break

if __name__ == '__main__':
    run_main()
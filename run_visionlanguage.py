import sys

sys.path.insert(0, "/home/bfcoelho/Documents/Scripts/CIBISA/src/")

# Imports
import os
import random
import csv

import numpy as np
# PyTorch Imports
import torch
import clip
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.data
import torchvision.transforms.v2
import tqdm
# Transformers Imports
import transformers

# from datasets import ConceptualCaptions, collate_fn_cc, ImagenetDataset, collect_fn_imagenet
import NIB.saliency
# Project Imports
import NIB.scripts.data
import src.metrics_NIB
import src.pibisa
import src.grad_eclip
from src.utils import create_save_folder

# Setup environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function: Min-Max Normalization
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


# Function: Set random seed
def setup_seed(seed=0):

    # Random
    random.seed(seed)

    # Python Environment
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return

def run_rise(args, model, image_processed, tid, im_f, tf):
    result, _ = NIB.saliency.rise(model, image_processed, tid, im_f, tf)
    return result

def run_m2ib(args, model, image_processed, tid, im_f, tf):
    result, _ = NIB.saliency.m2ib(model, tid, image_processed, args.beta)
    return result

def run_nib(args, model, image_processed, tid, im_f, tf):
    result, _ = NIB.saliency.nib(model, tid, image_processed, args.num_steps, args.target_layer)
    return result

def run_gradeclip(args, model, image_processed, tid, im_f, tf):
    outputs, last_feat, vs, qs, ks, attns, atten_outs, map_size = src.grad_eclip.clip_encode_dense(image_processed, n=1, clipmodel=model)    
    img_embedding = F.normalize(outputs[:,0], dim=-1)
    tf = F.normalize(tf, dim=-1)
    
    cosine = (img_embedding @ tf.T)[0]
            
    result = src.grad_eclip.grad_eclip(cosine, qs, ks, vs, atten_outs, map_size)
    # clean up everything
    del outputs, last_feat, vs, qs, ks, attns, atten_outs, map_size
    torch.cuda.empty_cache()
    result -= result.min()
    result /= result.max()
    
    h, w = image_processed.shape[-2:]
    resize = T.Resize((w,h))
    result = resize(result.unsqueeze(0))
    result = result.detach().cpu().numpy()
    return result

def run_pibisa(args, model, image_processed, tid, im_f, tf):
    result = pibisa.compute_attribution(query_image=image_processed, target_emb=tf)  
    return result

def run_pibisa_enc(args, model, image_processed, tid, im_f, tf):
    result = pibisa.compute_attribution(query_image=image_processed, target_emb=model.get_image_features(image_processed))  
    return result

# Dictionary for the saliency methods
SALIENCY_METHODS = {    
    "rise":run_rise, 
    "m2ib":run_m2ib, 
    "nib":run_nib, 
    "gradeclip":run_gradeclip,
    "pibisa":run_pibisa,
    "pibisa_enc":run_pibisa_enc
}


if __name__ == "__main__":
    # Imports
    import argparse

    # Command Line Interface
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="MSCocoDataset", choices=["Flickr8kDataset", "GoogleAIConceptualCations", "MSCXRDataset", "MSCocoDataset"])
    argparser.add_argument("--method", type=str, default="cibisa", choices=["chefer", "gradcam", "saliencymap", "fast_ig", "mfaba", "rise", "m2ib", "nib", "cibisa", "cibisa_enc"])
    argparser.add_argument("--beta", type=float, default=2)
    argparser.add_argument("--phi", type=float, default=2)
    argparser.add_argument("--learn", type=bool, default=False)
    argparser.add_argument("--num_steps", type=int, default=10)
    argparser.add_argument("--target_layer", type=int, default=9)
    argparser.add_argument("--batch_size", type=int, default=1, help="Batch size for experiments.")
    argparser.add_argument("--num_workers", type=int, default=0, help="Number of workers.")
    argparser.add_argument("--seed", type=int, default=0, help="Seed for the sake of reproducibility.")
    args = argparser.parse_args()

    # Setup seed
    setup_seed(seed=args.seed)

    # Constants
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_workers = args.num_workers
    loss_fn = torch.nn.CosineSimilarity(eps=1e-6)

    # Get model, processor, and tokenizer
    if args.method == 'gradeclip':
        # Patching monkey for grad-ECLip, since it uses CLIP from OpenAI
        model, _ = clip.load("ViT-B/32", device=device) #type:ignore
        model.get_image_features = model.encode_image   #type:ignore
        model.get_text_features = model.encode_text     #type:ignore                
    else:        
        model: transformers.CLIPModel = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device) # type:ignore
        
    processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    tokenizer = transformers.CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    # Get evaluation method
    method = SALIENCY_METHODS[args.method]    
    if args.method == 'pibisa':        
        pibisa_cfg = src.pibisa.PibisaConfig(beta=args.beta, phi=args.phi, device=device)
        pibisa = src.pibisa.PibisaAtt(encoder_forward=model.get_image_features, config=pibisa_cfg)

    # Create dataset
    if args.dataset == "GoogleAIConceptualCations":
        pass

    elif args.dataset == "Flickr8kDataset":
        dataset = NIB.scripts.data.Flickr8kDataset(
            image_preprocessor=processor, 
            transform=torchvision.transforms.v2.Compose([torchvision.transforms.v2.ToImage()])
        )
    elif args.dataset == "MSCXRDataset":
        dataset = NIB.scripts.data.MSCXRDataset(
            split='test',
            image_preprocessor=processor,
            transform=torchvision.transforms.v2.Compose([torchvision.transforms.v2.ToImage()])
        )
    elif args.dataset == "MSCocoDataset":
        dataset = NIB.scripts.data.MSCocoDataset(
            split='val',
            image_preprocessor=processor,
            transform=torchvision.transforms.v2.Compose([torchvision.transforms.v2.ToImage()])
        )
        
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=dataset.collate_fn, 
        pin_memory=True
    )

    # Initialize lists to store results
    image_fnames = list()
    image_saliency = list()
    text_saliency = list()
    images_processed = list()
    text_features = list()
    image_features = list()
    text_ids = list()

    # Go through the data
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader)):       
        image_fname = batch['image_fname']
        image = batch['image']
        captions = batch['captions']
        image_processed = batch['image_processed']

        # Special Case: Flickr8kDataset has several captions for the same image
        if isinstance(dataset, NIB.scripts.data.Flickr8kDataset) or isinstance(dataset, NIB.scripts.data.MSCocoDataset):
            with torch.no_grad():
                new_caption = []
                for idx, cp in enumerate(captions):
                    tids = [torch.tensor([tokenizer.encode(c, add_special_tokens=True)]).to(device) for c in cp]        
                        
                    tf = [model.get_text_features(t) for t in tids]
                    tf = torch.cat(tf, dim=0) # type:ignore

                    # This is the correct way of doing it: for each set of captions there is a specific image with index = idx
                    img_proc = image_processed[idx].unsqueeze(0)
                    im_features = model.get_image_features(img_proc.to(device))
                    prob = torch.nn.functional.softmax(loss_fn(im_features, tf), -1)
                    new_caption.append(cp[prob.argmax().item()])
                captions = new_caption
        
        tid = [torch.tensor([tokenizer.encode(cp, add_special_tokens=True)]).to(device) for cp in captions]
        tf = [model.get_text_features(t) for t in tid]
        tf = torch.cat(tf, dim=0) # type:ignore
        image_processed = image_processed.to(device)
        im_f = model.get_image_features(image_processed).detach().cpu()        
        
        # compute saliency
        v_saliency = method(args, model, image_processed, tid, im_f, tf)
        
        # image_fnames.append(image_fname)
        image_saliency.append(v_saliency)
        text_features.extend(tf.detach().cpu())
        images_processed.append(image_processed.cpu())
        image_features.append(im_f)
        text_ids.extend(tid)
    
    # image_fnames = np.concatenate(image_fnames, axis=0)
    image_saliency = np.concatenate(image_saliency, axis=0)
    text_features = torch.stack(text_features, dim=0)
    images_processed = torch.cat(images_processed, dim=0)
    image_features = torch.cat(image_features, dim=0)

    # Get results
    res = src.metrics_NIB.metric_evaluation_vision(
        model=model, 
        device=device, 
        images_processed=images_processed, 
        image_features=image_features,
        text_features=text_features, 
        saliency_v=image_saliency, 
    )

    # Get average values for results
    vdrops = [item['vdrop'] for item in res]
    vincrs = [item['vincr'] for item in res]
    
    vdrop = sum(vdrops) / len(vdrops)    
    vincr = sum(vincrs) / len(vincrs)
    
    print("Dataset: ", args.dataset)
    print("Results for beta equals to: " +  str(args.beta) + "Learnable: " + str(args.learn) )
    print("vdrop:", vdrop, "vincr:", vincr)
    
    
    csv_file = os.path.join('./results/vision_language/', args.method + args.beta, 'metrics.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([vdrops, vincrs])       
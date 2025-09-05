# Imports
import os
import random
import numpy as np
from tqdm import tqdm

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, ToImage, Resize, CenterCrop

# Transformers Imports
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast

# Project Imports
from scripts.data import Flickr8kDataset, GoogleAIConceptualCations
# from datasets import ConceptualCaptions, collate_fn_cc, ImagenetDataset, collect_fn_imagenet
from scripts.eval import metric_evaluation
from scripts.methods import vision_heatmap_iba, text_heatmap_iba
from saliency import chefer, gradcam, saliencymap, fast_ig, mfaba, rise, m2ib, nib



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



if __name__ == "__main__":

    # Imports
    import argparse

    # Command Line Interface
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="Flickr8kDataset", choices=["Flickr8kDataset", "GoogleAIConceptualCations"])
    argparser.add_argument("--method", type=str, default="nib", choices=["chefer", "gradcam", "saliencymap", "fast_ig", "mfaba", "rise", "m2ib", "nib"])
    argparser.add_argument("--beta", type=float, default=0.1)
    argparser.add_argument("--num_steps", type=int, default=10)
    argparser.add_argument("--target_layer", type=int, default=9)
    argparser.add_argument("--batch_size", type=int, default=32, help="Batch size for experiments.")
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
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print(processor)
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    # Get evaluation method
    method = eval(args.method)

    # Create dataset
    if args.dataset == "GoogleAIConceptualCations":
        dataset = GoogleAIConceptualCations(split='train')
        dataset = GoogleAIConceptualCations(split='val')
        print("GoogleAIConceptualCations download finished.")
        exit()

    elif args.dataset == "imagenet":
        
        # TODO
        # dataset = ImagenetDataset("datasets/tiny-imagenet-200", image_preprocessor=processor, split='val')
        # collate_fn = collect_fn_imagenet
        pass

    elif args.dataset == "Flickr8kDataset":

        # TODO: Review collate function
        # TODO: Review shape of list of captions
        transform = Compose([
            CenterCrop((224, 224)),
            ToImage(),
        ])
        dataset = Flickr8kDataset(image_preprocessor=processor, transform=transform)
        

    # Create dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)



    # Initialize lists to store results
    image_saliency = list()
    text_saliency = list()
    image_feats = list()
    text_features = list()
    image_features = list()
    text_ids = list()

    # Go through the data
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        x = batch['image']
        caption = batch['captions']
        batch_xs = batch['image_features']
        print(x.shape, np.shape(caption), batch_xs.shape)

        # Special Case: Flickr8kDataset has several captions for the same image
        if isinstance(dataloader.dataset, Flickr8kDataset):
            with torch.no_grad():
                new_caption = []
                for cp in caption:
                    tids = [torch.tensor([tokenizer.encode(c, add_special_tokens=True)]).to(device) for c in cp]
                    tf = [model.get_text_features(t) for t in tids]
                    tf = torch.cat(tf, dim=0)
                    im_features = model.get_image_features(batch_xs.to(device))[0]
                    prob = torch.nn.functional.softmax(loss_fn(im_features, tf), -1)
                    new_caption.append(cp[prob.argmax().item()])
                caption = new_caption


        # TODO: Review some of these dimensions
        tid = [torch.tensor([tokenizer.encode(cp, add_special_tokens=True)]).to(device) for cp in caption]
        tf = [model.get_text_features(t) for t in tid]
        tf = torch.cat(tf, dim=0)
        batch_xs = batch_xs.to(device)
        im_f = model.get_image_features(batch_xs).detach().cpu()

        if args.method in ['chefer', 'gradcam', 'saliencymap', 'fast_ig', 'mfaba']:
            v_saliency, t_saliency = method(model, processor, caption, x)
        elif args.method in ['rise']:
            v_saliency, t_saliency = rise(model, batch_xs, tid, im_f, tf)
        elif args.method in ['m2ib']:
            v_saliency, t_saliency = m2ib(model, tid, batch_xs, args.beta)
        elif args.method in ['nib']:
            v_saliency, t_saliency = nib(
                model, tid, batch_xs, args.num_steps, args.target_layer)
        else:
            v_saliency, t_saliency = method(model, tid, batch_xs)
        
        image_saliency.append(v_saliency)
        text_saliency.extend(t_saliency)
        text_features.extend(tf.detach().cpu())
        image_feats.append(batch_xs.cpu())
        image_features.append(im_f)
        text_ids.extend(tid)


    image_feats = torch.cat(image_feats, dim=0)
    image_features = torch.cat(image_features, dim=0)
    text_features = torch.stack(text_features, dim=0)
    image_saliency = np.concatenate(image_saliency, axis=0)


    # Get results
    res = metric_evaluation(
        model=model, 
        device=device, 
        image_feats=image_feats, 
        image_features=image_features,
        text_ids=text_ids,
        text_features=text_features, 
        saliency_v=image_saliency, 
        saliency_t=text_saliency
    )



    # Get average values for results
    vdrop = sum(k['vdrop'] for k in res) / len(res)
    vincr = sum(k['vincr'] for k in res) / len(res)
    tdrop = sum(k['tdrop'] for k in res) / len(res)
    tincr = sum(k['tincr'] for k in res) / len(res)
    print("vdrop:", vdrop, "vincr:", vincr, "tdrop:", tdrop, "tincr:", tincr)
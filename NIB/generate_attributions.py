# Imports
import os
import random
import numpy as np
from tqdm import tqdm

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, ToImage

# Transformers Imports
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast

# Project Imports
from scripts.data import Flickr8kDataset, MSCXRDataset, MSCocoDataset
from scripts.eval import metric_evaluation, metric_evaluation_vision
from saliency import chefer, gradcam, saliencymap, fast_ig, mfaba, rise, m2ib, nib



# Setup environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# Dictionary for the saliency methods
SALIENCY_METHODS = {
    "chefer":chefer, 
    "gradcam":gradcam, 
    "saliencymap":saliencymap, 
    "fast_ig":fast_ig, 
    "mfaba":mfaba, 
    "rise":rise, 
    "m2ib":m2ib, 
    "nib":nib
}



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
    argparser.add_argument("--dataset", type=str, default="Flickr8kDataset", choices=["Flickr8kDataset", "GoogleAIConceptualCations", "MSCXRDataset", "MSCocoDataset", "ImageNetDataset"])
    argparser.add_argument("--method", type=str, default="nib", choices=["chefer", "gradcam", "saliencymap", "fast_ig", "mfaba", "rise", "m2ib", "nib"])
    argparser.add_argument("--beta", type=float, default=0.1)
    argparser.add_argument("--num_steps", type=int, default=10)
    argparser.add_argument("--target_layer", type=int, default=9)
    argparser.add_argument("--batch_size", type=int, default=32, help="Batch size for experiments.")
    argparser.add_argument("--num_workers", type=int, default=0, help="Number of workers.")
    argparser.add_argument("--seed", type=int, default=0, help="Seed for the sake of reproducibility.")
    argparser.add_argument("--save_saliency_maps", action='store_true', default=False, help="Whether to save the saliency maps or not.")
    argparser.add_argument("--save_path", type=str, default="results/saliency_maps/", help="Path to save the saliency maps.")
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
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    # Get evaluation method
    method = SALIENCY_METHODS[args.method]

    # Create dataset
    if args.dataset == "GoogleAIConceptualCations":
        pass

    elif args.dataset == "ImageNetDataset":
        pass

    elif args.dataset == "Flickr8kDataset":
        dataset = Flickr8kDataset(
            image_preprocessor=processor, 
            transform=Compose([ToImage()])
        )
    
    elif args.dataset == "MSCXRDataset":
        dataset = MSCXRDataset(
            split='test',
            image_preprocessor=processor,
            transform=Compose([ToImage()])
        )
    
    elif args.dataset == "MSCocoDataset":
        dataset = MSCocoDataset(
            split='val',
            image_preprocessor=processor,
            transform=Compose([ToImage()])
        )

        

    # Create dataloader
    dataloader = DataLoader(
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
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        image_fname = batch['image_fname']
        image = batch['image']
        captions = batch['captions']
        image_processed = batch['image_processed']

        # Special Case: Flickr8kDataset has several captions for the same image
        if isinstance(dataset, Flickr8kDataset) or isinstance(dataset, MSCocoDataset):
            with torch.no_grad():
                new_caption = []
                print(len(captions), np.shape(captions))
                for idx, cp in enumerate(captions):
                    print(np.shape(cp))
                    tids = [torch.tensor([tokenizer.encode(c, add_special_tokens=True)]).to(device) for c in cp]
                    print(tids[0].shape, tids[1].shape)
                    tf = [model.get_text_features(t) for t in tids]
                    print(tf[0].shape, tf[1].shape)
                    tf = torch.cat(tf, dim=0)
                    print(tf.shape)

                    # TODO: Remove this part after testing, I believe the authors committed a mistake here
                    # im_features = model.get_image_features(image_processed.to(device))[0]

                    # This is the correct way of doing it: for each set of captions there is a specific image with index = idx
                    img_proc = image_processed[idx].unsqueeze(0)
                    im_features = model.get_image_features(img_proc.to(device))
                    print(im_features.shape, img_proc.shape, image_processed.shape)
                    prob = torch.nn.functional.softmax(loss_fn(im_features, tf), -1)
                    print(prob.shape)
                    new_caption.append(cp[prob.argmax().item()])
                    print(np.shape(new_caption))
                    exit()
                captions = new_caption


        # TODO: Review some of these dimensions
        tid = [torch.tensor([tokenizer.encode(cp, add_special_tokens=True)]).to(device) for cp in captions]
        tf = [model.get_text_features(t) for t in tid]
        tf = torch.cat(tf, dim=0)
        image_processed = image_processed.to(device)
        im_f = model.get_image_features(image_processed).detach().cpu()

        if args.method in ['chefer', 'gradcam', 'saliencymap', 'fast_ig', 'mfaba']:
            v_saliency, t_saliency = method(
                model=model, 
                processor=processor, 
                captions=captions, 
                image_feat=image
            )

        elif args.method in ['rise']:
            v_saliency, t_saliency = rise(
                model=model, 
                image_feat=image_processed, 
                tids=tid, 
                image_features=im_f, 
                text_features=tf
            )

        elif args.method in ['m2ib']:
            v_saliency, t_saliency = m2ib(
                model=model, 
                text_ids=tid, 
                image_feat=image_processed, 
                beta=args.beta
            )

        elif args.method in ['nib']:
            v_saliency, t_saliency = nib(
                model=model, 
                text_ids=tid, 
                image_feats=image_processed, 
                num_steps=args.num_steps, 
                target_layer=args.target_layer
            )

        else:
            v_saliency, t_saliency = method(model, tid, image_processed)
        
        image_fnames.append(image_fname)
        image_saliency.append(v_saliency)
        text_saliency.extend(t_saliency)
        text_features.extend(tf.detach().cpu())
        images_processed.append(image_processed.cpu())
        image_features.append(im_f)
        text_ids.extend(tid)


    # Prepare data
    image_fnames = np.concatenate(image_fnames, axis=0)
    image_saliency = np.concatenate(image_saliency, axis=0)
    text_features = torch.stack(text_features, dim=0)
    images_processed = torch.cat(images_processed, dim=0)
    image_features = torch.cat(image_features, dim=0)
    
    
    # Get results
    res = metric_evaluation_vision(
        model=model, 
        device=device, 
        images_processed=images_processed, 
        image_features=image_features,
        text_ids=text_ids,
        text_features=text_features, 
        saliency_v=image_saliency, 
        saliency_t=text_saliency
    )



    # Get average values for results
    vdrop = sum(k['vdrop'] for k in res) / len(res)
    vincr = sum(k['vincr'] for k in res) / len(res)
    print("vdrop:", vdrop, "vincr:", vincr)

    # tdrop = sum(k['tdrop'] for k in res) / len(res)
    # tincr = sum(k['tincr'] for k in res) / len(res)
    # print("tdrop:", tdrop, "tincr:", tincr)



    # Save saliency maps
    if args.save_saliency_maps:
        save_path = os.path.join(args.save_path, args.dataset, args.method)
        os.makedirs(save_path, exist_ok=True)

        for img_fname, img_smap in zip(image_fnames, image_saliency):
            np.save(
                file=os.path.join(save_path, img_fname.replace('.jpg','').replace('.png','')+'.npy'), 
                arr=img_smap, 
                allow_pickle=True
            )
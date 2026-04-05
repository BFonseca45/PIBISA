# Imports
import csv
import glob
import os
import random
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
# PyTorch Imports
from torch.utils.data import Dataset
from tqdm import tqdm


# Function: Build Flickr8k Dataset using Torchvision
class Flickr8kDataset(Dataset):

    """
    Flickr8kDataset class based on: https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/flickr.html#Flickr8k

    Data from: https://www.kaggle.com/datasets/adityajn105/flickr8k/data

    Official release: https://hockenmaier.cs.illinois.edu/8k-pictures.html
    """


    def __init__(self, root='/nas-ctm01/datasets/public/Flickr8k', image_preprocessor=None, transform=None) -> None:

        assert image_preprocessor is not None, 'Please provide an image_preprocessor function'

        self.root = root
        self.images_dir = os.path.join(root, "Images")
        annotations_txt = os.path.join(root, "captions.txt")
        annotations_df = pd.read_csv(annotations_txt, header=0)
        self.image_ids_dict, self.annotations_dict = self.annotations_to_dict(annotations_df=annotations_df)
        self.image_ids = list(self.image_ids_dict.keys())
        self.image_preprocessor = image_preprocessor
        self.transform = transform

        return


    # Method: Get a dictionary of annotations
    def annotations_to_dict(self, annotations_df):
        image_ids_dict = dict()
        annotations_dict = dict()

        # Get unique images
        images_unique = np.unique(annotations_df['image'].values)
        for image_id, image in enumerate(images_unique):
            image_ids_dict[image_id] = image
            annotations_dict[image] = list()

        # Populate dictionary
        for _, row in annotations_df.iterrows():
            image = row['image']
            caption = row['caption']
            annotations_dict[image].append(caption)

        return image_ids_dict, annotations_dict


    # Method: __getitem__
    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): Index

        Returns:
            dict: Dict(image, target). target is a list of captions for the image.
        """

        # Get image ID
        image_id = self.image_ids[idx]
        image_fname = self.image_ids_dict[image_id]

        # Image
        image = Image.open(os.path.join(self.images_dir, image_fname)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        image_features = self.image_preprocessor(images=image, return_tensors="pt")['pixel_values']

        # Captions
        captions = self.annotations_dict[image_fname]

        # Output dict
        output = {
            'image_id':image_id,
            'image_fname':image_fname,
            'image':image,
            "image_features":image_features,
            'captions':captions
        }

        return output


    # Method: __len__
    def __len__(self) -> int:
        return len(self.image_ids)



# Class: Google AI Conceptual Captions Dataset
class GoogleAIConceptualCations(Dataset):

    """

    Website: https://ai.google.com/research/ConceptualCaptions/download
    GitHub: https://github.com/google-research-datasets/conceptual-captions

    """

    def __init__(self, root="/nas-ctm01/datasets/public/GoogleAI-ConceptualCaptions", split=None, transform=None):

        assert split in ('train', 'val')

        self.root = root

        # Open Image Labels TSV
        image_labels_cols_names = [
              'image_caption',
              'image_url',
              'image_labels',
              'image_mids',
              'image_cscores'
        ]
        image_train_labels = pd.read_csv(os.path.join(root, "Image_Labels_Subset_Train_GCC-Labels-training.tsv"), sep='\t', header=None, names=image_labels_cols_names)
        if not os.path.exists(os.path.join(root, 'image_train_labels_gcc_complete.tsv')):
            self.download_image_labels_data(image_labels_df=image_train_labels)
        train_labels_images = pd.read_csv(os.path.join(root, 'image_train_labels_gcc_complete.tsv'), sep='\t')


        # Open Train/Validation TSV
        captions_cols_names = ['image_caption', 'image_url']
        captions_train = pd.read_csv(os.path.join(root, "Train_GCC-training.tsv"), sep='\t', header=None, names=captions_cols_names)
        captions_val = pd.read_csv(os.path.join(root, "Validation_GCC-1.1.0-Validation.tsv"), sep='\t', header=None, names=captions_cols_names)

        if split == 'train':
            if not os.path.exists(os.path.join(root, 'train_gcc_complete.tsv')):
                self.download_captions_data(captions_df=captions_train, split='train')
            self.captions_images = pd.read_csv(os.path.join(root, 'train_gcc_complete.tsv'), sep='\t')
        else:
            if not os.path.exists(os.path.join(root, 'val_gcc_complete.tsv')):
                self.download_captions_data(captions_df=captions_val, split='val')
            self.captions_images = pd.read_csv(os.path.join(root, 'val_gcc_complete.tsv'), sep='\t')
        print(self.captions_images.head())

        self.transform = transform

        return


    # Method: Download image labels data
    def download_image_labels_data(self, image_labels_df):

        os.makedirs(os.path.join(self.root, 'images', 'labels_subset', 'train'), exist_ok=True)

        # Convert DataDrame to dictionary
        labels_dict = image_labels_df.to_dict(orient='list')
        image_fname = ['' for i in range(len(image_labels_df))]
        for idx in tqdm(range(len(image_labels_df))):
            caption = labels_dict['image_caption'][idx]
            url = labels_dict['image_url'][idx]
            labels = labels_dict['image_labels'][idx]
            mids = labels_dict['image_mids'][idx]
            cscores = labels_dict['image_cscores'][idx]
            try:
                # Get image data
                urllib.request.urlretrieve(
                    url=url,
                    filename=os.path.join(self.root, 'images', 'labels_subset', 'train', f'image_idx{idx}.jpg')
                )
                Image.open(os.path.join(self.root, 'images', 'labels_subset', 'train', f'image_idx{idx}.jpg')).convert('RGB')
                image_fname[idx] = os.path.join(self.root, 'images', 'labels_subset', 'train', f'image_idx{idx}.jpg')
            except:
                image_fname[idx] = None

        # Save this into a new TSV file
        labels_dict['image_fname'] = image_fname

        # Convert dictionary to DataFrame
        labels_dict_new = pd.DataFrame.from_dict(labels_dict)
        labels_dict_new.to_csv(
            path_or_buf=os.path.join(self.root, "image_train_labels_gcc_complete.tsv"),
            sep='\t',
            index=False
        )

        return


    # Method: Download image captions data
    def download_captions_data(self, captions_df, split):

        assert split in ('train', 'val')

        os.makedirs(os.path.join(self.root, 'images', 'captions_subset', split), exist_ok=True)

        # Convert DataFrame to dictionary
        captions_dict = captions_df.to_dict(orient='list')
        image_fname = ['' for i in range(len(captions_df))]
        for idx in tqdm(range(len(captions_df))):
            caption = captions_dict['image_caption'][idx]
            url = captions_dict['image_url'][idx]
            try:
                # Get image data
                urllib.request.urlretrieve(
                    url=url,
                    filename=os.path.join(self.root, 'images', 'captions_subset', split, f'image_idx{idx}.jpg')
                )
                Image.open(os.path.join(self.root, 'images', 'captions_subset', split, f'image_idx{idx}.jpg')).convert('RGB')
                image_fname[idx] = os.path.join(self.root, 'images', 'captions_subset', split, f'image_idx{idx}.jpg')
            except:
                image_fname[idx] = None

        # Create final dictionary
        captions_dict['image_fname'] = image_fname

        # Convert dictionary to DataFrame
        captions_df_new = pd.DataFrame.from_dict(captions_dict)
        captions_df_new.to_csv(
            path_or_buf=os.path.join(self.root, f"{split}_gcc_complete.tsv"),
            sep='\t',
            index=False
        )

        return


    # Method: __len__
    def __len__(self):
        return len(self.captions_images)


    # Method: __getitem__
    def __getitem__(self, idx):
        image_caption = self.captions_images['image_caption'].iloc[idx]
        image_url = self.captions_images['image_url'].iloc[idx]
        image_fname = self.captions_images['image_fname'].iloc[idx]


        # Open image
        image = Image.open(image_fname).convert('RGB')

        data_dict = {
            'image_caption':image_caption,
            'image_url':image_url,
            'image_fname':image_fname,
            'image':image
        }

        return data_dict

class OxfordBuildings(Dataset):
    def __init__(self, path, transform = None):
        self.transform = transform
        self.ann_path = path + '/annotations/'
        self.img_path = path + '/images/'
        self.query_files = glob.glob('*query*', root_dir=self.ann_path)

    def __len__(self):
        return len(self.query_files)

    def __getitem__(self, idx:int):
        # Getting the query image
        query_file = self.query_files[idx]
        with open(self.ann_path + query_file, 'r') as file:
            contentq = file.read()
        filename = contentq.split(' ')[0]
        query_img = Image.open(self.img_path + filename.removeprefix('oxc1_') + '.jpg')
        if self.transform:
            query_img = self.transform(query_img)

        # Getting all the matched images
        gt_list = []
        with open(self.ann_path + query_file.replace('query', 'good'), 'r') as file:
            contentgt = file.read()

        for gt_file in contentgt.split('\n')[:-1]:
            gt_img = Image.open(self.img_path + gt_file + '.jpg')
            if self.transform:
                gt_img = self.transform(gt_img)
            gt_list.append(gt_img)

        return query_img, gt_list

class ParisBuildings(OxfordBuildings):
    def __init__(self, path, transform=None):
        super().__init__(path, transform)

    def __getitem__(self, idx:int):
        # Getting the query image
        query_file = self.query_files[idx]
        with open(self.ann_path + query_file, 'r') as file:
            contentq = file.read()
        filename = contentq.split(' ')[0]
        query_img = Image.open(self.img_path + filename + '.jpg')
        if self.transform:
            query_img = self.transform(query_img)

        # Getting all the matched images
        gt_list = []
        with open(self.ann_path + query_file.replace('query', 'ok'), 'r') as file:
            contentgt = file.read()

        for gt_file in contentgt.split('\n')[:-1]:
            gt_img = Image.open(self.img_path + gt_file + '.jpg')
            if self.transform:
                gt_img = self.transform(gt_img)
            gt_list.append(gt_img)

        return query_img, gt_list

@dataclass
class ImageSample:
    index: int
    subset: str
    image_path: Path
    bboxes: List[Tuple[int, int, int, int]]

class InstreDataset(Dataset):
    def __init__(self, csv_path: str, base_dir: str, max_boxes: int, transform=None, crop_mode: str = "bbox1", seed: int = 0, k: int = 6):
        """
        Args:
            samples: list of ImageSample objects
            transform: optional torchvision transform
            crop_mode: "none", "bbox1", "union"
        """
        self.samples = self._load_samples(csv_path, base_dir, max_boxes)
        # Organize by subset for fast retrieval
        self.by_subset = defaultdict(list)
        for idx, s in enumerate(self.samples):
            self.by_subset[s.subset].append(idx)

        self.transform = transform
        self.crop_mode = crop_mode
        self.rng = random.Random(seed)
        self.k = k

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        rimg_list = []
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        img = self._crop_image(img, s.bboxes)

        if self.transform:
            img = self.transform(img)

        ridx = self.sample_same_subset(anchor_idx=idx, k=self.k)
        
        for i in ridx:
            sr = self.samples[i]
            rimg = Image.open(sr.image_path).convert("RGB")
            # rimg = self._crop_image(rimg, sr.bboxes)
            if self.transform:
                rimg = self.transform(rimg)
            rimg_list.append(rimg)


        return img, rimg_list

    def _load_samples(self, csv_path: str, base_dir: str, max_boxes: int) -> List[ImageSample]:
        base_dir = Path(base_dir)
        samples = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        for idx, row in enumerate(rows):
            bboxes = []
            for i in range(1, max_boxes + 1):
                try:
                    x = int(row[f"bbox{i}_x"])
                    y = int(row[f"bbox{i}_y"])
                    w = int(row[f"bbox{i}_w"])
                    h = int(row[f"bbox{i}_h"])
                    if w > 0 and h > 0:
                        bboxes.append((x, y, w, h))
                except Exception:
                    continue

            samples.append(
                ImageSample(
                    index=idx,
                    subset=row["subset"],
                    image_path=base_dir / row["image_path"],
                    bboxes=bboxes,
                )
            )
        return samples

    def sample_same_subset(self, anchor_idx: int, k: int, include_anchor: bool = True) -> List[int]:
        subset = self.samples[anchor_idx].subset
        pool = self.by_subset.get(subset, [])
        if not include_anchor:
            pool = [i for i in pool if i != anchor_idx]
        if not pool:
            return []
        return pool if k >= len(pool) else self.rng.sample(pool, k)

    def _crop_image(self, img: Image.Image, bboxes: List[Tuple[int, int, int, int]]) -> Image.Image:
        if not bboxes or self.crop_mode == "none":
            return img
        if self.crop_mode == "bbox1":
            x, y, w, h = bboxes[0]
            return img.crop((x, y, x + w, y + h))
        if self.crop_mode == "union":
            x1 = min(b[0] for b in bboxes)
            y1 = min(b[1] for b in bboxes)
            x2 = max(b[0] + b[2] for b in bboxes)
            y2 = max(b[1] + b[3] for b in bboxes)
            return img.crop((x1, y1, x2, y2))
        return img

class ImageNETs(Dataset):
    def __init__(self, path: str, split: str, transform):
        self.files = glob.glob(os.path.join(path, split, "**", "*.JPEG"))
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        img = Image.open(self.files[key]).convert("RGB") 
        msk_path = self.files[key].replace(self.split, self.split + "-segmentation").replace("JPEG", "png")
        msk = Image.open(msk_path)

        if self.transform:
            img = self.transform(img)
            _,w,h = img.shape
            msk = self._preprocess_mask(mask=msk, width=w, height=h)

        return img, msk

    def _preprocess_mask(self, mask, width, height):
        mask = torchvision.transforms.functional.resize(mask, (width,height))
        mask = torchvision.transforms.functional.to_tensor(mask)
        mask = torch.sum(torch.squeeze(mask), axis=0)
        mask = (mask > 0)*1
        return mask
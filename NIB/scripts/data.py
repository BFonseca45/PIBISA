# Imports
import os
import numpy as np
import pandas as pd
from PIL import Image

# PyTorch Imports
import torch
from torch.utils.data import Dataset

# COCO Imports
from scripts.coco import COCO



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

        # Image processed (remove batch size)
        image_processed = self.image_preprocessor(images=image, return_tensors="pt")['pixel_values'][0]

        # Captions
        captions = self.annotations_dict[image_fname]

        return image_id, image_fname, image, image_processed, captions
    

    # Method: Collate function
    def collate_fn(self, batch):

        output = {
            "image_id":[i[0] for i in batch],
            "image_fname":[i[1] for i in batch],
            "image":[i[2] for i in batch],
            "image_processed":torch.stack([i[3] for i in batch], dim=0),
            "captions":[i[4] for i in batch]
        }
        
        return output


    # Method: __len__
    def __len__(self) -> int:
        return len(self.image_ids)



# Class: Google AI Conceptual Captions Dataset
class GoogleAIConceptualCaptions(Dataset):
    
    """
    
    Website: https://ai.google.com/research/ConceptualCaptions/download
    GitHub: https://github.com/google-research-datasets/conceptual-captions

    """

    def __init__(self, root="/nas-ctm01/datasets/public/GoogleAI-ConceptualCaptions", split=None, transform=None):

        assert split in ('train', 'val')

        self.root = root

        # Open Image Labels TSV for sanity check purposes
        image_labels_cols_names = [
              'image_caption',
              'image_url',
              'image_labels',
              'image_mids',
              'image_cscores'
        ]
        image_train_labels = pd.read_csv(os.path.join(root, "Image_Labels_Subset_Train_GCC-Labels-training.tsv"), sep='\t', header=None, names=image_labels_cols_names)
        assert os.path.exists(os.path.join(root, 'image_train_labels_gcc_complete.tsv')), "Please download captions data!"
        train_labels_images = pd.read_csv(os.path.join(root, 'image_train_labels_gcc_complete.tsv'), sep='\t')

        
        # Open Train/Validation TSV for sanity check purposes
        captions_cols_names = ['image_caption', 'image_url']
        captions_train = pd.read_csv(os.path.join(root, "Train_GCC-training.tsv"), sep='\t', header=None, names=captions_cols_names)
        captions_val = pd.read_csv(os.path.join(root, "Validation_GCC-1.1.0-Validation.tsv"), sep='\t', header=None, names=captions_cols_names)
        
        if split == 'train':
            assert os.path.exists(os.path.join(root, 'train_gcc_complete.tsv')), "Please download captions data!"
            self.captions_images = pd.read_csv(os.path.join(root, 'train_gcc_complete.tsv'), sep='\t')
        else:
            assert not os.path.exists(os.path.join(root, 'val_gcc_complete.tsv')),  "Please download captions data!"
            self.captions_images = pd.read_csv(os.path.join(root, 'val_gcc_complete.tsv'), sep='\t')
        print(self.captions_images.head())

        self.transform = transform

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



# Class: MS-CXR
class MSCXRDataset(Dataset):

    """
    Website: https://physionet.org/content/ms-cxr/1.1.0
    """


    # Method: __init__
    def __init__(self, 
                 root='/nas-ctm01/datasets/public/MEDICAL/mscxr-physionet-dataset',
                 mimicxr_root='/nas-ctm01/datasets/public/MEDICAL/MIMIC-CXR', 
                 split='train',
                 image_preprocessor=None, 
                 transform=None):

        assert image_preprocessor is not None, 'Please provide an image_preprocessor function'
        assert split in ('train', 'val', 'test')

        # Open CSV with MS-CXR annotations
        annotations_df = pd.read_csv(os.path.join(root, 'physionet.org/files/ms-cxr/1.1.0/MS_CXR_Local_Alignment_v1.1.0.csv'))

        # Filter annotations by split
        annotations_df_split = annotations_df[annotations_df["split"]==split]
        self.dicom_id = annotations_df_split['dicom_id'].values
        self.category_name = annotations_df_split['category_name'].values	
        self.label_text = annotations_df_split['label_text'].values	
        self.path = annotations_df_split['path'].values	
        self.x = annotations_df_split['x'].values
        self.y = annotations_df_split['y'].values
        self.w = annotations_df_split['w'].values
        self.h = annotations_df_split['h'].values
        self.image_width = annotations_df_split['image_width'].values
        self.image_height = annotations_df_split['image_height'].values

        self.annotations_df_split = annotations_df_split.copy()
        self.mimicxr_root = mimicxr_root
        self.image_preprocessor = image_preprocessor
        self.transform = transform

        return
    

    # Method: __len__
    def __len__(self):
        return len(self.annotations_df_split)
    

    # Method: __getitem__
    def __getitem__(self, idx:int):

        """
        Args:
            idx (int): Index

        Returns:
            dict: Dict(image, target). target is a list of captions for the image.
        """

        # Get image path
        image_fname = self.path[idx]

        # Image
        image = Image.open(os.path.join(self.mimicxr_root, image_fname)).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Image processed (remove batch size)
        image_processed = self.image_preprocessor(images=image, return_tensors="pt")['pixel_values'][0]

        # Captions
        captions = self.label_text[idx]

        return image_fname, image, image_processed, captions


    # Method: Collate function
    def collate_fn(self, batch):

        output = {
            "image_fname":[i[0] for i in batch],
            "image":[i[1] for i in batch],
            "image_processed":torch.stack([i[2] for i in batch], dim=0),
            "captions":[i[3] for i in batch]
        }
        
        return output



# Class: MSCocoDataset
class MSCocoDataset(Dataset):

    """
    Website: https://cocodataset.org/#home
    """


    # Method: __init__
    def __init__(self, 
                 root='/nas-ctm01/datasets/public/coco-dataset',
                 year=2017, 
                 split='train',
                 image_preprocessor=None, 
                 transform=None):

        assert image_preprocessor is not None, 'Please provide an image_preprocessor function'
        assert year in (2014, 2015, 2017), 'Please provide a valid year (2014, 2015, 2017)'
        assert split in ('train', 'val', 'test'), 'Please provide a valid split (train, val, test)'
        
        # Get images path
        image_folder = os.path.join(root, str(year), f"{split}{str(year)}")

        # Open CSV with MS-Coco annotations
        coco_ = COCO(
            annotation_file=os.path.join(root, str(year), "annotations", f"captions_{split}{year}.json")
        )
        

        # Images and Annotations
        self.image_to_annotations = coco_.imgToAnns
        self.categories_to_images = coco_.catToImgs
        self.images = coco_.imgs
        self.annotations = coco_.anns
        self.categories = coco_.cats
        self.image_folder = image_folder
        self.image_preprocessor = image_preprocessor
        self.transform = transform

        return
    

    # Method: __len__
    def __len__(self):
        return len(self.images)
    

    # Method: __getitem__
    def __getitem__(self, idx:int):

        """
        Args:
            idx (int): Index

        Returns:
            dict: Dict(image, target). target is a list of captions for the image.
        """

        # Get image path
        # image_fname = self.path[idx]

        # Image
        # image = Image.open(os.path.join(self.mimicxr_root, image_fname)).convert('RGB')
        # if self.transform:
        #     image = self.transform(image)

        # Image processed (remove batch size)
        # image_processed = self.image_preprocessor(images=image, return_tensors="pt")['pixel_values'][0]

        # Captions
        # captions = self.label_text[idx]

        # return image_fname, image, image_processed, captions
        pass


    # Method: Collate function
    # def collate_fn(self, batch):

    #     output = {
    #         "image_fname":[i[0] for i in batch],
    #         "image":[i[1] for i in batch],
    #         "image_processed":torch.stack([i[2] for i in batch], dim=0),
    #         "captions":[i[3] for i in batch]
    #     }
        
    #     return output
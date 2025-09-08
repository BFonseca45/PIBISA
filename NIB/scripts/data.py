# Imports
import os
import numpy as np
import pandas as pd
import urllib.request
from PIL import Image
from tqdm import tqdm

# PyTorch Imports
from torch.utils.data import Dataset



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

        # Image features (remove batch size)
        image_features = self.image_preprocessor(images=image, return_tensors="pt")['pixel_values'][0]

        # Captions
        captions = self.annotations_dict[image_fname]

        return image_id, image_fname, image, image_features, captions
    

    # Method: Collate function
    def collate_fn(self, batch):

        output = {
            "image_id":[i[0] for i in batch],
            "image_fname":[i[1] for i in batch],
            "image":[i[2] for i in batch],
            "image_features":[i[3] for i in batch],
            "captions":[i[4] for i in batch]
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
        # image_labels_cols_names = [
        #       'image_caption',
        #       'image_url',
        #       'image_labels',
        #       'image_mids',
        #       'image_cscores'
        # ]
        # image_train_labels = pd.read_csv(os.path.join(root, "Image_Labels_Subset_Train_GCC-Labels-training.tsv"), sep='\t', header=None, names=image_labels_cols_names)
        # if not os.path.exists(os.path.join(root, 'image_train_labels_gcc_complete.tsv')):
        #     self.download_image_labels_data(image_labels_df=image_train_labels)
        # train_labels_images = pd.read_csv(os.path.join(root, 'image_train_labels_gcc_complete.tsv'), sep='\t')

        
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
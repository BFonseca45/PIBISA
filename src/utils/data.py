# Imports
import os
import pandas as pd
import urllib.request
from PIL import Image

# PyTorch Imports
# from torchvision.datasets import Flickr8k
from torch.utils.data import Dataset



# Function: Build Flickr8k Dataset using Torchvision
# def build_flickr8k(
#         root='/nas-ctm01/datasets/public/Flickr8k/Images',
#         ann_file='/nas-ctm01/datasets/public/Flickr8k/captions.txt',
#         transform=None,
#         target_transform=None,
# ):
#     """
    
#     Build and return the Flickr8k dataset using torchvision.
#     See https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Flickr8k.html for more details.

#     Args:
#         root (str): Root directory of the dataset where images are stored.
#         ann_file (str): Path to the annotation file containing image captions.
#         transform (callable, optional): A function/transform that takes in an image
#             and returns a transformed version. E.g, `transforms.RandomCrop`
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it. E.g, `transforms.Lambda`
#     Returns:
#         dataset (Flickr8k): The constructed Flickr8k dataset.

#     """
    
#     dataset = Flickr8k(
#         root=root,
#         ann_file=ann_file,
#         transform=transform,
#         target_transform=target_transform
#     )

#     return dataset



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
        for idx in range(len(image_labels_df)):
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
        for idx in range(len(captions_df)):
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
    


if __name__ == "__main__":

    gcc = GoogleAIConceptualCations(split='train')
    gcc = GoogleAIConceptualCations(split='val')
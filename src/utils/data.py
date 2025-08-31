# Imports
import os
import pandas as pd
import urllib.request
from PIL import Image

# PyTorch Imports
from torchvision.datasets import Flickr8k
from torch.utils.data import Dataset



# Function: Build Flickr8k Dataset using Torchvision
def build_flickr8k(
        root='/nas-ctm01/datasets/public/Flickr8k/Images',
        ann_file='/nas-ctm01/datasets/public/Flickr8k/captions.txt',
        transform=None,
        target_transform=None,
):
    """
    
    Build and return the Flickr8k dataset using torchvision.
    See https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Flickr8k.html for more details.

    Args:
        root (str): Root directory of the dataset where images are stored.
        ann_file (str): Path to the annotation file containing image captions.
        transform (callable, optional): A function/transform that takes in an image
            and returns a transformed version. E.g, `transforms.RandomCrop`
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it. E.g, `transforms.Lambda`
    Returns:
        dataset (Flickr8k): The constructed Flickr8k dataset.

    """
    
    dataset = Flickr8k(
        root=root,
        ann_file=ann_file,
        transform=transform,
        target_transform=target_transform
    )

    return dataset



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
        
        # Open Train/Validation TSV
        captions_cols_names = ['image_caption', 'image_url']
        captions_train = pd.read_csv(os.path.join(root, "Train_GCC-training.tsv"), sep='\t', header=None, names=captions_cols_names)
        captions_val = pd.read_csv(os.path.join(root, "Validation_GCC-1.1.0-Validation.tsv"), sep='\t', header=None, names=captions_cols_names)
        
        if split == 'train':
            self.captions_images = captions_train
        else:
            self.captions_images = captions_val

        self.transform = transform

        return


    def __len__(self):
        return len(self.captions_images)


    def __getitem__(self, idx):
        image_caption = self.captions_images['image_caption'].iloc[idx]
        image_url = self.captions_images['image_url'].iloc[idx]
        image_fname = f"image_{idx}.jpg"

        # Get image data
        urllib.request.urlretrieve(image_url, image_fname)

        # Open image
        image = Image.open(image_fname).convert('RGB')

        data_dict = {
            'image_caption':image_caption,
            'image_url':image_url,
            'image':image
        }

        return data_dict
    


if __name__ == "__main__":

    gcc = GoogleAIConceptualCations(
        root="/Users/tiagofilipesousagoncalves/Downloads/GoogleAI-ConceptualCaptions",
        split='val'
    )

    for idx in range(len(gcc)):
        i = gcc.__getitem__(idx)
        print(i)
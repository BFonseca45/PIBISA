import torch
import torchvision
import matplotlib.pyplot as plt
import warnings
import clip
import evaluate
warnings.filterwarnings('ignore')

from PIL import Image
from utils import *
from src.metrics import *
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
from NIB.scripts.plot import visualize_vandt_heatmap
from NIB.saliency import chefer,fast_ig,gradcam,m2ib,mfaba,nib,rise,saliencymap

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

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

torch.manual_seed(0)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# dset = torchvision.datasets.CocoCaptions(root='/home/bfcoelho/Downloads/MSCoco2014/val2014/', annFile='/home/bfcoelho/Downloads/MSCoco2014/annotations/captions_val2014.json', transform=transform)
dset = torchvision.datasets.CocoCaptions(root='/home/bfcoelho/Downloads/coco/2017/val2017/', annFile='/home/bfcoelho/Downloads/coco/2017/annotations/captions_val2017.json')
dloader = torch.utils.data.DataLoader(dataset=dset, batch_size=1, shuffle=False)

processor_blip = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip = AutoModelForVision2Seq.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# model, preprocess = clip.load("ViT-B/16", device=device)
# model.eval()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

pil_img1, cap = dset[999]

x1 = processor(images=pil_img1, return_tensors="pt")['pixel_values'].to(device)

inputs = processor_blip(pil_img1, return_tensors="pt").to("cuda")
out = blip.generate(**inputs)
blip_input = processor_blip.decode(out[0], skip_special_tokens=True)

item = [cap[0]]
item.append(blip_input)

p = []
n = []
m = []

text = torch.tensor([tokenizer.encode(item, add_special_tokens=True)]).to(device)
y_target = model.get_text_features(text).to(device)

# p.append(compute_saliency(x=x1, x_target=y_target, model=model, n_masks=20, epochs=10, beta=2, return_negative=False, learnable_param=False))
# n.append(generate_plot(pil_img1,item,rise))
# m.append(generate_plot(pil_img1,item,m2ib))

# targets.append(y_target)

compute_saliency_text(x=text, x_target=x1, model=model, n_masks=20, epochs=10, beta=2, return_negative=False, learnable_param=False)

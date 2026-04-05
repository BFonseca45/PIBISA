import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from pibisa import PibisaConfig, PibisaAtt
from PIL import Image
from models import ResNET_emb
from utils import unnormalize
from metrics import DeletionMetrics, RetrievalMetrics

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                            torchvision.transforms.ToTensor()])

pil_img1 = Image.open('../test_images/Black1.JPEG').convert('RGB')
pil_img2 = Image.open('../test_images/Black2.JPEG').convert('RGB')

img1 = transform(pil_img1)
img2 = transform(pil_img2)

x1 = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img1)
x2 = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img2)

x1 = x1[None,...].to(device)
x2 = x2[None,...].to(device)

model = ResNET_emb()

config = PibisaConfig(number_masks=3, epochs=15)
pibisa = PibisaAtt(model.forward, config)

targetEmb = model.forward(x2)
saliencyMap = pibisa.compute_attribution(query_image=x1, target_emb=targetEmb)

retMetrics = RetrievalMetrics(encoder_forward=model.forward)
ci,cd = retMetrics.confidence_increase_drop(input_query=x1, input_targets=[x2], saliency_list=[saliencyMap])
print("Confidence Increase: ", ci, "Confidence Drop: ", cd)

# delMetrics = DeletionMetrics(encoderForward=model.forward, targetTask='similarity', numberSteps=9)
# delMetrics.compute_score(imageInput=x1, target=targetEmb, saliencyMap=saliencyMap)

plt.imshow(unnormalize(x1))
plt.imshow(saliencyMap.squeeze(), cmap='turbo', alpha=0.5)
plt.axis('off')
plt.show()
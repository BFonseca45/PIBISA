import numpy as np
import torch
import evaluate

from transformers import AutoProcessor
from tqdm import tqdm
from src.CIBISA_utils import unnormalize

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
np.random.seed(10)

class EvalMetric():
    def __init__(self, model, mode='logit') -> None:
        assert mode in ['logit', 'cosine','nle'], "Please, enter a valid mode, options ['logit', 'cosine']"

        self.model = model
        self.mode = mode
        if self.mode == 'nle':
            print("Using BLIP preprocessor")
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.nlp_metrics = evaluate.combine(["bleu", "meteor", "rouge"])

    def get_tiles(self, saliency, STEP=8):
        self.STEP = STEP
        importance = []
        coords = []

        for i in range(0, len(saliency[0,:]), self.STEP):
            for n in range(0, len(saliency[:,0]), self.STEP):
                aux = np.sum(saliency[i:i+self.STEP, n:n+self.STEP])
                importance.append(aux)
                coords.append([i, n])

        return coords, importance

    def sort_coordenates(self, importance, coords, reverse = True):
        importance_sorted = np.unique(importance)
        importance_sorted = sorted(importance_sorted, reverse=reverse) # Change this line to remove the least important pixels first

        coords_sorted = []
        for t in range(len(importance_sorted)):
            ind = np.where(importance == importance_sorted[t])
            for p in range(len(ind[0])):
                coords_sorted.append(coords[ind[0][p]])

        return coords_sorted

    def compute_sim(self, image, y):
        if self.mode == 'logit':
            preds = self.model(image)
            preds = torch.nn.functional.softmax(preds, dim=1)
            return preds[0,y].detach().cpu().numpy()

        elif self.mode == 'cosine':
            preds = self.model.encode_image(image)
            return torch.nn.functional.cosine_similarity(y, preds).detach().cpu().numpy()

        elif self.mode == 'nle':
            image = unnormalize(image)
            inputs = self.processor(image, return_tensors="pt").to("cuda")
            preds = self.model.generate(**inputs)
            preds = self.processor.decode(preds[0], skip_special_tokens=True)

            m = self.nlp_metrics.compute(predictions=[preds], references=[y])

            return [m['meteor'], m['bleu'], m['rouge1'], m['rouge2'], m['rougeL']]

        else:
            raise NameError("Insert a valid mode, options ['logit', 'cosine', 'nle]")

    def compute_score(self, img, y_target, saliency, metric):
        coords, importance = self.get_tiles(saliency=saliency)
        score = []

        similarity = self.compute_sim(image=img, y=y_target)
        score.append(similarity)

        if metric == 'MoRF':
            mask = torch.ones_like(img, dtype=torch.float32)
            ocl = 0
            coords_sorted = self.sort_coordenates(coords=coords, importance=importance, reverse=True)

        elif metric == 'LeRF':
            mask = torch.ones_like(img, dtype=torch.float32)
            ocl = 0
            coords_sorted = self.sort_coordenates(coords=coords, importance=importance, reverse=False)

        for i in range(len(coords_sorted) - 1):
            idx = coords_sorted[i][0]
            idy = coords_sorted[i][1]
            mask[:,:,idx:idx+self.STEP, idy:idy+self.STEP] = ocl

            # masking the input image
            img_in = img * mask
            img_in = img_in.to(device)

            similarity = self.compute_sim(image=img_in, y=y_target)
            score.append(similarity)

        score = np.array(score, dtype=np.float32)
        score = (score - score.min(axis=0))/(score.max(axis=0) - score.min(axis=0))

        x = np.linspace(start=0, stop=1, num=len(score))
        area = np.trapz(np.squeeze(score),x, axis=0)

        return score, area

    # def gauss_deg(self, saliency, scale_shape):
    #     imgg = self.img.detach().cpu().numpy()

    #     similarity0 = self.compute_sim(image=self.img)

    #     noise = np.random.randn(1,3,scale_shape,scale_shape)*np.std(imgg) + np.mean(imgg)

    #     saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    #     saliency_norm = np.stack((saliency_norm, saliency_norm, saliency_norm))
    #     saliency_norm = saliency_norm[None,...]

    #     img_in = saliency_norm*imgg + (1 - saliency_norm)*noise
    #     img_in = torch.Tensor(img_in)
    #     img_in = img_in.to(device)

    #     similarity = self.compute_sim(image=img_in)
    #     ratio = (similarity + 1e-6)/(similarity0 + 4e-6)

    #     return img_in, np.clip(ratio, 0, 1)

    def pointing_game(self, box_coord, saliency):
        ind = np.unravel_index(np.argmax(saliency), saliency.shape)
        if (ind[1] > box_coord[0] and ind[1] < box_coord[2]) and (ind[0] > box_coord[1] and ind[0] < box_coord[3]):
            return 1, ind
        else:
            return 0, ind

def kl_div_uni(x1, x2):
    mu_x1 = torch.mean(x1)
    mu_x2 = torch.mean(x2)

    std_x1 =torch.std(x1)
    std_x2 = torch.std(x2)

    dkl = torch.log(std_x2/std_x1) + (std_x1**2 + (mu_x1 - mu_x2)**2)/(2*std_x2**2) - 0.5

    return dkl

def create_sal_masks(saliency_list):
    sl = np.array(saliency_list)
    sl_mask = sl.copy()
    threshold = np.percentile(sl.reshape(len(sl),-1),50, axis=-1)
    
    thresh_matrix = np.zeros_like(sl)    
    for j,th in enumerate(threshold):
        thresh_matrix[j,:,:] = th
    
    sl_mask[sl < thresh_matrix] = 0
    sl_mask[sl > thresh_matrix] = 1
    
    sl_mask = torch.tensor(sl_mask, device=device)

    return sl_mask.view((-1,1,224,224))

def confidence_increase_drop(model, query, gts, saliency):
    gts = torch.squeeze(torch.stack(gts))
    sl_mask = create_sal_masks(saliency_list=saliency)    
    
    query_emb = model(query)
    gts_emb = model(gts)
    maks_emb = model(gts * sl_mask)

    sim_ori = torch.nn.functional.cosine_similarity(query_emb, gts_emb)
    sim_sal = torch.nn.functional.cosine_similarity(query_emb, maks_emb)

    ci = torch.nn.functional.relu(sim_sal - sim_ori).bool()*1
    cd = torch.nn.functional.relu(sim_ori - sim_sal)
    cd = cd[cd.nonzero()]
    
    return ci, cd
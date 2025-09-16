# Imports
import os
import numpy as np
import copy
from typing import List, Callable

# PyTorch and PyTorch Grad-Cam
import torch
from pytorch_grad_cam.metrics.cam_mult_image import DropInConfidence, IncreaseInConfidence
from pytorch_grad_cam.metrics.perturbation_confidence import PerturbationConfidenceMetric

# Project Imports
from scripts.utils import ImageFeatureExtractor, TextFeatureExtractor, CosSimilarity

# Environment Variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Function: Multiply an input tensor (after normalization) with a pixel attribution map
def multiply_tensor_with_cam(input_tensor: torch.Tensor, cam: torch.Tensor):
    """ 
    Multiply an input tensor (after normalization) with a pixel attribution map
    """
    return input_tensor * cam



# Class: Perturbation confidence
class PerturbationConfidenceMetric:

    # Method: __init__
    def __init__(self, perturbation):
        self.perturbation = perturbation

        return


    # Method: __call__
    def __call__(self, input_tensor: torch.Tensor,
                 cams: np.ndarray,
                 targets: List[Callable],
                 model1: torch.nn.Module,
                 model2: torch.nn.Module,
                 return_visualization=False,
                 return_diff=True):

        if return_diff:
            with torch.no_grad():
                outputs = model1(input_tensor)
                scores = [target(output).cpu().numpy()
                          for target, output in zip(targets, outputs)]
                scores = np.float32(scores)

        batch_size = input_tensor.size(0)
        perturbated_tensors = []
        for i in range(batch_size):
            cam = cams[i]
            tensor = self.perturbation(input_tensor[i, ...].cpu(),
                                       torch.from_numpy(cam))
            tensor = tensor.to(input_tensor.device)
            perturbated_tensors.append(tensor.unsqueeze(0))
        perturbated_tensors = torch.cat(perturbated_tensors)

        with torch.no_grad():
            outputs_after_imputation = model2(perturbated_tensors)
        scores_after_imputation = [
            target(output).cpu().numpy() for target, output in zip(
                targets, outputs_after_imputation)]
        scores_after_imputation = np.float32(scores_after_imputation)
        if return_diff:
            result = scores_after_imputation - scores
        else:
            result = scores_after_imputation
        if return_visualization:
            return result, perturbated_tensors
        else:
            return result



# Class: Multiply CAM by Image to understand change in confidence
class CamMultImageConfidenceChange(PerturbationConfidenceMetric):
    def __init__(self):
        super(CamMultImageConfidenceChange, self).__init__(multiply_tensor_with_cam)



# Class: Drop in confidence
class DropInConfidenceText(CamMultImageConfidenceChange):
    def __init__(self):
        super(DropInConfidenceText, self).__init__()

    def __call__(self, *args, **kwargs):
        scores = super(DropInConfidenceText, self).__call__(*args, **kwargs)
        scores = -scores
        return np.maximum(scores, 0)



# Class: Increase in confidence
class IncreaseInConfidenceText(CamMultImageConfidenceChange):
    def __init__(self):
        super(IncreaseInConfidenceText, self).__init__()

    def __call__(self, *args, **kwargs):
        scores = super(IncreaseInConfidenceText, self).__call__(*args, **kwargs)
        return np.float32(scores > 0)



# Function: Get metrics (vision and text)
def get_metrics_vt(image_feat, image_feature, text_id, text_feature, vmap, tmap, model):
    results = {}
    
    with torch.no_grad():
        vtargets = [CosSimilarity(text_feature)]
        # ttargets = [CosSimilarity(image_feature)]
        
        # # Remove start and end token
        # text_id = text_id[:,1:-1]
        # tmap = np.expand_dims(tmap, axis=0)[:,1:-1]
        # model_clone = copy.deepcopy(model)
        # temp = np.ones_like(tmap).astype(int)
        # for idx,i in enumerate(text_id[0]):
        #     i = i.item()
        #     model_clone.text_model.embeddings.token_embedding.weight[i] = model_clone.text_model.embeddings.token_embedding.weight[i] * tmap[0][idx]
        results['vdrop'] = DropInConfidence()(image_feat, vmap, vtargets, ImageFeatureExtractor(model))[0][0]*100
        results['vincr'] = IncreaseInConfidence()(image_feat, vmap, vtargets, ImageFeatureExtractor(model))[0][0]*100
        # results['tdrop'] = DropInConfidenceText()(text_id, temp, ttargets, TextFeatureExtractor(model),TextFeatureExtractor(model_clone))[0][0]*100
        # results['tincr'] = IncreaseInConfidenceText()(text_id, temp, ttargets, TextFeatureExtractor(model),TextFeatureExtractor(model_clone))[0][0]*100
    
    return results



# Function: Get metrics (vision and vision for retrieval)
def get_metrics_vv(image_q_processed, image_q_feature, image_ret_processed, image_ret_feature, image_q_smap, image_ret_smap, model):
    results = {}
    
    with torch.no_grad():
        q_targets = [CosSimilarity(image_ret_feature)]
        ret_targets = [CosSimilarity(image_q_feature)]

        # Get results
        results['qdrop'] = DropInConfidence()(image_q_processed, image_q_smap, q_targets, ImageFeatureExtractor(model))[0][0]*100
        results['qincr'] = IncreaseInConfidence()(image_q_processed, image_q_smap, q_targets, ImageFeatureExtractor(model))[0][0]*100
        results['retdrop'] = DropInConfidence()(image_ret_processed, image_ret_smap, ret_targets, ImageFeatureExtractor(model))[0][0]*100
        results['retincr'] = IncreaseInConfidence()(image_ret_processed, image_ret_smap, ret_targets, ImageFeatureExtractor(model))[0][0]*100
    
    return results



# Function: Compute metrics for all data
def metric_evaluation(model, device, images_processed, image_features, text_ids, text_features, saliency_v, saliency_t):
    all_results = []
    

    for image_processed,image_feature,text_id,text_feature,vmap,tmap in zip(images_processed,image_features,text_ids,text_features,saliency_v,saliency_t):
        image_processed = image_processed.unsqueeze(0).to(device)
        image_feature = image_feature.unsqueeze(0).to(device)
        text_feature = text_feature.unsqueeze(0).to(device)
        vmap = np.expand_dims(vmap, axis=0)

        # Get results
        results = get_metrics_vt(
            image_feat=image_processed,
            image_feature=image_feature,
            text_id=text_id,
            text_feature=text_feature,
            vmap=vmap,
            tmap=tmap,
            model=model
        )
        
        all_results.append(results)
    
    return all_results



# Function: Compute metrics for all data (retrieval model)
def metric_evaluation_retrieval(model, device, images_q_processed, image_q_features, images_ret_processed, images_ret_features, image_q_smaps, image_ret_smaps):
    all_results = []
    
    for image_q_processed, image_q_feature, image_ret_processed, image_ret_feature, image_q_smap, image_ret_smap in zip(images_q_processed, image_q_features, images_ret_processed, images_ret_features, image_q_smaps, image_ret_smaps):
        image_q_processed = image_q_processed.unsqueeze(0).to(device)
        image_q_feature = image_q_feature.unsqueeze(0).to(device)

        # Retrieved Image(s)
        image_ret_processed = image_ret_processed.unsqueeze(0).to(device)
        image_ret_feature = image_ret_feature.unsqueeze(0).to(device)
        
        # Visual Saliency Maps
        image_q_smap = np.expand_dims(image_q_smap, axis=0)
        image_ret_smap = np.expand_dims(image_ret_smap, axis=0)

        # Get results
        results = get_metrics_vv(
            image_q_processed=image_q_processed, 
            image_q_feature=image_q_feature, 
            image_ret_processed=image_ret_processed, 
            image_ret_feature=image_ret_feature, 
            image_q_smap=image_q_smap, 
            image_ret_smap=image_ret_smap, 
            model=model
        )
        
        all_results.append(results)
    
    return all_results
import torch
import numpy as np

from pytorch_grad_cam.metrics.cam_mult_image import DropInConfidence, IncreaseInConfidence

# Function: Get metrics (vision only)
def get_metrics_v(image_feat, text_feature, vmap, model):
    results = {}    
    with torch.no_grad():
        vtargets = [CosSimilarity(text_feature)]        
        results['vdrop'] = DropInConfidence()(image_feat, vmap, vtargets, ImageFeatureExtractor(model))[0][0] * 100
        results['vincr'] = IncreaseInConfidence()(image_feat, vmap, vtargets, ImageFeatureExtractor(model))[0][0] * 100 #type: ignore
    return results

# Function: Compute metrics for all data (vision only)
def metric_evaluation_vision(model, device, images_processed, image_features, text_features, saliency_v):
    all_results = []    

    for image_processed,image_feature,text_feature,vmap in zip(images_processed,image_features,text_features,saliency_v):
        image_processed = image_processed.unsqueeze(0).to(device)
        image_feature = image_feature.unsqueeze(0).to(device)
        text_feature = text_feature.unsqueeze(0).to(device)
        vmap = np.expand_dims(vmap, axis=0)

        # Get results
        results = get_metrics_v(
            image_feat=image_processed,
            text_feature=text_feature,
            vmap=vmap,
            model=model
        )
        
        all_results.append(results)
    
    return all_results

class CosSimilarity:
    """ Target function """
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity()
        return cos(model_output, self.features)
    
class ImageFeatureExtractor(torch.nn.Module):
    """ Image feature wrapper """
    def __init__(self, model):
        super(ImageFeatureExtractor, self).__init__()
        self.model = model
                
    def __call__(self, x):
        return self.model.get_image_features(x)

class TextFeatureExtractor(torch.nn.Module):
    """ Text feature wrapper """
    def __init__(self, model):
        super(TextFeatureExtractor, self).__init__()   
        self.model = model
                
    def __call__(self, x):
        return self.model.get_text_features(x)
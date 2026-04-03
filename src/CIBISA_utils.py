import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings

from src.CIBISA import *

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def compute_saliency(x:torch.Tensor, x_target:torch.Tensor, model, loss_mode='cosine_sim', return_negative:bool=False, n_masks:int=20, epochs:int=15, beta:int=10, learnable_param=False) -> np.array:
    # computing the target embedding
    k = 3*224*224
    with torch.enable_grad():
        if model._get_name() == 'CLIP' or model._get_name() == 'CLIPModel' or model._get_name() == 'CoCa':
            # assert x_target.shape == (1,512), "When using the CLIP model, you should pass the text/image embedding of shape (1,512) as target"
            y_target = x_target
        else:
            y_target = model(x_target)

        # setting the flag for computing positive/negative relations
        if return_negative:
            y_flag = -torch.ones(1, dtype=torch.long, device=device)
        else:
            y_flag = torch.ones(1, dtype=torch.long, device=device)

        # defining the bottleneck layer
        msw = int(np.ceil(x.shape[-2]/32))
        msh = int(np.ceil(x.shape[-1]/32))
        
        new_layer = InfoLayer(mask_range=3, mask_size_w=msw, mask_size_h=msh, input_size_w=x.shape[-2], input_size_h=x.shape[-1])
        new_layer = new_layer.to(device=device)

        # placing the bottleneck in the target model
        new_core = InfoModel(core=model, bottleneck=new_layer)
        new_core = new_core.to(device=device)

        # losses and parameters
        P = list(new_core.parameters())

        if learnable_param:
            loss_inf = InfoLoss(learnable_param=learnable_param).to(device)
            loss_sim = SimilarityLoss(mode=loss_mode, sigma=1, learnable_param=False).to(device)

            P.extend(list(loss_inf.parameters()))
            P.extend(list(loss_sim.parameters()))

            optimizer = torch.optim.Adam(P, lr=1)

        else:
            loss_inf = InfoLoss(beta=beta/k, phi=1/k).to(device)
            loss_sim = SimilarityLoss(mode=loss_mode, sigma=1).to(device)
            optimizer = torch.optim.Adam(P, lr=1)
            
        # computing the saliency
        final_saliency = 0
        hyperparams = []
        for _ in range(n_masks):
            new_core.reset_model()

            if learnable_param:
                loss_inf.reset_loss()
            
            saliency, hps, l = train_bottleneck(model=new_core, x=x, y=y_target, flag=y_flag, loss_ce=loss_sim, loss_inf=loss_inf, opt=optimizer, epochs=epochs)
            final_saliency +=  saliency
            hyperparams.append(hps)

        final_saliency = np.squeeze(final_saliency)
        final_saliency = (final_saliency - np.min(final_saliency))/(np.max(final_saliency) - np.min(final_saliency))

    return final_saliency, hyperparams

def compute_saliency_text(x:torch.Tensor, x_target:torch.Tensor, model, loss_mode='cosine_sim', return_negative:bool=False, n_masks:int=20, epochs:int=15, beta:int=10, learnable_param=False) -> np.array:
    # computing the target embedding
    k = 3*224*224
    with torch.enable_grad():
        
        y_target = model.get_image_features(x_target)

        # setting the flag for computing positive/negative relations
        if return_negative:
            y_flag = -torch.ones(1, dtype=torch.long, device=device)
        else:
            y_flag = torch.ones(1, dtype=torch.long, device=device)

        # defining the bottleneck layer
        # defining the bottleneck layer
        msw = int(np.ceil(x.shape[-1]))
        msh = int(np.ceil(512/32))
        new_layer = InfoLayer(mask_range=3, mask_size_w=msw, mask_size_h=msh, input_size_w=x.shape[-1], input_size_h=512)
        new_layer = new_layer.to(device=device)

        # placing the bottleneck in the target model
        new_core = InfoModel_text(core=model, bottleneck=new_layer)
        new_core = new_core.to(device=device)

        # losses and parameters
        P = list(new_core.parameters())

        if learnable_param:
            loss_inf = InfoLoss(learnable_param=learnable_param).to(device)
            loss_sim = SimilarityLoss(mode=loss_mode, sigma=1, learnable_param=False).to(device)

            P.extend(list(loss_inf.parameters()))
            P.extend(list(loss_sim.parameters()))

            optimizer = torch.optim.Adam(P, lr=1)

        else:
            loss_inf = InfoLoss(beta=beta/k, phi=1/k).to(device)
            loss_sim = SimilarityLoss(mode=loss_mode, sigma=1).to(device)
            optimizer = torch.optim.Adam(P, lr=1)
            
        # computing the saliency
        final_saliency = 0
        hyperparams = []
        for _ in range(n_masks):
            new_core.reset_model()

            if learnable_param:
                loss_inf.reset_loss()
            
            saliency, hps = train_bottleneck(model=new_core, x=x, y=y_target, flag=y_flag, loss_ce=loss_sim, loss_inf=loss_inf, opt=optimizer, epochs=epochs)
            final_saliency += saliency
            hyperparams.append(hps)

        final_saliency = np.squeeze(final_saliency)
        final_saliency = (final_saliency - np.min(final_saliency))/(np.max(final_saliency) - np.min(final_saliency))

    return final_saliency, hyperparams

def compute_pairs(x1, x2, model, mode='diff', loss_mode='cosine_sim', n_masks:int=20,beta:int=10, learnable_param=False):
    if mode == 'diff':
        s_pos,_ = compute_saliency(x=x1, x_target=x2, model=model, loss_mode=loss_mode, return_negative=False, n_masks=n_masks, beta=beta, learnable_param=learnable_param)
        s_neg,_ = compute_saliency(x=x1, x_target=x2, model=model, loss_mode=loss_mode, return_negative=True, n_masks=n_masks, beta=beta, learnable_param=learnable_param)
        sfinal_x1 = s_pos - s_neg

        s_pos,_ = compute_saliency(x=x2, x_target=x1, model=model, loss_mode=loss_mode, return_negative=False, n_masks=n_masks, beta=beta, learnable_param=learnable_param)
        s_neg,_ = compute_saliency(x=x2, x_target=x1, model=model, loss_mode=loss_mode, return_negative=True, n_masks=n_masks, beta=beta, learnable_param=learnable_param)
        sfinal_x2 = s_pos - s_neg

        return sfinal_x1, sfinal_x2

    elif mode == 'pos':
        sfinal_x1,_ = compute_saliency(x=x1, x_target=x2, model=model, loss_mode=loss_mode, return_negative=False, beta=beta, learnable_param=learnable_param)
        sfinal_x2,_ = compute_saliency(x=x2, x_target=x1, model=model, loss_mode=loss_mode, return_negative=False, beta=beta, learnable_param=learnable_param)

        return sfinal_x1, sfinal_x2

    elif mode == 'neg':
        sfinal_x1,_ = compute_saliency(x=x1, x_target=x2, model=model, loss_mode=loss_mode, return_negative=True, beta=beta, learnable_param=learnable_param)
        sfinal_x2,_ = compute_saliency(x=x2, x_target=x1, model=model, loss_mode=loss_mode, return_negative=True, beta=beta, learnable_param=learnable_param)

        return sfinal_x1, sfinal_x2

    else:
        raise NameError("Please, insert a valid mode, options: ['diff', 'pos', 'neg']")

def plot_pairs(x1_pil, x2_pil, s1, s2, cmap='RdBu_r'):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(8,8))

    size = s1.shape[-1]

    ax1.imshow(x1_pil.resize((size, size)))
    ax1.axis('off')

    ax2.imshow(x2_pil.resize((size, size)))
    ax2.axis('off')

    ax3.imshow(x1_pil.resize((size, size)))
    ax3.imshow(s1, cmap=cmap, alpha=0.5)
    ax3.axis('off')

    ax4.imshow(x2_pil.resize((size, size)))
    ax4.imshow(s2, cmap=cmap, alpha=0.5)
    ax4.axis('off')

    plt.show()

def patch_image(pil_img, grid_size=4):
    pil_img = pil_img.resize((224,224))
    window_size = np.floor(224/grid_size)
    patches_list = []
    for j in range(grid_size):
        for i in range(grid_size):
            patches_list.append(pil_img.crop((i*window_size,j*window_size, window_size+i*window_size, window_size+j*window_size)).resize((224,224)))
    return patches_list

def decode_imagenet(preds:torch.tensor):
    file = open('imagenet_labels.txt')
    imgnt_labels = file.read()
    file.close()

    imgnt_labels = imgnt_labels.split('\n')

    decoded_labels = []
    for y in preds:
        decoded_labels.append(imgnt_labels[y])

    return decoded_labels

def unnormalize(x:torch.Tensor, toPIL=True):
    invTrans = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])
    x = invTrans(x)
    if toPIL:
        return torchvision.transforms.functional.to_pil_image(torch.squeeze(x))
    else:
        return x

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
# def log_beta_phi():

def mutual_information(hgram):

    """ 
    
    Mutual information for joint histogram

    """

    # Convert bins counts to probability values

    pxy = hgram / float(np.sum(hgram))

    px = np.sum(pxy, axis=1) # marginal for x over y

    py = np.sum(pxy, axis=0) # marginal for y over x

    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals

    # Now we can do the calculation using the pxy, px_py 2D arrays

    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum

    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

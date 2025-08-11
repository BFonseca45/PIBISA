import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from src.CIBISA import *

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def compute_saliency(x:torch.Tensor, x_target:torch.Tensor, model:torch, loss_mode='cosine_sim', return_negative:bool=False, n_masks:int=20, beta:int=10, learnable_param=False) -> np.array:
    # computing the target embedding
    with torch.enable_grad():
        if model._get_name() == 'CLIP':
            assert x_target.shape == (1,512), "When using the CLIP model, you should pass the text/image embedding of shape (1,512) as target"
            y_target = x_target
        else:
            y_target = model(x_target)

        # setting the flag for computing positive/negative relations
        if return_negative:
            y_flag = -torch.ones(1, dtype=torch.long, device=device)
        else:
            y_flag = torch.ones(1, dtype=torch.long, device=device)

        # defining the bottleneck layer
        ms = int(np.ceil(x.shape[-1]/32))
        new_layer = InfoLayer(mask_range=3, mask_size=ms, input_size=x.shape[-1])
        new_layer = new_layer.to(device=device)

        # placing the bottleneck in the target model
        new_core = InfoModel(core=model, bottleneck=new_layer)
        new_core = new_core.to(device=device)

        # losses and parameters
        P = list(new_core.parameters())
        # optimizer = torch.optim.Adam(P, lr=1)
        optimizer = torch.optim.Adam(P, lr=0.2)
        
        if learnable_param:
            loss_inf = InfoLoss(learnable_param=learnable_param).to(device)
            loss_sim = SimilarityLoss(mode=loss_mode, learnable_param=learnable_param).to(device)
        else:
            loss_inf = InfoLoss(beta=beta, phi=1).to(device)
            loss_sim = SimilarityLoss(mode=loss_mode, sigma=1).to(device)

        # computing the saliency
        final_saliency = 0
        for _ in range(n_masks):
            new_core.reset_model()
            # final_saliency += train_bottleneck(model=new_core, x=x, y=y_target, flag=y_flag, loss_ce=loss_sim, loss_inf=loss_inf, opt=optimizer, epochs=10)
            final_saliency += train_bottleneck(model=new_core, x=x, y=y_target, flag=y_flag, loss_ce=loss_sim, loss_inf=loss_inf, opt=optimizer, epochs=20)

        final_saliency = np.squeeze(final_saliency)
        final_saliency = (final_saliency - np.min(final_saliency))/(np.max(final_saliency) - np.min(final_saliency))

    return final_saliency

def compute_pairs(x1, x2, model, mode='diff', loss_mode='cosine_sim', n_masks=20,beta=10):
    if mode == 'diff':
        s_pos = compute_saliency(x=x1, x_target=x2, model=model, loss_mode=loss_mode, return_negative=False, n_masks=n_masks, beta=beta)
        s_neg = compute_saliency(x=x1, x_target=x2, model=model, loss_mode=loss_mode, return_negative=True, n_masks=n_masks, beta=beta)
        sfinal_x1 = s_pos - s_neg

        s_pos = compute_saliency(x=x2, x_target=x1, model=model, loss_mode=loss_mode, return_negative=False, n_masks=n_masks, beta=beta)
        s_neg = compute_saliency(x=x2, x_target=x1, model=model, loss_mode=loss_mode, return_negative=True, n_masks=n_masks, beta=beta)
        sfinal_x2 = s_pos - s_neg

        return sfinal_x1, sfinal_x2

    elif mode == 'pos':
        sfinal_x1 = compute_saliency(x=x1, x_target=x2, model=model, loss_mode=loss_mode, return_negative=False, beta=beta)
        sfinal_x2 = compute_saliency(x=x2, x_target=x1, model=model, loss_mode=loss_mode, return_negative=False, beta=beta)

        return sfinal_x1, sfinal_x2

    elif mode == 'neg':
        sfinal_x1 = compute_saliency(x=x1, x_target=x2, model=model, loss_mode=loss_mode, return_negative=True, beta=beta)
        sfinal_x2 = compute_saliency(x=x2, x_target=x1, model=model, loss_mode=loss_mode, return_negative=True, beta=beta)

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

def unnormalize(x:torch.Tensor):
    invTrans = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])
    x = invTrans(x)
    return torchvision.transforms.functional.to_pil_image(torch.squeeze(x))
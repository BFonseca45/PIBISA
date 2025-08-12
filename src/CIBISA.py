import random
import torch
import torch.nn as nn
import numpy as np

random.seed(32984)
torch.manual_seed(2234)

class InfoModel(nn.Module):
    def __init__(self, core, bottleneck):
        super(InfoModel, self).__init__()
        self.core = core
        for parameter in self.core.parameters():
            parameter.requires_grad = False

        self.botteneck = bottleneck

    def forward(self, x):
        x = self.botteneck(x)
        if self.core._get_name() == 'CLIP':
            y = self.core.encode_image(x)
        else:
            y = self.core(x)
        return y

    def get_saliency(self):
        return self.botteneck.get_lamb()

    def reset_model(self):
        self.botteneck.reset_alpha()

    def activations_hook(self, module, inputs, act):
        self.activations = act

    def get_activations(self):
        return self.activations.detach()

class InfoLayer(nn.Module):
    def __init__(self, input_size, mask_size, mask_range):
        super(InfoLayer, self).__init__()

        self.input_size = input_size
        self.mask_size = mask_size
        self.mask_range = mask_range

        self.alpha = nn.Parameter(torch.empty((self.mask_size, self.mask_size), dtype=torch.float32))
        self.reset_alpha()

    def forward(self, x):
        alpha_ex = self.alpha.expand((1, 1, self.mask_size, self.mask_size))
        alpha_ex = nn.Upsample(size=(self.input_size, self.input_size), mode='bicubic')(alpha_ex)
        self.lamb = torch.sigmoid(alpha_ex)

        return torch.mul(x, self.lamb)

    def get_lamb(self):
        return self.lamb

    def reset_alpha(self):
        nn.init.uniform_(self.alpha, -self.mask_range, self.mask_range)

class InfoLoss(nn.Module):
    def __init__(self, beta=0.01, phi=0.01, learnable_param=False):
        super(InfoLoss, self).__init__()
        self.learnable_param = learnable_param

        if self.learnable_param:
            self.beta = torch.nn.Parameter(torch.empty((1,1), dtype=torch.float32))
            self.phi = torch.nn.Parameter(torch.empty((1,1), dtype=torch.float32))
        else:
            self.beta = beta
            self.phi = phi

    def forward(self, saliency_map):
        # Normalization term
        k = np.prod(saliency_map.shape)*3

        complexity_loss = torch.sum(torch.abs(saliency_map))/k
        variation_loss = total_variation(saliency=saliency_map)/k

        if self.learnable_param:
            # print('beta value', self.beta)
            # print('phi value', self.phi)
            return torch.exp(-self.beta)*complexity_loss + self.beta + torch.exp(-self.phi)*variation_loss + self.phi
        else:
            return self.beta*complexity_loss + self.phi*variation_loss

    def reset_loss(self):
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.phi)

# Horrible function
class SimilarityLoss(nn.Module):
    def __init__(self, mode='cosine_sim', sigma=1, learnable_param=False):
        super(SimilarityLoss, self).__init__()
        self.learnable_param = learnable_param
        self.mode = mode

        if self.learnable_param:
            self.sigma = torch.nn.Parameter(torch.empty((1,1), dtype=torch.float32))
        else:
            self.sigma = sigma

        if self.mode == 'cosine_sim':
            self.loss_sim = torch.nn.CosineEmbeddingLoss()
        elif self.mode == 'mse':
            self.loss_sim = torch.nn.MSELoss()
        else:
            raise AssertionError("Enter a valid mode: ['cosine_sim', 'mse']")

    def forward(self, y_pred, y_hat, y_flag):
        if self.mode == 'cosine_sim':
            if self.learnable_param:
                return torch.exp(-self.sigma)*self.loss_sim(y_pred,y_hat,y_flag) + self.sigma
            else:
                return self.sigma*self.loss_sim(y_pred,y_hat,y_flag)

        elif self.mode == 'mse':
            if y_flag == 1:
                if self.learnable_param:
                    return torch.exp(-self.sigma)*self.loss_sim(y_pred,y_hat) + self.sigma
                else:
                    return self.sigma*self.loss_sim(y_pred,y_hat)
            else:
                if self.learnable_param:
                    return torch.exp(-self.sigma)*1/self.loss_sim(y_pred,y_hat) + self.sigma
                else:
                    return self.sigma*1/self.loss_sim(y_pred,y_hat)

        else:
            raise AssertionError("Enter a valid mode: ['cosine_sim', 'mse']")

    def reset_loss(self):
        nn.init.zeros_(self.sigma)

def total_variation(saliency):
    saliency = torch.squeeze(saliency)
    x_diff = torch.abs(saliency[1:,:] - saliency[:-1,:])
    y_diff = torch.abs(saliency[:,1:] - saliency[:,:-1])

    return torch.sum(x_diff) + torch.sum(y_diff)

def train_bottleneck(model, x, y, flag, epochs, loss_ce, loss_inf, opt):
    for i in range(epochs):
        opt.zero_grad()
        y_pred = model(x)

        loss = loss_ce(y_pred, y, flag) + loss_inf(model.get_saliency())
        loss.backward(retain_graph=True)

        opt.step()

    return model.get_saliency().detach().cpu().numpy()
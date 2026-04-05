# class InfoModel_text(nn.Module):
#     def __init__(self, core, bottleneck):
#         super(InfoModel_text, self).__init__()
#         self.core = core
#         for parameter in self.core.parameters():
#             parameter.requires_grad = False

#         self.botteneck = bottleneck
#         self.model = copy.deepcopy(self.core)

#     def forward(self, x):
#         self.model.text_model.embeddings.token_embedding.weight.data[x,:] = self.botteneck(self.core.text_model.embeddings.token_embedding.weight.data[x,:])
#         return self.model.get_text_features(x)

#     def get_saliency(self):
#         return self.botteneck.get_lamb()

#     def reset_model(self):
#         self.botteneck.reset_smallMask()

# def compute_saliency_text(x:torch.Tensor, x_target:torch.Tensor, model, loss_mode='cosine_sim', return_negative:bool=False, n_masks:int=20, epochs:int=15, beta:int=10, learnable_param=False) -> np.array:
#     # computing the target embedding
#     k = 3*224*224
#     with torch.enable_grad():
        
#         y_target = model.get_image_features(x_target)

#         # setting the flag for computing positive/negative relations
#         if return_negative:
#             y_flag = -torch.ones(1, dtype=torch.long, device=config.device)
#         else:
#             y_flag = torch.ones(1, dtype=torch.long, device=config.device)

#         # defining the bottleneck layer
#         # defining the bottleneck layer
#         msw = int(np.ceil(x.shape[-1]))
#         msh = int(np.ceil(512/32))
#         new_layer = InfoLayer(mask_range=3, mask_size_w=msw, mask_size_h=msh, input_size_w=x.shape[-1], input_size_h=512)
#         new_layer = new_layer.to(device=config.device)

#         # placing the bottleneck in the target model
#         new_core = InfoModel_text(core=model, bottleneck=new_layer)
#         new_core = new_core.to(device=config.device)

#         # losses and parameters
#         P = list(new_core.parameters())

#         if learnable_param:
#             loss_inf = MaskLoss(learnable_param=learnable_param).to(config.device)
#             loss_sim = SimilarityLoss(simFunc=loss_mode, sigma=1, learnable_param=False).to(config.device)

#             P.extend(list(loss_inf.parameters()))
#             P.extend(list(loss_sim.parameters()))

#             optimizer = torch.optim.Adam(P, lr=1)

#         else:
#             loss_inf = MaskLoss(beta=beta/k, phi=1/k).to(config.device)
#             loss_sim = SimilarityLoss(simFunc=loss_mode, sigma=1).to(config.device)
#             optimizer = torch.optim.Adam(P, lr=1)
            
#         # computing the saliency
#         final_saliency = 0
#         hyperparams = []
#         for _ in range(n_masks):
#             new_core.reset_model()

#             if learnable_param:
#                 loss_inf.reset_loss()
            
#             saliency, hps = train_bottleneck(model=new_core, x=x, y=y_target, flag=y_flag, loss_ce=loss_sim, loss_inf=loss_inf, opt=optimizer, epochs=epochs)
#             final_saliency += saliency
#             hyperparams.append(hps)

#         final_saliency = np.squeeze(final_saliency)
#         final_saliency = (final_saliency - np.min(final_saliency))/(np.max(final_saliency) - np.min(final_saliency))

#     return final_saliency, hyperparams

import torch
import torchvision

from src.vision_transformer import VisionTransformer, vit_tiny, vit_small, vit_base, vit_large

class VGG_emb(torch.nn.Module):
    def __init__(self, level='full'):
        super().__init__()
        backbone = torchvision.models.vgg16(weights='DEFAULT')
        backbone.eval()

        for param in backbone.parameters():
            param.requires_grad = False

        if level == 'full':
            self.feature_extractor = backbone.features
        elif level == 3:
            self.feature_extractor = torch.nn.Sequential(*list(backbone.features.children())[:24])
        elif level == 2:
            self.feature_extractor = torch.nn.Sequential(*list(backbone.features.children())[:17])
        elif level == 1:
            self.feature_extractor = torch.nn.Sequential(*list(backbone.features.children())[:10])

        self.avg_pool = backbone.avgpool
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avg_pool(x)

        return self.flatten(x)

class ResNET_emb(torch.nn.Module):
    def __init__(self, level='full'):
        super().__init__()
        backbone = torchvision.models.resnet50(weights='DEFAULT')
        backbone.eval()

        for param in backbone.parameters():
            param.requires_grad = False

        if level == 'full':
            self.feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-2])
        elif level == 3:
            self.feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-3])
        elif level == 2:
            self.feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-4])
        elif level == 1:
            self.feature_extractor = torch.nn.Sequential(*list(backbone.children())[:5])

        self.avg_pool = backbone.avgpool
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avg_pool(x)

        return self.flatten(x)

class ViT16_emb(torch.nn.Module):
    def __init__(self, weights = 'ImageNET'):
        super().__init__()
        self.weights = weights
        
        if self.weights == 'ImageNET':
            self.backbone = torchvision.models.vit_b_16(weights='DEFAULT')
            self.backbone.eval()
        elif self.weights == 'iBOT':
            state_dict = torch.load("/nas-ctm01/homes/bfcoelho/CIBISA/pre_trained_models/checkpoint_teacher.pth")['state_dict']
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            self.backbone = vit_base(patch_size=16, return_all_tokens=True)
            self.backbone.load_state_dict(state_dict, strict=False)
            self.backbone.eval()
            
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        if self.weights == 'ImageNET':
            x = self.backbone._process_input(x)

            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.backbone.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.backbone.encoder(x)
            x = x[:, 0]

            return self.flatten(x)
        
        elif self.weights == 'iBOT':
            x = self.backbone(x)
            x = x[:,0,:]
            return self.flatten(x)
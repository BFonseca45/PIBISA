import torch
import torchvision

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
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.vit_b_16(weights='DEFAULT')
        self.backbone.eval()

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.backbone._process_input(x)

        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.backbone.encoder(x)
        x = x[:, 0]

        return self.flatten(x)
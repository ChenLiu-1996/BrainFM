import clip
import torch
import torchvision


class VisionEncoder():
    def __init__(self,
                 pretrained_model: str,
                 device: torch.device):
        if pretrained_model == 'resnet18':
            self.backbone = torchvision.models.resnet18(weights='DEFAULT')
            self.backbone.fc = torch.nn.Identity()
        elif pretrained_model == 'convnext_tiny':
            self.backbone = torchvision.models.convnext_tiny(weights='DEFAULT')
            self.backbone.classifier[-1] = torch.nn.Identity()
        elif pretrained_model == 'mobilenetv3_small':
            self.backbone = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
            self.backbone.classifier[-1] = torch.nn.Identity()
        elif pretrained_model == 'clip':
            self.backbone, _ = clip.load("ViT-B/32", device=device)
            self.preprocess = torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                 std=(0.26862954, 0.26130258, 0.27577711)),
            ])

        self.backbone.eval()
        self.backbone.to(device)
        self.pretrained_model = pretrained_model
        self.device = device

        for p in self.backbone.parameters():
            p.requires_grad = False

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        assert len(image.shape) == 4, f'Image shape should have length 4, but found {image.shape}.'
        assert image.shape[1] in [1, 3], f'Image channel should be 1 or 3, but found {image.shape[1]}.'
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        if self.pretrained_model == 'clip':
            latent_embedding = self.backbone.encode_image(self.preprocess(image.float().to(self.device)))
        else:
            latent_embedding = self.backbone(image.float().to(self.device))
        return latent_embedding


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vision_encoder = VisionEncoder(pretrained_model='clip', device=device)
    image = torch.rand(1, 1, 256, 256)
    vision_encoder.embed(image)
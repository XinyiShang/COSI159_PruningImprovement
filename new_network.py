import torch
import torchvision.models as models


class VGG(torch.nn.Module):
    def __init__(self, vgg='vgg16_bn', data_set='CIFAR10', pretrained=False):
        super(VGG, self).__init__()
        self.features = models.__dict__[vgg](pretrained=pretrained).features

        # Modify features.37
        self.features[37] = torch.nn.Conv2d(512, 52, kernel_size=3, padding=1)
        
        # Modify features.38
        self.features[38] = torch.nn.BatchNorm2d(52)
        self.features[38].weight.data = torch.ones_like(self.features[38].weight.data)
        self.features[38].bias.data = torch.zeros_like(self.features[38].bias.data)
        
        # Modify features.40
        self.features[40] = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        
        # Modify the classifier layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 10 if 'CIFAR' in data_set else None),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x



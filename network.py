import torch
import torchvision.models as models
from torch.quantization import QuantStub
from torch.quantization import DeQuantStub

class VGG(torch.nn.Module):
    def __init__(self, vgg='vgg16_bn', data_set='CIFAR10', pretrained=False, q=True):
        super(VGG, self).__init__()
        self.features = models.__dict__[vgg](pretrained=pretrained).features
        
        classifier = []
        if 'CIFAR' in data_set:
            num_class = int(data_set.split("CIFAR")[1])
            
            classifier.append(torch.nn.Linear(512, 512))
            classifier.append(torch.nn.BatchNorm1d(512))
            classifier.append(torch.nn.Linear(512, num_class))
        else:
            raise RuntimeError("Not expected data flag !!!")

        self.classifier = torch.nn.Sequential(*classifier)
        self.q = q
        if q:
          self.quant = QuantStub()
          self.dequant = DeQuantStub()
        self.network = network; 
    def forward(self, x):
        
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.q:
          x = self.quant(x)
          x = self.network(x)
            # manually specify where tensors will be converted from quantized
            # to floating point in the quantized model
          x = self.dequant(x)
        return x
import torch
import os
import torch.nn as nn
from new_network import VGG
from evaluate import test_network

# Load the model checkpoint
checkpoint = torch.load('/content/drive/MyDrive/Pruning_filters_for_efficient_convnets-master/Pruning_filters_for_efficient_convnets-master/trained_models/check_point.pth')
print(checkpoint.keys())
# Re-create the model architecture
model = VGG()
model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 1000),
)

# Load the pruned and retrained model state dict
model.load_state_dict(checkpoint['state_dict'], strict = False)

input_layer = torch.quantization.QuantStub()
output_layer = torch.quantization.DeQuantStub()

network = torch.nn.Sequential(
    input_layer,
    network.features,
    torch.nn.Flatten(),
    network.classifier,
    output_layer,
)

quantized_network = torch.quantization.quantize_dynamic(network, {torch.nn.Linear}, dtype=torch.qint8)
torch.save(quantized_network.state_dict(), 'quantized_network.pth')

network.load_state_dict(torch.load('/content/drive/MyDrive/Pruning_filters_for_efficient_convnets-master/Pruning_filters_for_efficient_convnets-master/quantized_network.pth'),strict=False)
test_network(data_set='CIFAR10', network=network)

# Get the size of the file
size_bytes = os.path.getsize("/content/drive/MyDrive/Pruning_filters_for_efficient_convnets-master/Pruning_filters_for_efficient_convnets-master/quantized_network.pth")
size_bytes2 = os.path.getsize("/content/drive/MyDrive/Pruning_filters_for_efficient_convnets-master/Pruning_filters_for_efficient_convnets-master/trained_models/check_point.pth")

print(f"Size of quantized model checkpoint: {size_bytes} bytes")
print(f"Size of unquantized model checkpoint: {size_bytes2} bytes")
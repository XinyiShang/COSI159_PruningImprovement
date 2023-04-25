import torch
import torch.nn as nn
import torch.quantization

# Load pre-pruned model
model = torch.load('check_point.pth')

# Define quantization layers for each type of layer in the model
quant_conv = nn.quantized.Conv2d( ... )  # Specify quantization parameters as needed
quant_fc = nn.quantized.Linear( ... )

# Replace each layer in the model with its corresponding quantization layer
# You can use the model.named_modules() function to iterate over all modules in the model
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        setattr(model, name, quant_conv)
    elif isinstance(module, nn.Linear):
        setattr(model, name, quant_fc)

# Quantize the model weights and activations based on a representative dataset
dataset = torch.utils.data.DataLoader( ... )  # Define a representative dataset
model = torch.quantization.quantize_dynamic(model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8, dataset=dataset)

# Save the quantized model to a new file
torch.save(model.state_dict(), 'pre_pruned_and_quantized_model.pth')

# Improvement based on "Pruning Filters For Efficient ConvNets"

**Unofficial PyTorch implementation of pruning VGG on CIFAR-10 Data set**

**Reference**: [Pruning Filters For Efficient ConvNets, ICLR2017](https://arxiv.org/abs/1608.08710)

Requirements
--
* torch (version: 1.2.0)
* torchvision (version: 0.4.0)
* Pillow (version: 6.1.0)
* matplotlib (version: 3.1.1)
* numpy (version: 1.16.5)

Usage
--

### Arguments
* `--train-flag`: Train VGG on CIFAR Data set
* `--save-path`: Path to save results, ex) trained_models/
* `--load-path`: Path to load checkpoint, add 'checkpoint.pht' with `save_path`, ex) trained_models/checkpoint.pth
* `--resume-flag`: Resume the training from checkpoint loaded with `load-path`
* `--prune-flag`: Prune VGG
* `--prune-layers`: List of target convolution layers for pruning, ex) conv1 conv2
* `--prune-channels`: List of number of channels for pruning the `prune-layers`, ex) 4 14
* `--independent-prune-flag`: Prune multiple layers by independent strategy
* `--retrain-flag`: Retrain the pruned nework
* `--retrain-epoch`: Number of epoch for retraining pruned network
* `--retrain-lr`: Number of epoch for retraining pruned network

Drop out & VGG-19
--
You can follow the procedure of prune filter in the .ipynb file but switch the network to the corresponding network file: drop_out_network.py and VGG_19_network.py

Quantization
--

quantization.py file apply quantization to the original VGG-16 model. we created a new network trying to fit the pruned and retrained network into the original model which unfortunately failed.

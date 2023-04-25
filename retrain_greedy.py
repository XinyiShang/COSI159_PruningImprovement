import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time
import numpy as np
import matplotlib.pyplot as plt

import torch
device = torch.device('cuda')

from evaluate import test_network
from prune import prune_network
import json

load_path = 'trained_models/'

args = torch.load(load_path+'arguments.pth')
for key, value in vars(args).items():
    print("%s: %s"%(key, value))

#################################
args.load_path = load_path + 'check_point.pth'
args.save_path = load_path+'%s/'%time.ctime().replace(' ', '_')
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

network, test_set, (top1, top5) = test_network(args)

prune_step_ratio = 1/8
max_channel_ratio = 0.90 

prune_channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
prune_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10', 'conv11', 'conv12', 'conv13']

args.retrain_flag = True
args.retrain_epoch = 10
args.independent_prune_flag = False
args.retrain_lr = 0.001

top1_accuracies = {}
top5_accuracies = {}

for conv, channel in zip(prune_layers, prune_channels):    
    top1_accuracies[conv] = []
    top5_accuracies[conv] = []
    
    # load new network and check accuracy
    network, _, _ = test_network(args, data_set=test_set)
        
    # remove 0 channels ~ M (max_channel_ratio) % of total channels
    step = np.linspace(0, int(channel*max_channel_ratio), int(1/prune_step_ratio), dtype=np.int)
    steps = (step[1:] - step[:-1]).tolist()
    
    for i in range(len(steps)):
        print("\n%s: %s Layer, %d Channels pruned"%(time.ctime(), conv, sum(steps[:i+1])))
        
        # set prune information
        args.prune_layers = [conv]
        args.prune_channels =[steps[i]]

        network = prune_network(args, network)
        
        network, _, (top1, top5) = test_network(args, network, test_set)
            
        top1_accuracies[conv].append(top1)
        top5_accuracies[conv].append(top5)

path_to_folder = 'trained_models_imp'

# Define the filename and extension for the file
filename1 = 'top1_accuracies_greedy'
filename2 = 'top5_accuracies_greedy'

# Join the path and filename to create the full file path
full_file_path1 = path_to_folder + filename1
full_file_path2 = path_to_folder + filename2

# Open the file for writing
with open(full_file_path1, 'w') as f:
    # Use json.dump() to write the dictionary to the file
    json.dump(top1_accuracies, f)

with open(full_file_path2, 'w') as f:
    # Use json.dump() to write the dictionary to the file
    json.dump(top5_accuracies, f)
import torch
import torchvision
import torchvision.transforms as transforms
import torch.quantization

class AverageMeter(object):    
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
def load_model(quantized_model, model):
    """ Loads in the weights into an object meant for quantization """
    state_dict = model.state_dict()
    model = model.to('cpu')
    quantized_model.load_state_dict(state_dict)


def print_size_of_model(model):
    """ Prints the real size of the model """
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
    
def get_normalizer(data_set, inverse=False):
    if data_set == 'CIFAR10':
        MEAN = (0.4914, 0.4822, 0.4465)
        STD = (0.2023, 0.1994, 0.2010)

    elif data_set == 'CIFAR100':
        MEAN = (0.5071, 0.4867, 0.4408)
        STD = (0.2675, 0.2565, 0.2761)

    else:
        raise RuntimeError("Not expected data flag !!!")

    if inverse:
        MEAN = [-mean/std for mean, std in zip(MEAN, STD)]
        STD = [1/std for std in STD]

    return transforms.Normalize(MEAN, STD)

def get_transformer(data_set, imsize=None, cropsize=None, crop_padding=None, hflip=None):
    transformers = [] 
    if imsize:
        transformers.append(transforms.Resize(imsize))
    if cropsize:
        ## https://github.com/kuangliu/pytorch-cifar
        transformers.append(transforms.RandomCrop(cropsize, padding=crop_padding))
    if hflip:
        transformers.append(transforms.RandomHorizontalFlip(hflip))

    transformers.append(transforms.ToTensor())
    transformers.append(get_normalizer(data_set))
    
    return transforms.Compose(transformers)

def get_data_set(args, train_flag=True):
    if train_flag:
        data_set = torchvision.datasets.__dict__[args.data_set](root=args.data_path, train=True, 
                                       transform=get_transformer(args.data_set, args.imsize,
                                           args.cropsize, args.crop_padding, args.hflip), download=True)
    else:
        data_set = torchvision.datasets.__dict__[args.data_set](root=args.data_path, train=False, 
                                           transform=get_transformer(args.data_set), download=True)    
    return data_set

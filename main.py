from parameter import get_parameter
from train import train_network
from evaluate import test_network
from prune import prune_network

if __name__ == '__main__':
    args = get_parameter()

    network = None
    if args.train_flag:
        network = train_network(args, network=network)
        

    if args.prune_flag:
        network = prune_network(args, network=network)
        qnet = VGG(q=True)
        load_model(qnet, network)
        print_size_of_model(qnet)

    test_network(args, network=network)

import argparse
import pprint
import sys


def load_arguments():
    parser = argparse.ArgumentParser(sys.argv[0], description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--source', type=str, default='mnist',
                        help='From mnist to fashionMnist or otherwise (default: mnist)')
    parser.add_argument('--graph_name', type=str, default='output_domain', help='Graph environment name')
    parser.add_argument('--one_sided', action='store_true', default=False, help='Do not train target domain')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', action='store_true', default=False, help='resume the model (default: False)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--model_target_path', type=str, default='vae/saved/MNIST.pt')
    parser.add_argument('--model_source_path', type=str, default='vae/saved/FashionMNIST.pt')
    args = parser.parse_args()
    print('-' * 10)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print('-' * 10)
    return args

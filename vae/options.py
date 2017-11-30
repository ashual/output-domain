import argparse
import pprint
import sys
import torch


def load_arguments():
    parser = argparse.ArgumentParser(sys.argv[0], description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', action='store_true', default=True, help='resume the model')
    parser.add_argument('--svhn_path', type=str, default='../data_SVHN')
    parser.add_argument('--mnist_path', type=str, default='../data_MNIST')
    parser.add_argument('--save_path', type=str, default='./saved')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=28)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print('-' * 10)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print('-' * 10)
    return args

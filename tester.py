from __future__ import print_function

import os.path

import torch.utils.data

from data_loader import get_data_loader
from options import load_arguments
from tests import Tests
from utils.graph import Graph
from utils.mnist_classifier.classify import ClassifyMNIST

args = load_arguments()
args.graph_name = args.graph_name + '_test'
args.cuda = not args.no_cuda and torch.cuda.is_available()
graph = Graph(args.graph_name)
classifyMNIST = ClassifyMNIST(args)

if args.source == 'mnist':
    train_loader_source = get_data_loader(args, True, 'mnist')
    train_loader_target = get_data_loader(args, True, 'fashionMnist')
elif args.source == 'fashionMnist':
    train_loader_source = get_data_loader(args, True, 'fashionMnist')
    train_loader_target = get_data_loader(args, True, 'mnist')
else:
    raise Exception('args.source does not defined')

if not os.path.isfile(args.model_source_path):
    raise Exception('No source model')
else:
    print('Loading source model from {}'.format(args.model_source_path))
    model_source = torch.load(args.model_source_path)
    source_resume = True

if not os.path.isfile(args.model_target_path):
    raise Exception('No target model')
else:
    print('Loading target model from {}'.format(args.model_target_path))
    model_target = torch.load(args.model_target_path)

if args.cuda:
    model_target.cuda()
    model_source.cuda()

if args.source == 'mnist':
    tests = Tests(model_source, model_target, classifyMNIST, 'mnist', 'fashionMnist', args, graph)
elif args.source == 'fashionMnist':
    tests = Tests(model_source, model_target, classifyMNIST, 'fashionMnist', 'mnist', args, graph)
else:
    raise Exception('args.source does not defined')

tests.source_to_target_test()
tests.args.one_sided = not tests.args.one_sided
tests.source_to_target_test()
tests.args.one_sided = not tests.args.one_sided
tests.gaussian_input()
tests.tsne()
tests.reconstruction(0)
accuracy = tests.test_matching()
print('Diagonal accuracy is {}'.format(accuracy))

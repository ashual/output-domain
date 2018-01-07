from __future__ import print_function

import os.path

import torch.utils.data
from torch import nn
from torch import optim
from torch.autograd import Variable

from data_loader import get_data_loader
from utils.mnist_classifier.classify import ClassifyMNIST
from tests import Tests
from models.complex_model import VAE as SourceModel
from models.simple_model import VAE as TargetModel
from models.discriminator import Discriminator
from utils.graph import Graph
from utils.loss import complex_loss_function as source_loss
from utils.loss import simple_loss_function as target_loss
from options import load_arguments


args = load_arguments()
args.cuda = not args.no_cuda and torch.cuda.is_available()
graph = Graph(args.graph_name)

torch.manual_seed(args.seed)
classifyMNIST = ClassifyMNIST(args)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.source == 'mnist':
    train_loader_source = get_data_loader(args, True, 'mnist')
    train_loader_target = get_data_loader(args, True, 'fashionMnist')
elif args.source == 'fashionMnist':
    train_loader_source = get_data_loader(args, True, 'mnist')
    train_loader_target = get_data_loader(args, True, 'mnist')
else:
    raise Exception('args.source does not defined')

if not (args.resume and os.path.isfile(args.model_target_path)):
    print('Creating new target model')
    model_target = SourceModel()
else:
    print('Loading target model from {}'.format(args.model_target_path))
    model_target = torch.load(args.model_target_path)

if not (args.resume and os.path.isfile(args.model_source_path)):
    print('Creating new source model')
    model_source = SourceModel()
else:
    print('Loading source model from {}'.format(args.model_source_path))
    model_source = torch.load(args.model_source_path)
discriminator_model = Discriminator(20, 20)

if args.cuda:
    model_target.cuda()
    model_source.cuda()
    discriminator_model.cuda()

# target_optimizer_encoder_params = [{'params': model_target.fc1.parameters()}, {'params': model_target.fc2.parameters()}]
target_optimizer = optim.Adam(model_target.parameters(), lr=args.lr)
# target_optimizer_encoder = optim.Adam(target_optimizer_encoder_params, lr=args.lr)
source_optimizer = optim.Adam(model_source.parameters(), lr=args.lr)
d_optimizer = optim.Adam(discriminator_model.parameters(), lr=args.lr)

criterion = nn.BCELoss()

if args.source == 'mnist':
    tests = Tests(model_source, model_target, classifyMNIST, 'mnist', 'fashionMnist', args, graph)
elif args.source == 'fashionMnist':
    tests = Tests(model_source, model_target, classifyMNIST, 'mnist', 'mnist', args, graph)
else:
    raise Exception('args.source does not defined')


def reset_grads():
    model_target.zero_grad()
    model_source.zero_grad()
    discriminator_model.zero_grad()


running_counter = 0
overall_accuracy = 0.
for epoch in range(1, args.epochs + 1):
    print('---- Epoch {} ----'.format(epoch))
    model_source.train()
    model_target.train()
    discriminator_model.train()

    # ---------- Train --------------
    for i, ((source_input, _), (target_input, _)) in enumerate(zip(train_loader_source, train_loader_target)):
        running_counter += 1
        if source_input.size()[0] is not target_input.size()[0]:
            continue
        source_input = Variable(source_input)
        target_input = Variable(target_input)
        size = source_input.size()[0]
        ones = Variable(torch.ones(size))
        zeros = Variable(torch.zeros(size))

        if args.cuda:
            source_input = source_input.cuda()
            target_input = target_input.cuda()
            ones = ones.cuda()
            zeros = zeros.cuda()

        # Train generators
        reset_grads()
        decode_s, mu_s, logvar_s, z_s = model_source(source_input)
        s_loss_generator = source_loss(decode_s, source_input, mu_s, logvar_s, args.batch_size)

        decode_t, mu_t, logvar_t, z_t = model_target(target_input)
        t_loss_generator = source_loss(decode_t, target_input, mu_t, logvar_t, args.batch_size)

        # Train encoder
        d_fake_t = discriminator_model(z_t)[:, 0]
        t_loss_discriminator = criterion(d_fake_t, ones)
        if args.one_sided:
            t_loss = t_loss_discriminator
        else:
            t_loss = t_loss_generator + args.h_tg * t_loss_discriminator

        d_fake_s = discriminator_model(z_s)[:, 0]
        s_loss_discriminator = criterion(d_fake_s, ones)

        if args.apply_source_to_discriminator:
            s_loss = s_loss_generator + s_loss_discriminator
        else:
            s_loss = s_loss_generator

        s_loss.backward()
        source_optimizer.step()
        t_loss.backward()
        target_optimizer.step()

        # reset_grads()
        # Train Discriminator
        # z_s = z_s.detach()
        _, _, _, z_s = model_source(source_input)
        d_real_decision = discriminator_model(z_s)[:, 0]
        d_real_error = criterion(d_real_decision, ones)  # ones = true
        d_real_error.backward()

        # z_t = z_t.detach()
        _, _, _, z_t = model_target(target_input)
        d_fake_decision = discriminator_model(z_t)[:, 0]
        d_fake_error = criterion(d_fake_decision, zeros)  # zeros = fake
        d_fake_error.backward()
        d_optimizer.step()

        # for p in discriminator_model.parameters():
        #     p.data.clamp_(-0.1, 0.1)

        graph.last1 = s_loss_generator.data[0]
        graph.last2 = t_loss_generator.data[0]
        graph.last3 = 0#s_loss_discriminator.data[0]
        graph.last4 = 0#t_loss_discriminator.data[0]
        graph.last5 = 0#d_real_error.data[0]
        graph.last6 = 0#d_fake_error.data[0]
        graph.add_point(running_counter)

    # ---------- Tests --------------
    tests.source_to_target_test()
    tests.args.one_sided = not tests.args.one_sided
    tests.source_to_target_test()
    tests.args.one_sided = not tests.args.one_sided
    tests.gaussian_input()
    tests.tsne()
    if not args.one_sided:
        tests.reconstruction(epoch)
        certain, sparse = tests.test_matching()
        accuracy = certain + sparse
        print('certain: {}, sparse: {}, all: {} old max: {}'.format(certain, sparse, accuracy, overall_accuracy))
        if epoch > 10 and accuracy > overall_accuracy:
            overall_accuracy = accuracy
    print('saving mnist model')
    if not os.path.isdir('results/{}'.format(args.graph_name)):
        os.mkdir('results/{}'.format(args.graph_name))
    torch.save(model_source, 'results/{}/model_source.pt'.format(args.graph_name))
    torch.save(model_target, 'results/{}/model_target.pt'.format(args.graph_name))

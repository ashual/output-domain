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
    train_loader_source = get_data_loader(args, True, 'fashionMnist')
    train_loader_target = get_data_loader(args, True, 'mnist')
else:
    raise Exception('args.source does not defined')

if not (args.resume and os.path.isfile(args.model_target_path)):
    print('Creating new target model')
    model_target = TargetModel()
else:
    print('Loading target model from {}'.format(args.model_target_path))
    model_target = torch.load(args.model_target_path)

if not (args.resume and os.path.isfile(args.model_source_path)):
    print('Creating new source model')
    model_source = SourceModel()
else:
    print('Loading source model from'.format(args.model_source_path))
    model_source = torch.load(args.model_source_path)
discriminator_model = Discriminator(20, 20)

if args.cuda:
    model_target.cuda()
    model_source.cuda()
    discriminator_model.cuda()

mnist_optimizer_encoder_params = [{'params': model_target.fc1.parameters()}, {'params': model_target.fc2.parameters()}]
target_optimizer = optim.Adam(model_target.parameters(), lr=args.lr)
mnist_optimizer_encoder = optim.Adam(mnist_optimizer_encoder_params, lr=args.lr)
source_optimizer = optim.Adam(model_source.parameters(), lr=args.lr)
d_optimizer = optim.Adam(discriminator_model.parameters(), lr=args.lr)

criterion = nn.BCELoss()

if args.source == 'mnist':
    tests = Tests(model_source, model_target, classifyMNIST, 'mnist', 'fashionMnist', args, graph)
elif args.source == 'fashionMnist':
    tests = Tests(model_source, model_target, classifyMNIST, 'fashionMnist', 'mnist', args, graph)
else:
    raise Exception('args.source does not defined')


def reset_grads():
    model_target.zero_grad()
    model_source.zero_grad()
    discriminator_model.zero_grad()
    target_optimizer.zero_grad()
    mnist_optimizer_encoder.zero_grad()
    source_optimizer.zero_grad()
    d_optimizer.zero_grad()


running_counter = 0
overall_accuracy = 0.
for epoch in range(1, args.epochs + 1):
    print('---- Epoch {} ----'.format(epoch))
    model_source.train()
    model_target.train()
    discriminator_model.train()

    # ---------- Train --------------
    for i, ((source_input, source_labels), (target_input, target_labels)) in enumerate(
            zip(train_loader_source, train_loader_target)):
        running_counter += 1
        break
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
        s_loss = source_loss(decode_s, source_input, mu_s, logvar_s, args)
        s_loss.backward()
        source_optimizer.step()

        decode_t, z_t = model_target(target_input)
        t_loss_generator = target_loss(decode_t, target_input)
        if not args.one_sided:
            t_loss_generator.backward()
            target_optimizer.step()

        # Train Discriminator
        reset_grads()
        z_s = z_s.detach()
        d_real_decision = discriminator_model(z_s)[:, 0]
        d_real_error = criterion(d_real_decision, ones)  # ones = true
        d_real_error.backward()

        z_t = z_t.detach()
        d_fake_decision = discriminator_model(z_t)[:, 0]
        d_fake_error = criterion(d_fake_decision, zeros)  # zeros = fake
        d_fake_error.backward()
        d_optimizer.step()

        # for p in discriminator_model.parameters():
        #     p.data.clamp_(-0.1, 0.1)

        # Train encoder
        reset_grads()
        z_m = model_target.encoder_only(target_input)
        d_fake_m = discriminator_model(z_m)[:, 0]
        m_loss_discriminator = criterion(d_fake_m, ones)
        m_loss_discriminator.backward()
        mnist_optimizer_encoder.step()

        graph.last1 = s_loss.data[0]
        graph.last2 = t_loss_generator.data[0]
        graph.last3 = d_real_error.data[0]
        graph.last4 = d_fake_error.data[0]
        graph.last5 = m_loss_discriminator.data[0]
        graph.add_point(running_counter, 'mnist encoder')

    # ---------- Tests --------------
    tests.reconstruction(epoch)
    tests.source_to_target_test(epoch)
    certain, sparse = tests.test_matching(epoch)
    accuracy = certain + sparse
    print('certain: {}, sparse: {}, all: {} old max: {}'.format(certain, sparse, accuracy, overall_accuracy))
    if accuracy > overall_accuracy:
        overall_accuracy = accuracy
        print('saving mnist model')

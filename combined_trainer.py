from __future__ import print_function

import os.path

import torch.utils.data
from torch import nn
from torch import optim
from torch.autograd import Variable

from data_loader import get_data_loader
from utils.mnist_classifier.classify import ClassifyMNIST
from combined_tests import Tests
from models.fc_model import CoVAE32x32
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

if not (args.resume and os.path.isfile(args.model_combined_path)):
    print('Creating new combined model')
    model_combined = CoVAE32x32()
    source_resume = False
else:
    print('Loading source model from {}'.format(args.model_combined_path))
    model_source = torch.load(args.model_combined_path)
    source_resume = True

discriminator_model = Discriminator(20, 20)

if args.cuda:
    discriminator_model.cuda()
    model_combined.cuda()
    model_combined.xy = model_combined.xy.cuda()

combined_optimizer = optim.Adam(model_combined.parameters(), lr=args.lr)
d_optimizer = optim.Adam(discriminator_model.parameters(), lr=args.lr)

criterion = nn.BCELoss()
ll_criterion = nn.MSELoss()

if args.source == 'mnist':
    tests = Tests(model_combined, classifyMNIST, 'mnist', 'fashionMnist', args, graph)
elif args.source == 'fashionMnist':
    tests = Tests(model_combined, classifyMNIST, 'fashionMnist', 'mnist', args, graph)
else:
    raise Exception('args.source does not defined')


def reset_grads():
    discriminator_model.zero_grad()
    model_combined.zero_grad()


running_counter = 0
overall_accuracy = 0.
for epoch in range(1, args.epochs + 1):
    print('---- Epoch {} ----'.format(epoch))
    model_combined.train()
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
        x_aa, x_ba, x_ab, x_bb, [codes] = model_combined(source_input, target_input)
        s_loss_generator = ll_criterion(source_input, x_aa)
        s_loss = s_loss_generator
        # decode_s, mu_s, logvar_s, z_s = model_source(source_input)
        # s_loss_generator = source_loss(decode_s, source_input, mu_s, logvar_s, args.batch_size)

        # decode_t, mu_t, logvar_t, z_t = model_target(target_input)
        t_loss_generator = ll_criterion(target_input, x_bb)
        t_loss = t_loss_generator
        # t_loss_generator = source_loss(decode_t, target_input, mu_t, logvar_t, args.batch_size)

        # Train encoder
        # d_fake_t = discriminator_model(z_t)[:, 0]
        # t_loss_discriminator = criterion(d_fake_t, ones)
        # if args.one_sided:
        #     t_loss = t_loss_discriminator
        # else:
        #     t_loss = t_loss_generator + args.h_tg * t_loss_discriminator

        # d_fake_s = discriminator_model(z_s)[:, 0]
        # s_loss_discriminator = criterion(d_fake_s, ones)
        #
        # if args.apply_source_to_discriminator:
        #     s_loss = s_loss_generator + s_loss_discriminator
        # else:
        #     s_loss = s_loss_generator

        if not source_resume:
            s_loss.backward()
            combined_optimizer.step()
        t_loss.backward()
        combined_optimizer.step()

        # reset_grads()
        # Train Discriminator
        # z_s = z_s.detach()
        # _, _, _, z_s = model_source(source_input)
        # d_real_decision = discriminator_model(z_s)[:, 0]
        # d_real_error = criterion(d_real_decision, ones)  # ones = true
        # d_real_error.backward()

        # z_t = z_t.detach()
        # _, _, _, z_t = model_target(target_input)
        # d_fake_decision = discriminator_model(z_t)[:, 0]
        # d_fake_error = criterion(d_fake_decision, zeros)  # zeros = fake
        # d_fake_error.backward()
        # if t_loss_discriminator.data[0] < 0.5 or d_fake_error.data[0] > 0.3 or d_real_error.data[0] > 0.3:
        #     d_optimizer.step()
        #
        # for p in discriminator_model.parameters():
        #     p.data.clamp_(-0.1, 0.1)

        graph.last1 = s_loss_generator.data[0]
        graph.last2 = t_loss_generator.data[0]
        graph.last3 = 0.  # s_loss_discriminator.data[0]
        graph.last4 = 0. #t_loss_discriminator.data[0]
        graph.last5 = 0. #d_real_error.data[0]
        graph.last6 = 0. #d_fake_error.data[0]
        graph.add_point(running_counter)

    # ---------- Tests --------------
    # tests.source_to_target_test()
    # tests.args.one_sided = not tests.args.one_sided
    # tests.source_to_target_test()
    # tests.args.one_sided = not tests.args.one_sided
    # tests.gaussian_input()
    # tests.tsne()
    if not args.one_sided:
        tests.reconstruction(epoch)
    #     certain, sparse = tests.test_matching()
    #     accuracy = certain + sparse
    #     print('certain: {}, sparse: {}, all: {} old max: {}'.format(certain, sparse, accuracy, overall_accuracy))
    #     if epoch > 10 and accuracy > overall_accuracy:
    #         overall_accuracy = accuracy
    # print('saving mnist model')
    # if not os.path.isdir('results/{}'.format(args.graph_name)):
    #     os.mkdir('results/{}'.format(args.graph_name))
    # torch.save(model_source, 'results/{}/model_source.pt'.format(args.graph_name))
    # torch.save(model_target, 'results/{}/model_target.pt'.format(args.graph_name))

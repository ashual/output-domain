from __future__ import print_function

import os.path

import numpy as np
import torch
import torch.utils.data
from torch import optim
from torch.autograd import Variable
from torch import nn

from torchvision.utils import save_image
from vae.data_loader import get_loader
from vae.graph import Graph
from vae.discriminator import Discriminator
from vae.loss import target_loss
from vae.target_model import VAE as TargetModel
from vae.source_model import VAE as SourceModel
from vae.options import load_arguments

args = load_arguments()
SAVED_MODEL_SOURCE_PATH = '{}/{}.pt'.format(args.save_path, 'source')
SAVED_MODEL_TARGET_PATH = '{}/{}.pt'.format(args.save_path, 'target')

lr = 1e-3
graph = Graph()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader_target, train_loader_source = get_loader(args, train=True)
test_loader_target, test_loader_source = get_loader(args, train=False)

test_loader_source_iter = iter(test_loader_source)
test_loader_target_iter = iter(test_loader_target)

criterion = nn.BCELoss()


def test_reconstruction(epoch):
    target_model.eval()
    print('-- test epoch {} --'.format(epoch))
    test_loss = 0.
    for i, (data, _) in enumerate(test_loader_target):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        decode_t, mu_t, logvar_t, _ = target_model(data)
        test_loss += target_loss(decode_t, data, mu_t, logvar_t, args).data[0]

        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], decode_t.view(args.batch_size, 3, args.image_size, args.image_size)[:n]])
            graph.add_images(comparison.data.cpu().numpy(), n, title='recon target epoch'.format(epoch))
        break
            # save_image(comparison.data.cpu(), 'results/recon_{}.png'.format(epoch), nrow=n)
    # test_loss /= len(test_loader_target.dataset)
    print('====> Epoch {}, Target reconstruction loss: {:.6f}'.format(epoch, test_loss))


def transfer(epoch):
    source_iter = iter(test_loader_source)
    sample, labels = source_iter.next()
    for idx in range(10):
        one_digit = np.where(labels.numpy() == idx)[0]
        sample_digit = sample.numpy()[one_digit]
        sample_digit_torch = torch.FloatTensor(sample_digit)
        sample_digit = Variable(sample_digit_torch)
        # save_image(sample_digit.data.view(len(sample_digit), 1, 28, 28),
        #            'results/{}_sample_{}_{}.png'.format('EXAMPLE_Fashion_MNIST', epoch, idx))
        if args.cuda:
            sample_digit = sample_digit.cuda()
        sample_digit = source_model(sample_digit.view(-1, 784))
        sample_digit = target_model.decode(sample_digit)
        graph.add_images(sample_digit_torch.numpy(), title='source epoch {}'.format(epoch))
        graph.add_images(sample_digit.view(-1, 3, args.image_size, args.image_size).data.cpu().numpy(),
                         title='target epoch {}'.format(epoch))
        # save_image(concat_data.view(len(sample_digit) * 2, 3, 28, 28),
        #            'results/source_to_target_{}_{}.png'.format(epoch, idx), nrow=len(sample_digit))


def train_target_generator():
    target_model.train()
    for epoch in range(1, args.epochs + 1):
        print('---- Epoch {} ----'.format(epoch))
        for target_batch, _ in train_loader_target:
            reset_grads()
            target_batch = Variable(target_batch)
            if args.cuda:
                target_batch = target_batch.cuda()

            decode_t, mu_t, logvar_t, _ = target_model(target_batch)
            t_loss = target_loss(decode_t, target_batch, mu_t, logvar_t, args)
            t_loss.backward()
            target_optimizer.step()
        torch.save(target_model, SAVED_MODEL_TARGET_PATH)
        test_reconstruction(epoch)


def train_source_generator_and_discriminator():
    source_model.train()
    discriminator_model.train()
    for epoch in range(1, args.epochs + 1):
        print('---- Epoch {} ----'.format(epoch))
        train_loader_target_iter = iter(train_loader_target)
        train_loader_source_iter = iter(train_loader_source)
        counter = 0
        times = 0
        running_counter = 0
        while True:
            try:
                target_batch, _ = train_loader_target_iter.next()
                source_batch, _ = train_loader_source_iter.next()
                if len(source_batch) != len(target_batch):
                    continue
                target_batch = Variable(target_batch)
                source_batch = Variable(source_batch)
                if args.cuda:
                    target_batch = target_batch.cuda()
                    source_batch = source_batch.cuda()
            except StopIteration:
                break

            reset_grads()

            # Train Discriminator
            if counter % 2 == 0:
                counter = train_discriminator(source_batch, target_batch, running_counter, counter)

            if counter % 2 == 1:
                times, counter = train_source_generator(source_batch, running_counter, times, counter)
            running_counter += 1
        torch.save(source_model, SAVED_MODEL_SOURCE_PATH)
        transfer(epoch)


def train_discriminator(source_b, target_b, running_counter, counter):
    _, mu_f, _, _ = target_model(target_b)
    mu_f = mu_f.detach()
    z_s = source_model(source_b)
    z_s = z_s.detach()

    d_real_decision = discriminator_model(mu_f)[:, 0]
    size = d_real_decision.size()[0]
    ones = Variable(torch.ones(size))
    zeros = Variable(torch.zeros(size))
    if args.cuda:
        ones = ones.cuda()
        zeros = zeros.cuda()

    d_real_error = criterion(d_real_decision, ones)  # ones = true
    d_real_error.backward()

    d_fake_decision = discriminator_model(z_s)[:, 0]
    d_fake_error = criterion(d_fake_decision, zeros)  # zeros = fake
    d_fake_error.backward()
    d_optimizer.step()

    graph.last3 = d_real_error.data[0]
    graph.last4 = d_fake_error.data[0]
    graph.add_point(running_counter, 'discriminator')
    # print('d lost real {:.4f}'.format(d_real_error.data[0]))
    # print('d lost fake {:.4f}'.format(d_fake_error.data[0]))

    if d_real_error.data[0] < 0.3 and d_fake_error.data[0] < 0.3:
        counter += 1
    return counter


def train_source_generator(mnist_batch, running_counter, times, counter):
    z_m = source_model(mnist_batch)
    d_fake_m = discriminator_model(z_m)[:, 0]
    size = d_fake_m.size()[0]

    ones = Variable(torch.ones(size))
    if args.cuda:
        ones = ones.cuda()
    m_loss_discriminator = criterion(d_fake_m, ones)
    m_loss_discriminator.backward()
    source_optimizer.step()
    graph.last5 = m_loss_discriminator.data[0]
    graph.add_point(running_counter, 'mnist encoder')
    # print('mnist loss discriminator {:.4f}'.format(m_loss_discriminator.data[0]))
    if times >= 10:
        counter += 1
        times = 0
    else:
        times += 1
    return times, counter


def reset_grads():
    if source_model:
        source_model.zero_grad()
    if target_model:
        target_model.zero_grad()
    if discriminator_model:
        discriminator_model.zero_grad()
    if source_optimizer:
        source_optimizer.zero_grad()
    if target_optimizer:
        target_optimizer.zero_grad()
    if d_optimizer:
        d_optimizer.zero_grad()


if args.resume and os.path.isfile(SAVED_MODEL_TARGET_PATH):
    print('loading target model')
    target_model = torch.load(SAVED_MODEL_TARGET_PATH)
    target_optimizer = None
else:
    target_model = TargetModel()
    if args.cuda:
        target_model = target_model.cuda()
    target_optimizer = optim.Adam(target_model.parameters(), lr=lr)

if args.resume and os.path.isfile(SAVED_MODEL_SOURCE_PATH):
    print('loading source model')
    source_model = torch.load(SAVED_MODEL_SOURCE_PATH)
    discriminator_model = Discriminator(100, 100)
else:
    source_model = SourceModel()
    discriminator_model = Discriminator(100, 100)
    if args.cuda:
        source_model = source_model.cuda()
        discriminator_model = discriminator_model.cuda()
    source_optimizer = optim.Adam(source_model.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator_model.parameters(), lr=lr)

if not (args.resume and os.path.isfile(SAVED_MODEL_TARGET_PATH)):
    print('---- Training target generator ----')
    train_target_generator()
print('---- Training source generator ----')
train_source_generator_and_discriminator()

from __future__ import print_function

import os.path

import numpy as np
import torch
import torch.utils.data
from torch import optim
from torch.autograd import Variable

from torchvision import datasets, transforms
from torchvision.utils import save_image
from vae.discriminator import Discriminator
from vae.loss import generator_loss_function, discriminator_loss_function
from vae.model import VAE
from vae.options import load_arguments

SAVED_MODEL_MNIST_PATH = 'saved/MNIST.pt'
SAVED_MODEL_FASHION_MNIST_PATH = 'saved/FashionMNIST.pt'
lr = 1e-3

args = load_arguments()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader_mnist = torch.utils.data.DataLoader(
    datasets.MNIST('../data_MNIST', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader_mnist = torch.utils.data.DataLoader(
    datasets.MNIST('../data_MNIST', train=False, transform=transforms.ToTensor()), batch_size=args.batch_size,
    shuffle=True, **kwargs)

train_loader_mnist_iter = iter(train_loader_mnist)
test_loader_mnist_iter = iter(test_loader_mnist)

train_loader_fashion_mnist = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data_FashionMNIST', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader_fashion_mnist = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data_FashionMNIST', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

if args.resume and os.path.isfile(SAVED_MODEL_MNIST_PATH):
    model_mnist = torch.load(SAVED_MODEL_MNIST_PATH)
else:
    model_mnist = VAE()

if False and args.resume and os.path.isfile(SAVED_MODEL_FASHION_MNIST_PATH):
    model_fashion_mnist = torch.load(SAVED_MODEL_FASHION_MNIST_PATH)
else:
    model_fashion_mnist = VAE()

discriminator_model = Discriminator()

if args.cuda:
    model_mnist.cuda()
    model_fashion_mnist.cuda()
    discriminator_model.cuda()

optimizer = optim.Adam(model_fashion_mnist.parameters(), lr=lr)
D_optimizer = optim.Adam(discriminator_model.parameters(), lr=lr)


def train(epoch):
    model_fashion_mnist.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader_fashion_mnist):
        train_discriminator(data)
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model_fashion_mnist(data)
        if batch_idx % 2 == 0:
            loss = generator_loss_function(recon_batch, data, mu, logvar, args)
            loss.backward()
        else:
            d_loss = discriminator_loss_function(discriminator_model(mu))
            d_loss.backward()

        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader_fashion_mnist.dataset),
                                                                           100. * batch_idx / len(
                                                                               train_loader_fashion_mnist),
                                                                           loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader_fashion_mnist.dataset)))


def test(epoch):
    model_fashion_mnist.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader_fashion_mnist):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model_fashion_mnist(data)
        test_loss += generator_loss_function(recon_batch, data, mu, logvar, args).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(), 'results/{}_reconstruction_{}.png'.format('FASHION_MNIST', epoch), nrow=n)

    test_loss /= len(test_loader_fashion_mnist.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def train_discriminator(data):
    data = Variable(data)

    # Sample data
    for index, (mnist_data, _) in enumerate(train_loader_mnist):
        mnist_data = Variable(mnist_data)

        # Dicriminator forward-loss-backward-update
        z_data = model_fashion_mnist.encoder_only(data)
        z_mnist_data = model_mnist.encoder_only(mnist_data)
        D_real = discriminator_model(z_mnist_data)
        D_fake = discriminator_model(z_data)

        D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
        # print('loss D_real {:.4f}, loss D_fake {:.4f}'.format(torch.mean(D_real).data.numpy()[0],
        #                                                       torch.mean(D_fake).data.numpy()[0]))
        D_loss.backward()
        D_optimizer.step()

        # Weight clipping
        for p in discriminator_model.parameters():
            p.data.clamp_(-0.01, 0.01)

            # Housekeeping - reset gradient
            D_optimizer.zero_grad()
        if index >= 5:
            break


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    # sample = Variable(torch.randn(64, 20))
    sample, labels = test_loader_mnist_iter.next()
    for idx in range(9):
        one_digit = np.where(labels.numpy() == idx)[0]
        sample_digit = sample.numpy()[one_digit]
        sample_digit = Variable(torch.FloatTensor(sample_digit))
        save_image(sample_digit.data.view(len(sample_digit), 1, 28, 28),
                   'results/{}_sample_{}_{}.png'.format('EXAMPLE_MNIST', epoch, idx))
        if args.cuda:
            sample_digit = sample_digit.cuda()
        sample_digit, _ = model_mnist.encode(sample_digit.view(-1, 784))
        sample_digit = model_fashion_mnist.decode(sample_digit).cpu()
        save_image(sample_digit.data.view(len(sample_digit), 1, 28, 28),
                   'results/{}_sample_{}_{}.png'.format('FASHION_MNIST', epoch, idx))
    torch.save(model_fashion_mnist, SAVED_MODEL_FASHION_MNIST_PATH)

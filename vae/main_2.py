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
from vae.loss import complex_loss_function as fashion_loss
from vae.loss import simple_loss_function as mnist_loss
from vae.loss import discriminator_loss_function
from vae.complex_model import VAE as FashionMnistModel
from vae.simple_model import VAE as MnistModel
from vae.options import load_arguments


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

test_loader_mnist_iter = iter(test_loader_mnist)

train_loader_fashion_mnist = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data_FashionMNIST', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader_fashion_mnist = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data_FashionMNIST', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader_fashion_mnist_iter = iter(test_loader_fashion_mnist)


model_mnist = MnistModel()
model_fashion_mnist = FashionMnistModel()
discriminator_model = Discriminator()

if args.cuda:
    model_mnist.cuda()
    model_fashion_mnist.cuda()
    discriminator_model.cuda()

mnist_optimizer = optim.Adam(model_mnist.parameters(), lr=lr)
fashion_optimizer = optim.Adam(model_fashion_mnist.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator_model.parameters(), lr=lr)


def train(epoch):
    model_fashion_mnist.train()
    train_loss = 0
    discriminator_loss = 0
    counter = 0
    d_loss = None
    train_loader_mnist_iter = iter(train_loader_mnist)

    for batch_idx, (data, _) in enumerate(train_loader_fashion_mnist):
        #train_discriminator(data, train_loader_mnist_iter)
        # if batch_idx % 3 != 0:
        #     continue
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch = model_fashion_mnist(data)
        z = model_fashion_mnist.encoder_only(data)
        d_loss = None # discriminator_loss_function(discriminator_model(z))
        if batch_idx % 10 != 9:
            loss = generator_loss_function2(recon_batch, data)
            loss.backward()
        # else:
        #
        #     d_loss.backward()

        train_loss += loss.data[0]
        discriminator_loss += 0. #d_loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} d loss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data),
                                                                           len(train_loader_fashion_mnist.dataset),
                                                                           100. * batch_idx / len(train_loader_fashion_mnist),
                                                                           loss.data[0] / len(data),
                                                                           0 / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader_fashion_mnist.dataset)))


def test(epoch):
    model_fashion_mnist.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader_mnist):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, _ = model_mnist(data)
        test_loss += mnist_loss(recon_batch, data).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(), 'results/{}_reconstruction_{}.png'.format('MNIST', epoch), nrow=n)

    for i, (data, _) in enumerate(test_loader_fashion_mnist):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, _, _ = model_fashion_mnist(data)
        test_loss += mnist_loss(recon_batch, data).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(), 'results/{}_reconstruction_{}.png'.format('FASHION_MNIST', epoch), nrow=n)
    test_loss /= len(test_loader_mnist.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def train_discriminator(data, train_loader_mnist_iter):
    data = Variable(data)

    # Sample data
    mnist_data, _ = train_loader_mnist_iter.next()
    D_optimizer.zero_grad()
    mnist_data = Variable(mnist_data)

    # Dicriminator forward-loss-backward-update
    z_data = model_fashion_mnist.encoder_only(data)
    z_mnist_data = model_mnist.encoder_only(mnist_data)
    D_real = discriminator_model(z_mnist_data)
    D_fake = discriminator_model(z_data)

    D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
    if D_loss.data.numpy()[0] < -1.:
        print('.', end='')
        return
    print('T', end='')
    D_loss.backward()
    D_optimizer.step()

    # Weight clipping
    # for p in discriminator_model.parameters():
    #     p.data.clamp_(-0.01, 0.01)

    # Housekeeping - reset gradient

    # print(
    #     'loss D_real {:.4f}, loss D_fake {:.4f}, total loss: {:.4f}'.format(torch.mean(D_real).data.numpy()[0],
    #                                                                         torch.mean(D_fake).data.numpy()[0],
    #                                                                         D_loss.data.numpy()[0]))


for epoch in range(1, args.epochs + 1):
    print('---- Epoch {} ----'.format(epoch))
    model_fashion_mnist.train()
    model_mnist.train()
    discriminator_model.train()

    # ---------- Train --------------
    train_loader_fashion_mnist_iter = iter(train_loader_fashion_mnist)
    train_loader_mnist_iter = iter(train_loader_mnist)
    while True:
        try:
            fashion_batch, _ = train_loader_fashion_mnist_iter.next()
            mnist_batch, _ = train_loader_mnist_iter.next()
            fashion_batch = Variable(fashion_batch)
            mnist_batch = Variable(mnist_batch)
            if args.cuda:
                fashion_batch = fashion_batch.cuda()
                mnist_batch = mnist_batch.cuda()
        except StopIteration:
            break

        mnist_optimizer.zero_grad()
        fashion_optimizer.zero_grad()
        d_optimizer.zero_grad()

        decode_f, mu_f, logvar_f = model_fashion_mnist(fashion_batch)
        f_loss = fashion_loss(decode_f, fashion_batch, mu_f, logvar_f, args)
        f_loss.backward()
        fashion_optimizer.step()

        decode_m, z_m = model_mnist(mnist_batch)

        d_real = discriminator_model(Variable(mu_f.data, requires_grad=True))
        d_fake = discriminator_model(Variable(z_m.data, requires_grad=True))

        d_loss = -(torch.mean(d_real) - torch.mean(d_fake))
        d_loss.backward()
        d_optimizer.step()
        d_optimizer.zero_grad()

        d_fake_m = discriminator_model(z_m)

        m_loss_discriminator = discriminator_loss_function(d_fake_m)
        m_loss_generator = mnist_loss(decode_m, mnist_batch)
        m_loss = m_loss_generator + 0.001 * m_loss_discriminator
        m_loss.backward()
        mnist_optimizer.step()

        for p in discriminator_model.parameters():
            p.data.clamp_(-1., 1.)

    print('fashion lost {}'.format(f_loss.data[0]))
    print('mnist lost {} from d {}'.format(m_loss_generator.data[0], m_loss_discriminator.data[0]))
    print('d lost real {}'.format(torch.mean(d_real).data[0]))
    print('d lost fake {}'.format(torch.mean(d_fake).data[0]))
    # ---------- Test --------------
    # sample = Variable(torch.randn(64, 20))
    test(epoch)
    sample, labels = test_loader_fashion_mnist_iter.next()
    for idx in range(9):
        one_digit = np.where(labels.numpy() == idx)[0]
        sample_digit = sample.numpy()[one_digit]
        sample_digit = Variable(torch.FloatTensor(sample_digit))
        # save_image(sample_digit.data.view(len(sample_digit), 1, 28, 28),
        #            'results/{}_sample_{}_{}.png'.format('EXAMPLE_Fashion_MNIST', epoch, idx))
        if args.cuda:
            sample_digit = sample_digit.cuda()
        sample_digit = model_fashion_mnist.encoder_only(sample_digit.view(-1, 784))
        sample_digit = model_mnist.decode(sample_digit).cpu()
        save_image(sample_digit.data.view(len(sample_digit), 1, 28, 28),
                   'results/{}_sample_{}_{}.png'.format('MNIST', epoch, idx))
    #torch.save(model_fashion_mnist, SAVED_MODEL_FASHION_MNIST_PATH)

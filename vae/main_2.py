from __future__ import print_function

import os.path

import numpy as np
import torch
import torch.utils.data
from torch import optim
from torch.autograd import Variable
from torch import nn

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

mnist_optimizer_encoder_params = [{'params': model_mnist.fc1.parameters()},
                                  {'params':  model_mnist.fc2.parameters()}]
mnist_optimizer = optim.Adam(model_mnist.parameters(), lr=lr)
mnist_optimizer_encoder = optim.Adam(mnist_optimizer_encoder_params, lr=lr)
# mnist_optimizer_encoder = optim.Adam([model_mnist.fc1, model_mnist.fc2], lr=lr)
fashion_optimizer = optim.Adam(model_fashion_mnist.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator_model.parameters(), lr=lr)

criterion = nn.BCELoss()

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


def reset_grads():
    model_mnist.zero_grad()
    model_fashion_mnist.zero_grad()
    discriminator_model.zero_grad()
    mnist_optimizer.zero_grad()
    mnist_optimizer_encoder.zero_grad()
    fashion_optimizer.zero_grad()
    d_optimizer.zero_grad()

for epoch in range(1, args.epochs + 1):
    print('---- Epoch {} ----'.format(epoch))
    model_fashion_mnist.train()
    model_mnist.train()
    discriminator_model.train()

    # ---------- Train --------------
    train_loader_fashion_mnist_iter = iter(train_loader_fashion_mnist)
    train_loader_mnist_iter = iter(train_loader_mnist)
    counter = 0
    times = 0
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

        reset_grads()

        # Train generators
        if counter % 3 == 0:
            decode_f, mu_f, logvar_f = model_fashion_mnist(fashion_batch)
            f_loss = fashion_loss(decode_f, fashion_batch, mu_f, logvar_f, args)
            f_loss.backward()
            fashion_optimizer.step()

            decode_m, z_m = model_mnist(mnist_batch)
            m_loss_generator = mnist_loss(decode_m, mnist_batch)
            m_loss_generator.backward()
            mnist_optimizer.step()
            print('fashion lost {:.4f}'.format(f_loss.data[0]))
            print('mnist lost {:.4f}'.format(m_loss_generator.data[0]))
            if times > 100:
                counter += 1
                times = 0
            else:
                times += 1

        # Train Discriminator
        if counter % 3 == 1:
            _, mu_f, _ = model_fashion_mnist(fashion_batch)
            mu_f = mu_f.detach()
            _, z_m = model_mnist(mnist_batch)
            z_m = z_m.detach()

            d_real_decision = discriminator_model(mu_f)
            size = d_real_decision.size()[0]
            d_real_error = criterion(d_real_decision, Variable(torch.ones(size, 1)))  # ones = true
            d_real_error.backward()

            d_fake_decision = discriminator_model(z_m)
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(size, 1)))  # zeros = fake
            d_fake_error.backward()
            d_optimizer.step()
            print('d lost real {:.4f}'.format(torch.mean(d_real_error).data[0]))
            print('d lost fake {:.4f}'.format(torch.mean(d_fake_error).data[0]))
            if torch.mean(d_real_error).data[0] < 0.2 and torch.mean(d_fake_error).data[0] < 0.2:
                counter += 1
            # for p in discriminator_model.parameters():
            #     p.data.clamp_(-1., 1.)

        if counter % 3 == 2:
            _, z_m = model_mnist(mnist_batch)
            d_fake_m = discriminator_model(z_m)
            size = d_fake_m.size()[0]

            m_loss_discriminator = criterion(d_fake_m, Variable(torch.ones(size, 1)))
            m_loss_discriminator.backward()
            mnist_optimizer_encoder.step()
            print('mnist lost discriminator {:.4f}'.format(m_loss_discriminator.data[0]))
            if times > 100:
                counter += 1
                times = 0
            else:
                times += 1

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

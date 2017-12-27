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
from vae.graph import Graph
from vae.discriminator import Discriminator
from vae.loss import complex_loss_function as fashion_loss
from vae.loss import simple_loss_function as mnist_loss
from vae.complex_model import VAE as FashionMnistModel
from vae.simple_model import VAE as MnistModel
from vae.options import load_arguments
from mnist_classifier.classify import ClassifyMNIST
from vae.plot import plot_results, calculate_accuracy

SAVED_MODEL_MNIST_PATH = 'vae/saved/MNIST.pt'
SAVED_MODEL_FASHION_MNIST_PATH = 'vae/saved/FashionMNIST.pt'

lr = 1e-4
graph = Graph()
args = load_arguments()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
classifyMNIST = ClassifyMNIST(args)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader_fashion_mnist = torch.utils.data.DataLoader(
    datasets.MNIST('./data_MNIST', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader_fashion_mnist = torch.utils.data.DataLoader(
    datasets.MNIST('./data_MNIST', train=False, transform=transforms.ToTensor()), batch_size=args.batch_size,
    shuffle=True, **kwargs)


train_loader_mnist = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data_FashionMNIST', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader_mnist = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data_FashionMNIST', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader_mnist_iter = iter(test_loader_mnist)
test_loader_fashion_mnist_iter = iter(test_loader_fashion_mnist)


if args.resume and os.path.isfile(SAVED_MODEL_MNIST_PATH):
    print('loading model mnist')
    model_mnist = torch.load(SAVED_MODEL_MNIST_PATH)
else:
    model_mnist = MnistModel()

if args.resume and os.path.isfile(SAVED_MODEL_FASHION_MNIST_PATH):
    print('loading model fashion mnist')
    model_fashion_mnist = torch.load(SAVED_MODEL_FASHION_MNIST_PATH)
else:
    model_fashion_mnist = FashionMnistModel()
discriminator_model = Discriminator(20, 20)

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


def test_matching():
    n_categories = 10
    confusion = torch.zeros(n_categories, n_categories)
    test_loader_mnist_iter = iter(test_loader_mnist)
    for i in range(20):
        sample, labels = test_loader_mnist_iter.next()
        sample_digit = Variable(sample)
        if args.cuda:
            sample_digit = sample_digit.cuda()
        sample_digit = model_mnist.encoder_only(sample_digit.view(-1, 784))
        sample_digit = model_fashion_mnist.decode(sample_digit).cpu()
        results = classifyMNIST.test(sample_digit)
        for i, label in enumerate(labels):
            confusion[label][results[i]] += 1
    # plot_results(confusion)
    return calculate_accuracy(confusion)


def test(epoch):
    model_fashion_mnist.eval()
    test_loss = 0.
    for i, (data, _) in enumerate(test_loader_mnist):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, _ = model_mnist(data)
        test_loss += mnist_loss(recon_batch, data).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(), 'vae/results/{}_reconstruction_{}.png'.format('MNIST', epoch), nrow=n)
    test_loss /= len(test_loader_mnist.dataset)
    print('====> Test mnist loss: {:.6f}'.format(test_loss))
    test_loss = 0.

    # for i, (data, _) in enumerate(test_loader_fashion_mnist):
    #     if args.cuda:
    #         data = data.cuda()
    #     data = Variable(data, volatile=True)
    #     recon_batch, _, _, _ = model_fashion_mnist(data)
    #     test_loss += mnist_loss(recon_batch, data).data[0]
    #     if i == 0:
    #         n = min(data.size(0), 8)
    #         comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
    #         save_image(comparison.data.cpu(), 'results/{}_reconstruction_{}.png'.format('FASHION_MNIST', epoch), nrow=n)
    # test_loss /= len(test_loader_mnist.dataset)
    # print('====> Test fashion mnist loss: {:.6f}'.format(test_loss))


def reset_grads():
    model_mnist.zero_grad()
    model_fashion_mnist.zero_grad()
    discriminator_model.zero_grad()
    mnist_optimizer.zero_grad()
    mnist_optimizer_encoder.zero_grad()
    fashion_optimizer.zero_grad()
    d_optimizer.zero_grad()


running_counter = 0
overall_accuracy = 0.
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
            # counter += 1

            decode_f, mu_f, logvar_f, _ = model_fashion_mnist(fashion_batch)
            f_loss = fashion_loss(decode_f, fashion_batch, mu_f, logvar_f, args)
            f_loss.backward()
            fashion_optimizer.step()

            # decode_m, z_m = model_mnist(mnist_batch)
            # m_loss_generator = mnist_loss(decode_m, mnist_batch)
            # m_loss_generator.backward()
            # mnist_optimizer.step()

            if running_counter % 100 == 0:
                graph.last1 = f_loss.data[0]
                graph.last2 = 0.  # m_loss_generator.data[0]
                graph.add_point(running_counter, 'generator')
            # print('fashion lost {:.4f}'.format(f_loss.data[0]))
            # print('mnist lost {:.4f}'.format(m_loss_generator.data[0]))
            if times > 1:
                counter += 1
                times = 0
            else:
                times += 1

        # Train Discriminator
        if counter % 3 == 1:
            _, _, _, z_f = model_fashion_mnist(fashion_batch)
            z_f = z_f.detach()
            z_m = model_mnist.encoder_only(mnist_batch)
            z_m = z_m.detach()

            d_real_decision = discriminator_model(z_f)[:, 0]
            size = d_real_decision.size()[0]
            ones = Variable(torch.ones(size))
            zeros = Variable(torch.zeros(size))
            if args.cuda:
                ones = ones.cuda()
                zeros = zeros.cuda()
            d_real_error = criterion(d_real_decision, ones)  # ones = true
            d_real_error.backward()

            d_fake_decision = discriminator_model(z_m)[:, 0]
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
            # for p in discriminator_model.parameters():
            #     p.data.clamp_(-0.1, 0.1)

        if counter % 3 == 2:
            z_m = model_mnist.encoder_only(mnist_batch)
            d_fake_m = discriminator_model(z_m)[:, 0]
            size = d_fake_m.size()[0]

            m_loss_discriminator = criterion(d_fake_m, Variable(torch.ones(size)))
            m_loss_discriminator.backward()
            mnist_optimizer_encoder.step()
            graph.last5 = m_loss_discriminator.data[0]
            graph.add_point(running_counter, 'mnist encoder')
            # print('mnist loss discriminator {:.4f}'.format(m_loss_discriminator.data[0]))
            if times >= 10:
                counter += 1
                times = 0
            else:
                times += 1
        running_counter += 1
    # ---------- Test --------------
    # test(epoch)
    try:
        sample, labels = test_loader_mnist_iter.next()
    except Exception:
        test_loader_mnist_iter = iter(test_loader_mnist)
        sample, labels = test_loader_mnist_iter.next()
    for idx in range(10):
        one_digit = np.where(labels.numpy() == idx)[0]
        sample_digit = sample.numpy()[one_digit]
        if len(sample_digit) == 0:
            continue
        sample_digit_torch = torch.FloatTensor(sample_digit)
        sample_digit = Variable(sample_digit_torch)
        # save_image(sample_digit.data.view(len(sample_digit), 1, 28, 28),
        #            'results/{}_sample_{}_{}.png'.format('EXAMPLE_Fashion_MNIST', epoch, idx))
        if args.cuda:
            sample_digit = sample_digit.cuda()
        sample_digit = model_mnist.encoder_only(sample_digit.view(-1, 784))
        sample_digit = model_fashion_mnist.decode(sample_digit).cpu()
        concat_data = torch.cat((sample_digit_torch.view(-1, 784), sample_digit.data), 0)
        graph.draw(str(idx), concat_data.view(len(sample_digit)*2, 1, 28, 28).cpu().numpy())
        # save_image(concat_data.view(len(sample_digit)*2, 1, 28, 28),
        #            'vae/results/{}_sample_{}_{}.png'.format('MNIST', epoch, idx), nrow=len(sample_digit))
    certain, sparse = test_matching()
    accuracy = certain + sparse
    print('certain: {}, sparse: {}, all: {} old max: {}'.format(certain, sparse, accuracy, overall_accuracy))
    if accuracy > overall_accuracy:
        overall_accuracy = accuracy
        print('saving mnist model')
        torch.save(model_mnist, SAVED_MODEL_MNIST_PATH)
    torch.save(model_fashion_mnist, SAVED_MODEL_FASHION_MNIST_PATH)

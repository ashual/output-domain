from __future__ import print_function

import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable
from torchvision.utils import save_image

from data_loader import get_data_loader
from vae.loss import simple_loss_function as target_loss
from vae.plot import plot_results, calculate_accuracy


class Tests:
    def __init__(self, model_source, model_target, classify_model, source, target, args, graph):
        self.test_loader_source = get_data_loader(args, False, source)
        self.test_loader_target = get_data_loader(args, False, target)
        self.model_source = model_source
        self.model_target = model_target
        self.classify_model = classify_model
        self.args = args
        self.graph = graph
        self.cuda = args.cuda

    def source_to_target_test(self, epoch):
        self.model_source.eval()
        for i, (sample, labels) in enumerate(self.test_loader_target):
            for idx in range(10):
                one_digit = np.where(labels.numpy() == idx)[0]
                sample_digit = sample.numpy()[one_digit]
                if len(sample_digit) == 0:
                    continue
                sample_digit_torch = torch.FloatTensor(sample_digit)
                sample_digit = Variable(sample_digit_torch)
                # save_image(sample_digit.data.view(len(sample_digit), 1, 28, 28),
                #            'results/{}_sample_{}_{}.png'.format('EXAMPLE_Fashion_MNIST', epoch, idx))
                if self.cuda:
                    sample_digit = sample_digit.cuda()
                sample_digit = self.model_target.encoder_only(sample_digit.view(-1, 784))
                sample_digit = self.model_source.decode(sample_digit).cpu()
                concat_data = torch.cat((sample_digit_torch.view(-1, 784), sample_digit.data), 0)
                self.graph.draw(str(idx), concat_data.view(len(sample_digit) * 2, 1, 28, 28).cpu().numpy())
                if False:
                    save_image(concat_data.view(len(sample_digit) * 2, 1, 28, 28),
                               'vae/results/{}_sample_{}_{}.png'.format('MNIST', epoch, idx), nrow=len(sample_digit))

    def test_matching(self, epoch):
        n_categories = 10
        confusion = torch.zeros(n_categories, n_categories).long().cpu()
        for i, (sample, labels) in self.test_loader_target:
            sample_digit = Variable(sample)
            if self.cuda:
                sample_digit = sample_digit.cuda()
            sample_digit = self.model_target.encoder_only(sample_digit.view(-1, 784))
            sample_digit = self.model_source.decode(sample_digit)
            results = self.classify_model.test(sample_digit).cpu()
            for indx, label in enumerate(labels):
                confusion[label][results[indx]] += 1
        plot_results(confusion, epoch)
        return calculate_accuracy(confusion)

    def reconstruction(self, epoch):
        self.model_source.eval()
        test_loss = 0.
        for i, (data, _) in enumerate(self.test_loader_target):
            data = Variable(data, volatile=True)
            if self.cuda:
                data = data.cuda()
            recon_batch, _ = self.model_target(data)
            test_loss += target_loss(recon_batch, data).data[0]
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.data.cpu(), 'vae/results/{}_reconstruction_{}.png'.format('MNIST', epoch), nrow=n)
        test_loss /= len(self.test_loader_target.dataset)
        print('====> Test mnist loss: {:.6f}'.format(test_loss))

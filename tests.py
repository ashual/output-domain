from __future__ import print_function

import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable

from data_loader import get_data_loader
from plot import plot_results, calculate_accuracy
from utils.loss import simple_loss_function


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
        for i, (sample, labels) in enumerate(self.test_loader_source):
            for idx in range(10):
                one_digit = np.where(labels.numpy() == idx)[0]
                sample_digit = sample.numpy()[one_digit]
                if len(sample_digit) == 0:
                    continue
                sample_digit_torch = torch.FloatTensor(sample_digit)
                sample_digit = Variable(sample_digit_torch)
                if self.cuda:
                    sample_digit = sample_digit.cuda()
                sample_digit = self.model_source.encoder_only(sample_digit.view(-1, 784))
                sample_digit = self.model_target.decode(sample_digit).cpu()
                concat_data = torch.cat((sample_digit_torch.view(-1, 784), sample_digit.data), 0)
                self.graph.draw(str(idx), concat_data.view(len(sample_digit) * 2, 1, 28, 28).cpu().numpy())
            break

    def test_matching(self, epoch):
        n_categories = 10
        confusion = torch.zeros(n_categories, n_categories).long().cpu()
        for i, (sample, labels) in enumerate(self.test_loader_target):
            sample_digit = Variable(sample)
            if self.cuda:
                sample_digit = sample_digit.cuda()
            sample_digit = self.model_target.encoder_only(sample_digit.view(-1, 784))
            sample_digit = self.model_source.decode(sample_digit)
            results = self.classify_model.test(sample_digit).cpu()
            for index, label in enumerate(labels):
                confusion[label][results[index]] += 1
        plot_results(confusion, self.graph)
        return calculate_accuracy(confusion)

    def reconstruction(self, epoch):
        self.model_source.eval()
        self.model_target.eval()
        source_loss = 0.
        target_loss = 0.
        for i, ((source, _), (target, _)) in enumerate(zip(self.test_loader_source, self.test_loader_target)):
            source = Variable(source, volatile=True)
            target = Variable(target, volatile=True)
            if self.cuda:
                source = source.cuda()
                target = target.cuda()
            recon_source, _, _, _ = self.model_source(source)
            recon_target, _ = self.model_target(target)

            source_loss += simple_loss_function(recon_source, source).data[0]
            target_loss += simple_loss_function(recon_target, target).data[0]
            if i == 0:
                n = min(source.size(0), 8)
                comparison_source = torch.cat([source[:n], recon_source.view(-1, 1, 28, 28)[:n]])
                n = min(target.size(0), 8)
                comparison_target = torch.cat([target[:n], recon_target.view(-1, 1, 28, 28)[:n]])
                self.graph.draw('reconstruction_source', comparison_source.data.cpu().numpy())
                self.graph.draw('reconstruction_target', comparison_target.data.cpu().numpy())
        source_loss /= len(self.test_loader_source.dataset)
        target_loss /= len(self.test_loader_target.dataset)
        print('====> Epoch: {}, Reconstruction source loss: {:.6f},'
              'Reconstruction target loss: {:.6f}'.format(epoch, source_loss, target_loss))
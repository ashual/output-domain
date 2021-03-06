from __future__ import print_function

import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable

from data_loader import get_data_loader
from plot import plot_results, calculate_accuracy
from utils.loss import simple_loss_function
from utils.tsne import run as run_tsne
import matplotlib.pyplot as plt


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

    def source_to_target_test(self):
        if self.args.one_sided:
            test_loader = self.test_loader_target
            model_target = self.model_source
            model_source = self.model_target
            text = ' t2s'
        else:
            test_loader = self.test_loader_source
            model_source = self.model_source
            model_target = self.model_target
            text = ' s2t'
        self.model_source.eval()
        for i, (sample, labels) in enumerate(test_loader):
            for idx in range(10):
                one_digit = np.where(labels.numpy() == idx)[0]
                sample_digit = sample.numpy()[one_digit]
                if len(sample_digit) == 0:
                    continue
                sample_digit_torch = torch.FloatTensor(sample_digit)
                sample_digit = Variable(sample_digit_torch)
                if self.cuda:
                    sample_digit = sample_digit.cuda()
                sample_digit = model_source.encoder_only(sample_digit.view(-1, 784))
                sample_digit_t = model_target.decode(sample_digit).cpu()
                sample_digit_s = model_source.decode(sample_digit).cpu()
                concat_data = torch.cat((sample_digit_torch.view(-1, 784), sample_digit_t.data, sample_digit_s.data), 0)
                self.graph.draw(str(idx)+text, concat_data.view(len(sample_digit) * 3, 1, 28, 28).cpu().numpy())
            break

    def test_matching(self):
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
            recon_target, _, _, _ = self.model_target(target)

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

    def tsne(self, test_target, model_target):
        all_enc_target = None
        all_t_labels = None
        for i, (target, t_labels) in enumerate(test_target):
            if i == 4:
                break
            target = Variable(target, volatile=True)
            if self.cuda:
                target = target.cuda()
            enc_target = model_target.encoder_only(target).cpu().data
            if i == 0:
                all_enc_target = enc_target
                all_t_labels = t_labels
            else:
                all_enc_target = torch.cat([all_enc_target, enc_target], 0)
                all_t_labels = torch.cat([all_t_labels, t_labels])
        # fig = run_tsne(all_enc_source.numpy(), all_s_labels.numpy())
        # self.graph.draw_figure('source tsne', fig)
        # plt.close(fig)
        fig = run_tsne(all_enc_target.numpy(), all_t_labels.numpy())
        self.graph.draw_figure('target tsne', fig)
        plt.close(fig)


    def gaussian_input(self):
        self.model_source.eval()
        self.model_target.eval()
        sample = Variable(torch.randn(64, 40))
        if self.cuda:
            sample = sample.cuda()
        sample_source = self.model_source.decode(sample).cpu()
        sample_target = self.model_target.decode(sample).cpu()
        self.graph.draw('gaussian source', sample_source.data.view(64, 1, 28, 28).cpu().numpy())
        self.graph.draw('gaussian target', sample_target.data.view(64, 1, 28, 28).cpu().numpy())

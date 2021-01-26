from __future__ import print_function
import datetime
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
from data_loader import *
import models
import numpy as np
from torch.utils import model_zoo
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter
import warnings
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from easydl import *
warnings.filterwarnings("ignore", category=UserWarning)

# Training settings
num_experiments = 1
batch_size = 32
max_iter = 10000
lr = 0.01
momentum = 0.9
no_cuda =False
seed = 999
l2_decay = 5e-4
num_classes = 31
dim_structure = 1000
alpha = 5
structure_regulizer = 10
root_path = "/data1/yinmingyue/Datasets/Office/domain_adaptation_images"
source_name = "dslr/images"  # amazon dslr webcam
target_name = "amazon/images"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
cuda = not no_cuda and torch.cuda.is_available()
device_ids = [0]
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
kwargs = {'num_workers': 4, 'pin_memory': False} if cuda else {}

source_loader = load_training(root_path, source_name, batch_size, kwargs)
target_train_loader = load_training(root_path, target_name, batch_size, kwargs)
target_test_loader = load_testing(root_path, target_name, batch_size, kwargs)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)

now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = f'log/{now}'
writer = SummaryWriter(log_dir)
scheduler = lambda global_step, lr: inverseDecaySheduler(global_step, lr, gamma=10, power=0.75, max_iter=max_iter)

def train(epoch, feature_extractor,classifier, Metric_Generator, Discriminator):
    i = 1
    global_step = (epoch - 1) * len_source_loader + i
    # LEARNING_RATE = scheduler(global_step, lr)
    LEARNING_RATE = lr
    print("learning rate：", LEARNING_RATE)
    if torch.cuda.device_count() > 1:
        optimizer = torch.optim.SGD([
            {'params': feature_extractor.module.parameters(), 'lr': LEARNING_RATE / 10},
            {'params': classifier.module.parameters(), 'lr': LEARNING_RATE},
            {'params': Metric_Generator.module.parameters(), 'lr': LEARNING_RATE},
            {'params': Discriminator.module.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay, nesterov=True)#
    else:
        optimizer = torch.optim.SGD([
            {'params': feature_extractor.parameters(), 'lr': LEARNING_RATE / 10},
            {'params': classifier.parameters(), 'lr': LEARNING_RATE},
            {'params': Metric_Generator.parameters(), 'lr': LEARNING_RATE},
            {'params': Discriminator.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay, nesterov=True)#
    data_source_iter = iter(source_loader)
    data_target_iter = iter(target_train_loader)
    while i <= len_source_loader:
        ### initialization ###
        classifier.train()
        source_data, source_label = data_source_iter.next()
        triplet_loss = torch.zeros(1)
        margin = torch.zeros(num_classes)
        class_in_batch = torch.zeros(num_classes)
        feature_list = np.zeros(shape=(batch_size * 2, 2048))
        label_list = np.zeros(shape=(1, batch_size * 2))
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            triplet_loss, margin ,class_in_batch = triplet_loss.cuda(), margin.cuda(), class_in_batch.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        ### source feature extract and domain output ###
        feature_src = feature_extractor(source_data)
        clabel_src = classifier(feature_src)
        clabel_src = F.softmax(clabel_src)
        structure_src = Metric_Generator(feature_src)
        structure_regulize_loss = torch.mean((Metric_Generator(feature_src.detach()).norm(2,dim=1)-structure_regulizer)**2)
        discriminator_src = Discriminator(feature_src)
        label_loss = F.nll_loss(clabel_src.log(), source_label)
        ### target feature extract and domain output ###
        target_data, target_label = data_target_iter.next()
        if i % len_target_loader == 0:
            data_target_iter = iter(target_train_loader)
        if cuda:
            target_data, target_label = target_data.cuda(), target_label.cuda()
        target_data = Variable(target_data)
        feature_tgt = feature_extractor(target_data)
        clabel_tgt = classifier(feature_tgt)
        clabel_tgt = F.softmax(clabel_tgt)
        discriminator_tgt = Discriminator(feature_tgt)
        with torch.no_grad():
            ### update alpha to punish samples around the classification boundry ###
            for batch_index in range(batch_size):
                pseudo_label = clabel_tgt[batch_index].max(dim=-1)[1]
                subpeak_softmax = torch.sort(clabel_tgt[batch_index], dim=-1, descending=True)[0][1].detach()
                margin[pseudo_label] += subpeak_softmax * num_classes
                class_in_batch[pseudo_label] += 1
            for label in range(num_classes):
                margin[label] = margin[label] / (class_in_batch[label] + 1e-6) + alpha
            ### metric learning ###
            for batch_index in range(batch_size):
                true_label = source_label[batch_index]
                same_category = torch.eq(source_label,source_label[batch_index]).type(torch.FloatTensor).cuda().detach()
                different_category = torch.eq(same_category,-same_category).type(torch.FloatTensor).cuda().detach()
                max_same_distance = torch.max(same_category * torch.norm((structure_src - structure_src[batch_index]),dim=1))[0]
                min_confused_distance = torch.min(different_category * torch.norm((structure_src - structure_src[batch_index]),dim=1) + 1e2 * same_category,dim=-1)[0]
                distance_discrepancy = max_same_distance - min_confused_distance + margin[source_label[batch_index]]
                if distance_discrepancy > 0:
                    triplet_loss += distance_discrepancy
        triplet_loss = triplet_loss/batch_size
        discriminator_loss = nn.BCELoss()(discriminator_src, torch.ones_like(discriminator_src)) \
                                 + nn.BCELoss()(discriminator_tgt, torch.zeros_like(discriminator_tgt))
        target_entropy_loss = -torch.mean((clabel_tgt * torch.log(clabel_tgt + 1e-6)).sum(dim=1))
        total_loss = label_loss + discriminator_loss + 0.08 * triplet_loss #+ 0.1 * target_entropy_loss

        print('Epoch: [{}/{}], iter: [{}/{}],max same/min different distance:[{:.2f}/{:.2f}], max margin:[{:.2f}]'
              .format(epoch, epochs, i, len_source_loader, max_same_distance, min_confused_distance, margin.max(-1)[0]))

        ## Training shared network and label classifier ###
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ### write summary of loss and feature map ###
        if i % len_source_loader == 0:
            writer.add_scalar('loss/source_label_loss', label_loss.data[0], (epoch - 1) * len_source_loader + i)
            writer.add_scalar('loss/target_entropy_loss', target_entropy_loss.data[0],
                              (epoch - 1) * len_source_loader + i)
            writer.add_scalar('loss/discriminator_loss', discriminator_loss.data[0],
                              (epoch - 1) * len_source_loader + i)
            writer.add_scalar('loss/triplet_loss', triplet_loss.data[0],
                              (epoch - 1) * len_source_loader + i)
            writer.add_scalar('loss/structure_regulize_loss', structure_regulize_loss.data[0],
                              (epoch - 1) * len_source_loader + i)
        ### test model accuracy ###
        if epoch % 2 == 0  and  i % len_source_loader == 0:
            with torch.no_grad():
                true_correct, average_acc, Dict_acc = test(feature_extractor,classifier, epoch)
                t_correct_true = 0
                t_correct_average = 0
                if true_correct > t_correct_true:
                    t_correct_true = true_correct
                if average_acc > t_correct_average:
                    t_correct_average = average_acc
                print(
                    'Experi-No: {}, source: {} to target: {}, max correct: {}, max true accuracy: {:.2f}%, max average accuracy: {:.2f}%\n'.format(
                        ex + 1, source_name[:-7], target_name[:-7], t_correct_true, 100. * t_correct_true / len_target_dataset, t_correct_average))
                writer.add_scalar('accuracy/target_average_correct', t_correct_average, (epoch-1) * len_source_loader + i)
        i = i + 1

def test(feature_extractor,classifier, epoch):
    classifier.eval()
    test_loss = 0
    correct = 0
    Dict_all = list(np.zeros(num_classes))
    Dict_acc = list(np.zeros(num_classes))
    for target_data, target_label in target_test_loader:
        target_label = target_label.long()
        if cuda:
            target_data, target_label = target_data.cuda(), target_label.cuda()
        target_data, target_label = Variable(target_data), Variable(target_label)
        feature = feature_extractor(target_data)
        out_tgt = classifier(feature)
        out_tgt = F.softmax(out_tgt)
        test_loss += F.nll_loss(out_tgt.log(), target_label, size_average=False).data[0]  # sum up batch loss
        pred = out_tgt.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target_label.data.view_as(pred)).cpu().sum()
        target_label = target_label.data.cpu()
        pred = pred.data.cpu()
        for j in range(target_label.numpy().shape[0]):
            Dict_all[target_label[j].item()] += 1
            if target_label[j] == pred[j]:
                Dict_acc[pred[j].item()] += 1
    test_loss /= len_target_dataset
    for j in range(len(Dict_all)):
        Dict_acc[j] = Dict_acc[j] / Dict_all[j] * 100.
    all_acc =0
    for i in range(len(Dict_acc)):
        all_acc += Dict_acc[i]
    average_acc = all_acc/num_classes
    print('average_acc: ',average_acc)
    return correct, average_acc ,Dict_acc

if __name__ == '__main__':
    true_accuracy = list(np.zeros(num_experiments))
    true_acc_max = list(np.zeros(num_experiments))
    avg_accuracy = list(np.zeros(num_experiments))
    avg_acc_max = list(np.zeros(num_experiments))
    class_list = list(np.zeros(num_experiments))
    for ex in range(num_experiments):
        feature_extractor = models.resnet50(pretrained=True)
        classifier = models.FClayers(out_dim=num_classes)
        Metric_Generator = models.FClayers(out_dim=dim_structure)
        Discriminator = models.AdversarialNetwork(2048)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            feature_extractor = torch.nn.DataParallel(feature_extractor, device_ids)
            classifier = torch.nn.DataParallel(classifier, device_ids)
            Metric_Generator = torch.nn.DataParallel(Metric_Generator, device_ids)
            Discriminator = torch.nn.DataParallel(Discriminator, device_ids)
        if cuda:
            feature_extractor = feature_extractor.cuda(device=device_ids[0])
            classifier = classifier.cuda(device=device_ids[0])
            Metric_Generator = Metric_Generator.cuda(device=device_ids[0])
            Discriminator = Discriminator.cuda(device=device_ids[0])
        epochs = torch.LongTensor([max_iter / len_source_loader]).data[0]
        print('Total Epochs: [{}]'.format(epochs))
        for epoch in range(1, epochs + 1):
            train(epoch, feature_extractor,classifier, Metric_Generator, Discriminator)

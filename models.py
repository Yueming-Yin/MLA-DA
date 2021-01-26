import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Function
from torchvision import models
from easydl import *
from torchvision import utils as vutils

__all__ = ['ResNet', 'resnet50', 'resnet101', 'TotalNet101', 'TotalNet50']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        # for key, value in pretrained_dict.items():
        #     print(key)
        for k, v in model_dict.items():
            if not "fc.weight" in k and not "fc.bias" in k and not "num_batches_tracked" in k:
                model_dict[k] = pretrained_dict[k]
        model.load_state_dict(model_dict)
    return model


def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            if not "fc.weight" in k and not "fc.bias" in k and not "num_batches_tracked" in k:
                model_dict[k] = pretrained_dict[k]
        model.load_state_dict(model_dict)
    return model


class FClayers(nn.Module):

    def __init__(self, in_dim=2048, out_dim=12, bottleneck=1000):
        super(FClayers, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_dim, bottleneck)
        self.bn1 = nn.BatchNorm1d(bottleneck)
        self.fc3 = nn.Linear(bottleneck, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        return x


class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
    """

    def __init__(self, in_feature, bottleneck=1024):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, bottleneck),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(bottleneck, bottleneck),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(bottleneck, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y


def l2_norm(input, dim):
    norm = torch.norm(input, dim=dim, keepdim=True)
    output = torch.div(input, norm)
    return output


def normalize_perturbation(d):
    output = l2_norm(l2_norm(d, dim=2), dim=3)
    return output


def perturb_image(x, p, feature_extractor, classifier, radius=3.5):
    eps = 1e-6 * normalize_perturbation(torch.randn(x.shape))
    eps = Variable(eps, requires_grad=True)
    # Predict on randomly perturbed image
    eps_f = feature_extractor(x + eps.cuda())
    eps_p = classifier(eps_f)
    eps_p = F.softmax(eps_p)
    loss = F.nll_loss(torch.log(eps_p + 1e-6), p)
    loss.backward()

    # Based on perturbed image, get direction of greatest error
    eps_adv = eps.grad
    # Use that direction as adversarial perturbation
    eps_adv = normalize_perturbation(eps_adv)
    x_adv = x + radius * eps_adv.cuda()
    return eps_adv, x_adv


def vat_loss(x, p, feature_extractor, classifier):
    eps_adv, x_adv = perturb_image(x, p, feature_extractor, classifier)
    f_adv = feature_extractor(x_adv)
    p_adv = classifier(f_adv)
    p_adv = F.softmax(p_adv)
    loss = torch.mean(F.nll_loss(torch.log(p_adv + 1e-6), p))
    return eps_adv, x_adv, loss


def save_image_tensor(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)

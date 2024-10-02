import os
from collections import namedtuple
import torch
import torch.nn as nn
from torchvision import models as tv


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True) + eps)
    return in_feat / (norm_factor + eps)


class RerangeLayer(nn.Module):
    # Change the input from range [-1., 1.] to [0., 1.]
    def __init__(self):
        super(RerangeLayer, self).__init__()

    def forward(self, inp):
        return (inp + 1.) / 2.


class NetLinLayer(nn.Module):
    ''' A single linear layer used as placeholder for LPIPS learnt weights '''
    def __init__(self):
        super(NetLinLayer, self).__init__()
        self.weight = None

    def forward(self, inp):
        out = self.weight * inp
        return out


class ScalingLayer(nn.Module):
    # For rescaling the input to vgg16
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


# Learned perceptual network, modified from https://github.com/richzhang/PerceptualSimilarity
class LPNet(nn.Module):
    def __init__(self, pnet_type='vgg', version='0.1', path='.'):
        super(LPNet, self).__init__()

        self.scaling_layer = ScalingLayer()
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.L = 5
        self.lins = [NetLinLayer() for _ in range(self.L)]

        model_path = os.path.abspath(
            os.path.join(path, 'weights/v%s/%s.pth' % (version, pnet_type)))
        print('Loading model from: %s' % model_path)
        weights = torch.load(model_path)
        for i in range(self.L):
            self.lins[i].weight = torch.sqrt(weights["lin%d.model.1.weight" % i])

    def forward(self, in0, avg=False):
        in0_input = self.scaling_layer(in0)
        outs0 = self.net.forward(in0_input)
        feats0 = {}
        shapes = []
        res = []

        for kk in range(self.L):
            feats0[kk] = normalize_tensor(outs0[kk])

        if avg:
            res = [self.lins[kk](feats0[kk]).mean([2,3],keepdim=False) for kk in range(self.L)]
        else:
            for kk in range(self.L):
                cur_res = self.lins[kk](feats0[kk])
                shapes.append(cur_res.shape[-1])
                res.append(cur_res.reshape(cur_res.shape[0], -1))

        return res, shapes


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out
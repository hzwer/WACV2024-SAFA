import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from warplayer import warp
from head import Head

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.bernoulli(x)
        return y

    @staticmethod
    def backward(ctx, grad):
        return grad, None

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True, groups=groups),        
        nn.PReLU(out_planes)
    )

class Resblock(nn.Module):
    def __init__(self, c, dilation=1):
        super(Resblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1),
            nn.PReLU(c),
            nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1),
        )
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.prelu = nn.PReLU(c)

    def forward(self, x):
        y = self.conv(x)
        return self.prelu(y * self.beta + x)

class RecurrentBlock(nn.Module):
    def __init__(self, c, depth=6):
        super(RecurrentBlock, self).__init__()
        self.conv_stem = conv(3*c+6+1, c, 3, 1, 1, groups=1)
        self.conv_backbone = torch.nn.ModuleList([])
        self.depth = depth
        for i in range(depth):
            self.conv_backbone.append(Resblock(c))
    
    def forward(self, x, i0, i1, flow, timestep, convflow, getscale):
        flow_down = F.interpolate(flow, scale_factor=0.5, mode="bilinear")
        i0 = warp(i0, flow_down[:, :2] * 0.5)
        i1 = warp(i1, flow_down[:, 2:4] * 0.5)
        x = torch.cat((x, flow_down, i0, i1, timestep), 1)
        scale = RoundSTE.apply(getscale(x)).unsqueeze(2).unsqueeze(3)
        feat = 0
        if scale.shape[0] != 1 or (scale[:, 0:1].mean() > 0.5 and scale[:, 1:2].mean() > 0.5):
            x0 = self.conv_stem(x)
            for i in range(self.depth):
                x0 = self.conv_backbone[i](x0)
            feat = feat + x0 * scale[:, 0:1] * scale[:, 1:2]

        if scale.shape[0] != 1 or (scale[:, 0:1].mean() < 0.5 and scale[:, 1:2].mean() > 0.5):
            x1 = self.conv_stem(F.interpolate(x, scale_factor=0.5, mode="bilinear"))
            for i in range(self.depth):
                x1 = self.conv_backbone[i](x1)
            feat = feat + F.interpolate(x1, scale_factor=2.0, mode="bilinear") * (1 - scale[:, 0:1]) * scale[:, 1:2]

        if scale.shape[0] != 1 or scale[:, 1:2].mean() < 0.5:
            x2 = self.conv_stem(F.interpolate(x, scale_factor=0.25, mode="bilinear"))
            for i in range(self.depth):
                x2 = self.conv_backbone[i](x2)
            feat = feat + F.interpolate(x2, scale_factor=4.0, mode="bilinear") * (1 - scale[:, 1:2])
        return feat, convflow(feat) + flow

class FlowModule(nn.Module):
    def __init__(self, in_planes, c=64):
        super(FlowModule, self).__init__()
        self.encoder = Head(c)
        self.flowdecoder = nn.Sequential(
            nn.Conv2d(c, 4*6, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.getscale = nn.Sequential(
            conv(3*c+6+1, c, 1, 1, 0),
            conv(c, c, 1, 2, 0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c, 3),
            nn.Sigmoid()
        )
        self.init = conv(2*c, c, 3, 1, 1, groups=1)
        self.convblock = torch.nn.ModuleList([])
        for i in range(6):
            self.convblock.append(RecurrentBlock(c, 2))

    def extract_feat(self, x):
        i0 = self.encoder(x[:, :3])
        i1 = self.encoder(x[:, 3:6])
        feat = self.init(torch.cat((i0, i1), 1))
        return feat, i0, i1
        
    def forward(self, i0, i1, feat, timestep, flow):
        flow_list = []
        feat_list = []
        for i in range(6):
            feat, flow = self.convblock[i](feat, i0, i1, flow, timestep, self.flowdecoder, self.getscale)
            flow_list.append(flow)
            feat_list.append(feat)
        return flow_list, feat_list

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model.warplayer import warp
from model.head import Head

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True, groups=groups),        
        nn.PReLU(out_planes)
    )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
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

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.bernoulli(x)
        return y

    @staticmethod
    def backward(ctx, grad):
        return grad, None
    
class RecurrentBlock(nn.Module):
    def __init__(self, c, dilation=1, depth=6):
        super(RecurrentBlock, self).__init__()
        self.conv_stem = conv(3*c+6+1, c, 3, 1, 1, groups=1)
        self.conv_backbone = torch.nn.ModuleList([])
        self.depth = depth
        for i in range(depth):
            self.conv_backbone.append(Resblock(c, dilation))
        
    def forward(self, x, i0, i1, flow, timestep, convflow, getscale):
        flow_down = F.interpolate(flow, scale_factor=0.5, mode="bilinear")
        i0 = warp(i0, flow_down[:, :2] * 0.5)
        i1 = warp(i1, flow_down[:, 2:4] * 0.5)
        x = torch.cat((x, flow_down, i0, i1, timestep), 1)
        scale = RoundSTE.apply(getscale(x)).unsqueeze(2).unsqueeze(3)
        feat = 0
        if scale.shape[0] != 1 or (scale[:, 0:1].mean() >= 0.5 and scale[:, 1:2].mean() >= 0.5):
            x0 = self.conv_stem(x)
            for i in range(self.depth):
                x0 = self.conv_backbone[i](x0)
            feat = feat + x0 * scale[:, 0:1] * scale[:, 1:2] 

        if scale.shape[0] != 1 or (scale[:, 0:1].mean() < 0.5 and scale[:, 1:2].mean() >= 0.5):
            x1 = self.conv_stem(F.interpolate(x, scale_factor=0.5, mode="bilinear"))
            for i in range(self.depth):
                x1 = self.conv_backbone[i](x1)
            feat = feat + F.interpolate(x1, scale_factor=2.0, mode="bilinear") * (1 - scale[:, 0:1]) * scale[:, 1:2]

        if scale.shape[0] != 1 or scale[:, 1:2].mean() < 0.5:
            x2 = self.conv_stem(F.interpolate(x, scale_factor=0.25, mode="bilinear"))
            for i in range(self.depth):
                x2 = self.conv_backbone[i](x2)
            feat = feat + F.interpolate(x2, scale_factor=4.0, mode="bilinear") * (1 - scale[:, 1:2])
        return feat, convflow(feat) + flow, i0, i1, scale

class Flownet(nn.Module):
    def __init__(self, block_num, c=64):
        super(Flownet, self).__init__()
        self.convimg = Head(c)
        self.shuffle = conv(2*c, c, 3, 1, 1, groups=1)
        self.convblock = torch.nn.ModuleList([])
        self.block_num = block_num
        self.convflow = nn.Sequential(
            nn.Conv2d(c, 4*6, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.getscale = nn.Sequential(
            conv(3*c+6+1, c, 1, 1, 0),
            conv(c, c, 1, 2, 0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c, 2),
            nn.Sigmoid()
        )
        for i in range(self.block_num):
            self.convblock.append(RecurrentBlock(c, 1, 2))

    def extract_feat(self, x):
        i0 = self.convimg(x[:, :3])
        i1 = self.convimg(x[:, 3:6])
        feat = self.shuffle(torch.cat((i0, i1), 1))
        return feat, i0, i1
        
    def forward(self, i0, i1, feat, timestep, flow):
        flow_list = []
        feat_list = []
        scale_list = []
        for i in range(self.block_num):
            feat, flow, w0, w1, scale = self.convblock[i](feat, i0, i1, flow, timestep, self.convflow, self.getscale)
            flow_list.append(flow)
            feat_list.append(feat)
            scale_list.append(scale)
        return flow_list, feat_list, torch.cat(scale_list, 1)
        
class SAFA(nn.Module):
    def __init__(self):
        super(SAFA, self).__init__()
        c=80
        self.block = Flownet(6, c=c)
        self.lastconv = nn.Sequential(
            conv(2*c+1, c, 3, 2, 1),
            Resblock(c),
            Resblock(c),
            Resblock(c),
            Resblock(c),
            Resblock(c),
            Resblock(c),
            Resblock(c),
            Resblock(c),
            Resblock(c),
            Resblock(c),
            Resblock(c),
            Resblock(c),
            conv(c, 2*c),
            nn.PixelShuffle(2),
            nn.Conv2d(c//2, 12, 3, 1, 1),
            nn.PixelShuffle(2),
        )

    def inference(self, lowres, timestep=0.5):
        if isinstance(timestep, list):
            one = torch.tensor([1.]).reshape(1, 1, 1, 1).repeat(lowres.shape[0], 1, lowres.shape[2], lowres.shape[3]).to(device)
            one = F.interpolate(one, scale_factor=0.5, mode="bilinear")
            timestep_list = []
            for i in timestep:
                timestep_list.append(one * i)
        else:
            if not torch.is_tensor(timestep):
                timestep = torch.tensor(timestep).reshape(1, 1, 1, 1).repeat(lowres.shape[0], 1, 1, 1).to(device)
            timestep = timestep.repeat(1, 1, lowres.shape[2], lowres.shape[3])
            timestep = F.interpolate(timestep, scale_factor=0.5, mode="bilinear")
            one = 1-timestep*0
            timestep_list = [one*0, timestep, one]
        merged = []
        feat, i0, i1 = self.block.extract_feat(lowres)
        for timestep in timestep_list:
            flow_list, feat_list, soft_scale = self.block(i0, i1, feat, timestep, (lowres[:, :6] * 0).detach())
            flow_sum = flow_list[-1]
            warped_i0 = warp(lowres[:, :3], flow_sum[:, :2])
            warped_i1 = warp(lowres[:, -3:], flow_sum[:, 2:4])
            mask = torch.sigmoid(flow_sum[:, 4:5])
            warped = warped_i0 * mask + warped_i1 * (1 - mask)
            flow_down = F.interpolate(flow_sum, scale_factor=0.5, mode="bilinear")
            w0 = warp(i0, flow_down[:, :2] * 0.5)
            w1 = warp(i1, flow_down[:, 2:4] * 0.5)
            img = self.lastconv(torch.cat((timestep, w0, w1), 1))
            merged.append(torch.clamp(warped + img, 0, 1))
        return merged
        
    def forward(self, lowres, timestep=0.5, training=False):
        img0 = lowres[:, :3]
        img1 = lowres[:, -3:]
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep).reshape(1, 1, 1, 1).repeat(img0.shape[0], 1, 1, 1).to(device)
        timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
        timestep = F.interpolate(timestep, scale_factor=0.5, mode="bilinear")
        one = 1-timestep*0
        timestep_list = [timestep*0, one/8, one/8*2, one/8*3, one/8*4, one/8*5, one/8*6, one/8*7, one]
        merged = []
        feat, i0, i1 = self.block.extract_feat(torch.cat((img0, img1), 1))
        feat_loss_sum = 0
        soft_scale_list = []
        for timestep in timestep_list:
            flow_list, feat_list, soft_scale = self.block(i0, i1, feat, timestep, (lowres[:, :6] * 0).detach())
            soft_scale_list.append(soft_scale)
            flow_sum = flow_list[-1]
            warped_i0 = warp(img0, flow_sum[:, :2])
            warped_i1 = warp(img1, flow_sum[:, 2:4])
            mask = torch.sigmoid(flow_sum[:, 4:5])
            warped = warped_i0 * mask + warped_i1 * (1 - mask)
            flow_down = F.interpolate(flow_sum, scale_factor=0.5, mode="bilinear")
            w0 = warp(i0, flow_down[:, :2] * 0.5)
            w1 = warp(i1, flow_down[:, 2:4] * 0.5)
            img = self.lastconv(torch.cat((timestep, w0, w1), 1))
            merged.append(torch.clamp(warped + img, 0, 1))
            flow_list.append(flow_sum)
        return torch.cat(flow_list, 3), soft_scale_list[1], merged

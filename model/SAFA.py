import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from warplayer import warp
from head import Head
from flow import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

c=80
class SAFA(nn.Module):
    def __init__(self):
        super(SAFA, self).__init__()
        self.safa = FlowModule(6, c=c)
        self.reconstruction = nn.Sequential(
            conv(2*c, c, 3, 2, 1),
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
        
    def forward(self, lowres, timestep=0.5, training=False):
        img0 = lowres[:, :3]
        img1 = lowres[:, -3:]
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep).reshape(1, 1, 1, 1).repeat(img0.shape[0], 1, 1, 1).to(device)
        timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
        timestep = F.interpolate(timestep, scale_factor=0.5, mode="bilinear")
        one = 1-timestep*0
        timestep_list = [one*0, timestep, one]
        result = []
        feat, i0, i1 = self.safa.extract_feat(torch.cat((img0, img1), 1))
        for timestep in timestep_list:
            weight_list = []
            flow_list, feat_list = self.safa(i0, i1, feat, timestep, (lowres[:, :6] * 0).detach())
            final_flow = flow_list[-1]
            warped_i0 = warp(img0, final_flow[:, :2])
            warped_i1 = warp(img1, final_flow[:, 2:4])
            mask = torch.sigmoid(final_flow[:, 4:5])
            warped = warped_i0 * mask + warped_i1 * (1 - mask)
            flow_down = F.interpolate(final_flow, scale_factor=0.5, mode="bilinear")
            w0 = warp(i0, flow_down[:, :2] * 0.5)
            w1 = warp(i1, flow_down[:, 2:4] * 0.5)
            delta = self.reconstruction(torch.cat((w0, w1), 1))
            result.append(torch.clamp(warped + delta, 0, 1))
        return result

model = SAFA()
img = torch.zeros([1, 6, 256, 256])
print(len(model(img)), model(img)[0].shape)

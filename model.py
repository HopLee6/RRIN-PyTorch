import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unet import UNet

def warp(img, flow):
    _, _, H, W = img.size()
    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
    gridX = torch.tensor(gridX, requires_grad=False).cuda()
    gridY = torch.tensor(gridY, requires_grad=False).cuda()
    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v
    x = 2*(x/W - 0.5)
    y = 2*(y/H - 0.5)
    grid = torch.stack((x,y), dim=3)
    imgOut = F.grid_sample(img, grid)
    return imgOut

class Net(nn.Module):
    def __init__(self,level=3):
        super(Net, self).__init__()
        self.Mask = UNet(16,2,4)
        self.Flow_L = UNet(6,4,5)
        self.refine_flow = UNet(10,4,4)
        self.final = UNet(9,3,4)

    def process(self,x0,x1,flow_net,t):
        x = torch.cat((x0,x1),1)
        Flow = flow_net(x)
        Flow_0_1, Flow_1_0 = Flow[:,:2,:,:], Flow[:,2:4,:,:]
        Flow_t_0 = -(1-t)*t*Flow_0_1+t*t*Flow_1_0
        Flow_t_1 = (1-t)*(1-t)*Flow_0_1-t*(1-t)*Flow_1_0
        Flow_t = torch.cat((Flow_t_0,Flow_t_1,x),1)
        Flow_t = self.refine_flow(Flow_t)
        Flow_t_0 = Flow_t_0+Flow_t[:,:2,:,:]
        Flow_t_1 = Flow_t_1+Flow_t[:,2:4,:,:]
        xt1 = warp(x0,Flow_t_0)
        xt2 = warp(x1,Flow_t_1)
        temp = torch.cat((Flow_t_0,Flow_t_1,x,xt1,xt2),1)
        Mask = F.sigmoid(self.Mask(temp))
        w1, w2 = (1-t)*Mask[:,0:1,:,:], t*Mask[:,1:2,:,:]
        output = (w1*xt1+w2*xt2)/(w1+w2+1e-8)
        return output

    def forward(self, input0, input1, t=0.5):

        output = self.process(input0,input1,self.Flow_L,t)
        compose = torch.cat((input0, input1, output),1)
        final = self.final(compose)+output
        final = final.clamp(0,1)
        return output

import dnnlib
import legacy
import torch
from wrapper import Generator
import numpy as np
from training.networks import modulated_conv2d
from torch_utils.misc import copy_params_and_buffers


if __name__ == '__main__':
    device = torch.device('cuda:0')
    ckpt = 'pretrained/ffhq.pkl'


    G = Generator(ckpt, device)

    seed = 1
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.G.z_dim)).to(device)

    ws = G.mapping(z)
    styles = G.mapping_stylespace(ws)

    synth = G.synthesis_from_stylespace(ws, styles)
    breakpoint()
#    w_b4 = ws.narrow(1,0,2)

#    x = img = None
#    x, img = G.G.synthesis.b4(x, img, w_b4.to(device), styles=None)
#    breakpoint()
#    x = img = None
#    style = G.G.synthesis.b4.conv1.affine(ws.unbind(dim=1)[0].to(device))
#    x2, img2 = G.G.synthesis.b4(x, img, w_b4.to(device), styles=style)
#    breakpoint()

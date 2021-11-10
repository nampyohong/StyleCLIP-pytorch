import argparse
from tqdm import tqdm

import numpy as np
import torch

import dnnlib
import legacy
from wrapper import Generator


class StyleManipulator():
    """
    Manipulator for style editing

    in paper, use 100 image pairs to estimate the mean for alpha(magnitude of the perturbation) [-5, 5]

    *** Args ***
    G : Genertor wrapper for synthesis styles
    device : torch.device
    lst_alpha : magnitude of the perturbation
    num_images : num images to process

    *** Methods ***
    get_boundary : get style's mean, std of given layer, channel index
    edit : style space edit for given layer, channel index, lst_alpha
    generate : generate images from styles edited

    *** Attributes ***
    S :  List[dict(str, torch.Tensor)] # length 2,000
    styles : List[dict(str, torch.Tensor)] # length of num_images
    lst_alpha : List[int]
    boundary : (num_images, len_alpha)
    edited_style : (num_images, len_alpha, style)
    edited_images : (num_images, len_alpha, 3, 1024, 1024)
    """
    def __init__(self, G, device, lst_alpha, num_images):
        self.S = torch.load('tensor/S.pt')
        self.S_mean = torch.load('tensor/S_mean.pt')
        self.S_std = torch.load('tensor/S_std.pt')

        # device cpu -> cuda # TODO : refactoring
        self.S = {layer: self.S[layer].to(device) for layer in G.style_layers}
        self.S_mean = {layer: self.S_mean[layer].to(device) for layer in G.style_layers}
        self.S_std = {layer: self.S_std[layer].to(device) for layer in G.style_layers}

        self.G = G
        self.num_images = num_images
        self.lst_alpha = lst_alpha

        breakpoint()


def extract_global_direction(num_images, lst_alpha):
    device = torch.device('cuda:0')
    ckpt = 'pretrained/ffhq.pkl'
    G = Generator(ckpt, device)
    
    manipulate_layers = [layer for layer in G.style_layers if 'torgb' not in layer] # lindex in original tf version

    # total channel: 6048
    latent = torch.randn([1,18,512]).to(device)
    style = G.mapping_stylespace(latent)
    cnt = 0
    for layer in manipulate_layers:
        cnt += style[layer].shape[1]
    del latent
    del style
    print(f"total channels : {cnt}")

    breakpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('runtype', type=str, default='test')
    args = parser.parse_args()
    
    runtype = args.runtype

    assert runtype in ['test', 'extract'] 

    device = torch.device('cuda:0')
    ckpt = 'pretrained/ffhq.pkl'
    G = Generator(ckpt, device)

    if runtype == 'test': # test manipulator
        num_images = 10
        lst_alpha = [-5, 0, 5]
        manipulator = StyleManipulator(G, device, lst_alpha, num_images=10)
    elif runtype == 'extract': # extract global style direction from "tensor/S.pt"
        num_images = 100
        lst_alpha = [-5, 5]
        extract_global_direction(num_images, lst_alpha)

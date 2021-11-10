import argparse
import copy
from tqdm import tqdm
from pprint import pprint

import numpy as np
import torch

import dnnlib
import legacy
from wrapper import Generator


class StyleManipulator():
    """Manipulator for style editing

    in paper, use 100 image pairs to estimate the mean for alpha(magnitude of the perturbation) [-5, 5]

    *** Args ***
    G : Genertor wrapper for synthesis styles
    device : torch.device
    lst_alpha : magnitude of the perturbation
    num_images : num images to process

    *** Attributes ***
    S :  List[dict(str, torch.Tensor)] # length 2,000
    styles : List[dict(str, torch.Tensor)] # length of num_images
                (num_images, style)
    lst_alpha : List[int]
    boundary : (num_images, len_alpha)
    edited_styles : List[styles]
    edited_images : List[(num_images, 3, 1024, 1024)]
    """
    def __init__(self, G, device, lst_alpha, num_images):
        self.W = torch.load('tensor/W.pt')
        self.S = torch.load('tensor/S.pt')
        self.S_mean = torch.load('tensor/S_mean.pt')
        self.S_std = torch.load('tensor/S_std.pt')

        self.S = {layer: self.S[layer].to(device) for layer in G.style_layers}
        self.styles = {layer: self.S[layer][:num_images] for layer in G.style_layers}
        self.latent = self.W[:num_images].to(device)
        del self.W
        del self.S
        self.S_mean = {layer: self.S_mean[layer].to(device) for layer in G.style_layers}
        self.S_std = {layer: self.S_std[layer].to(device) for layer in G.style_layers}

        self.G = G
        self.device = device
        self.num_images = num_images
        self.lst_alpha = lst_alpha

    def __call__(self, layer, channel_ind:int):
        assert layer in G.style_layers
        assert 0 <= channel_ind < self.styles[layer].shape[1]

        boundary = self.S_std[layer][channel_ind].item()
        # apply self.S_mean value for given layer, channel_ind
        for ind in range(self.num_images):
            self.styles[layer][ind][channel_ind] = self.S_mean[layer][channel_ind]
        styles = [copy.deepcopy(self.styles) for _ in range(len(self.lst_alpha))]
        
        perturbation = (torch.Tensor(self.lst_alpha) * boundary).numpy().tolist()
       
        # apply edit 
        for img_ind in range(self.num_images):
            for edit_ind, delta_s in enumerate(perturbation):
                styles[edit_ind][layer][img_ind][channel_ind] += delta_s

        return styles

    def synthesis_from_styles(self, styles, num):
        for i in range(len(styles)):
            for layer_ in G.style_layers:
                styles[i][layer_] = styles[i][layer_][:num]
        return torch.cat([G.synthesis_from_stylespace(self.latent[:num], style) for style in styles], dim=0).cpu()


def extract_global_direction(num_images, lst_alpha):
    device = torch.device('cuda:0')
    ckpt = 'pretrained/ffhq.pkl'
    G = Generator(ckpt, device)
    
    # lindex in original tf version
    manipulate_layers = [layer for layer in G.style_layers if 'torgb' not in layer] 

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
        num_images = 100
        lst_alpha = [-5, 0, 5]
        layer = G.style_layers[6]
        channel_ind = 501
        manipulator = StyleManipulator(G, device, lst_alpha, num_images=num_images)
        styles = manipulator(layer, channel_ind)
        test_num = 10
        print(manipulator.synthesis_from_styles(styles, test_num).shape)

    elif runtype == 'extract': # extract global style direction from "tensor/S.pt"
        num_images = 100
        lst_alpha = [-5, 5]
        extract_global_direction(num_images, lst_alpha)

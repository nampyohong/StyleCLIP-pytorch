import argparse
from tqdm import tqdm

import dnnlib
import legacy
import torch
from wrapper import Generator
import numpy as np


def concat_style(s_lst, layers):
    result = {layer:list() for layer in layers}
    for layer in layers:
        for s_ in s_lst:
            result[layer].append(s_[layer])
    for layer in layers:
        result[layer] = torch.cat(result[layer])
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt', type=str, default='pretrained/ffhq.pkl')
    parser.add_argument('--dataset_name', type=str, default='')
    args = parser.parse_args()

    device = torch.device('cuda:0')
    ckpt = args.ckpt
    dataset_name = args.dataset_name

    G = Generator(ckpt, device)

    # get intermediate latent of 100000 samples
    seed = 2324
    w_lst = list()

    z = torch.from_numpy(np.random.RandomState(seed).randn(100_000, G.G.z_dim))
    for i in tqdm(range(1000)): # 100 * 1000 = 100000 # 1000
        start, end = 100 * i, 100 * (i+1)
        z_ = z[start:end].to(device)
        # apply truncation_psi=.7, first 8 layers
        w_ = G.mapping(z_.to(device), truncation_psi=0.7, truncation_cutoff=8)
        w_lst.append(w_.cpu())
    w_lst = torch.cat(w_lst)
    torch.save(w_lst, f'tensor/W{dataset_name}.pt')

    # get style of first 2000 sample in W.pt
    sample_ws = w_lst[:2000] # 2000
    sample_s = G.mapping_stylespace(sample_ws.to(device))
    for layer in G.style_layers:
        sample_s[layer] = sample_s[layer].cpu()
    torch.save(sample_s, f'tensor/S{dataset_name}.pt')
    del sample_s

    # get  std, mean of 100000 style samples
    s_lst = list()
    for i in tqdm(range(1000)): # 100 * 1000
        start, end = 100 * i, 100 * (i+1)
        w_ = w_lst[start:end]
        s_ = G.mapping_stylespace(w_.to(device))
        for layer in G.style_layers:
            s_[layer] = s_[layer].cpu()
        s_lst.append(s_)
    s_lst = concat_style(s_lst, G.style_layers)
   
    s_mean = {layer: torch.mean(s_lst[layer], axis=0) for layer in G.style_layers}
    s_std = {layer: torch.std(s_lst[layer], axis=0) for layer in G.style_layers}
    torch.save(s_mean, f'tensor/S_mean{dataset_name}.pt')
    torch.save(s_std, f'tensor/S_std{dataset_name}.pt')
    print("Done.")

import clip
import numpy as np
import torch

from embedding import get_delta_t
from manipulator import Manipulator
from mapper import get_delta_s
from wrapper import Generator


if __name__ == '__main__':
    device = torch.device('cuda:3')
    ckpt = 'pretrained/ffhq.pkl'
    G = Generator(ckpt, device)
    model, preprocess = clip.load("ViT-B/32", device=device)

    fs3 = np.load('tensor/fs3_.npy')
    
    classnames=['face', 'face with glasses']
    delta_t = get_delta_t(classnames, model)

    lst_alpha = [-20, -10, -5, 0, 5, 10, 20]
    num_images = 1
    # from pre-saved latent
    manipulator = Manipulator(G, device, lst_alpha, num_images)

    beta_threshold = 0.13
    delta_s, num_channel = get_delta_s(fs3, delta_t, manipulator, beta_threshold=beta_threshold)

    print(f'{num_channel} channels will be manipulated under the beta threshold {beta_threshold}')

    styles = manipulator.manipulate(delta_s)

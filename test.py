import torch
from manipulator import Manipulator
from wrapper import Generator


if __name__ == '__main__':
    # preprocess
    device = torch.device('cuda:2')
    ckpt = 'pretrained/ffhq.pkl'
    G = Generator(ckpt, device)
    expdir = 'samples'
    manipulator = Manipulator(G, device)
    # test e4e
    #manipulator.set_real_img_projection(expdir, mode='w+')
    # test w_pti
    manipulator.set_real_img_projection(expdir, mode='w_pti')
    # test s_pti
    #manipulator.set_real_img_projection(expdir, mode='s_pti')
    breakpoint()

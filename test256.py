import torch
from manipulator import Manipulator
from wrapper import Generator


if __name__ == '__main__':
    # preprocess
    device = torch.device('cuda:2')
    ckpt = 'pretrained/ffhq256.pkl'
    G = Generator(ckpt, device)
    expdir = 'samples/test01.jpeg'
    manipulator = Manipulator(G, device)
    # test e4e
    #manipulator.set_real_img_projection(expdir, inv_mode='w', pti_mode=None)
    #manipulator.set_real_img_projection(expdir, inv_mode='w', pti_mode='w')
    manipulator.set_real_img_projection(expdir, inv_mode='w', pti_mode='s')
    breakpoint()

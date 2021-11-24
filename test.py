import torch
from manipulator import Manipulator
from wrapper import Generator


if __name__ == '__main__':
    # preprocess
    device = torch.device('cuda:2')
    ckpt = 'pretrained/ffhq.pkl'
    G = Generator(ckpt, device)
    expdir = 'samples/test01.jpeg'
    manipulator = Manipulator(G, device)

    # w projector
    #manipulator.set_real_img_projection(expdir, inv_mode='w', pti_mode=None)
    # e4e
    #manipulator.set_real_img_projection(expdir, inv_mode='w+', pti_mode=None)
    # w projector + w pivot pti
    #manipulator.set_real_img_projection(expdir, inv_mode='w', pti_mode='w')
    # e4e + w pivot pti
    #manipulator.set_real_img_projection(expdir, inv_mode='w+', pti_mode='w')
    # w projector + s pivot pti
    #manipulator.set_real_img_projection(expdir, inv_mode='w', pti_mode='s')

    # e4e + s pivot pti
    manipulator.set_real_img_projection(expdir, inv_mode='w+', pti_mode='s')
    breakpoint()

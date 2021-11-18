import time

import clip
import numpy as np
import torch
from torchvision.transforms import transforms
import PIL.Image
import os
from embedding import get_delta_t
from manipulator import Manipulator
from mapper import get_delta_s
from wrapper import Generator, FaceLandmarksDetector, e4eEncoder

from pivot_tuning_inversion.utils.ImagesDataset import ImagesDataset
from pivot_tuning_inversion.training.coaches.multi_id_coach import MultiIDCoach

if __name__ == '__main__':
    # preprocess
#    timestamp = int(time.time())
#    timestamp = 'test'
    device = torch.device('cuda:2')
    ckpt = 'pretrained/ffhq.pkl'
    G = Generator(ckpt, device)
    expdir = 'pivot_tuning_inversion/aligned2'
    manipulator = Manipulator(G, device)
    manipulator.set_real_img_projection(expdir, mode='w+')
    breakpoint()
#
#    dataset = ImagesDataset(
#        expdir, 
#        device,
#        transforms.Compose([
#            transforms.ToTensor(),
#            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
#        align_face=True
#    )
#    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
#
#    coach = MultiIDCoach(dataloader, use_wandb=False, device=device)
#    latents = list()
#    for fname, image in dataloader:
#        image_name = fname[0]
#        embedding_dir = f'tmp/{timestamp}'
#        latents.append(coach.get_e4e_inversion(image))
#    latents = torch.cat(latents)
#    breakpoint()





    # test e4e encoder

    #ckpt = 'pivot_tuning_inversion/checkpoints/model_PLPLIHFZUBII_multi_id.pt'
    #G = Generator(ckpt, device)
    #model, preprocess = clip.load("ViT-B/32", device=device)
    #fs3 = np.load('tensor/fs3.npy')
    #expdir = 'pivot_tuning_inversion/aligned2'
#    num_images = 1
#    lst_alpha = [0]
#
#
#    manipulator = Manipulator(G, device, lst_alpha, num_images)
#    manipulator.set_real_img_projection(expdir, mode='w+')
#    breakpoint()

import os

import torch
from tqdm import tqdm

from configs import pti_hparams, pti_global_cfgs
from pivot_tuning_inversion.training.coaches.base_coach import BaseCoach


class MultiIDCoach(BaseCoach):

    def __init__(self, data_loader, device=None, generator=None, mode='w'):
        super().__init__(data_loader, device, generator, mode)


    def train_from_latent(self):
        '''
        train mode : self.mode
        - w : train from latent pivot
        - s : train from style pivot
        '''
        use_ball_holder = True
        self.G.synthesis.train()
        self.G.mapping.train()

        for i in tqdm(range(pti_hparams.max_pti_steps)):
            self.image_counter = 0

            for image, w_pivot in self.data_loader:
                if self.image_counter >= pti_hparams.max_images_to_invert:
                    break
                real_images_batch = image
                generated_images = self.forward(w_pivot)

                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, '',
                                      self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                use_ball_holder = pti_global_cfgs.training_step % pti_hparams.locality_regularization_interval == 0

                pti_global_cfgs.training_step += 1
                self.image_counter += 1

        return self.G.requires_grad_(False)

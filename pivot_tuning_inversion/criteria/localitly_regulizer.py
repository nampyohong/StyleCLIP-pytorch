import torch
import numpy as np

from configs import PTI_HPARAMS
from pivot_tuning_inversion.criteria import l2_loss

pti_hparams = PTI_HPARAMS()

class Space_Regulizer:
    def __init__(self, original_G, lpips_net):
        self.original_G = original_G
        self.morphing_regulizer_alpha = pti_hparams.regulizer_alpha
        self.lpips_loss = lpips_net

    def get_morphed_w_code(self, new_w_code, fixed_w):
        interpolation_direction = new_w_code - fixed_w # w_z - w_p
        interpolation_direction_norm = torch.norm(interpolation_direction, p=2) # ||w_z - w_p||_2
        direction_to_move = pti_hparams.regulizer_alpha * interpolation_direction / interpolation_direction_norm
        # regulizer_alpha : 30
        result_w = fixed_w + direction_to_move # w_r
        self.morphing_regulizer_alpha * fixed_w + (1 - self.morphing_regulizer_alpha) * new_w_code

        return result_w

    def get_image_from_ws(self, w_codes, G):
        return torch.cat([G.synthesis(w_code, noise_mode='none', force_fp32=True) for w_code in w_codes])

    def ball_holder_loss_lazy(self, new_G, num_of_sampled_latents, w_batch, device):
        loss = 0.0

        z_samples = np.random.randn(num_of_sampled_latents, self.original_G.z_dim)
        w_samples = self.original_G.mapping(torch.from_numpy(z_samples).to(device), None,
                                            truncation_psi=0.5)
        territory_indicator_ws = [self.get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_samples]

        for w_code in territory_indicator_ws:
            new_img = new_G.synthesis(w_code, noise_mode='none', force_fp32=True)
            with torch.no_grad():
                old_img = self.original_G.synthesis(w_code, noise_mode='none', force_fp32=True)

            if pti_hparams.regulizer_l2_lambda > 0:
                l2_loss_val = l2_loss.l2_loss(old_img, new_img)
                loss += l2_loss_val * pti_hparams.regulizer_l2_lambda

            if pti_hparams.regulizer_lpips_lambda > 0:
                loss_lpips = self.lpips_loss(old_img, new_img)
                loss_lpips = torch.mean(torch.squeeze(loss_lpips))
                loss += loss_lpips * pti_hparams.regulizer_lpips_lambda

        return loss / len(territory_indicator_ws)

    def space_regulizer_loss(self, new_G, w_batch, device):
        ret_val = self.ball_holder_loss_lazy(new_G, pti_hparams.latent_ball_num_of_samples, w_batch, device)
        return ret_val

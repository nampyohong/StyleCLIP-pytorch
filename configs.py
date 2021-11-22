import numpy as np


class GENERATOR_CONFIGS:
    """StyleGAN2-ada generator configuration
    """
    def __init__(self, resolution=1024):
        channel_base = 32768 if resolution >= 1024 else 16384
        self.G_kwargs = {
            'class_name': 'training.networks.Generator',
            'z_dim': 512,
            'w_dim': 512,
            'mapping_kwargs': {'num_layers': 8},
            'synthesis_kwargs': {
                'channel_base': channel_base,
                'channel_max': 512,
                'num_fp16_res': 4,
                'conv_clamp': 256
            }
        }
        self.common_kwargs = {'c_dim': 0, 'img_resolution': resolution, 'img_channels': 3}
        self.w_idx_lst = [
            0,1,        # 4
            1,2,3,      # 8
            3,4,5,      # 16
            5,6,7,      # 32
            7,8,9,      # 64
            9,10,11,    # 128
            11,12,13,   # 256
            13,14,15,   # 512
            15,16,17,   # 1024
        ]
        cutoff_idx = int(np.log2(1024/resolution) * (-3))
        if resolution < 1024:
            self.w_idx_lst = self.w_idx_lst[:cutoff_idx]


class PATH_CONFIGS:
    """Paths configuration
    """
    def __init__(self): 
        self.e4e = 'pretrained/e4e_ffhq_encode.pt'
        self.stylegan2_ada_ffhq = 'pretrained/ffhq.pkl'
        self.ir_se50 = 'pretrained/model_ir_se50.pth'
        self.dlib = 'pretrained/shape_predictor_68_face_landmarks.dat'

class PTI_HPARAMS:
    """Pivot-tuning-inversion related hyper-parameters
    """
    def __init__(self):
        # Architectures
        self.lpips_type = 'alex'
        self.first_inv_type = 'w+'
        self.optim_type = 'adam'

        # Locality regularization
        self.latent_ball_num_of_samples = 1
        self.locality_regularization_interval = 1
        self.use_locality_regularization = False
        self.regulizer_l2_lambda = 0.1
        self.regulizer_lpips_lambda = 0.1
        self.regulizer_alpha = 30
        
        ## Loss
        self.pt_l2_lambda = 1
        self.pt_lpips_lambda = 1
        
        ## Steps
        self.LPIPS_value_threshold = 0.06
        self.max_pti_steps = 350
        self.first_inv_steps = 450
        self.max_images_to_invert = 30
        
        ## Optimization
        self.pti_learning_rate = 3e-4
        self.first_inv_lr = 5e-3
        self.train_batch_size = 1


class PTI_GLOBAL_CFGS:
    def __init__(self):
        self.training_step = 1
        self.imgage_rec_result_log_snapshot = 100
        self.pivotal_training_steps = 0
        self.model_snapshot_interval = 400
        self.run_name = ''

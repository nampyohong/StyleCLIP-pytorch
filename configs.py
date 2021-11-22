import os
import os.path as osp

class path_configs:
    """Paths configuration
    """
    def __init__(self): 
        self.e4e = 'pretrained/e4e_ffhq_encode.pt'
        self.stylegan2_ada_ffhq = 'pretrained/ffhq.pkl'
        self.ir_se50 = 'pretrained/model_ir_se50.pth'
        self.dlib = 'pretrained/shape_predictor_68_face_landmarks.dat'

class pti_hprams:
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


class pti_global_cfgs:
    def __init__(self):
        self.training_step = 1
        self.imgage_rec_result_log_snapshot = 100
        self.pivotal_training_steps = 0
        self.model_snapshot_interval = 400
        self.run_name = ''

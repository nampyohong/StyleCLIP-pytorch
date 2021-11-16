import time

## Pretrained models paths
e4e = 'pretrained/e4e_ffhq_encode.pt'
stylegan2_ada_ffhq = 'pretrained/ffhq.pkl'
style_clip_pretrained_mappers = ''
ir_se50 = 'pretrained/model_ir_se50.pth'
dlib = 'pretrained/shape_predictor_68_face_landmarks.dat'

## Dirs for output files
checkpoints_dir = 'pivot_tuning_inversion/checkpoints'
embedding_base_dir = 'pivot_tuning_inversion/embeddings'
styleclip_output_dir = ''
experiments_output_dir = 'pivot_tuning_inversion/output'

## Input info
### Input dir, where the images reside
input_data_path = 'pivot_tuning_inversion/aligned2'
#
### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator
input_data_id = 'test'
#input_data_id = f'{int(time.time())}'

## Keywords
pti_results_keyword = 'PTI'
e4e_results_keyword = 'e4e'
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

## Edit directions
interfacegan_age = 'editings/interfacegan_directions/age.pt'
interfacegan_smile = 'editings/interfacegan_directions/smile.pt'
interfacegan_rotation = 'editings/interfacegan_directions/rotation.pt'
ffhq_pca = 'editings/ganspace_pca/ffhq_pca.pt'

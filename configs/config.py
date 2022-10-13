# ------------------------------------------------------------
# configs for depth completion
# @author:                  jokerWRN
# @data:                    Mon 2021.1.22 16:53
# @latest modified data:    Mon 2020.1.22 16.53
# ------------------------------------------------------------
# reference source code: https://github.com/abdo-eldesokey/pncnn
# ------------------------------------------------------------

from yacs.config import CfgNode as CN

cfg = CN()

# TRAINING SETTINGS
cfg.debug_dp = False
cfg.debug_loss_txt = ''

# Hardware
cfg.seed = 7240
cfg.gpus = (8, )
cfg.port = 29000
cfg.num_threads = 0
cfg.no_multiprocessing = False
cfg.syncbn = False
cfg.cudnn_deterministic = False
cfg.cudnn_benchmark = True

# Dataset
cfg.data_folder = ''
cfg.max_depth = 0.0
cfg.dataset = ['rgb', 'dep', 'gt', 'penetgt', 's2dgt', 'structure']
cfg.penetgt_mixed_gt = False
cfg.s2dgt_mixed_gt = False

cfg.fill_type = 'multiscale'
cfg.extrapolate = True
cfg.blur_type = 'gaussian'

cfg.toy_test = False
cfg.toy_test_number = 30

cfg.val = 'select'
cfg.val_h = 352
cfg.val_w = 1216
cfg.not_random_crop = False
cfg.val_not_random_crop = True
cfg.random_crop_height = 320
cfg.random_crop_width = 1216
cfg.jitter = 0.1

cfg.val_epoch = 10
cfg.val_iters = 500

# Network
cfg.model = ''
cfg.summary_name = ''
cfg.fusion_block = 'Vanilla_add'
cfg.aspp_block = ''
cfg.ben_aspp = False
cfg.jin_aspp = False
cfg.init = ''
cfg.convolutional_layer_encoding = ''
cfg.round1 = 1
cfg.weight_an1 = 0.0
cfg.weight_ben1 = 0.0
cfg.weight_jin1 = 0.0
cfg.round2 = 3
cfg.weight_an2 = 0.0
cfg.weight_ben2 = 0.0
cfg.weight_jin2 = 0.0
cfg.weight_an3 = 0.0
cfg.weight_ben3 = 0.0
cfg.weight_jin3 = 0.0
cfg.output = ''
cfg.ben_supervised = ''
cfg.jin_supervised = ''
cfg.an_supervised = ''

# Resume
cfg.resume = False
cfg.pretrain = ''
cfg.wandb_id_resume = ''

# Test
cfg.test = False
cfg.test_option = ''
cfg.test_name = ''
cfg.tta = True
cfg.test_not_random_crop = False
cfg.wandb_id_test = ''
cfg.test_dir = ''
cfg.test_model = ''
cfg.save_test_image = False

# Training
cfg.log_itr = 1
cfg.start_epoch = 0
cfg.epochs = 0
cfg.batch_size = 0

cfg.accumulation_gradient = False
cfg.accumulation_steps = 0

# Mixed precision
cfg.opt_level = 'O0'

# warm_up
cfg.warm_up = False
cfg.no_warm_up = True

# Loss
cfg.loss_fixed = True
cfg.partial_supervised_index = 0.0
cfg.loss_ben = ''
cfg.loss_jin = ''
cfg.loss_an = ''

# Optimizer
cfg.lr = 0.01
cfg.optimizer = 'ADAM'
# * ADAM
cfg.momentum = 0.9
cfg.betas = (0.9, 0.999)
cfg.epsilon = 1e-8
cfg.weight_decay = 0.0
cfg.scheduler = 'stepLR'    # choices:(stepLR, lambdaLR)
# * lambdaLR
cfg.decay = (7, )
cfg.gamma = (7, )
# * stepLR
cfg.decay_step = 3
cfg.decay_factor = 0.1

# Logs
cfg.vis_step = 10
cfg.num_summary = 4
cfg.record_by_wandb_online = True
cfg.test_record_by_wandb_online = True

cfg.ben_online_loss=True
cfg.ben_online_metric=True
cfg.ben_online_rmse_only=True
cfg.ben_online_img=True
cfg.summary_jin=False
cfg.jin_online_loss=True
cfg.jin_online_metric=True
cfg.jin_online_rmse_only=True
cfg.jin_online_img=True
cfg.summary_an=False
cfg.an_online_loss=True
cfg.an_online_metric=True
cfg.an_online_rmse_only=True
cfg.an_online_img=True

cfg.save_result_only = True


def get_cfg_defaults():
    """
    :return: global local has an error (2020.12.30)
    """
    return cfg.clone()


if __name__ == '__main__':
    my_cfg = get_cfg_defaults()
    print(my_cfg)

# CONFIG
import argparse
arg = argparse.ArgumentParser(description='depth completion')
arg.add_argument('-p', '--project_name', type=str, default='CUNet')
arg.add_argument('-n', '--model_name', type=str, default='CUNet')
arg.add_argument('-c', '--configuration', type=str, default='train.yml')
arg = arg.parse_args()
from configs import get as get_cfg
config = get_cfg(arg)
assert len(config.output) != 0, 'the output of network should not be zero!'
if config.output == 'ben_depth':
    assert len(config.ben_supervised) != 0 and len(config.jin_supervised) == 0 and len(config.an_supervised) == 0
elif config.output == 'jin_depth':
    assert len(config.jin_supervised) != 0 and len(config.an_supervised) == 0
elif config.output == 'an_depth':
    assert len(config.an_supervised) != 0

# ENVIRONMENT SETTINGS
import os
rootPath = os.path.abspath(os.path.dirname(__file__))
import functools
if len(config.gpus) == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpus[0])
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = functools.reduce(lambda x, y: str(x) + ',' + str(y), config.gpus)
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = str(config.port)
if not config.record_by_wandb_online:
    os.environ["WANDB_MODE"] = 'dryrun'

# BASIC PACKAGES
import time
import emoji
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# MULTI-GPU AND MIXED PRECISION
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# MODULES
from dataloaders.kitti_loader import KittiDepth
from model import get as get_model
from optimizer_scheduler import make_optimizer_scheduler
from summary import get as get_summary
from metric import get as get_metric
from utility import *
from loss import get as get_loss

# VARIANCES
pbar, pbar_val = None, None
batch, batch_val = None, None
checkpoint, best_metric = None, None
writer_ben_train, writer_ben_val = None, None
writer_jin_train, writer_jin_val = None, None
writer_an_train, writer_an_val = None, None
warm_up_cnt, warm_up_max_cnt = None, None
loss_sum_ben_val, loss_sum_jin_val, loss_sum_an_val = torch.from_numpy(np.array(0)), torch.from_numpy(np.array(0)), torch.from_numpy(np.array(0))
sample, sample_val, sample_, output, output_, output_val = None, None, None, None, None, None
log_itr, log_cnt, log_loss, log_cnt_val, log_loss_val, log_val = None, None, None, None, None, 0
loss_jin, loss_an = None, None
ben_val_metric, jin_val_metric, an_val_metric = None, None, None

def train(gpu, args):

    # GLOBAL INVARIANCE
    global checkpoint, best_metric, log_val, loss_jin, loss_an, an_val_metric, jin_val_metric, ben_val_metric
    global warm_up_cnt, warm_up_max_cnt
    global pbar, pbar_val
    global sample, sample_val, output, output_val
    global writer_ben_train, writer_ben_val, writer_jin_train, writer_jin_val, writer_an_train, writer_an_val 
    global batch, batch_val
    global log_itr, log_cnt, log_loss, log_cnt_val, log_loss_val
    global loss_sum_ben_val, loss_sum_jin_val, loss_sum_an_val
    if gpu == 0:
        print(args.dump())

    # INITIALIZE
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.num_gpus, rank=gpu)

    # MINIMIZE RANDOMNESS
    rank = torch.distributed.get_rank()
    torch.manual_seed(config.seed + rank)
    torch.cuda.manual_seed(config.seed + rank)
    torch.cuda.manual_seed_all(config.seed + rank)
    np.random.seed(config.seed + rank)
    random.seed(config.seed + rank)
    torch.backends.cudnn.deterministic = config.cudnn_deterministic
    torch.backends.cudnn.benchmark = config.cudnn_benchmark

    # WANDB
    if gpu == 0:
        best_metric = 1e8
        import wandb
        wandb.login()
        if not args.resume:
            wandb.init(dir=rootPath, config=args, project=args.project_name)
            args.defrost()
            args.save_dir = os.path.split(wandb.run.dir)[0]
            args.freeze()
            with open(args.save_dir + '/' + 'config.txt', 'w') as f:
                f.write(args.dump())
    if gpu == 0:
        if args.pretrain is not None:
            if args.resume:
                import wandb
                if len(args.wandb_id_resume) == 0:
                    wandb.init(dir=rootPath, config=args, project=args.project_name, resume=True)
                else:
                    assert len(args.wandb_id_resume) != 0, 'wandb_id should not be empty when resuming'
                    wandb.init(id=args.wandb_id_resume, dir=rootPath, config=args, project=args.project_name,
                               resume='must')
                args.defrost()
                args.save_dir = os.path.split(wandb.run.dir)[0]
                args.freeze()
                print('=> Resume wandb from : {}'.format(args.wandb_id_resume))

    # DATASET
    if gpu == 0:
        print(emoji.emojize('Prepare data... :writing_hand:', variant="emoji_type"))
    batch_size = args.batch_size // args.num_gpus
    data_train = KittiDepth('train', args)
    sampler_train = DistributedSampler(data_train)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                               shuffle=False, num_workers=args.num_threads,
                                               pin_memory=True, sampler=sampler_train, drop_last=True)
    if gpu == 0:
        len_train_dataset = len(loader_train) * batch_size * args.num_gpus
        print('=> Train dataset: {} samples'.format(len_train_dataset))
    data_val = KittiDepth('val', args)
    sampler_val = SequentialDistributedSampler(data_val, batch_size=batch_size)
    loader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size,
                                             shuffle=False, num_workers=args.num_threads,
                                             pin_memory=True, sampler=sampler_val)
    if gpu == 0:
        len_val_dataset = len(loader_val) * batch_size * args.num_gpus
        print('=> Val dataset: {} samples'.format(len_val_dataset))

    # NETWORK
    if gpu == 0:
        print(emoji.emojize('Prepare network... :writing_hand:', variant="emoji_type"))
    models = get_model(args)
    net = models(args)
    if gpu == 0:
        total_params = count_parameters(net)
    if gpu == 0:
        if len(args.pretrain) != 0:
            assert os.path.isfile(args.pretrain), "file not found: {}".format(args.pretrain)
            checkpoint = torch.load(args.pretrain)
            net.load_state_dict(checkpoint['net'])
            print('=> Load network parameters from : {}'.format(args.pretrain))
    if args.syncbn:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda(gpu)
    net = DDP(net, device_ids=[gpu], output_device=gpu)

    # OPTIMIZER
    if gpu == 0:
        print(emoji.emojize('Prepare optimizer... :writing_hand:', variant="emoji_type"))
    optimizer, scheduler = make_optimizer_scheduler(args, net)

    # IF RESUME
    if gpu == 0:
        if args.pretrain is not None:
            if args.resume:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    args.defrost()
                    args.start_epoch = checkpoint['epoch'] + 1
                    args.log_itr = checkpoint['log_itr']
                    args.freeze()
                    print('=> Resume optimizer, scheduler and amp '
                          'from : {}'.format(args.pretrain))
                except KeyError:
                    print('=> State dicts for resume are not saved. '
                          'Use --save_full argument')
            del checkpoint

    # LOSSES
    if gpu == 0:
        print(emoji.emojize('Prepare loss... :writing_hand:', variant="emoji_type"))
        print('=> Loss_ben: {}; Loss_jin: {}; Loss_an: {}'.format(args.loss_ben, args.loss_jin, args.loss_an))
    loss = get_loss(args)
    loss_ben = loss(args, args.loss_ben)
    loss_ben.cuda()
    if len(args.jin_supervised) != 0:
        loss_jin = loss(args, args.loss_jin)
        loss_jin.cuda()
    if len(args.an_supervised) != 0:
        loss_an = loss(args, args.loss_an)
        loss_an.cuda()

    # METRIC
    if gpu == 0:
        print(emoji.emojize('Prepare metric... :writing_hand:', variant="emoji_type"))
    metric = get_metric(args)
    metric = metric(args)

    # SUMMARY
    if gpu == 0:
        print(emoji.emojize('Prepare summary... :writing_hand:', variant="emoji_type"))
        summary = get_summary(args)
        writer_ben_train = summary(args.save_dir, 'ben_train', args,
                               loss_ben.loss_name, metric.metric_name)
        writer_ben_val = summary(args.save_dir, 'ben_val', args,
                             loss_ben.loss_name, metric.metric_name)
        if args.summary_jin:
            writer_jin_train = summary(args.save_dir, 'jin_train', args,
                                   loss_jin.loss_name, metric.metric_name)
            writer_jin_val = summary(args.save_dir, 'jin_val', args,
                                 loss_jin.loss_name, metric.metric_name)
        if args.summary_an:
            writer_an_train = summary(args.save_dir, 'an_train', args,
                                   loss_an.loss_name, metric.metric_name)
            writer_an_val = summary(args.save_dir, 'an_val', args,
                                 loss_an.loss_name, metric.metric_name)

    if gpu == 0:
        log_itr = args.log_itr
        backup_source_code(args.save_dir + '/backup_code')
        try:
            assert os.path.isdir(args.save_dir)
            os.makedirs(args.save_dir, exist_ok=True)
            os.makedirs(args.save_dir + '/train', exist_ok=True)
            os.makedirs(args.save_dir + '/val', exist_ok=True)
        except OSError:
            pass
        print('=> Save backup source code and makedirs done')

    # WARM UP
    if args.warm_up:
        warm_up_cnt = 0.0
        warm_up_max_cnt = len(loader_train) + 1.0

    # GO
    for epoch in range(args.start_epoch, args.epochs + 1):

        # TRAIN
        net.train()
        loader_train.sampler.set_epoch(epoch)

        # LOG
        if gpu == 0:
            print(emoji.emojize('Let\'s do something interesting :oncoming_fist:', variant="emoji_type"))
            current_time = time.strftime('%y%m%d@%H:%M:%S')
            list_lr = []
            for g in optimizer.param_groups:
                list_lr.append(g['lr'])
            print('=======> Epoch {:5d} / {:5d} | Lr : {} | {} | {} <======='.format(
                epoch, args.epochs, list_lr, current_time, args.save_dir
            ))
            num_sample = len(loader_train) * loader_train.batch_size * args.num_gpus
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        # ROUND
        for batch, sample in enumerate(loader_train):
            sample = {k: value.cuda(gpu) for k, value in sample.items() if torch.is_tensor(value)}

            # WARM UP
            if epoch == 1 and args.warm_up:
                warm_up_cnt += 1
                for param_group in optimizer.param_groups:
                    lr_warm_up = param_group['initial_lr'] * warm_up_cnt / warm_up_max_cnt
                    param_group['lr'] = lr_warm_up

            output = net(sample)

            # LOSS
            loss_sum_ben, loss_val_ben = loss_ben(output['ben_depth'], sample[args.ben_supervised])

            if len(args.jin_supervised) == 0:
                loss_sum_jin, loss_val_jin = torch.from_numpy(np.array(0)), torch.from_numpy(np.array(0))
            else:
                loss_sum_jin, loss_val_jin = loss_jin(output['jin_depth'], sample[args.jin_supervised])    
            if len(args.an_supervised) == 0:
                loss_sum_an, loss_val_an = torch.from_numpy(np.array(0)), torch.from_numpy(np.array(0))
            else:
                loss_sum_an, loss_val_an = loss_an(output['an_depth'], sample[args.an_supervised])

            if epoch <= args.round1:
                loss_sum = args.weight_ben1 * loss_sum_ben + \
                           args.weight_jin1 * loss_sum_jin + args.weight_an1 * loss_sum_an
            elif epoch <= args.round2:
                loss_sum = args.weight_ben2 * loss_sum_ben + \
                           args.weight_jin2 * loss_sum_jin + args.weight_an2  * loss_sum_an
            else:
                loss_sum = args.weight_ben3 * loss_sum_ben +\
                           args.weight_jin3 * loss_sum_jin + args.weight_an3 * loss_sum_an

            # ACCUMULATION GRADIENT
            if args.accumulation_gradient:
                # TODO accelerate without sync
                loss_accumulation = loss_sum / args.accumulation_steps
                loss_accumulation.backward()
                if ((batch + 1) % args.accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                loss_sum.backward()
                optimizer.step()
                optimizer.zero_grad()

            # METRIC
            if gpu == 0:
                metric_ben_train = metric.evaluate(output['ben_depth'], sample['gt'], 'train')
                writer_ben_train.add(loss_val_ben, metric_ben_train, log_itr)
                if args.summary_jin:
                    metric_jin_train = metric.evaluate(output['jin_depth'], sample['gt'], 'train')
                    writer_jin_train.add(loss_val_jin, metric_jin_train, log_itr)
                if args.summary_an:
                    metric_an_train = metric.evaluate(output['an_depth'], sample['gt'], 'train')
                    writer_an_train.add(loss_val_an, metric_an_train, log_itr)
                log_itr += 1
                log_cnt += 1
                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{}|Ls={:.4f}|Lb={:.4f}|Lj={:.4f}|La={:.4f}'.format(
                    'Train', loss_sum.item(), loss_sum_ben.item(), loss_sum_jin.item(), loss_sum_an.item())
                if epoch == 1 and args.warm_up:
                    list_lr = []
                    for g in optimizer.param_groups:
                        list_lr.append(round(g['lr'], 6))
                    error_str = '{} | Lr Warm Up : {}'.format(error_str, list_lr)
                pbar.set_description(error_str)
                pbar.update(loader_train.batch_size * args.num_gpus)

            # VAL IF NEED
            if (epoch <= args.val_epoch and batch == len(loader_train) - 1) or (
                    epoch > args.val_epoch and (batch + 1) % (args.val_iters // (loader_train.batch_size * args.num_gpus)) == 0):
                # ENVIRONMENT SETTING
                torch.set_grad_enabled(False)
                net.eval()

                # LOG
                if gpu == 0:
                    num_sample_val = len(loader_val) * loader_val.batch_size * args.num_gpus
                    pbar_val = tqdm(total=num_sample_val)
                    log_cnt_val = 0.0
                    log_loss_val = 0.0
                loss_ben_val_list, metric_ben_val_list = [], []
                loss_jin_val_list, metric_jin_val_list = [], []
                loss_an_val_list, metric_an_val_list = [], []

                # ROUND
                for batch_val, sample_val in enumerate(loader_val):
                    sample_val = {key: val.cuda(gpu) for key, val in sample_val.items()
                                  if val is not None}

                    output_val = net(sample_val)

                    # LOG
                    for ben_depth, ben_supervised, gt in zip(torch.chunk(output_val['ben_depth'], batch_size, dim=0),
                                                         torch.chunk(sample_val[args.ben_supervised], batch_size, dim=0),
                                                         torch.chunk(sample_val['gt'], batch_size, dim=0)):

                        loss_sum_ben_val, loss_val_ben_val = loss_ben(ben_depth, ben_supervised)
                        loss_ben_val_list.append(loss_val_ben_val)
                        metric_ben_val_list.append(metric.evaluate(ben_depth, gt, 'val'))

                    if args.summary_jin:
                        for jin_depth, jin_supervised, gt in zip(
                                torch.chunk(output_val['jin_depth'], batch_size, dim=0),
                                torch.chunk(sample_val[args.jin_supervised], batch_size, dim=0),
                                torch.chunk(sample_val['gt'], batch_size, dim=0)):
                            loss_sum_jin_val, loss_val_jin_val = loss_jin(jin_depth, jin_supervised)
                            loss_jin_val_list.append(loss_val_jin_val)
                            metric_jin_val_list.append(metric.evaluate(jin_depth, gt, 'val'))

                    if args.summary_an:
                        for an_depth, an_supervised, gt in zip(
                                torch.chunk(output_val['an_depth'], batch_size, dim=0),
                                torch.chunk(sample_val[args.an_supervised], batch_size, dim=0),
                                torch.chunk(sample_val['gt'], batch_size, dim=0)):
                            loss_sum_an_val, loss_val_an_val = loss_an(an_depth, an_supervised)
                            loss_an_val_list.append(loss_val_an_val)
                            metric_an_val_list.append(metric.evaluate(an_depth, gt, 'val'))

                    if gpu == 0:
                        current_time = time.strftime('%y%m%d@%H:%M:%S')
                        error_str = '{}|Lb={:.4f}|Lj={:.4f}|La={:.4f}'.format(
                            'Val', loss_sum_ben_val.item(), loss_sum_jin_val.item(), loss_sum_an_val.item())
                        pbar_val.set_description(error_str)
                        pbar_val.update(loader_val.batch_size * args.num_gpus)

                loss_ben_val_all = distributed_concat(torch.cat(loss_ben_val_list, axis=0),
                                                  len(loader_val.dataset))
                metric_ben_val_all = distributed_concat(torch.cat(metric_ben_val_list, axis=0),
                                                    len(loader_val.dataset))
                if gpu == 0:
                    pbar_val.close()
                    for i, j in zip(loss_ben_val_all, metric_ben_val_all):
                        writer_ben_val.add(i.unsqueeze(0), j.unsqueeze(0))
                    ben_val_metric = writer_ben_val.update(log_val, sample_val, output_val,
                                                   online_loss=args.ben_online_loss, online_metric=args.ben_online_metric,
                                                   online_rmse_only=args.ben_online_rmse_only, online_img=args.ben_online_img)
                if args.summary_jin:
                    loss_jin_val_all = distributed_concat(torch.cat(loss_jin_val_list, axis=0),
                                                          len(loader_val.dataset))
                    metric_jin_val_all = distributed_concat(torch.cat(metric_jin_val_list, axis=0),
                                                            len(loader_val.dataset))
                    if gpu == 0:
                        for i, j in zip(loss_jin_val_all, metric_jin_val_all):
                            writer_jin_val.add(i.unsqueeze(0), j.unsqueeze(0))
                        jin_val_metric = writer_jin_val.update(log_val, sample_val, output_val,
                                                               online_loss=args.jin_online_loss,
                                                               online_metric=args.jin_online_metric,
                                                               online_rmse_only=args.jin_online_rmse_only,
                                                               online_img=args.jin_online_img)
                if args.summary_an:
                    loss_an_val_all = distributed_concat(torch.cat(loss_an_val_list, axis=0),
                                                          len(loader_val.dataset))
                    metric_an_val_all = distributed_concat(torch.cat(metric_an_val_list, axis=0),
                                                            len(loader_val.dataset))
                    if gpu == 0:
                        for i, j in zip(loss_an_val_all, metric_an_val_all):
                            writer_an_val.add(i.unsqueeze(0), j.unsqueeze(0))
                        an_val_metric = writer_an_val.update(log_val, sample_val, output_val,
                                                               online_loss=args.an_online_loss,
                                                               online_metric=args.an_online_metric,
                                                               online_rmse_only=args.an_online_rmse_only, online_img=args.an_online_img)

                # SAVE CHECKPOINT
                if gpu == 0:
                    if args.output == 'jin_depth':
                        val_metric = jin_val_metric
                    elif args.output == 'an_depth':
                        val_metric = an_val_metric
                    else:
                        val_metric = ben_val_metric

                    tmp = args.val_iters // (loader_train.batch_size * args.num_gpus)
                    if (epoch <= args.val_epoch and batch == len(loader_train) - 1) or (epoch > args.val_epoch
                        and batch + 1 == (len(loader_train) // tmp) * tmp):
                        writer_ben_val.save(epoch, batch + 1, sample_val, output_val)

                    if val_metric < best_metric:
                        best_metric = val_metric
                        state = {
                            'net': net.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch,
                            'log_itr': log_itr,
                            'args': args
                        }
                        torch.save(state, '{}/best_model.pt'.format(args.save_dir))
                    log_val += 1
                torch.set_grad_enabled(True)
                net.train()

        # LOG
        if gpu == 0:
            pbar.close()
            _ = writer_ben_train.update(epoch, sample, output,
                                    online_loss=args.ben_online_loss, online_metric=args.ben_online_metric,
                                    online_rmse_only=args.ben_online_rmse_only, online_img=args.ben_online_img)
            if args.summary_jin:
                _ = writer_jin_train.update(epoch, sample, output,
                                            online_loss=args.jin_online_loss, online_metric=args.jin_online_metric,
                                            online_rmse_only=args.jin_online_rmse_only, online_img=args.jin_online_img)
            if args.summary_an:
                _ = writer_an_train.update(epoch, sample, output,
                                            online_loss=args.an_online_loss, online_metric=args.an_online_metric,
                                            online_rmse_only=args.an_online_rmse_only, online_img=args.an_online_img)
            state = {
                'net': net.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'log_itr': log_itr,
                'args': args
            }
            torch.save(state, '{}/latest_model.pt'.format(args.save_dir))
        scheduler.step()


def main(args):

    if args.no_multiprocessing:
        train(0, args)
    else:
        assert args.num_gpus > 0

        spawn_context = mp.spawn(train, nprocs=args.num_gpus, args=(args,),
                                 join=False)

        while not spawn_context.join():
            pass

        for process in spawn_context.processes:
            if process.is_alive():
                process.terminate()
            process.join()


if __name__ == '__main__':
    main(config)


























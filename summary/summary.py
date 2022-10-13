from . import BaseSummary
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import wandb
from PIL import Image

cm = plt.get_cmap('plasma')
log_metric_val = None

def pad_rep(image, ori_size):
    h, w = image.shape
    (oh, ow) = ori_size
    pl = (ow - w) // 2
    pr = ow - w - pl
    pt = oh - h
    image_pad = np.pad(image, pad_width=((pt, 0), (pl, pr)), mode='edge')
    return image_pad


class Summary(BaseSummary):
    def __init__(self, log_dir, mode, args, loss_name, metric_name):
        # assert mode in ['train', 'val', 'test'], \
        #     "mode should be one of ['train', 'val', 'test'] " \
        #     "but got {}".format(mode)

        super(Summary, self).__init__(log_dir, mode, args)

        self.log_dir = log_dir
        self.mode = mode
        self.args = args

        self.loss = []
        self.metric = []

        self.loss_name = loss_name
        self.metric_name = metric_name

        self.path_output = None

    def add(self, loss=None, metric=None, log_itr=None):
        # loss and metric should be numpy arrays
        if loss is not None:
            self.loss.append(loss.data.cpu().numpy())
        if metric is not None:
            self.metric.append(metric.data.cpu().numpy())

        if 'train' in self.mode and log_itr % self.args.vis_step == 0:
            log_dict = {}
            for idx, loss_type in enumerate(self.loss_name):
                val = loss.data.cpu().numpy()[0, idx]
                log_dict[self.mode + '_all_' + loss_type] = val

                # log by tb
                self.add_scalar('All/' + loss_type, val, log_itr)
            log_dict['custom_step_loss'] = log_itr
            wandb.log(log_dict)

    def update(self, global_step, sample, output,
               online_loss=True, online_metric=True, online_rmse_only=True, online_img=True):
        """
        update results
        """
        global log_metric_val
        log_dict = {}
        if self.loss_name is not None:
            self.loss = np.concatenate(self.loss, axis=0)
            self.loss = np.mean(self.loss, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format(self.mode + '_Loss')]
            for idx, loss_type in enumerate(self.loss_name):
                val = self.loss[0, idx]
                if online_loss:
                    log_dict[self.mode + '_' + loss_type] = val
                self.add_scalar('Loss/' + loss_type, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(loss_type, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_loss = open(self.f_loss, 'a')
            f_loss.write('{:04d} | {}\n'.format(global_step, msg))
            f_loss.close()

        if self.metric_name is not None:
            self.metric = np.concatenate(self.metric, axis=0)
            self.metric = np.mean(self.metric, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format(self.mode + '_Metric')]
            for idx, name in enumerate(self.metric_name):
                val = self.metric[0, idx]
                if online_metric:
                    if online_rmse_only:
                        if name == 'RMSE':
                            log_metric_val = val
                            log_dict[self.mode + '_' + name] = val
                        else:
                            pass
                    else:
                        if name == 'RMSE':
                            log_metric_val = val
                        log_dict[self.mode + '_' + name] = val
                self.add_scalar('Metric/' + name, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(name, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_metric = open(self.f_metric, 'a')
            f_metric.write('{:04d} | {}\n'.format(global_step, msg))
            f_metric.close()

            if self.args.test:
                f_metric = open(os.path.join(self.args.test_dir + '/test/result_metric.txt'), 'a')
                f_metric.write('\n{:04d} | {}\n'.format(global_step, msg))
                f_metric.close()

        if online_img:
            # Un-normalization
            rgb = sample['rgb'].detach().clone().data.cpu().numpy()
            dep = sample['dep'].detach().clone().data.cpu().numpy()
            gt = sample['gt'].detach().clone().data.cpu().numpy()
            ipbasicgt = sample['ipbasicgt'].detach().clone().data.cpu().numpy()
            rgb_depth = output['ben_depth'].detach().clone().data.cpu().numpy()
            rgb_mask = output['ben_mask'].detach().clone().data.cpu().numpy()
            rgb_conf = output['ben_conf'].detach().clone().data.cpu().numpy()
            d_depth = output['jin_depth'].detach().clone().data.cpu().numpy()
            d_conf = output['jin_conf'].detach().clone().data.cpu().numpy()
            pred = output['an_depth'].detach().clone().data.cpu().numpy()

            num_summary = gt.shape[0]
            if num_summary > self.args.num_summary:
                num_summary = self.args.num_summary

                rgb = rgb[0:num_summary, :, :, :]
                dep = dep[0:num_summary, :, :, :]
                gt = gt[0:num_summary, :, :, :]
                ipbasicgt = ipbasicgt[0:num_summary, :, :, :]
                rgb_depth = rgb_depth[0:num_summary, :, :, :]
                rgb_mask = rgb_mask[0:num_summary, :, :, :]
                rgb_conf = rgb_conf[0:num_summary, :, :, :]
                d_depth = d_depth[0:num_summary, :, :, :]
                d_conf = d_conf[0:num_summary, :, :, :]
                pred = pred[0:num_summary, :, :, :]

            rgb = np.clip(rgb, a_min=0, a_max=255.0)
            dep = np.clip(dep, a_min=0, a_max=self.args.max_depth)
            gt = np.clip(gt, a_min=0, a_max=self.args.max_depth)
            ipbasicgt = np.clip(ipbasicgt, a_min=0, a_max=self.args.max_depth)
            rgb_depth = np.clip(rgb_depth, a_min=0, a_max=self.args.max_depth)
            rgb_mask = np.clip(rgb_mask, a_min=0, a_max=self.args.max_depth)
            rgb_conf = np.clip(rgb_conf, a_min=0, a_max=1.0)
            d_depth = np.clip(d_depth, a_min=0, a_max=self.args.max_depth)
            d_conf = np.clip(d_conf, a_min=0, a_max=1.0)
            pred = np.clip(pred, a_min=0, a_max=self.args.max_depth)

            list_img = []

            for b in range(0, num_summary):
                rgb_tmp = rgb[b, :, :, :]
                dep_tmp = dep[b, 0, :, :]
                gt_tmp = gt[b, 0, :, :]
                ipbasicgt_tmp = ipbasicgt[b, 0, :, :]
                rgb_depth_tmp = rgb_depth[b, 0, :, :]
                rgb_mask_tmp = rgb_mask[b, 0, :, :]
                rgb_conf_tmp = rgb_conf[b, 0, :, :]
                d_depth_tmp = d_depth[b, 0, :, :]
                d_conf_tmp = d_conf[b, 0, :, :]
                pred_tmp = pred[b, 0, :, :]

                rgb_tmp = rgb_tmp / 255.0
                dep_tmp = 255.0 * dep_tmp / self.args.max_depth
                gt_tmp = 255.0 * gt_tmp / self.args.max_depth
                ipbasicgt_tmp = 255.0 * ipbasicgt_tmp / self.args.max_depth
                rgb_depth_tmp = 255.0 * rgb_depth_tmp / self.args.max_depth
                rgb_mask_tmp = 255.0 * rgb_mask_tmp / self.args.max_depth
                rgb_conf_tmp = 255.0 * rgb_conf_tmp
                d_depth_tmp = 255.0 * d_depth_tmp / self.args.max_depth
                d_conf_tmp = 255.0 * d_conf_tmp
                pred_tmp = 255.0 * pred_tmp / self.args.max_depth

                dep_tmp = cm(dep_tmp.astype('uint8'))
                gt_tmp = cm(gt_tmp.astype('uint8'))
                ipbasicgt_tmp = cm(ipbasicgt_tmp.astype('uint8'))
                rgb_depth_tmp = cm(rgb_depth_tmp.astype('uint8'))
                rgb_mask_tmp = cm(rgb_mask_tmp.astype('uint8'))
                rgb_conf_tmp = cm(rgb_conf_tmp.astype('uint8'))
                d_depth_tmp = cm(d_depth_tmp.astype('uint8'))
                d_conf_tmp = cm(d_conf_tmp.astype('uint8'))
                pred_tmp = cm(pred_tmp.astype('uint8'))

                dep_tmp = np.transpose(dep_tmp[:, :, :3], (2, 0, 1))
                gt_tmp = np.transpose(gt_tmp[:, :, :3], (2, 0, 1))
                ipbasicgt_tmp = np.transpose(ipbasicgt_tmp[:, :, :3], (2, 0, 1))
                rgb_depth_tmp = np.transpose(rgb_depth_tmp[:, :, :3], (2, 0, 1))
                rgb_mask_tmp = np.transpose(rgb_mask_tmp[:, :, :3], (2, 0, 1))
                rgb_conf_tmp = np.transpose(rgb_conf_tmp[:, :, :3], (2, 0, 1))
                d_depth_tmp = np.transpose(d_depth_tmp[:, :, :3], (2, 0, 1))
                d_conf_tmp = np.transpose(d_conf_tmp[:, :, :3], (2, 0, 1))
                pred_tmp = np.transpose(pred_tmp[:, :, :3], (2, 0, 1))

                img = np.concatenate((rgb_tmp, dep_tmp, gt_tmp,
                                      rgb_depth_tmp, rgb_conf_tmp,
                                      rgb_mask_tmp,
                                      d_depth_tmp, d_conf_tmp,
                                      pred_tmp), axis=1)
                list_img.append(img)

            img_total = np.concatenate(list_img, axis=2)
            img_total = torch.from_numpy(img_total)

            log_dict[self.mode + '_' + "examples"] = wandb.Image(img_total, caption=self.mode)

        if 'train' in self.mode:
            log_dict['custom_step_train'] = global_step
        elif 'val' in self.mode:
            log_dict['custom_step_val'] = global_step
        elif 'test' in self.mode:
            log_dict['custom_step_test'] = global_step

        # Log by wandb
        if len(log_dict) != 0 and 'test' not in self.mode:
                wandb.log(log_dict)

        # # Log by tb
        # self.add_image(self.mode + '/images', img_total, global_step)
        #
        # self.flush()

        # Reset
        self.loss = []
        self.metric = []

        return log_metric_val

    def save(self, epoch, idx, sample, output):
        with torch.no_grad():
            if self.args.save_result_only:
                if not self.args.test:
                    self.path_output = '{}/{}/epoch{:04d}'.format(self.log_dir,
                                                              'result_pred', epoch)
                else:
                    self.path_output = '{}/{}/{}'.format(self.log_dir,
                                                                  'test', 'depth_gray')

                os.makedirs(self.path_output, exist_ok=True)

                path_save_pred = '{}/{:010d}.png'.format(self.path_output, idx)

                pred = output[self.args.output].detach()

                pred = torch.clamp(pred, min=0)

                pred = pred[0, 0, :, :].data.cpu().numpy()

                if not self.args.test_not_random_crop:
                    org_size = (352, 1216)
                    pred = pad_rep(pred, org_size)

                pred = (pred*256.0).astype(np.uint16)
                pred = Image.fromarray(pred)
                pred.save(path_save_pred)
            else:
                rgb = sample['rgb'].detach().data.cpu().numpy()
                dep = sample['dep'].detach().data.cpu().numpy()
                gt = sample['gt'].detach().data.cpu().numpy()
                ipbasicgt = sample['gt'].detach().data.cpu().numpy()
                rgb_depth = output['ben_depth'].detach().data.cpu().numpy()
                rgb_mask = output['ben_mask'].detach().data.cpu().numpy()
                rgb_conf = output['ben_conf'].detach().data.cpu().numpy()
                d_depth = output['jin_depth'].detach().data.cpu().numpy()
                d_conf = output['jin_conf'].detach().data.cpu().numpy()
                pred = output['an_depth'].detach().data.cpu().numpy()

                num_summary = gt.shape[0]
                if num_summary > self.args.num_summary:
                    num_summary = self.args.num_summary

                    rgb = rgb[0:num_summary, :, :, :]
                    dep = dep[0:num_summary, :, :, :]
                    gt = gt[0:num_summary, :, :, :]
                    ipbasicgt = ipbasicgt[0:num_summary, :, :, :]
                    rgb_depth = rgb_depth[0:num_summary, :, :, :]
                    rgb_mask = rgb_mask[0:num_summary, :, :, :]
                    rgb_conf = rgb_conf[0:num_summary, :, :, :]
                    d_depth = d_depth[0:num_summary, :, :, :]
                    d_conf = d_conf[0:num_summary, :, :, :]
                    pred = pred[0:num_summary, :, :, :]

                rgb = np.clip(rgb, a_min=0, a_max=255.0)
                dep = np.clip(dep, a_min=0, a_max=self.args.max_depth)
                gt = np.clip(gt, a_min=0, a_max=self.args.max_depth)
                ipbasicgt = np.clip(ipbasicgt, a_min=0, a_max=self.args.max_depth)
                rgb_depth = np.clip(rgb_depth, a_min=0, a_max=self.args.max_depth)
                rgb_mask = np.clip(rgb_mask, a_min=0, a_max=self.args.max_depth)
                rgb_conf = np.clip(rgb_conf, a_min=0, a_max=1.0)
                d_depth = np.clip(d_depth, a_min=0, a_max=self.args.max_depth)
                d_conf = np.clip(d_conf, a_min=0, a_max=1.0)
                pred = np.clip(pred, a_min=0, a_max=self.args.max_depth)

                list_imgv, list_imgh = [], []
                for b in range(0, num_summary):
                    rgb_tmp = rgb[b, :, :, :]
                    dep_tmp = dep[b, 0, :, :]
                    gt_tmp = gt[b, 0, :, :]
                    ipbasicgt_tmp = ipbasicgt[b, 0, :, :]
                    rgb_depth_tmp = rgb_depth[b, 0, :, :]
                    rgb_mask_tmp = rgb_mask[b, 0, :, :]
                    rgb_conf_tmp = rgb_conf[b, 0, :, :]
                    d_depth_tmp = d_depth[b, 0, :, :]
                    d_conf_tmp = d_conf[b, 0, :, :]
                    pred_tmp = pred[b, 0, :, :]

                    rgb_tmp = rgb_tmp / 255.0
                    dep_tmp = 255.0 * dep_tmp / self.args.max_depth
                    gt_tmp = 255.0 * gt_tmp / self.args.max_depth
                    ipbasicgt_tmp = ipbasicgt_tmp / self.args.max_depth
                    rgb_depth_tmp = rgb_depth_tmp / self.args.max_depth
                    rgb_mask_tmp = 255.0 * rgb_mask_tmp / self.args.max_depth
                    rgb_conf_tmp = rgb_conf_tmp
                    d_depth_tmp = d_depth_tmp / self.args.max_depth
                    d_conf_tmp = d_conf_tmp
                    pred_tmp = pred_tmp / self.args.max_depth

                    dep_tmp = (255.0 * cm(dep_tmp)).astype('uint8')
                    gt_tmp = (255.0 * cm(gt_tmp)).astype('uint8')
                    ipbasicgt_tmp = (255.0 * cm(ipbasicgt_tmp)).astype('uint8')
                    rgb_depth_tmp = (255.0 * cm(rgb_depth_tmp)).astype('uint8')
                    rgb_mask_tmp = (255.0 * cm(rgb_mask_tmp)).astype('uint8')
                    rgb_conf_tmp = (255.0 * cm(rgb_conf_tmp)).astype('uint8')
                    d_depth_tmp = (255.0 * cm(d_depth_tmp)).astype('uint8')
                    d_conf_tmp = (255.0 * cm(d_conf_tmp)).astype('uint8')
                    pred_tmp = (255.0 * cm(pred_tmp)).astype('uint8')

                    rgb_tmp = 255.0 * np.transpose(rgb_tmp, (1, 2, 0))
                    rgb_tmp = np.clip(rgb_tmp, 0, 256).astype('uint8')
                    rgb_tmp = Image.fromarray(rgb_tmp, 'RGB')
                    dep_tmp = Image.fromarray(dep_tmp[:, :, :3], 'RGB')
                    gt_tmp = Image.fromarray(gt_tmp[:, :, :3], 'RGB')
                    ipbasicgt_tmp = Image.fromarray(ipbasicgt_tmp[:, :, :3], 'RGB')
                    rgb_depth_tmp = Image.fromarray(rgb_depth_tmp[:, :, :3], 'RGB')
                    rgb_mask_tmp = Image.fromarray(rgb_mask_tmp[:, :, :3], 'RGB')
                    rgb_conf_tmp = Image.fromarray(rgb_conf_tmp[:, :, :3], 'RGB')
                    d_depth_tmp = Image.fromarray(d_depth_tmp[:, :, :3], 'RGB')
                    d_conf_tmp = Image.fromarray(d_conf_tmp[:, :, :3], 'RGB')
                    pred_tmp = Image.fromarray(pred_tmp[:, :, :3], 'RGB')

                    # FIXME
                    list_imgv = [rgb_tmp,
                                 dep_tmp,
                                 gt_tmp,
                                 rgb_depth_tmp,
                                 rgb_conf_tmp,
                                 rgb_mask_tmp,
                                 d_depth_tmp,
                                 d_conf_tmp,
                                 pred_tmp]

                    widths, heights = zip(*(i.size for i in list_imgv))
                    max_width = max(widths)
                    total_height = sum(heights)
                    new_im = Image.new('RGB', (max_width, total_height))
                    y_offset = 0
                    for im in list_imgv:
                        new_im.paste(im, (0, y_offset))
                        y_offset += im.size[1]

                    list_imgh.append(new_im)

                widths, heights = zip(*(i.size for i in list_imgh))
                total_width = sum(widths)
                max_height = max(heights)
                img_total = Image.new('RGB', (total_width, max_height))
                x_offset = 0
                for im in list_imgh:
                    img_total.paste(im, (x_offset, 0))
                    x_offset += im.size[0]

                if not self.args.test:
                    self.path_output = '{}/{}'.format(self.log_dir, 'result_analy')
                else:
                    self.path_output = '{}/{}'.format(self.log_dir, 'test')
                os.makedirs(self.path_output, exist_ok=True)
                if not self.args.test:
                    path_save = '{}/epoch{:04d}_{:08d}_result.png'.format(self.path_output, epoch, idx)
                else:
                    os.makedirs('{}/depth_analy'.format(self.path_output), exist_ok=True)
                    os.makedirs('{}/depth_rgb'.format(self.path_output), exist_ok=True)
                    path_save = '{}/depth_analy/{}'.format(self.path_output, sample['d_path'][0].split('/')[-1])
                    pred_tmp.save('{}/depth_rgb/{}'.format(self.path_output, sample['d_path'][0].split('/')[-1]))
                img_total.save(path_save)






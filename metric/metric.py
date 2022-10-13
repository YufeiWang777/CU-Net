import torch
from . import BaseMetric


class Metric(BaseMetric):
    def __init__(self, args):
        super(Metric, self).__init__(args)

        self.args = args
        self.t_valid = 0.0001

        self.metric_name = [
            'RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3'
        ]

    def evaluate(self, output, sample, mode):
        with torch.no_grad():
            # FIXME
            # pred = output['pred'].detach()
            # gt = sample['gt'].detach()
            pred, gt = output.detach(), sample.detach()

            pred_inv = 1.0 / (pred + 1e-8)
            gt_inv = 1.0 / (gt + 1e-8)

            # For numerical stability
            mask = gt > self.t_valid
            num_valid = mask.sum()

            pred = pred[mask]
            gt = gt[mask]

            # tmp_pred = pred.detach().data.cpu().numpy()
            # tmp_gt = gt.detach().data.cpu().numpy()

            pred_inv = pred_inv[mask]
            gt_inv = gt_inv[mask]

            pred_inv[pred <= self.t_valid] = 0.0
            gt_inv[gt <= self.t_valid] = 0.0

            # RMSE / MAE
            diff = pred - gt
            diff_abs = torch.abs(diff)
            diff_sqr = torch.pow(diff, 2)

            # tmp = diff_sqr.sum()
            rmse = diff_sqr.sum() / (num_valid + 1e-8)
            rmse = torch.sqrt(rmse)

            mae = diff_abs.sum() / (num_valid + 1e-8)

            # iRMSE / iMAE
            diff_inv = pred_inv - gt_inv
            diff_inv_abs = torch.abs(diff_inv)
            diff_inv_sqr = torch.pow(diff_inv, 2)

            irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
            irmse = torch.sqrt(irmse)

            imae = diff_inv_abs.sum() / (num_valid + 1e-8)

            # Rel
            rel = diff_abs / (gt + 1e-8)
            rel = rel.sum() / (num_valid + 1e-8)

            # delta
            r1 = gt / (pred + 1e-8)
            r2 = pred / (gt + 1e-8)
            ratio = torch.max(r1, r2)

            del_1 = (ratio < 1.25).type_as(ratio)
            del_2 = (ratio < 1.25**2).type_as(ratio)
            del_3 = (ratio < 1.25**3).type_as(ratio)

            del_1 = del_1.sum() / (num_valid + 1e-8)
            del_2 = del_2.sum() / (num_valid + 1e-8)
            del_3 = del_3.sum() / (num_valid + 1e-8)

            result = [rmse, mae, irmse, imae, rel, del_1, del_2, del_3]
            result = torch.stack(result)
            result = torch.unsqueeze(result, dim=0).detach()

        return result

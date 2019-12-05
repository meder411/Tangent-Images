import torch
import torch.nn.functional as F

import math
import shutil
import os.path as osp

from mapped_convolution.util import *


def visualize_rgb(rgb, data_format='pano'):
    # Scale back to [0,255]

    mean = torch.tensor(
        [0.5490018535423888, 0.5305735878705738,
         0.477942701493174]).view(-1, 1, 1).to(rgb.get_device())
    std = torch.tensor(
        [0.19923060886633429, 0.20050344344849544,
         0.21290783912717565]).view(-1, 1, 1).to(rgb.get_device())
    C = rgb.shape[0]
    rgb *= std[:C, ...]
    rgb += mean[:C, ...]
    return (255 * rgb).byte()


def visualize_mask(mask):
    '''Visualize the data mask'''
    mask /= mask.max()
    return (255 * mask).byte()


def iou_score(pred_cls, true_cls, nclass=14, drop=[0]):
    """
    compute the intersection-over-union score
    both inputs should be categorical (as opposed to one-hot)
    """
    intersect_ = []
    union_ = []
    for i in range(nclass):
        if i not in drop:
            intersect = ((pred_cls == i).byte() +
                         (true_cls == i).byte()).eq(2).sum().item()
            union = ((pred_cls == i).byte() +
                     (true_cls == i).byte()).ge(1).sum().item()
            intersect_.append(intersect)
            union_.append(union)
    return torch.tensor(intersect_), torch.tensor(union_)


def accuracy(pred_cls, true_cls, nclass=14, drop=[0]):
    if true_cls.max().item() > 13:
        raise Exception()

    positive = torch.histc(true_cls.cpu().float(),
                           bins=nclass,
                           min=0,
                           max=nclass,
                           out=None)
    per_cls_counts = []
    tpos = []
    for i in range(nclass):
        if i not in drop:
            true_positive = ((pred_cls == i).byte() +
                             (true_cls == i).byte()).eq(2).sum().item()
            tpos.append(true_positive)
            per_cls_counts.append(positive[i])

    return torch.tensor(tpos), torch.tensor(per_cls_counts)


class TextureBakedTestingManagerSemSeg(NetworkManager):

    def __init__(self,
                 network,
                 checkpoint_dir,
                 dataloader,
                 path_to_color_map,
                 image_shape,
                 base_order,
                 max_sample_order,
                 evaluation_sample_freq=-1,
                 sample_dir=None,
                 device=None,
                 data_format='pano',
                 drop_unknown=False):

        super().__init__(network, '', None, None, dataloader, None, None, None,
                         0, 0, 0, evaluation_sample_freq, device)

        # Directory to store checkpoints
        self.checkpoint_dir = checkpoint_dir

        # Directory to store output samples
        self.sample_dir = sample_dir

        # Mapping of labels to color
        # 14 x 3
        self.color_map = torch.from_numpy(
            np.loadtxt(path_to_color_map, dtype=np.uint8)[:, 1:])

        # Tangent patches to equirectangular texture
        self.base_order = base_order
        self.max_sample_order = max_sample_order
        self.nearest_resample_to_texture = ResampleToUVTexture(
            image_shape, base_order, max_sample_order, 1, 'nearest').to(device)
        self.bispherical_resample_to_texture = ResampleToUVTexture(
            image_shape, base_order, max_sample_order, 1,
            'bispherical').to(device)
        self.nearest_resample_from_uv_layer = ResampleFromUVTexture(
            image_shape, base_order, max_sample_order, 'nearest').to(device)
        self.bilinear_resample_from_uv_layer = ResampleFromUVTexture(
            image_shape, base_order, max_sample_order, 'bilinear').to(device)

        # Containers to hold validation results
        self.iou = 0
        self.accuracy = 0
        self.intersections = 0
        self.unions = 0
        self.true_positives = 0
        self.per_cls_counts = 0
        self.count = 0

        self.drop_unknown = drop_unknown
        self.data_format=data_format

    def parse_data(self, data):
        '''
        Returns a list of the inputs as first output, a list of the GT as a second output, and a list of the remaining info as a third output. Must be implemented.
        '''

        # Parse the relevant data
        rgb = data[0].to(self.device)
        labels = data[1].to(self.device)
        basename = data[-1]

        rgb, labels = self.pano_data_to_patches(rgb, labels)

        inputs = rgb
        gt = labels
        other = basename

        return inputs, gt, other

    def pano_data_to_patches(self, rgb=None, gt=None):
        """
        rgb : B x 3 x H x W
        gt : B x 1 x H x W

        returns
            RGB: N x B x 3 x H x W
            gt: N x B x 1 x H x W
        """
        if rgb is not None:
            rgb = self.bispherical_resample_to_texture(rgb)
            rgb = rgb.permute(2, 0, 1, 3, 4).contiguous()
            rgb = torch.flip(rgb, (-2, )).contiguous()

        if gt is not None:
            gt = self.nearest_resample_to_texture(gt)
            gt = gt.permute(2, 0, 1, 3, 4).contiguous()
            gt = torch.flip(gt, (-2, )).contiguous()
        return rgb, gt

    def patches_to_pano_data(self, rgb=None, gt_labels=None, pred_labels=None):
        """
        RGB: N x B x 3 x H x W
        gt_labels: N x B x 1 x H x W
        pred_labels: N x B x 1 x H x W

        returns
            rgb : B x 3 x H x W
            gt_labels : B x 1 x H x W
            pred_labels : B x 1 x H x W
        """

        # Process RGB
        if rgb is not None:
            rgb = torch.flip(rgb, (-2, )).contiguous()
            rgb = rgb.permute(1, 2, 0, 3, 4).contiguous()
            rgb = self.bilinear_resample_from_uv_layer(rgb)

        # Process GT
        if gt_labels is not None:
            gt_labels = torch.flip(gt_labels, (-2, )).contiguous()
            gt_labels = gt_labels.permute(1, 2, 0, 3, 4).contiguous()
            gt_labels = self.nearest_resample_from_uv_layer(
                gt_labels.float()).long()

        # Process outputs
        if pred_labels is not None:
            pred_labels = torch.flip(pred_labels, (-2, )).contiguous()
            pred_labels = pred_labels.permute(1, 2, 0, 3, 4).contiguous()
            pred_labels = self.nearest_resample_from_uv_layer(
                pred_labels.float()).long()

        return rgb, gt_labels, pred_labels

    def compute_labels(self, output, gt_labels):
        """Converts raw outputs to numerical labels"""

        pred_labels = F.softmax(output, dim=2)
        pred_labels = pred_labels.argmax(dim=2, keepdim=True)

        # Mask out invalid areas in predictions
        pred_labels[gt_labels == 0] = 0
        pred_labels[gt_labels == 14] = 14

        return pred_labels

    def evaluate(self, checkpoint_path):
        print('Evaluating model....')

        # Put the model in eval mode
        self.network = self.network.eval()

        # Load the checkpoint to test
        self.load_checkpoint(checkpoint_path, True)

        # Reset meter
        self.reset_eval_metrics()

        # Load data
        s = time.time()
        with torch.no_grad():
            for batch_num, data in enumerate(self.test_dataloader):
                print('Evaluating batch {}/{}'.format(
                    batch_num, len(self.test_dataloader)),
                      end='\r')

                # Parse the data
                inputs, gt, other = self.parse_data(data)

                # Run a forward pass on each pass separately
                N, B, C, H, W = inputs.shape
                output = torch.zeros(N, B, 14, H, W).to(self.device)
                for i in range(N):
                    output[i, ...] = self.forward_pass(inputs[i, ...])

                # Compute the evaluation metrics
                self.compute_eval_metrics(output, gt)

                # If trying to save intermediate outputs
                if self.evaluation_sample_freq >= 0:
                    # Save the intermediate outputs
                    if batch_num % self.evaluation_sample_freq == 0:
                        self.save_samples(inputs, gt, other, output)

        # Print a report on the validation results
        print('Evaluation finished in {} seconds'.format(time.time() - s))
        self.print_evaluation_report()

    def forward_pass(self, inputs):
        '''
        Returns the network output
        '''
        output = self.network(inputs)
        if isinstance(output, dict):
            output = output['out']
        if self.drop_unknown:
            output[:, 0] = -1e8
        return output

    def compute_eval_metrics(self, output, gt):
        '''
        Computes metrics used to evaluate the model
        '''
        gt_labels = gt.long()
        pred_labels = self.compute_labels(output, gt_labels)

        # Convert back to pano view
        _, pred_labels, gt_labels = self.patches_to_pano_data(
            None, gt_labels, pred_labels)

        # Compute scores
        int_, uni_ = iou_score(pred_labels, gt_labels)
        true_positive, per_class_count = accuracy(pred_labels, gt_labels)
        self.intersections += int_.float()
        self.unions += uni_.float()
        self.true_positives += true_positive.float()
        self.per_cls_counts += per_class_count.float()
        self.count += output.shape[0]

    def load_checkpoint(self, checkpoint_path=None, weights_only=False):
        '''
        Initializes network and/or optimizer with pretrained parameters
        '''
        if checkpoint_path is not None:
            print('Loading checkpoint \'{}\''.format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            load_partial_model(self.network, checkpoint['state_dict'])
            print('Loaded checkpoint \'{}\' (epoch {})'.format(
                checkpoint_path, checkpoint['epoch']))
        else:
            print('WARNING: No checkpoint found')

    def print_evaluation_report(self):
        '''
        Prints a report of the evaluation results
        '''

        self.iou = self.intersections / torch.clamp_min(self.unions, 1e-6)
        self.accuracy = self.true_positives / torch.clamp_min(self.per_cls_counts, 1e-6)
        mean_iou = self.iou.mean()
        mean_acc = self.accuracy.mean()

        print('Epoch: {}\n'
              '  Avg. IOU: {:.4f}\n'
              '  Avg. Pixel Accuracy: {:.4f}\n\n'.format(
                  self.epoch + 1, mean_iou, mean_acc))

        print('Per-Class IOU')
        print(self.iou)

        print('Per-Class Accuracy')
        print(self.accuracy)

    def save_samples(self, inputs, gt, other, output):

        if self.sample_dir is None:
            print('No sample directory specified')
            return

        # Parse loaded data
        rgb = inputs
        gt_labels = gt.long()
        pred_labels = self.compute_labels(output, gt_labels)
        basename = other

        pano_rgb, pano_gt_labels, pano_pred_labels = self.patches_to_pano_data(
            rgb, gt_labels, pred_labels)
        pano_rgb = pano_rgb[:, :3]

        for b in range(rgb.shape[1]):
            out_dir = osp.join(self.sample_dir, 'pano', basename[b])
            patch_dir = osp.join(out_dir, 'patches')
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(patch_dir, exist_ok=True)

            # Save the pano versions of the image
            io.imsave(
                osp.join(out_dir, 'pano_input.png'),
                visualize_rgb(pano_rgb[b],
                              self.data_format).permute(1, 2,
                                                        0).byte().cpu().numpy())
            io.imsave(
                osp.join(out_dir, 'pano_gt.png'), self.color_map[
                    pano_gt_labels[b].squeeze()].byte().cpu().numpy())
            io.imsave(
                osp.join(out_dir, 'pano_pred.png'), self.color_map[
                    pano_pred_labels[b].squeeze()].byte().cpu().numpy())

            # for p in range(rgb.shape[0]):
            #     io.imsave(
            #         osp.join(patch_dir, 'patch{:06d}_input.png'.format(p)),
            #         visualize_rgb(rgb[p, b, :, ...],
            #                       self.data_format).permute(
            #                           1, 2, 0).byte().cpu().numpy())
            #     io.imsave(
            #         osp.join(patch_dir, 'patch{:06d}_pred.png'.format(p)),
            #         self.color_map[
            #             pred_labels[p, b, 0, ...]].byte().cpu().numpy())
            #     io.imsave(
            #         osp.join(patch_dir, 'patch{:06d}_gt.png'.format(p)),
            #         self.color_map[
            #             gt_labels[p, b, 0, ...]].byte().cpu().numpy())

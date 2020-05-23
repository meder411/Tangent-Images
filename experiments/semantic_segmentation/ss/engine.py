import time
import torch.nn.functional as F

from spherical_distortion.util import *
from spherical_distortion.metrics import *

from .utils import *


class Engine(TrainingEngine):

    def __init__(self,
                 network,
                 name='',
                 train_dataloader=None,
                 test_dataloader=None,
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 num_epochs=20,
                 validation_freq=1,
                 checkpoint_freq=1,
                 visualization_freq=5,
                 display_samples=True,
                 evaluation_sample_freq=-1,
                 logfile=None,
                 checkpoint_root='',
                 sample_root='',
                 op_mode=OpMode.STANDARD,
                 distributed=False,
                 device=None,
                 count_epochs=False,
                 visdom=None,
                 image_shape=None,
                 base_order=0,
                 sample_order=7,
                 data_format='pano',
                 per_patch=False,
                 random_sample_size=0,
                 path_to_color_map=None,
                 eval_format='ico',
                 mean_type='simple',
                 drop_unknown=False,
                 norm_stats=None):

        super().__init__(
            network=network,
            name=name,
            train_dataloader=train_dataloader,
            val_dataloader=test_dataloader,
            test_dataloader=test_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            validation_freq=validation_freq,
            checkpoint_freq=checkpoint_freq,
            visualization_freq=visualization_freq,
            display_samples=display_samples,
            evaluation_sample_freq=evaluation_sample_freq,
            logfile=logfile,
            checkpoint_root=checkpoint_root,
            sample_root=sample_root,
            op_mode=op_mode,
            distributed=distributed,
            device=device,
            count_epochs=count_epochs,
            higher_is_better=True)

        # Mapping of labels to color
        self.color_map = torch.from_numpy(
            np.loadtxt(path_to_color_map, dtype=np.uint8)[:, 1:])

        # 'pano' or perspective ('data') images
        self.data_format = data_format

        # Whether to eval on the projection of each icosahedral face ('ico') or on the re-rendered pano ('rect')
        self.eval_format = eval_format
        
        # Compute a simple average or weight per-class
        self.mean_type = mean_type

        # Whether to run val/test on a single patch at a time vs on the whole image as once. This comes into play with level 10 inputs to address memory limitations
        self.per_patch = per_patch

        if self.data_format == 'pano':
            # Only in pano mode, set up the necessary maps and parameters
            self.random_sample_size = random_sample_size
            self.num_patches = compute_num_faces(base_order)
            self.base_order = base_order
            self.sample_order = sample_order
            self.rect2tan_sample_map = create_equirectangular_to_tangent_images_sample_map(
                image_shape, base_order, sample_order).to(device)
            self.tan2rect_quad_map, self.tan2rect_uv_map = create_tangent_images_to_equirectangular_uv_sample_map(
                image_shape, base_order, sample_order)
            self.tan2rect_quad_map = self.tan2rect_quad_map.to(device)
            self.tan2rect_uv_map = self.tan2rect_uv_map.to(device)
            self.tan_dim = tangent_image_dim(base_order, sample_order)
            if self.eval_format == 'ico':
                self.icosahedron_face_masks = compute_icosahedron_face_mask(
                    base_order, sample_order).to(device)

        # Containers to hold validation results
        self.iou = 0
        self.accuracy = 0
        self.intersections = 0
        self.unions = 0
        self.true_positives = 0
        self.per_cls_counts = 0
        self.mean_val_loss = 0

        # Track the best pixel accuracy recorded so far
        self.best_metric_key = 'best_pixel_accuracy'

        # List of length 2 [Visdom instance, env]
        self.vis = visdom

        # Loss trackers
        self.loss = AverageMeter()
        self.val_loss = AverageMeter()

        self.drop_unknown = drop_unknown
        self.norm_stats = norm_stats

    def parse_data(self, data, train=True):
        """
        Returns a list of the inputs as first output,
        a list of the GT as a second output,
        and a list of the remaining info as a third output.
        Must be implemented.
        """
        # Parse the relevant data
        rgb = data[0].to(self.device)
        labels = data[1].to(self.device)
        basename = data[-1]

        if self.data_format == 'pano':
            rgb, labels = rect2tan(self.rect2tan_sample_map, self.tan_dim, rgb,
                                   labels)

            if train:
                # Randomly sample patches
                k = torch.randperm(self.num_patches)[:self.random_sample_size]
                rgb = rgb[k, ...].contiguous()
                labels = labels[k, ...].contiguous()

            # Collapse patches and batches into one dimension
            c, h, w = rgb.shape[-3:]
            rgb = rgb.view(-1, c, h, w)
            labels = labels.view(-1, 1, h, w)

        inputs = rgb
        gt = labels
        other = basename
        return inputs, gt, other

    @staticmethod
    def compute_labels(output, gt_labels):
        """Converts raw outputs to numerical labels"""
        pred_labels = F.softmax(output, dim=1)
        pred_labels = pred_labels.argmax(dim=1, keepdim=True)

        # Mask out invalid areas in predictions
        pred_labels[gt_labels == 0] = 0
        pred_labels[gt_labels == 14] = 14
        return pred_labels

    @staticmethod
    def parse_output(output):
        """
        For ResNet
        """
        if isinstance(output, dict):
            output = output['out']
        return output

    def forward_pass(self, inputs, train=True):
        """
        Returns the network output
        """
        if self.data_format == 'data':
            output = self.parse_output(self.network(inputs))
        else:
            if self.per_patch > 0 and not train:
                # Run a forward pass on tangent image separately (in groups)
                nb, c, h, w = inputs.shape
                pb = self.per_patch * (nb // self.num_patches)
                inputs = inputs.view(-1, pb, *inputs.shape[-3:])
                n = inputs.shape[0]
                output = torch.zeros(n, pb, 14, h, w).to(self.device)
                for i in range(n):
                    output[i, ...] = self.parse_output(
                        self.network(inputs[i, ...]))
                output = output.view(nb, -1, h, w)
            else:
                output = self.parse_output(self.network(inputs))

        if self.drop_unknown:
            # Not predicting an unknown output
            output[:, 0] = -1e8
        return output

    def compute_loss(self, output, gt, update_running_loss=True):
        """
        Returns a loss as a list where the first element is the total loss
        """
        if self.drop_unknown:
            # NB: criterion must ignore -1 in this case, and not 0
            output = output[:, 1:]
            gt = gt - 1

        loss = self.criterion(output, gt.squeeze(1).long())

        if update_running_loss:
            self.loss.update(loss.item(), output.shape[0])
        return loss

    def reset_eval_metrics(self):
        """
        Resets metrics used to evaluate the model
        """
        self.iou = 0
        self.accuracy = 0
        self.intersections = 0
        self.unions = 0
        self.true_positives = 0
        self.per_cls_counts = 0
        self.mean_val_loss = 0
        self.is_best = False

    def compute_eval_metrics(self, output, gt):
        """
        Computes metrics used to evaluate the model
        """
        # Compute the validation loss
        if self.criterion is not None:
            val_loss = self.compute_loss(output, gt, False)
            self.val_loss.update(val_loss.item(), output.shape[0])

        # Compute the labels from the network outputs
        gt_labels = gt.long()
        pred_labels = self.compute_labels(output, gt_labels)

        if self.data_format == 'pano':
            # Reshape the tangent images in pano mode
            pred_labels = pred_labels.view(self.num_patches, -1,
                                           *pred_labels.shape[1:])
            gt_labels = gt_labels.view(self.num_patches, -1,
                                       *gt_labels.shape[1:])

            if self.eval_format == 'rect':
                # Convert back to pano view
                _, pred_labels, gt_labels = tan2rect(
                    self.tan2rect_quad_map, self.tan2rect_uv_map, None,
                    gt_labels, pred_labels)

        # Compute scores
        int_, uni_ = iou_score(
            pred_labels,
            gt_labels,
            nclass=14,
            drop=(0, ),
            mask=self.icosahedron_face_masks.unsqueeze(1).unsqueeze(1)
            if (self.eval_format == 'ico') and (self.data_format == 'pano')
            else None)
        true_positive, per_class_count = accuracy(
            pred_labels,
            gt_labels,
            nclass=14,
            drop=(0, ),
            mask=self.icosahedron_face_masks.unsqueeze(1).unsqueeze(1)
            if (self.eval_format == 'ico') and (self.data_format == 'pano')
            else None)
        self.intersections += int_.float()
        self.unions += uni_.float()
        self.true_positives += true_positive.float()
        self.per_cls_counts += per_class_count.float()

    def compute_final_metrics(self):
        """
        Called after the eval loop to perform any final computations (like aggregating metrics)
        """
        val_sample_count = torch.tensor(
            [self.val_loss.count], device=self.intersections.get_device())
        accum_val_loss = torch.tensor(
            [self.val_loss.sum], device=self.intersections.get_device())

        if self.distributed:
            # Synchronize the different processes before reduce (otherwise this hangs)
            synchronize()

            # Reduce the accumulated values to pid 0
            torch.distributed.reduce(self.intersections, 0,
                                     torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(self.unions, 0,
                                     torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(self.true_positives, 0,
                                     torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(self.per_cls_counts, 0,
                                     torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(accum_val_loss, 0,
                                     torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(val_sample_count, 0,
                                     torch.distributed.ReduceOp.SUM)

        # Only aggregate on the on pid 0
        if get_rank() != 0:
            return

        # Compute the values
        self.iou = self.intersections / torch.clamp_min(self.unions, 1e-6)
        self.accuracy = self.true_positives / torch.clamp_min(
            self.per_cls_counts, 1e-6)
        self.mean_val_loss = accum_val_loss.float() / torch.clamp_min(
            val_sample_count, 1e-6)

    def initialize_visualizations(self):
        """
        Initializes visualizations
        """

        if get_rank() != 0:
            return

        if self.vis is None:
            return

        self.vis.line(
            X=torch.zeros(1, 1).long(),
            Y=torch.zeros(1, 1).float(),
            win='losses',
            opts=dict(
                title='Loss Plot',
                markers=False,
                xlabel='Iteration',
                ylabel='Loss',
                legend=['Total Loss']))

        self.vis.line(
            X=torch.zeros(1, 2).long(),
            Y=torch.zeros(1, 2).float(),
            win='error_metrics',
            opts=dict(
                title='Semantic Segmentation Error Metrics',
                markers=True,
                xlabel='Epoch',
                ylabel='Error',
                legend=['Mean IOU', 'Mean Class Accuracy']))

        self.vis.line(
            X=torch.zeros(1, 1).long(),
            Y=torch.zeros(1, 1).float(),
            win='val_loss',
            opts=dict(
                title='Mean Validation Loss Plot',
                markers=False,
                xlabel='Epoch',
                ylabel='Loss',
                legend=['Mean Validation Loss']))

    def visualize_loss(self, batch_num, loss):
        """
        Updates the visdom display with the loss
        """
        if get_rank() != 0:
            return

        if self.vis is None:
            return

        self.vis.line(
            X=torch.tensor([batch_num]),
            Y=torch.tensor([self.loss.avg]),
            win='losses',
            update='append',
            opts=dict(legend=['Total Loss']))

    def visualize_samples(self, inputs, gt, other, output):
        """
        Updates the visdom display with samples
        """
        if get_rank() != 0:
            return

        if self.vis is None:
            return

        # Parse loaded data
        rgb = inputs[:, :3, :, :]
        gt_labels = gt.long()
        pred_labels = self.compute_labels(output, gt_labels)

        # In 'pano' mode, visualize a random subset of the tangent images
        if self.data_format == 'pano':
            k = torch.randperm(rgb.shape[0])[:16]
            rgb_img = visualize_rgb(rgb[k, ...], self.norm_stats).cpu()
            pred_img = self.color_map[pred_labels[k, ...].squeeze(
                1)].byte().permute(0, 3, 1, 2).cpu()
            gt_img = self.color_map[gt_labels[k, ...].squeeze(
                1)].byte().permute(0, 3, 1, 2).cpu()

        # In 'data' mode, visualize all the input images
        else:
            rgb_img = visualize_rgb(rgb, self.norm_stats).cpu()
            gt_img = self.color_map[gt_labels.squeeze(1)].byte().cpu().permute(
                0, 3, 1, 2)
            pred_img = self.color_map[pred_labels.squeeze(
                1)].byte().cpu().permute(0, 3, 1, 2)

        self.vis.images(
            rgb_img,
            win='rgb',
            opts=dict(title='Input RGB Image', caption='Input RGB Image'))

        self.vis.images(
            gt_img,
            win='gt',
            opts=dict(
                title='Ground Truth Segmentation',
                caption='Ground Truth Segmentation'))

        self.vis.images(
            pred_img,
            win='output',
            opts=dict(
                title='Output Segmentation', caption='Output Segmentation'))

    def compute_mean_stats(self):
        if self.mean_type == 'weighted':
            mean_acc = (self.accuracy * self.per_cls_counts /
                        self.per_cls_counts.sum()).sum()
            mean_iou = (self.iou * self.per_cls_counts /
                        self.per_cls_counts.sum()).sum()
        elif self.mean_type == 'simple':
            mean_acc = self.accuracy.mean()
            mean_iou = self.iou.mean()
        else:
            assert False, 'Unknown mean type {}'.format(self.mean_type)

        return mean_acc, mean_iou

    def visualize_metrics(self):
        """
        Updates the visdom display with the metrics
        """
        if get_rank() != 0:
            return

        if self.vis is None:
            return

        mca, miou = self.compute_mean_stats()
        miou = miou.cpu()
        mca = mca.cpu()
        accuracy = self.accuracy.cpu()
        iou = self.iou.cpu()

        errors = torch.tensor([miou, mca])
        errors = errors.view(1, -1)
        epoch_expanded = torch.ones_like(errors) * (self.epoch + 1)
        self.vis.line(
            X=epoch_expanded,
            Y=errors,
            win='error_metrics',
            update='append',
            opts=dict(legend=['Mean IOU', 'Mean Class Accuracy']))

        self.vis.line(
            X=torch.tensor([self.epoch + 1]),
            Y=torch.tensor([self.val_loss.avg]),
            win='val_loss',
            update='append',
            opts=dict(legend=['Mean Validation Loss']))

        self.vis.bar(X=iou, win='class_iou', opts=dict(title='Per-Class IOU'))

        accuracy[torch.isnan(accuracy)] = 0
        self.vis.bar(
            X=accuracy,
            win='class_accuracy',
            opts=dict(title='Per-Class Accuracy'))

    def print_batch_report(self, batch_num, loss=None):
        """
        Prints a report of the current batch
        """
        if get_rank() != 0:
            return

        dprint('Epoch: [{0}][{1}/{2}]\t'
               'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
               'Loss {loss.val:.4f} ({loss.avg:.4f})\n\n'.format(
                   self.epoch + 1,
                   batch_num + 1,
                   len(self.train_dataloader),
                   batch_time=self.batch_time_meter,
                   loss=self.loss))

    def print_evaluation_report(self):
        """
        Prints a report of the evaluation results
        """
        if get_rank() != 0:
            return

        mca, miou = self.compute_mean_stats()
        self.logger.info('Epoch: {}\n'
                         '  Avg. IOU: {:.4f}\n'
                         '  Avg. Pixel Accuracy: {:.4f}\n\n'.format(
                             self.epoch + 1, miou, mca))

        self.logger.info('Per-Class IOU')
        self.logger.info(self.iou)
        self.logger.info('Per-Class Accuracy')
        self.logger.info(self.accuracy)
        self.logger.info('\n\n')

        # Also update the best state tracker
        if self.best_metric < mca:
            self.best_metric = mca
            self.is_best = True

    def save_samples(self, inputs, gt, other, output):

        if get_rank() != 0:
            return

        if self.sample_dir is None:
            dprint('No sample directory specified')
            return

        # Parse loaded data
        rgb = inputs
        gt_labels = gt.long()
        pred_labels = self.compute_labels(output, gt_labels)
        basename = other

        pano_rgb, pano_gt_labels, pano_pred_labels = tan2rect(
            self.tan2rect_quad_map, self.tan2rect_uv_map, rgb, gt_labels,
            pred_labels)
        pano_rgb = pano_rgb[:, :3]

        for b in range(rgb.shape[1]):
            out_dir = osp.join(self.sample_dir, 'pano', basename[b])
            patch_dir = osp.join(out_dir, 'patches')
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(patch_dir, exist_ok=True)

            # Save the pano versions of the image
            io.imsave(
                osp.join(out_dir, 'pano_input.png'),
                visualize_rgb(pano_rgb[b], self.data_format).permute(
                    1, 2, 0).byte().cpu().numpy())
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

import torch
import torch.nn as nn

import time


def xavier_init(m):
    '''Provides Xavier initialization for the network weights and
    normally distributes batch norm params'''
    classname = m.__class__.__name__
    if (classname.find('Conv2d') !=
            -1) or (classname.find('ConvTranspose2d') !=
                    -1) or (classname.find('MappedConvolution') != -1) or (
                        classname.find('MappedTransposedConvolution') != -1):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

    if classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# From https://github.com/fyu/drn
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'avg': self.avg
        }

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']


class NetworkManager(object):

    def __init__(self,
                 network,
                 name='',
                 train_dataloader=None,
                 val_dataloader=None,
                 test_dataloader=None,
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 num_epochs=20,
                 validation_freq=1,
                 visualization_freq=5,
                 evaluation_sample_freq=-1,
                 device=None):

        # Name of this experiment
        self.name = name

        # Class instances
        self.network = network
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Training options
        self.num_epochs = num_epochs
        self.validation_freq = validation_freq
        self.visualization_freq = visualization_freq
        self.evaluation_sample_freq = evaluation_sample_freq

        # CUDA info
        self.device = device

        # Some timers
        self.batch_time_meter = AverageMeter()
        self.forward_time_meter = AverageMeter()
        self.backward_time_meter = AverageMeter()

        # Some trackers
        self.epoch = 0

    def forward_pass(self, inputs):
        '''
        Accepts the inputs to the network as a Python list
        Returns the network output
        '''
        return self.network(*inputs)

    def compute_loss(self, output, gt):
        '''
        Returns the total loss
        '''
        return self.criterion(output, gt)

    def backward_pass(self, loss):
        # Computes the backward pass and updates the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_one_epoch(self):

        # Put the model in train mode
        self.network = self.network.train()

        # Load data
        end = time.time()
        for batch_num, data in enumerate(self.train_dataloader):

            # Parse the data into inputs, ground truth, and other
            inputs, gt, other = self.parse_data(data)

            # Run a forward pass
            forward_time = time.time()
            output = self.forward_pass(inputs)
            self.forward_time_meter.update(time.time() - forward_time)

            # Compute the loss(es)
            loss = self.compute_loss(output, gt)

            # Compute the training metrics (if desired)
            self.compute_training_metrics(output, gt)

            # Backpropagation of the total loss
            backward_time = time.time()
            self.backward_pass(loss)
            self.backward_time_meter.update(time.time() - backward_time)

            # Update batch times
            self.batch_time_meter.update(time.time() - end)
            end = time.time()

            # Every few batches
            if batch_num % self.visualization_freq == 0:

                # Visualize the loss
                self.visualize_loss(batch_num, loss)
                self.visualize_training_metrics(batch_num)
                self.visualize_samples(inputs, gt, other, output)
                self.print_batch_report(batch_num, loss)

    def train(self, checkpoint_path=None, weights_only=False):
        print('Starting training')

        # Load pretrained parameters if desired
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, weights_only)
            if weights_only:
                self.initialize_visualizations()
        else:
            # Initialize any training visualizations
            self.initialize_visualizations()

        # Train for specified number of epochs
        for self.epoch in range(self.epoch, self.num_epochs):

            # Run an epoch of training
            self.train_one_epoch()

            if self.epoch % self.validation_freq == 0:
                self.save_checkpoint()
                self.validate()
                self.visualize_metrics()

            # Increment the LR scheduler
            if self.scheduler is not None:
                self.scheduler.step()

    def validate(self):
        print('Validating model....')

        # Put the model in eval mode
        self.network = self.network.eval()

        # Reset meter
        self.reset_eval_metrics()

        # Load data
        s = time.time()
        with torch.no_grad():
            for batch_num, data in enumerate(self.val_dataloader):
                print('Validating batch {}/{}'.format(batch_num,
                                                      len(self.val_dataloader)),
                      end='\r')

                # Parse the data
                inputs, gt, other = self.parse_data(data)

                # Run a forward pass
                output = self.forward_pass(inputs)

                # Compute the evaluation metrics
                self.compute_eval_metrics(output, gt)

                # If trying to save intermediate outputs
                if self.evaluation_sample_freq >= 0:
                    # Save the intermediate outputs
                    if batch_num % self.evaluation_sample_freq == 0:
                        self.save_samples(inputs, gt, other, output)

        # Print a report on the validation results
        print('Validation finished in {} seconds'.format(time.time() - s))
        self.print_evaluation_report()

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

                # Run a forward pass
                output = self.forward_pass(inputs)

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

    def parse_data(self, data):
        '''
        Returns a list of the inputs as first output, a list of the GT as a second output, and a list of the remaining info as a third output. Must be implemented.
        '''
        raise NotImplementedError('Must implement the `parse_data` method')

    def compute_training_metrics(self, output, gt):
        '''
        Computes metrics used to evaluate the training of the model
        '''
        pass

    def reset_eval_metrics(self):
        '''
        Resets metrics used to evaluate the model
        '''
        pass

    def compute_eval_metrics(self, output, gt):
        '''
        Computes metrics used to evaluate the model
        '''
        pass

    def load_checkpoint(self, checkpoint_path=None, weights_only=False):
        '''
        Initializes network with pretrained parameters
        '''
        pass

    def initialize_visualizations(self):
        '''
        Initializes visualizations
        '''
        pass

    def visualize_loss(self, loss):
        '''
        Updates the loss visualization
        '''
        pass

    def visualize_training_metrics(self, batch_num):
        '''
        Updates training metrics visualization
        '''
        pass

    def visualize_samples(self, inputs, gt, other, output):
        '''
        Updates the output samples visualization
        '''
        pass

    def visualize_metrics(self):
        '''
        Updates the metrics visualization
        '''
        pass

    def print_batch_report(self, batch_num, loss):
        '''
        Prints a report of the current batch
        '''
        pass

    def print_evaluation_report(self):
        '''
        Prints a report of the validation results
        '''
        pass

    def save_checkpoint(self):
        '''
        Saves the model state
        '''
        pass

    def save_samples(self, inputs, gt, other, outputs):
        '''
        Saves samples of the network inputs and outputs
        '''
        pass


def save_checkpoint(state, is_best, filename):
    '''Saves a training checkpoints'''
    torch.save(state, filename)
    if is_best:
        basename = osp.basename(filename)  # File basename
        idx = filename.find(basename)  # Where path ends and basename begins
        # Copy the file to a different filename in the same directory
        shutil.copyfile(filename, osp.join(filename[:idx], 'model_best.pth'))


def load_partial_model(model, loaded_state_dict):
    '''Loaded a save model, even if the model is not a perfect match. This will run even if there is are layers from the current network missing in the saved model.
    However, layers without a perfect match will be ignored.'''
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in loaded_state_dict.items() if k in model_dict
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def load_optimizer(optimizer, loaded_optimizer_dict, device):
    '''Loads the saved state of the optimizer and puts it back on the GPU if necessary.  Similar to loading the partial model, this will load only the optimization parameters that match the current parameterization.'''
    optimizer_dict = optimizer.state_dict()
    pretrained_dict = {
        k: v
        for k, v in loaded_optimizer_dict.items()
        if k in optimizer_dict and k != 'param_groups'
    }
    optimizer_dict.update(pretrained_dict)
    optimizer.load_state_dict(optimizer_dict)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
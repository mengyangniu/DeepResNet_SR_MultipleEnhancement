import torch
import time
import os
import math
import random

from torch.utils.data import DataLoader
from utils import SSIM as _SSIM
from collections import OrderedDict
from model import res_sr_net
from dataset import TrainSet, ValSet
from utils import fft_loss

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--upscale-factor', type=int, choices=[2, 3, 4, 8], required=True, help='upscale factor')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'],
                    help='optimizer in the training process')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--lr-halve-step', type=int,
                    help='reduce learning rate by half every certain epochs', required=True)
parser.add_argument('--total-epochs', type=int, default=300,
                    help='total training epochs')
parser.add_argument('--batch-size', type=int,
                    help='training batch size', required=True)
parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'l2', 'smoothl1', 'fftloss'],
                    help='which loss function to use.')
parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                    help='the root dir of checkpoints saved')
parser.add_argument('--save-subdir', type=str, default=None, required=True)
parser.add_argument('--save-prefix', type=str, default=None, required=True)
parser.add_argument('--load-subdir', type=str, default=None)
parser.add_argument('--load-prefix', type=str, default=None)
parser.add_argument('--load-epoch', type=int, default=None)
args = parser.parse_args()

if not os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)


class Trainer(object):
    def __init__(self, model, optimizer, train_loader, val_loader):
        self.device = torch.device('cuda')  # if use cpu, rewrite it to self.device=torch.device('cpu').
        self.epoch = 0
        self.model = torch.nn.DataParallel(model.to(self.device))
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        print('Trainer initialization complete.\n')
        print('model info:', '=' * 100)
        print(self.model)
        print('optimizer info:', '=' * 100)
        print(self.optimizer)
        self.checkpoint_info = {}

        # uncomment following 2 lines if want to add more random factors
        # torch.manual_seed(random.randint(0, 100))
        # torch.cuda.manual_seed(random.randint(0, 100))

    def save_checkpoint(self):
        """
        save checkpoint dict to a specified directory for every training epoch.
        checkpoint dict contains model's params, optimizer, epoch's training information and validation information.
        saving dir is set to '<checkpoint dir>/<save subdir>/<save prefix>-<current epoch>.dict'.
        """
        save_dir = os.path.join(args.checkpoint_dir, args.save_subdir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        state_dict = self.model.state_dict()

        # since model can be set to be paralleled by 'torch.nn.DataParallel(model)',
        # we save the model's dict without parallel.
        if 'module' in state_dict.keys().__repr__():
            new_odict = OrderedDict()
            for k, v in state_dict.items():
                new_odict[k.split('module.')[1]] = v
            state_dict = new_odict
        self.checkpoint_info['state_dict'] = state_dict
        torch.save(self.checkpoint_info,
                   os.path.join(save_dir, '{}_{}.dict'.format(args.save_prefix, self.epoch)))
        print('checkpoint {} saved to {}'.format(self.epoch, save_dir))

    def load_checkpoint(self):
        """
        load params in a specified checkpoint into the model and print checkpoint information.
        """
        print('loading checkpoint from {}/{}/{}-epoch{}\n'.format(args.checkpoints_dir, args.load_subdir,
                                                                  args.load_prefix,
                                                                  args.load_epoch))
        checkpoint = torch.load(os.path.join(args.checkpoints_dir, args.load_subdir,
                                             '{}_{}.dict'.format(args.load_prefix, args.load_epoch)))
        self.model.load_state_dict(checkpoint['state_dict'])
        print('loading checkpoint info:')
        print('optimizer:', self.checkpoint_info['optimizer'])
        print('avg_training_loss: ', self.checkpoint_info['avg_training_loss'])
        print('validation:', self.checkpoint_info['validation'])
        print('loading complete.\n\n')
        print('-' * 50)

    def lr_step(self):
        """
        reduce the learning rate by a half every <args.lr_halve_step> steps.
        """
        lr = args.lr / (2 ** (self.epoch // args.lr_halve_step))
        print('LR = {}'.format(lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_one_epoch(self):
        """
        train the model for 1 epoch with l1 loss or l2 loss or smooth l1 loss.
        save information into checkpoint dict and also print it.
        """
        self.checkpoint_info = {}
        t0 = time.time()
        print('\n{}epoch {}{}'.format('-' * 25, self.epoch, '-' * 25))
        self.model.train(True)
        self.lr_step()

        # use 4 loss functions in total.
        L1 = torch.nn.L1Loss()
        total_L1_loss = 0
        L2 = torch.nn.MSELoss()
        total_L2_loss = 0
        SmoothL1 = torch.nn.SmoothL1Loss()
        total_SmoothL1_Loss = 0
        total_FFT_loss = 0

        # display training loss every 5% epoch.
        display_times = 20
        display_count = 0

        # save some info into checkpoint dict.
        self.checkpoint_info['optimizer'] = self.optimizer
        self.checkpoint_info['avg_training_loss'] = {}

        # starting to train
        for batch_index, (LR, HR) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            LR = LR.to(self.device)
            HR = HR.to(self.device)
            SR = self.model(LR)

            # calculate 4 loss functions and choose one to train the model.
            L1_loss = L1(SR, HR)
            L2_loss = L2(SR, HR)
            Smooth_L1_loss = SmoothL1(SR, HR)
            FFT_loss = fft_loss(SR, HR)
            total_L1_loss += L1_loss.item()
            total_L2_loss += L2_loss.item()
            total_SmoothL1_Loss += Smooth_L1_loss.item()
            total_FFT_loss += FFT_loss.item()
            if args.loss == 'l1':
                loss = L1_loss
            elif args.loss == 'l2':
                loss = L2_loss
            elif args.loss == 'smoothl1':
                loss = Smooth_L1_loss
            elif args.loss == 'fftloss':
                loss = FFT_loss
            else:
                raise ValueError('loss incorrect.')
            loss.backward()
            self.optimizer.step()

            # print training information.
            if batch_index % (len(self.train_loader) // display_times) == 0:
                print('{:.0f}%\tl1_loss={:.8f}\tl2_loss={:.8f}\tsmoothl1_loss={:.8f}\tfft_loss={:.8f}'.format(
                    100 * display_count / display_times,
                    L1_loss.item(), L2_loss.item(), Smooth_L1_loss.item(), FFT_loss.item()))
                display_count += 1

        self.checkpoint_info['optimizer'] = self.optimizer
        self.checkpoint_info['avg_training_loss']['l1'] = total_L1_loss / len(self.train_loader)
        self.checkpoint_info['avg_training_loss']['l2'] = total_L2_loss / len(self.train_loader)
        self.checkpoint_info['avg_training_loss']['sml1'] = total_SmoothL1_Loss / len(self.train_loader)
        self.checkpoint_info['avg_training_loss']['fft'] = total_FFT_loss / len(self.train_loader)

        print('epoch_avg_l1_loss   = {:.16f}'.format(total_L1_loss / len(self.train_loader)))
        print('epoch_avg_l2_loss   = {:.16f}'.format(total_L2_loss / len(self.train_loader)))
        print('epoch_avg_sml1_loss = {:.16f}'.format(total_SmoothL1_Loss / len(self.train_loader)))
        print('epoch_avg_fft_loss  = {:.16f}'.format(total_FFT_loss / len(self.train_loader)))
        print('training epoch {} complete. ({:.2f}s)'.format(self.epoch, time.time() - t0))

    def validate(self):
        """
        validate the model by loss and psnr and ssim on several test data sets.
        """
        t0 = time.time()
        print('{}validation{}'.format('-' * 1, '-' * 1))
        self.model.train(False)
        L1 = torch.nn.L1Loss()
        L2 = torch.nn.MSELoss()
        SmoothL1 = torch.nn.SmoothL1Loss()
        SSIM = _SSIM()
        self.checkpoint_info['validation'] = {}
        for set_name, val_loader in self.val_loader.items():

            # init total loss
            total_l1_loss = 0
            total_l2_loss = 0
            total_sml1_loss = 0
            total_psnr = 0
            total_ssim = 0

            # iterator in every test data set
            for LR, HR in val_loader:
                LR, HR = LR.to(self.device), HR.to(self.device)
                SR = self.model(LR)

                l1_loss = L1(SR, HR).item()
                l2_loss = L2(SR, HR).item()
                sml1_loss = SmoothL1(SR, HR).item()
                psnr = 10 * math.log10(1 / l2_loss)
                ssim = SSIM(SR, HR).item()

                total_l1_loss += l1_loss
                total_l2_loss += l2_loss
                total_sml1_loss += sml1_loss
                total_psnr += psnr
                total_ssim += ssim

            avg_psnr = total_psnr / len(val_loader)
            avg_ssim = total_ssim / len(val_loader)
            avg_l1_loss = total_l1_loss / len(val_loader)
            avg_l2_loss = total_l2_loss / len(val_loader)
            avg_sml1_loss = total_sml1_loss / len(val_loader)
            self.checkpoint_info['validation']['avg_psnr_of_{}'.format(set_name)] = avg_psnr
            self.checkpoint_info['validation']['avg_ssim_of_{}'.format(set_name)] = avg_ssim
            self.checkpoint_info['validation']['avg_l1loss_of_{}'.format(set_name)] = avg_l1_loss
            self.checkpoint_info['validation']['avg_l2loss_of_{}'.format(set_name)] = avg_l2_loss
            self.checkpoint_info['validation']['avg_sml1loss_of_{}'.format(set_name)] = avg_sml1_loss
            print('avg_psnr on {}     \t= {:.8f}'.format(set_name, avg_psnr))
            print('avg_ssim on {}     \t= {:.8f}'.format(set_name, avg_ssim))
            print('avg_l1_loss on {}  \t= {:.8f}'.format(set_name, avg_l1_loss))
            print('avg_l2_loss on {}  \t= {:.8f}'.format(set_name, avg_l2_loss))
            print('avg_sml1_loss on {}\t= {:.8f}'.format(set_name, avg_sml1_loss))

        print('validation complete. ({:.2f}s)'.format(time.time() - t0))


if __name__ == '__main__':
    model = res_sr_net(img_channel=3, base_channel=40, wn=False, bn=True, conv_dcp=True, ca=True,
                       upscale_factor=args.upscale_factor, num_blocks=1, wa_rate=1.2)
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise ValueError('optimizer {} not defined yet.'.format(args.optimizer))

    training_dataset_json_file = '/SSD2/SR/data/lst/train_fixHR_192to64_nooverlap_BICUBIC_DIV2K_55510.json'
    train_set = TrainSet(json_file=training_dataset_json_file)
    train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

    # use following validation data sets:
    val_set = {'Set5': ValSet(json_file='/SSD2/SR/data/lst/val_Set5_X3_BICUBIC_pair.json'),
               'Set14': ValSet(json_file='/SSD2/SR/data/lst/val_Set14_X3_BICUBIC_pair.json')}
    val_loader = {key: DataLoader(dataset=value, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
                  for key, value in val_set.items()}

    trainer = Trainer(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader)

    if args.load_subdir and args.load_prefix and args.load_epoch:
        trainer.load_checkpoint()
    elif not (args.load_subdir or args.load_prefix or args.load_epoch):
        pass
    else:
        raise ValueError('load_dir & load_prefix & load_epoch should all be filled or empty.')
    trainer.checkpoint_info['trainset_json_file'] = training_dataset_json_file
    for trainer.epoch in range(args.total_epochs):
        trainer.train_one_epoch()
        trainer.validate()
        if args.save_subdir and args.save_prefix:
            trainer.save_checkpoint()

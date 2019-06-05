# -*- coding: UTF-8 -*-

import torch.utils.data.dataset
import json
import PIL.Image as Image
import torchvision.transforms.functional as f
import random


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, json_file, mode='RGB'):
        """
        Args:
            :param json_file(string): path of json file
            :param mode(string): accept 'RGB' or 'YCbCr'
        """
        torch.utils.data.Dataset.__init__(self)
        self.image_pair_list = json.load(open(json_file, 'r'))
        random.shuffle(self.image_pair_list)
        if mode not in ['RGB', 'YCbCr']:
            raise ValueError('mode incorrect')
        self.mode = mode

    def __getitem__(self, index):
        """
        according to the json file and current index,
        load the LR image and its corresponding HR image,
        execute augmentation,
        convert to 3d tensor.

        Args:
            :param index: which image pair should read
            :return: LR-HR tensor pair
        """
        path = self.image_pair_list[index]
        LR_path = path['LR']
        HR_path = path['HR']

        LR = Image.open(LR_path).convert(self.mode)
        HR = Image.open(HR_path).convert(self.mode)

        LR, HR = CustomTransform(LR, HR,
                                 hflip=True,
                                 vflip=False,
                                 rotate=True,
                                 adjust_brightness=False,
                                 adjust_contrast=False,
                                 adjust_saturation=False,
                                 adjust_hue=False,
                                 adjust_gamma=False)

        LR = f.to_tensor(LR)
        HR = f.to_tensor(HR)

        return LR, HR

    def __len__(self):
        return len(self.image_pair_list)


class ValSet(torch.utils.data.Dataset):
    """
    basically an augmentation-free version of class 'TrainSet'.
    """

    def __init__(self, json_file, mode='RGB'):
        torch.utils.data.Dataset.__init__(self)
        self.image_pair_list = json.load(open(json_file, 'r'))
        random.shuffle(self.image_pair_list)
        if mode not in ['RGB', 'YCbCr']:
            raise ValueError('mode incorrect')
        self.mode = mode

    def __getitem__(self, index):
        HR_path = self.image_pair_list[index]['HR']
        LR_path = self.image_pair_list[index]['LR']

        LR = Image.open(LR_path).convert(self.mode)
        HR = Image.open(HR_path).convert(self.mode)

        LR, HR = f.to_tensor(LR), f.to_tensor(HR)
        return LR, HR

    def __len__(self):
        return len(self.image_pair_list)


def CustomTransform(LR, HR,
                    adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue, adjust_gamma, rotate, hflip,
                    vflip):
    """
    to execute data augmentation.
    except :param LR and :param HR, all other params accept bool args.
    """
    if adjust_brightness:
        brightness_factor = random.gauss(1, 0.05)
        while brightness_factor > 2 or brightness_factor < 0:
            brightness_factor = random.gauss(1, 0.05)
        HR = f.adjust_brightness(HR, brightness_factor)
        LR = f.adjust_brightness(LR, brightness_factor)

    if adjust_contrast:
        contrast_factor = random.gauss(1, 0.05)
        while contrast_factor > 2 or contrast_factor < 0:
            contrast_factor = random.gauss(1, 0.05)
        HR = f.adjust_contrast(HR, contrast_factor)
        LR = f.adjust_contrast(LR, contrast_factor)

    if adjust_saturation:
        saturation_factor = random.gauss(1, 0.05)
        while saturation_factor > 2 or saturation_factor < 0:
            saturation_factor = random.gauss(1, 0.05)
        HR = f.adjust_saturation(HR, saturation_factor)
        LR = f.adjust_saturation(LR, saturation_factor)

    if adjust_hue:
        hue_factor = random.gauss(0, 0.025)
        while hue_factor > 0.5 or hue_factor < -0.5:
            hue_factor = random.gauss(0, 0.025)
        HR = f.adjust_hue(HR, hue_factor)
        LR = f.adjust_hue(LR, hue_factor)

    if adjust_gamma:
        gamma = random.gauss(1, 0.025)
        while gamma > 0.5 or gamma < -0.5:
            gamma = random.gauss(0, 0.025)
        HR = f.adjust_gamma(HR, gamma)
        LR = f.adjust_gamma(LR, gamma)

    if rotate:
        angle = random.choice([-90, 0, 90])
        HR = f.rotate(HR, angle)
        LR = f.rotate(LR, angle)

    if hflip:
        flip = random.choice([True, False])
        if flip:
            HR = f.hflip(HR)
            LR = f.hflip(LR)

    if vflip:
        flip = random.choice([True, False])
        if flip:
            HR = f.vflip(HR)
            LR = f.vflip(LR)

    return LR, HR

from __future__ import division
import torch
import torch.nn.functional
import random
import numpy as np
import numbers
import torchvision.transforms.functional as F
import scipy.ndimage as ndimage

class RandomHorizontalFlip(object):
    """
    Randomly horizontally flips the given torch.floatTensor with a probability of 0.5
    All in [Channels, Height, Width] layout, can be type torch.Tensor
    """
    def __call__(self, sample):
        if random.random() < 0.5:

            for k in sample.keys():
                if 'flow' in k:
                    sample[k] = torch.flip(sample[k], [-1])
                    sample[k][0, :, :] *= -1
                else:
                    for kk in sample[k].keys():
                        if sample[k][kk] is not None and isinstance(sample[k][kk], (torch.Tensor)):
                            sample[k][kk] = torch.flip(sample[k][kk], [-1])


        return sample


class RandomVerticalFlip(object):
    """
    Randomly horizontally flips the given torch.floatTensor with a probability of 0.5
    All in [Channels, Height, Width] layout, can be type torch.Tensor
    """
    def __call__(self, sample):
        if random.random() < 0.5:

            for k in sample.keys():
                if 'flow' in k:
                    sample[k] = torch.flip(sample[k], [-2])
                    sample[k][1, :, :] *= -1
                else:
                    for kk in sample[k].keys():
                        if sample[k][kk] is not None and isinstance(sample[k][kk], (torch.Tensor)):
                            sample[k][kk] = torch.flip(sample[k][kk], [-2])
        return sample

class ToHWC(object):
    """
    CHW->HWC, only tensor
    """
    def __call__(self, sample):
        for k in sample.keys():
            if 'flow' in k:
                if len(sample[k].shape) == 3:
                    if sample[k].shape[-1] < 4:
                        pass
                    else:
                        sample[k] = sample[k].permute(1, 2, 0)
            else:
                for kk in sample[k].keys():
                    if len(sample[k][kk].shape) == 3:
                        if sample[k][kk].shape[-1] < 4:
                            pass
                        else:
                            sample[k][kk] = sample[k][kk].permute(1, 2, 0)
                        sample[k][kk].long()

        return sample

class ToTensorCHW(object):
    def __call__(self, sample):
        for k in sample.keys():
            if 'flow' in k:
                assert isinstance(sample[k], np.ndarray) or isinstance(sample[k], torch.Tensor)
                if len(sample[k].shape) == 4:
                    assert sample[k].shape[0] == 1
                    sample[k] = sample[k].squeeze()
                assert len(sample[k].shape) in (2, 3)
                if isinstance(sample[k], np.ndarray): sample[k] = torch.FloatTensor(sample[k])
                if len(sample[k].shape) == 2:
                    sample[k] = 255 * sample[k][..., None].repeat(1, 1, 3)
                if len(sample[k].shape) == 3:
                    if sample[k].shape[0] < 4:
                        pass
                    else:
                        sample[k] = sample[k].permute(2, 0, 1)
            else:
                for kk in sample[k].keys():
                    assert isinstance(sample[k][kk], np.ndarray) or isinstance(sample[k][kk], torch.Tensor)
                    if len(sample[k][kk].shape) == 4:
                        assert sample[k][kk].shape[0] == 1
                        sample[k][kk] = sample[k][kk].squeeze()
                    assert len(sample[k][kk].shape) in (2, 3)
                    if isinstance(sample[k][kk], np.ndarray): sample[k][kk] = torch.FloatTensor(sample[k][kk])
                    if len(sample[k][kk].shape) == 2:
                        sample[k][kk] = 255 * sample[k][kk][..., None].repeat(1, 1, 3)
                    if len(sample[k][kk].shape) == 3:
                        if sample[k][kk].shape[0] < 4:
                            pass
                        else:
                            sample[k][kk] = sample[k][kk].permute(2, 0, 1)
                    sample[k][kk].long()
        return sample

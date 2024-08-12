import os
import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import CMPDataset


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    #shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = CMPDataset(opt)

    #sampler = get_bal_sampler(dataset) if opt.class_bal else None
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=True,
                                              sampler=None,
                                              num_workers=int(opt.num_threads))
    return data_loader

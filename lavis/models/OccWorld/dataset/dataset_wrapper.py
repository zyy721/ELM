
import numpy as np, torch
from torch.utils import data
import torch.nn.functional as F
from copy import deepcopy
from mmengine import MMLogger
logger = MMLogger.get_instance('genocc')
from . import OPENOCC_DATAWRAPPER


@OPENOCC_DATAWRAPPER.register_module()
class tpvformer_dataset_nuscenes(data.Dataset):
    def __init__(
            self, 
            in_dataset, 
            phase='train', 
        ):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.phase = phase

    def __len__(self):
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        occ, cond0, cond1 = self.point_cloud_dataset[index]
        return (
            torch.from_numpy(occ).unsqueeze(0),
            torch.from_numpy(cond0).unsqueeze(0),
            torch.from_numpy(cond1).unsqueeze(0)
        )


def custom_collate_fn_temporal(data):
    data_tuple = []
    for i, item in enumerate(data[0]):
        if isinstance(item, torch.Tensor):
            data_tuple.append(torch.stack([d[i] for d in data]))
        elif isinstance(item, (dict, str)):
            data_tuple.append([d[i] for d in data])
        elif item is None:
            data_tuple.append(None)
        else:
            raise NotImplementedError
    return data_tuple

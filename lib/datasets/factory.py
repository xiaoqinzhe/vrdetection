from datasets.vg_hdf5 import vg_hdf5
from vrd import vrd
from fast_rcnn.config import cfg

def get_db(split, num_im = -1):
    dataset_name = cfg.DATASET
    if dataset_name == 'vrd':
        return vrd(cfg.VRD_DIR, split, num_im)
    elif dataset_name == 'vg':
        return vg_hdf5('VG-SGG.h5', 'VG-SGG-dicts.json', 'imdb_1024.h5', 'proposals.h5', split=split, num_im=num_im)
    else:
        raise AttributeError("dataset name does not exist")

def get_val_db(num_im = -1):
    dataset_name = cfg.DATASET
    if dataset_name == 'vrd':
        return vrd(cfg.VRD_DIR, 1, num_im)
    elif dataset_name == 'vg':
        return vg_hdf5('VG-SGG.h5', 'VG-SGG-dicts.json', 'imdb_1024.h5', 'proposals.h5', split=1, num_im=num_im)
    else:
        raise AttributeError("dataset name does not exist")
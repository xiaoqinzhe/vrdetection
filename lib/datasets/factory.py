from datasets.vg import vg
from vrd import vrd
from fast_rcnn.config import cfg

def get_db(split, num_im = -1):
    dataset_name = cfg.DATASET
    if dataset_name == 'vrd':
        return vrd(cfg.VRD_DIR, split, num_im)
    elif dataset_name == 'vg':
        return vg(cfg.VG_DIR, split=split, num_im=num_im)
    else:
        raise AttributeError("dataset name does not exist")

def get_val_db(num_im = -1):
    dataset_name = cfg.DATASET
    if dataset_name == 'vrd':
        return vrd(cfg.VRD_DIR, 1, num_im)
    elif dataset_name == 'vg':
        return vg(cfg.VG_DIR, split=2, num_im=num_im)
    else:
        raise AttributeError("dataset name does not exist")

def get_detections_filename(iter=75000, net="res50"):
    dataset_name = cfg.DATASET
    if dataset_name == 'vrd':
        return "tf_faster_rcnn/output/{}/vrd_test/default/{}_faster_rcnn_iter_{}/detections.pkl".format(net, net, iter)
    elif dataset_name == 'vg':
        return ""
    else:
        raise AttributeError("dataset name does not exist")
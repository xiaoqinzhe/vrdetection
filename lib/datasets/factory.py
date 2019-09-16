from datasets.vg import vg
from datasets.vrd import vrd
from fast_rcnn.config import cfg

def get_db(split, num_im = -1):
    dataset_name = cfg.DATASET
    if dataset_name == 'vrd':
        return vrd('train', num_im=num_im)
    elif dataset_name == 'vg_drnet':
        return vrd('train', version='drnet', num_im=num_im)
    elif dataset_name == 'vg_msdn':
        return vrd('train', version='msdn', num_im=num_im)
    elif dataset_name == 'vg_vtranse':
        return vrd('train', version='vtranse', num_im=num_im)
    else:
        raise AttributeError("dataset name does not exist")

def get_val_db(num_im = -1):
    dataset_name = cfg.DATASET
    if dataset_name == 'vrd':
        return vrd('test', num_im=num_im)
    elif dataset_name == 'vg_drnet':
        return vrd('test', version='drnet', num_im=num_im)
    elif dataset_name == 'vg_msdn':
        return vrd('test', version='msdn', num_im=num_im)
    elif dataset_name == 'vg_vtranse':
        return vrd('test', version='vtranse', num_im=num_im)
    else:
        raise AttributeError("dataset name does not exist")

def get_detections_filename(iter=450000, net="res50"):
# def get_detections_filename(iter=75000, net="vgg16"):
    dataset_name = cfg.DATASET
    if dataset_name == 'vrd':
        iter = 75000
        net = 'res50'
        return "tf_faster_rcnn/output/{}/vrd_test/default/{}_faster_rcnn_iter_{}/detections.npy".format(net, net, iter)
    elif dataset_name == 'vg_drnet':
        iter = 450000
        net = 'res50'
        return "tf_faster_rcnn/output/{}/vg_drnet_test/default/{}_faster_rcnn_iter_{}/detections.npy".format(net, net, iter)
    elif dataset_name == 'vg_msdn':
        iter = 450000
        net = 'res50'
        return "tf_faster_rcnn/output/{}/vg_msdn_test/default/{}_faster_rcnn_iter_{}/detections.npy".format(net, net, iter)
    elif dataset_name == 'vg_vtranse':
        iter = 450000
        net = 'res50'
        return "tf_faster_rcnn/output/{}/vg_vtranse_test/default/{}_faster_rcnn_iter_{}/detections.npy".format(net, net, iter)
    else:
        raise AttributeError("dataset name does not exist")
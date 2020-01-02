from datasets.vg import vg
from datasets.vrd import vrd
from fast_rcnn.config import cfg

def get_db(num_im = -1):
    dataset_name = cfg.DATASET
    path = './data/'
    if dataset_name.startswith("vg"):
        path += "vg/"
    return vrd('train', dataset_name, path, num_im=num_im)

def get_val_db(num_im = -1):
    dataset_name = cfg.DATASET
    path = './data/'
    if dataset_name.startswith("vg"):
        path += "vg/"
    return vrd('test', dataset_name, path, num_im=num_im)

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
    elif dataset_name == 'tl_vrd':
        iter = 90000
        net = 'vgg16'
        return "tf_faster_rcnn/output/{}/tl_vrd_test/default/{}_faster_rcnn_iter_{}/detections.npy".format(net, net, iter)
    elif dataset_name == 'tl_vg':
        iter = 150000
        net = 'vgg16'
        return "tf_faster_rcnn/output/{}/tl_vg_test/default/{}_faster_rcnn_iter_{}/detections.npy".format(net, net, iter)
    else:
        raise AttributeError("dataset name does not exist")

# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# --------------------------------------------------------

"""Factory method for easily getting networks by name."""

__sets = {}

from networks.models import *
from networks.omodels import *
from networks.attnets import *
from networks.graphnet import *
from networks.cainet import *
from networks.weightednet import *
from networks.sptnet import *

__sets['vrdnet'] = vrdnet # VRD baseline
__sets['vggnet'] = basenet # VRD baseline
__sets['vtranse'] = vtranse # VtransE
__sets['sptnet'] = sptnet # sptnet
__sets['sptnet2'] = sptnet2 # sptnet
__sets['sptnet3'] = sptnet3 # sptnet
__sets['zoomnet'] = zoomnet # sptnet
__sets['zoomnet2'] = zoomnet2 # sptnet
__sets['attnet'] = attnet # sptnet
__sets['ctxnet'] = ctxnet
__sets['ctxnet2'] = ctxnet2
__sets['visualnet'] = visualnet
__sets['multinet'] = multinet
__sets['graphnet'] = graphnet
__sets['grunet'] = grunet
__sets['cainet'] = cainet
__sets['weightednet'] = weightednet
__sets['weightednet2'] = weightednet2
__sets['weightednet3'] = weightednet3
__sets['ranknet'] = ranknet
__sets['simplenet'] = simplenet
# __sets['dual_graph_vrd_avgpool'] = dual_graph_vrd_avgpool  # avg pooling baseline
# __sets['dual_graph_vrd_maxpool'] = dual_graph_vrd_maxpool  # max pooling baseline
# __sets['dual_graph_vrd_final'] = dual_graph_vrd_final  # final model


def get_network(name):
    """Get a network by name."""
    if not name in  __sets:
        raise KeyError('Unknown network: {}'.format(name))
    return __sets[name]

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()

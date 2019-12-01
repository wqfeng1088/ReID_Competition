from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cross_entropy_loss import CrossEntropyLoss
from .hard_mine_triplet_loss import TripletLoss
from .oim_loss import OIMLoss
from .center_loss import CenterLoss
from .local_loss import local_loss, Local_TripletLoss
from .ranked_loss import RankedLoss
from .focal_loss import FocalLoss

__model_support_loss_factory = {
    'CrossEntropyLoss': CrossEntropyLoss,
    'TripletLoss': TripletLoss,
    'OIMLoss': OIMLoss,
    'CenterLoss': CenterLoss,
    'Local_TripletLoss': Local_TripletLoss,
    'local_loss': local_loss,
    'rank_loss': RankedLoss,
    'focal_loss': FocalLoss,
    # OtherLoss  # you can write other loss in here
}


def show_avi_loss():
    """Displays available loss design.
    Examples::
        >>> from torchreid import losses
        >>> losses.show_avi_loss()
    """
    print(list(__model_support_loss_factory.keys()))


def DeepSupervision(criterion, xs, y):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss
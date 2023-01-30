from .barlow_twins import BarlowTwins
from .bootstrap import BootstrapLatent
from .infonce import InfoNCE, InfoNCESP, DebiasedInfoNCE, HardnessInfoNCE, HardMixingLoss, RingLoss, DynamicLoss
from .jsd import JSD, DebiasedJSD, HardnessJSD
from .losses import Loss
from .triplet import TripletMargin, TripletMarginSP
from .vicreg import VICReg

__all__ = [
    'Loss',
    'InfoNCE',
    'InfoNCESP',
    'DebiasedInfoNCE',
    'HardnessInfoNCE',
    'JSD',
    'DebiasedJSD',
    'HardnessJSD',
    'TripletMargin',
    'TripletMarginSP',
    'VICReg',
    'BarlowTwins',
    'HardMixingLoss',
    'RingLoss',
    'DynamicLoss'
]

classes = __all__

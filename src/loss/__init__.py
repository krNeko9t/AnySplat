from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_opacity import LossOpacity, LossOpacityCfgWrapper
from .loss_depth_gt import LossDepthGT, LossDepthGTCfgWrapper
from .loss_lod import LossLOD, LossLODCfgWrapper
from .loss_depth_consis import LossDepthConsis, LossDepthConsisCfgWrapper
from .loss_normal_consis import LossNormalConsis, LossNormalConsisCfgWrapper
from .loss_chamfer_distance import LossChamferDistance, LossChamferDistanceCfgWrapper
from .loss_mvc import LossMVC, LossMvcCfgWrapper
from .loss_disc import LossDisc, LossDiscCfgWrapper
from .loss_phys import LossPhys, LossPhysCfgWrapper
from .loss_phys_prop import LossPhysProp, LossPhysPropCfgWrapper
LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossOpacityCfgWrapper: LossOpacity,
    LossDepthGTCfgWrapper: LossDepthGT,
    LossLODCfgWrapper: LossLOD,
    LossDepthConsisCfgWrapper: LossDepthConsis,
    LossNormalConsisCfgWrapper: LossNormalConsis,
    LossChamferDistanceCfgWrapper: LossChamferDistance,
    LossMvcCfgWrapper: LossMVC,
    LossDiscCfgWrapper: LossDisc,
    LossPhysCfgWrapper: LossPhys,
    LossPhysPropCfgWrapper: LossPhysProp,
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper | LossOpacityCfgWrapper | LossDepthGTCfgWrapper | LossLODCfgWrapper | LossDepthConsisCfgWrapper | LossNormalConsisCfgWrapper | LossChamferDistanceCfgWrapper | LossMvcCfgWrapper | LossDiscCfgWrapper | LossPhysCfgWrapper | LossPhysPropCfgWrapper

def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]

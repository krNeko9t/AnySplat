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
from .loss_physgm import LossPhysGM, LossPhysGMCfgWrapper
from .loss_segvggt import LossSegVGGT, LossSegVGGTCfgWrapper
from .loss_segvggt_geo import LossSegVGGTGeo, LossSegVGGTGeoCfgWrapper
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
    LossPhysGMCfgWrapper: LossPhysGM,
    LossSegVGGTCfgWrapper: LossSegVGGT,
    LossSegVGGTGeoCfgWrapper: LossSegVGGTGeo,
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper | LossOpacityCfgWrapper | LossDepthGTCfgWrapper | LossLODCfgWrapper | LossDepthConsisCfgWrapper | LossNormalConsisCfgWrapper | LossChamferDistanceCfgWrapper | LossMvcCfgWrapper | LossDiscCfgWrapper | LossPhysCfgWrapper | LossPhysPropCfgWrapper | LossPhysGMCfgWrapper | LossSegVGGTCfgWrapper | LossSegVGGTGeoCfgWrapper

def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]

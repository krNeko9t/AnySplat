from typing import Union

from .encoder import Encoder
from .anysplat import EncoderAnySplat, EncoderAnySplatCfg
from .iggt import EncoderIGGT, EncoderIGGTCfg

EncoderCfg = Union[EncoderAnySplatCfg, EncoderIGGTCfg]

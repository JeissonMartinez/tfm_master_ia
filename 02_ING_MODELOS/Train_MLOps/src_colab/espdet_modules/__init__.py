"""Official Espressif ESP-Detection building blocks.

Ported from https://github.com/espressif/esp-detection/tree/main/nn/modules
(AGPL-3.0). These blocks require ``ultralytics`` to be installed because
they inherit from Ultralytics base classes (Conv, C2f, Bottleneck, etc.).
"""

from .esp_conv import DSConv
from .esp_block import C3k, DSBottleneck, DSC3k2, ESPSerial, ESPSerialLite, ESPBlock, ESPBlockLite
from .esp_head import ESPDetectHead, ESPDetect, ESPDLDetect

__all__ = (
    "DSConv",
    "C3k",
    "DSBottleneck",
    "DSC3k2",
    "ESPSerial",
    "ESPSerialLite",
    "ESPBlock",
    "ESPBlockLite",
    "ESPDetectHead",
    "ESPDetect",
    "ESPDLDetect",
)

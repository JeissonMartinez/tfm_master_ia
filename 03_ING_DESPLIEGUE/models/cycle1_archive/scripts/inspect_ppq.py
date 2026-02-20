"""Inspect esp-ppq quantization options."""
from esp_ppq import *

setting = QuantizationSettingFactory.espdl_setting()

print("=== Boolean options (defaults) ===")
for attr in ['equalization', 'bias_correct', 'blockwise_reconstruction',
             'lsq_optimization', 'channel_split', 'weight_split',
             'matrix_factorization', 'ssd_equalization', 'extension',
             'convtranspose_decomposition', 'quantize_activation', 'quantize_parameter']:
    val = getattr(setting, attr, 'N/A')
    print(f"  {attr:35s} = {val}")

print("\n=== Activation calib setting ===")
act = setting.quantize_activation_setting
for attr in dir(act):
    if not attr.startswith('_'):
        print(f"  {attr} = {getattr(act, attr)}")

print("\n=== Blockwise Reconstruction Setting ===")
br = setting.blockwise_reconstruction_setting
for attr in dir(br):
    if not attr.startswith('_'):
        v = getattr(br, attr)
        if not callable(v):
            print(f"  {attr} = {v}")

print("\n=== LSQ Setting ===")
lsq = setting.lsq_optimization_setting
for attr in dir(lsq):
    if not attr.startswith('_'):
        v = getattr(lsq, attr)
        if not callable(v):
            print(f"  {attr} = {v}")

print("\n=== Equalization Setting ===")
eq = setting.equalization_setting
for attr in dir(eq):
    if not attr.startswith('_'):
        v = getattr(eq, attr)
        if not callable(v):
            print(f"  {attr} = {v}")

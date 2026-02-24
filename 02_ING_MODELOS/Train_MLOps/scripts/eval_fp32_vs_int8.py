#!/usr/bin/env python3
"""
eval_fp32_vs_int8.py — Evaluación comparativa FP32 vs INT8 (simulación)
=========================================================================
Compara mAP, Precision, Recall y F1 de los 3 modelos seleccionados
(FCOS T3, YOLO26 T2, ESPDet T4) ejecutando:
  - Inferencia FP32 vía onnxruntime
  - Inferencia INT8 simulada vía esp-ppq TorchExecutor

Genera:
  - outputs/espdl/eval_fp32_vs_int8.json           (métricas)
  - outputs/espdl/eval_visualizations/*.png          (grids visuales)

Uso:
    cd 02_ING_MODELOS/Train_MLOps
    python scripts/eval_fp32_vs_int8.py                       # todos
    python scripts/eval_fp32_vs_int8.py --models fcos_v3s_t3  # solo FCOS
    python scripts/eval_fp32_vs_int8.py --skip-int8           # solo FP32
    python scripts/eval_fp32_vs_int8.py --skip-viz            # sin visuales
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent                           # Train_MLOps/
DATASET_DIR = BASE_DIR / ".." / "datasets" / "IODC" / "coco"
TEST_IMAGES_DIR = DATASET_DIR / "test" / "images"
TEST_ANNOTATIONS = DATASET_DIR / "test" / "_annotations.coco.json"
CALIB_IMAGES_DIR = DATASET_DIR / "train" / "images"
OUTPUT_DIR = BASE_DIR / "outputs" / "espdl"
VIZ_DIR = OUTPUT_DIR / "eval_visualizations"

IMGSZ = 224
NC = 5
CLASS_NAMES = ["dog", "door", "obstacle", "person", "stair"]
STRIDES = [8, 16, 32]

# ---------------------------------------------------------------------------
# Model configuration — same paths as convert_onnx_to_espdl.py
# ---------------------------------------------------------------------------
MODELS = {
    "fcos_v3s_t3": {
        "onnx": str(BASE_DIR / "outputs/fcos_v3s_v1-1771690809/export/fcos_v3s.onnx"),
        "input_name": "input",
        "input_shape": [1, 3, IMGSZ, IMGSZ],
        "family": "fcos",
        "conf_threshold": 0.40,
        "nms_threshold": 0.45,
        "label": "FCOS T3",
    },
    "yolo26n_t2": {
        "onnx": str(BASE_DIR / "outputs/yolo26n_custom_v2-run1/export/best.onnx"),
        "input_name": "images",
        "input_shape": [1, 3, IMGSZ, IMGSZ],
        "family": "yolo26",
        "conf_threshold": 0.25,
        "nms_threshold": 0.45,
        "label": "YOLO26 T2",
    },
    "espdet_pico_t4": {
        "onnx": str(BASE_DIR / "outputs/espdet-pico-v4-t4/export/espdet_pico.onnx"),
        "input_name": "input",
        "input_shape": [1, 3, IMGSZ, IMGSZ],
        "family": "espdet",
        "conf_threshold": 0.35,
        "nms_threshold": 0.40,
        "label": "ESPDet T4",
    },
    "yolo26n_t2_esp": {
        "onnx": str(BASE_DIR / "outputs/yolo26n_custom_v2-run1/export/best_esp.onnx"),
        "input_name": "images",
        "input_shape": [1, 3, IMGSZ, IMGSZ],
        "family": "yolo26_esp",
        "conf_threshold": 0.25,
        "nms_threshold": 0.45,
        "label": "YOLO26 T2 ESP",
    },
    "fcos_v3s_t3_mixed": {
        "onnx": str(BASE_DIR / "outputs/fcos_v3s_v1-1771690809/export/fcos_v3s.onnx"),
        "input_name": "input",
        "input_shape": [1, 3, IMGSZ, IMGSZ],
        "family": "fcos",
        "conf_threshold": 0.40,
        "nms_threshold": 0.45,
        "label": "FCOS T3 Mixed",
        "mixed_precision": True,
    },
}

# ─── Full head + FPN ops to dispatch to FP32 for FCOS mixed-precision ───
# Strategy: entire detection head + FPN in FP32, backbone stays INT8
# Rationale: InstanceNorm-only FP32 failed because quantized Conv outputs
# feeding InstanceNorm already lose precision; the whole head path must be FP32
FCOS_FP32_OPS = [
    # ── FPN (10 ops) ──
    "/m/fpn/lateral_convs.0/Conv",
    "/m/fpn/lateral_convs.1/Conv",
    "/m/fpn/lateral_convs.2/Conv",
    "/m/fpn/Resize",
    "/m/fpn/Add",
    "/m/fpn/Resize_1",
    "/m/fpn/Add_1",
    "/m/fpn/smooth_convs.0/Conv",
    "/m/fpn/smooth_convs.1/Conv",
    "/m/fpn/smooth_convs.2/Conv",
    # ── Head level 0 (32 ops) ──
    "/m/head/cls_tower/cls_tower.0/Conv",
    "/m/head/cls_tower/cls_tower.1/Reshape",
    "/m/head/cls_tower/cls_tower.1/InstanceNormalization",
    "/m/head/cls_tower/cls_tower.1/Reshape_1",
    "/m/head/cls_tower/cls_tower.1/Mul",
    "/m/head/cls_tower/cls_tower.1/Add",
    "/m/head/cls_tower/cls_tower.2/Relu",
    "/m/head/cls_tower/cls_tower.3/Conv",
    "/m/head/cls_tower/cls_tower.4/Reshape",
    "/m/head/cls_tower/cls_tower.4/InstanceNormalization",
    "/m/head/cls_tower/cls_tower.4/Reshape_1",
    "/m/head/cls_tower/cls_tower.4/Mul",
    "/m/head/cls_tower/cls_tower.4/Add",
    "/m/head/cls_tower/cls_tower.5/Relu",
    "/m/head/reg_tower/reg_tower.0/Conv",
    "/m/head/reg_tower/reg_tower.1/Reshape",
    "/m/head/reg_tower/reg_tower.1/InstanceNormalization",
    "/m/head/reg_tower/reg_tower.1/Reshape_1",
    "/m/head/reg_tower/reg_tower.1/Mul",
    "/m/head/reg_tower/reg_tower.1/Add",
    "/m/head/reg_tower/reg_tower.2/Relu",
    "/m/head/reg_tower/reg_tower.3/Conv",
    "/m/head/reg_tower/reg_tower.4/Reshape",
    "/m/head/reg_tower/reg_tower.4/InstanceNormalization",
    "/m/head/reg_tower/reg_tower.4/Reshape_1",
    "/m/head/reg_tower/reg_tower.4/Mul",
    "/m/head/reg_tower/reg_tower.4/Add",
    "/m/head/reg_tower/reg_tower.5/Relu",
    "/m/head/cls_logits/Conv",
    "/m/head/bbox_pred/Conv",
    "/m/head/Relu",
    "/m/head/centerness/Conv",
    # ── Head level 1 (32 ops) ──
    "/m/head/cls_tower/cls_tower.0_1/Conv",
    "/m/head/cls_tower/cls_tower.1_1/Reshape",
    "/m/head/cls_tower/cls_tower.1_1/InstanceNormalization",
    "/m/head/cls_tower/cls_tower.1_1/Reshape_1",
    "/m/head/cls_tower/cls_tower.1_1/Mul",
    "/m/head/cls_tower/cls_tower.1_1/Add",
    "/m/head/cls_tower/cls_tower.2_1/Relu",
    "/m/head/cls_tower/cls_tower.3_1/Conv",
    "/m/head/cls_tower/cls_tower.4_1/Reshape",
    "/m/head/cls_tower/cls_tower.4_1/InstanceNormalization",
    "/m/head/cls_tower/cls_tower.4_1/Reshape_1",
    "/m/head/cls_tower/cls_tower.4_1/Mul",
    "/m/head/cls_tower/cls_tower.4_1/Add",
    "/m/head/cls_tower/cls_tower.5_1/Relu",
    "/m/head/reg_tower/reg_tower.0_1/Conv",
    "/m/head/reg_tower/reg_tower.1_1/Reshape",
    "/m/head/reg_tower/reg_tower.1_1/InstanceNormalization",
    "/m/head/reg_tower/reg_tower.1_1/Reshape_1",
    "/m/head/reg_tower/reg_tower.1_1/Mul",
    "/m/head/reg_tower/reg_tower.1_1/Add",
    "/m/head/reg_tower/reg_tower.2_1/Relu",
    "/m/head/reg_tower/reg_tower.3_1/Conv",
    "/m/head/reg_tower/reg_tower.4_1/Reshape",
    "/m/head/reg_tower/reg_tower.4_1/InstanceNormalization",
    "/m/head/reg_tower/reg_tower.4_1/Reshape_1",
    "/m/head/reg_tower/reg_tower.4_1/Mul",
    "/m/head/reg_tower/reg_tower.4_1/Add",
    "/m/head/reg_tower/reg_tower.5_1/Relu",
    "/m/head/cls_logits_1/Conv",
    "/m/head/bbox_pred_1/Conv",
    "/m/head/Relu_1",
    "/m/head/centerness_1/Conv",
    # ── Head level 2 (32 ops) ──
    "/m/head/cls_tower/cls_tower.0_2/Conv",
    "/m/head/cls_tower/cls_tower.1_2/Reshape",
    "/m/head/cls_tower/cls_tower.1_2/InstanceNormalization",
    "/m/head/cls_tower/cls_tower.1_2/Reshape_1",
    "/m/head/cls_tower/cls_tower.1_2/Mul",
    "/m/head/cls_tower/cls_tower.1_2/Add",
    "/m/head/cls_tower/cls_tower.2_2/Relu",
    "/m/head/cls_tower/cls_tower.3_2/Conv",
    "/m/head/cls_tower/cls_tower.4_2/Reshape",
    "/m/head/cls_tower/cls_tower.4_2/InstanceNormalization",
    "/m/head/cls_tower/cls_tower.4_2/Reshape_1",
    "/m/head/cls_tower/cls_tower.4_2/Mul",
    "/m/head/cls_tower/cls_tower.4_2/Add",
    "/m/head/cls_tower/cls_tower.5_2/Relu",
    "/m/head/reg_tower/reg_tower.0_2/Conv",
    "/m/head/reg_tower/reg_tower.1_2/Reshape",
    "/m/head/reg_tower/reg_tower.1_2/InstanceNormalization",
    "/m/head/reg_tower/reg_tower.1_2/Reshape_1",
    "/m/head/reg_tower/reg_tower.1_2/Mul",
    "/m/head/reg_tower/reg_tower.1_2/Add",
    "/m/head/reg_tower/reg_tower.2_2/Relu",
    "/m/head/reg_tower/reg_tower.3_2/Conv",
    "/m/head/reg_tower/reg_tower.4_2/Reshape",
    "/m/head/reg_tower/reg_tower.4_2/InstanceNormalization",
    "/m/head/reg_tower/reg_tower.4_2/Reshape_1",
    "/m/head/reg_tower/reg_tower.4_2/Mul",
    "/m/head/reg_tower/reg_tower.4_2/Add",
    "/m/head/reg_tower/reg_tower.5_2/Relu",
    "/m/head/cls_logits_2/Conv",
    "/m/head/bbox_pred_2/Conv",
    "/m/head/Relu_2",
    "/m/head/centerness_2/Conv",
]


# =====================================================================
#  COCO test set loader
# =====================================================================

def load_test_set() -> Tuple[List[dict], Dict[int, List[dict]]]:
    """Load test images metadata and ground-truth annotations.

    Returns:
        images: list of {id, file_name, width, height}
        gt_by_image: {image_id: [{class_id, bbox_xyxy_norm}]}
    """
    with open(str(TEST_ANNOTATIONS)) as f:
        coco = json.load(f)

    images = coco["images"]
    gt_by_image: Dict[int, List[dict]] = defaultdict(list)

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        # Find image dimensions
        img_info = next(im for im in images if im["id"] == img_id)
        w_img, h_img = img_info["width"], img_info["height"]

        # COCO bbox: [x, y, w, h] → normalized xyxy
        bx, by, bw, bh = ann["bbox"]
        x1 = bx / w_img
        y1 = by / h_img
        x2 = (bx + bw) / w_img
        y2 = (by + bh) / h_img

        gt_by_image[img_id].append({
            "class_id": ann["category_id"],
            "bbox": (x1, y1, x2, y2),
        })

    print(f"  Test set: {len(images)} images, "
          f"{sum(len(v) for v in gt_by_image.values())} annotations")
    return images, gt_by_image


def preprocess_image(img_path: str) -> np.ndarray:
    """Load, resize, normalize → NCHW float32 array [1,3,224,224]."""
    from PIL import Image
    img = Image.open(img_path).convert("RGB").resize((IMGSZ, IMGSZ))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)       # HWC → CHW
    return np.expand_dims(arr, 0)       # [1,3,H,W]


# =====================================================================
#  Calibration dataset (reuses convert script logic)
# =====================================================================

def create_calibration_dataset(n_samples: int = 500) -> list:
    """Create calibration dataset from train images (torch tensors)."""
    from PIL import Image

    calib_path = Path(CALIB_IMAGES_DIR)
    image_files = sorted(calib_path.glob("*.jpg"))[:n_samples]
    samples = []
    for img_path in image_files:
        img = Image.open(img_path).convert("RGB").resize((IMGSZ, IMGSZ))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        samples.append(torch.from_numpy(np.expand_dims(arr, 0)))
    print(f"  Calibration: {len(samples)} samples from {calib_path}")
    return samples


# =====================================================================
#  Fix negative axes (for YOLO26)
# =====================================================================

def fix_negative_axes(onnx_path: str) -> str:
    """Replace negative axis attrs in ONNX graph with positive equivalents."""
    import onnx

    model = onnx.load(onnx_path)
    onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=False)

    fixed = False
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == "axis" and attr.i < 0:
                rank = _get_tensor_rank(model, node.input[0])
                if rank and rank > 0:
                    new_axis = attr.i + rank
                    attr.i = new_axis
                    fixed = True

    if fixed:
        fixed_path = onnx_path.replace(".onnx", "_fixed.onnx")
        if not os.path.exists(fixed_path):
            onnx.save(model, fixed_path)
        return fixed_path
    return onnx_path


def _get_tensor_rank(model, tensor_name: str):
    for vi in model.graph.value_info:
        if vi.name == tensor_name:
            shape = vi.type.tensor_type.shape
            if shape and shape.dim:
                return len(shape.dim)
    for inp in model.graph.input:
        if inp.name == tensor_name:
            shape = inp.type.tensor_type.shape
            if shape and shape.dim:
                return len(shape.dim)
    for out in model.graph.output:
        if out.name == tensor_name:
            shape = out.type.tensor_type.shape
            if shape and shape.dim:
                return len(shape.dim)
    return None


# =====================================================================
#  INT8 quantization via esp-ppq
# =====================================================================

def quantize_for_evaluation(onnx_path: str, config: dict, calib_data: list):
    """Quantize ONNX → PPQ IR and return TorchExecutor for INT8 inference.

    Returns:
        (executor, output_names) or (None, None) on failure.
    """
    from esp_ppq import QuantizationSettingFactory, TargetPlatform
    from esp_ppq.api import espdl_quantize_onnx
    from esp_ppq.executor import TorchExecutor

    # For YOLO26, fix negative axes before quantization
    actual_onnx = onnx_path
    if config["family"] == "yolo26":
        actual_onnx = fix_negative_axes(onnx_path)

    # Temporary ESPDL path (we don't need the file, just the PPQ graph)
    tmp_espdl = str(OUTPUT_DIR / "tmp_eval.espdl")

    setting = QuantizationSettingFactory.espdl_setting()

    # Apply mixed-precision dispatching for FCOS
    if config.get("mixed_precision"):
        for op_name in FCOS_FP32_OPS:
            setting.dispatching_table.append(op_name, TargetPlatform.FP32)
        print(f"    ↳ Mixed-precision: {len(FCOS_FP32_OPS)} ops → FP32")

    ppq_graph = espdl_quantize_onnx(
        onnx_import_file=actual_onnx,
        espdl_export_file=tmp_espdl,
        calib_dataloader=calib_data,
        calib_steps=min(len(calib_data), 500),
        input_shape=config["input_shape"],
        target="esp32s3",
        setting=setting,
        do_quantize=True,
    )

    executor = TorchExecutor(ppq_graph, device="cpu")
    output_names = list(ppq_graph.outputs)

    # Clean up temp file
    if os.path.exists(tmp_espdl):
        os.remove(tmp_espdl)
    for ext in [".info", ".json"]:
        tmp = tmp_espdl.replace(".espdl", ext)
        if os.path.exists(tmp):
            os.remove(tmp)

    return executor, output_names


# =====================================================================
#  FP32 inference (onnxruntime)
# =====================================================================

def infer_fp32(session, input_name: str, img_np: np.ndarray) -> dict:
    """Run FP32 ONNX inference → dict of {output_name: ndarray}."""
    outputs = session.run(None, {input_name: img_np})
    output_names = [o.name for o in session.get_outputs()]
    return {name: arr for name, arr in zip(output_names, outputs)}


# =====================================================================
#  INT8 inference (TorchExecutor)
# =====================================================================

def infer_int8(executor, img_np: np.ndarray) -> list:
    """Run INT8 simulated inference → list of tensors."""
    img_tensor = torch.from_numpy(img_np).float()
    raw = executor.forward(img_tensor)
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    return raw


# =====================================================================
#  Decoders — FCOS
# =====================================================================

def decode_fcos(
    outputs: dict,
    conf_threshold: float = 0.40,
    nms_threshold: float = 0.45,
    is_int8: bool = False,
) -> List[Tuple[int, float, Tuple[float, float, float, float]]]:
    """Decode FCOS 9-output tensors → list of (class_id, conf, bbox_xyxy_norm).

    Output order from ONNX: cls_lvl{0,1,2}, reg_lvl{0,1,2}, centerness_lvl{0,1,2}
    For INT8 TorchExecutor, outputs is a dict keyed by PPQ output names.
    """
    boxes_all, scores_all, labels_all = [], [], []

    for lvl in range(3):
        stride = STRIDES[lvl]

        cls_key = f"cls_lvl{lvl}"
        reg_key = f"reg_lvl{lvl}"
        ctr_key = f"centerness_lvl{lvl}"

        cls_t = _to_tensor(outputs.get(cls_key))   # [1, 5, H, W]
        reg_t = _to_tensor(outputs.get(reg_key))    # [1, 4, H, W]
        ctr_t = _to_tensor(outputs.get(ctr_key))    # [1, 1, H, W]

        if cls_t is None or reg_t is None or ctr_t is None:
            continue

        h_feat, w_feat = cls_t.shape[2], cls_t.shape[3]

        cls_scores = cls_t[0].sigmoid()      # [5, H, W]
        centerness = ctr_t[0].sigmoid()       # [1, H, W]
        reg_vals = reg_t[0]                   # [4, H, W]

        cls_flat = cls_scores.permute(1, 2, 0).reshape(-1, NC)   # [HW, 5]
        ctr_flat = centerness.reshape(-1)                         # [HW]
        reg_flat = reg_vals.permute(1, 2, 0).reshape(-1, 4)      # [HW, 4]

        # Grid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h_feat), torch.arange(w_feat), indexing="ij"
        )
        cx = (x_grid.flatten().float() + 0.5) * stride
        cy = (y_grid.flatten().float() + 0.5) * stride

        max_cls, max_labels = cls_flat.max(dim=-1)
        mask = max_cls > conf_threshold
        if mask.sum() == 0:
            continue

        cls_sel = max_cls[mask]
        labels_sel = max_labels[mask]
        ctr_sel = ctr_flat[mask]
        reg_sel = reg_flat[mask]
        cx_sel, cy_sel = cx[mask], cy[mask]

        # Score = cls × centerness
        scores_sel = cls_sel * ctr_sel

        # Decode boxes: FCOS predicts (l, t, r, b) in stride-normalized units
        x1 = (cx_sel - reg_sel[:, 0] * stride) / IMGSZ
        y1 = (cy_sel - reg_sel[:, 1] * stride) / IMGSZ
        x2 = (cx_sel + reg_sel[:, 2] * stride) / IMGSZ
        y2 = (cy_sel + reg_sel[:, 3] * stride) / IMGSZ

        boxes = torch.stack([x1, y1, x2, y2], dim=1).clamp(0, 1)
        boxes_all.append(boxes)
        scores_all.append(scores_sel)
        labels_all.append(labels_sel)

    return _nms_and_collect(boxes_all, scores_all, labels_all, nms_threshold)


# =====================================================================
#  Decoders — YOLO26
# =====================================================================

def decode_yolo26(
    outputs: dict,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    is_int8: bool = False,
) -> List[Tuple[int, float, Tuple[float, float, float, float]]]:
    """Decode YOLO26 single output [1, 9, 1029] → detections.

    output0 layout: [1, 4+5, 1029] = [boxes(cx,cy,w,h), scores(5 classes)]
    Boxes are in pixel coords [0-224]. Scores are post-sigmoid.
    """
    out_key = "output0"
    out_t = _to_tensor(outputs.get(out_key))  # [1, 9, 1029]
    if out_t is None:
        return []

    pred = out_t[0]  # [9, 1029]

    # Split: first 4 = box, last 5 = scores
    boxes_raw = pred[:4]    # [4, 1029]
    scores_raw = pred[4:]   # [5, 1029]

    # For INT8 TorchExecutor, scores may still need sigmoid if not applied
    # In the ONNX export, sigmoid is baked in. TorchExecutor preserves graph
    # semantics, so scores should already be post-sigmoid.

    # Transpose to [1029, ...] for easier handling
    boxes_t = boxes_raw.permute(1, 0).float()    # [1029, 4] cx,cy,w,h pixels
    scores_t = scores_raw.permute(1, 0).float()  # [1029, 5]

    # Max score per anchor
    max_scores, max_cls = scores_t.max(dim=1)
    mask = max_scores > conf_threshold

    if mask.sum() == 0:
        return []

    boxes_sel = boxes_t[mask]     # [N, 4] cx,cy,w,h
    scores_sel = max_scores[mask]
    labels_sel = max_cls[mask]

    # Convert cx,cy,w,h → x1,y1,x2,y2 normalized
    cx, cy, w, h = boxes_sel[:, 0], boxes_sel[:, 1], boxes_sel[:, 2], boxes_sel[:, 3]
    x1 = (cx - w / 2) / IMGSZ
    y1 = (cy - h / 2) / IMGSZ
    x2 = (cx + w / 2) / IMGSZ
    y2 = (cy + h / 2) / IMGSZ

    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1).clamp(0, 1)

    # NMS
    try:
        from torchvision.ops import nms
        keep = nms(boxes_xyxy, scores_sel, nms_threshold)
    except ImportError:
        keep = torch.arange(len(scores_sel))

    dets = []
    for i in keep.tolist():
        dets.append((
            int(labels_sel[i]),
            float(scores_sel[i]),
            tuple(boxes_xyxy[i].tolist()),
        ))
    return dets


# =====================================================================
#  Decoders — ESPDet
# =====================================================================

def decode_espdet(
    outputs: dict,
    conf_threshold: float = 0.35,
    nms_threshold: float = 0.40,
    is_int8: bool = False,
) -> List[Tuple[int, float, Tuple[float, float, float, float]]]:
    """Decode ESPDet 6-output tensors → detections.

    Outputs: box{0,1,2} [1,4,H,W], score{0,1,2} [1,5,H,W]
    Scores need sigmoid. Boxes are l,t,r,b distances in stride units.
    """
    boxes_all, scores_all, labels_all = [], [], []

    for lvl in range(3):
        stride = STRIDES[lvl]

        box_key = f"box{lvl}"
        score_key = f"score{lvl}"

        box_t = _to_tensor(outputs.get(box_key))     # [1, 4, H, W]
        score_t = _to_tensor(outputs.get(score_key))  # [1, 5, H, W]

        if box_t is None or score_t is None:
            continue

        h_feat, w_feat = box_t.shape[2], box_t.shape[3]

        cls_scores = score_t[0].sigmoid()  # [5, H, W]
        reg_vals = box_t[0]                # [4, H, W]

        cls_flat = cls_scores.permute(1, 2, 0).reshape(-1, NC)
        reg_flat = reg_vals[:4].permute(1, 2, 0).reshape(-1, 4)

        y_grid, x_grid = torch.meshgrid(
            torch.arange(h_feat), torch.arange(w_feat), indexing="ij"
        )
        cx = (x_grid.flatten().float() + 0.5) * stride
        cy = (y_grid.flatten().float() + 0.5) * stride

        max_scores, max_labels = cls_flat.max(dim=-1)
        mask = max_scores > conf_threshold
        if mask.sum() == 0:
            continue

        scores_sel = max_scores[mask]
        labels_sel = max_labels[mask]
        reg_sel = torch.relu(reg_flat[mask])
        cx_sel, cy_sel = cx[mask], cy[mask]

        x1 = (cx_sel - reg_sel[:, 0] * stride) / IMGSZ
        y1 = (cy_sel - reg_sel[:, 1] * stride) / IMGSZ
        x2 = (cx_sel + reg_sel[:, 2] * stride) / IMGSZ
        y2 = (cy_sel + reg_sel[:, 3] * stride) / IMGSZ

        boxes = torch.stack([x1, y1, x2, y2], dim=1).clamp(0, 1)
        boxes_all.append(boxes)
        scores_all.append(scores_sel)
        labels_all.append(labels_sel)

    return _nms_and_collect(boxes_all, scores_all, labels_all, nms_threshold)


# =====================================================================
#  Decoders — YOLO26 ESP (6 separate outputs, reg_max=16 with DFL)
# =====================================================================

def _dfl_integral(box_flat: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Apply DFL (Distribution Focal Loss) integral over raw logits.

    Args:
        box_flat: [N, reg_max*4] raw logits
        reg_max: number of bins (16)
    Returns:
        [N, 4] decoded distances
    """
    box = box_flat.reshape(-1, 4, reg_max)  # [N, 4, reg_max]
    box = box.softmax(dim=2)  # probability distribution
    arange = torch.arange(reg_max, dtype=box.dtype, device=box.device)
    box = (box * arange).sum(dim=2)  # [N, 4] expected distance
    return box


def decode_yolo26_esp(
    outputs: dict,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    is_int8: bool = False,
) -> List[Tuple[int, float, Tuple[float, float, float, float]]]:
    """Decode YOLO26 ESP 6-output tensors → detections.

    Outputs: box{0,1,2} [1, 64, H, W] (reg_max=16, 16*4=64)
             score{0,1,2} [1, 5, H, W] (nc=5, raw logits → need sigmoid)
    Boxes need DFL integral + dist2bbox decoding.
    """
    REG_MAX = 16
    boxes_all, scores_all, labels_all = [], [], []

    for lvl in range(3):
        stride = STRIDES[lvl]

        box_t = _to_tensor(outputs.get(f"box{lvl}"))    # [1, 64, H, W]
        score_t = _to_tensor(outputs.get(f"score{lvl}"))  # [1, 5, H, W]

        if box_t is None or score_t is None:
            continue

        B, _, H, W = box_t.shape
        N = H * W

        # Reshape to [N, C] format
        box_flat = box_t[0].reshape(REG_MAX * 4, N).permute(1, 0)   # [N, 64]
        score_flat = score_t[0].reshape(NC, N).permute(1, 0)         # [N, 5]

        # DFL integral: [N, 64] → [N, 4] (l, t, r, b distances in stride units)
        distances = _dfl_integral(box_flat, REG_MAX)  # [N, 4]

        # Generate grid centers
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij"
        )
        cx = (grid_x.flatten() + 0.5) * stride  # [N] pixels
        cy = (grid_y.flatten() + 0.5) * stride   # [N] pixels

        # dist2bbox: (center ± distance * stride) → xyxy pixels
        x1 = (cx - distances[:, 0] * stride) / IMGSZ
        y1 = (cy - distances[:, 1] * stride) / IMGSZ
        x2 = (cx + distances[:, 2] * stride) / IMGSZ
        y2 = (cy + distances[:, 3] * stride) / IMGSZ

        # Sigmoid on scores (raw logits)
        cls_scores = score_flat.sigmoid()  # [N, 5]

        max_scores, max_labels = cls_scores.max(dim=-1)
        mask = max_scores > conf_threshold
        if mask.sum() == 0:
            continue

        boxes = torch.stack([x1, y1, x2, y2], dim=1)[mask].clamp(0, 1)
        boxes_all.append(boxes)
        scores_all.append(max_scores[mask])
        labels_all.append(max_labels[mask])

    return _nms_and_collect(boxes_all, scores_all, labels_all, nms_threshold)


# =====================================================================
#  Shared helpers
# =====================================================================

def _to_tensor(x) -> Optional[torch.Tensor]:
    """Convert numpy/torch/None → float tensor."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().float()
    return torch.from_numpy(np.asarray(x)).float()


def _nms_and_collect(
    boxes_all, scores_all, labels_all, nms_threshold
) -> List[Tuple[int, float, Tuple[float, float, float, float]]]:
    """Concatenate multi-level detections and apply NMS."""
    if not boxes_all:
        return []

    boxes_cat = torch.cat(boxes_all)
    scores_cat = torch.cat(scores_all)
    labels_cat = torch.cat(labels_all)

    try:
        from torchvision.ops import batched_nms
        keep = batched_nms(boxes_cat, scores_cat, labels_cat, nms_threshold)
    except ImportError:
        keep = _greedy_nms(boxes_cat, scores_cat, labels_cat, nms_threshold)

    dets = []
    for i in keep.tolist():
        dets.append((
            int(labels_cat[i]),
            float(scores_cat[i]),
            tuple(boxes_cat[i].tolist()),
        ))
    return dets


def _greedy_nms(boxes, scores, cls_ids, iou_thr):
    """Fallback greedy per-class NMS."""
    keep = []
    order = torch.argsort(scores, descending=True)
    boxes_np = boxes.cpu().numpy()
    cls_np = cls_ids.cpu().numpy()
    remaining = set(range(len(order)))

    for idx in order.tolist():
        if idx not in remaining:
            continue
        keep.append(idx)
        remaining.discard(idx)
        to_remove = []
        for other in remaining:
            if cls_np[idx] == cls_np[other]:
                iou_val = _iou(boxes_np[idx], boxes_np[other])
                if iou_val >= iou_thr:
                    to_remove.append(other)
        for r in to_remove:
            remaining.discard(r)
    return torch.tensor(keep, dtype=torch.long)


def _iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-8)


# =====================================================================
#  Dispatcher
# =====================================================================

DECODERS = {
    "fcos": decode_fcos,
    "yolo26": decode_yolo26,
    "espdet": decode_espdet,
    "yolo26_esp": decode_yolo26_esp,
}


def run_inference_and_decode(
    outputs_raw,
    output_names: list,
    config: dict,
    is_int8: bool = False,
) -> List[Tuple[int, float, Tuple]]:
    """Dispatch to the correct decoder based on model family."""
    # Build name→tensor dict
    if isinstance(outputs_raw, dict):
        outputs_dict = outputs_raw
    else:
        # INT8 TorchExecutor returns a list of tensors
        outputs_dict = {}
        for name, val in zip(output_names, outputs_raw):
            outputs_dict[name] = val

    decoder = DECODERS[config["family"]]
    return decoder(
        outputs_dict,
        conf_threshold=config["conf_threshold"],
        nms_threshold=config["nms_threshold"],
        is_int8=is_int8,
    )


# =====================================================================
#  mAP computation (101-point interpolation, COCO-style)
# =====================================================================

def compute_metrics(
    all_detections: List[Tuple],
    all_ground_truths: List[Tuple],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute mAP@50, mAP@50-95, per-class AP, P, R, F1.

    Args:
        all_detections: [(img_idx, class_id, conf, bbox_xyxy_norm), ...]
        all_ground_truths: [(img_idx, class_id, bbox_xyxy_norm), ...]

    Returns:
        dict with all metrics.
    """
    num_classes = NC
    aps_50 = []

    gt_by_img_cls: Dict[tuple, list] = defaultdict(list)
    for img_idx, cls_id, bbox in all_ground_truths:
        gt_by_img_cls[(img_idx, cls_id)].append({"bbox": bbox, "matched": False})

    per_class = {}

    for c in range(num_classes):
        dets_c = [(d[0], d[2], d[3]) for d in all_detections if d[1] == c]
        dets_c.sort(key=lambda x: x[1], reverse=True)

        tp = np.zeros(len(dets_c))
        fp = np.zeros(len(dets_c))
        n_gt_c = sum(1 for gt in all_ground_truths if gt[1] == c)

        # Reset matched
        for key in gt_by_img_cls:
            for gt in gt_by_img_cls[key]:
                gt["matched"] = False

        for i, (img_idx, conf, bbox) in enumerate(dets_c):
            gts = gt_by_img_cls.get((img_idx, c), [])
            best_iou, best_gt = 0.0, -1
            for g_idx, gt in enumerate(gts):
                iv = _iou(bbox, gt["bbox"])
                if iv > best_iou:
                    best_iou = iv
                    best_gt = g_idx
            if best_iou >= iou_threshold and best_gt >= 0 and not gts[best_gt]["matched"]:
                tp[i] = 1
                gts[best_gt]["matched"] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / (n_gt_c + 1e-8)
        prec = tp_cum / (tp_cum + fp_cum + 1e-8)

        ap = _ap_101(rec, prec)
        aps_50.append(ap)

        final_p = float(prec[-1]) if len(prec) > 0 else 0.0
        final_r = float(rec[-1]) if len(rec) > 0 else 0.0
        f1_c = 2 * final_p * final_r / (final_p + final_r + 1e-8)

        per_class[CLASS_NAMES[c]] = {
            "ap50": round(ap, 4),
            "precision": round(final_p, 4),
            "recall": round(final_r, 4),
            "f1": round(f1_c, 4),
            "n_gt": n_gt_c,
        }

    mAP50 = float(np.mean(aps_50)) if aps_50 else 0.0

    # mAP@50-95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    maps_per_iou = []
    for iou_t in iou_thresholds:
        aps_t = _compute_aps_at_iou(all_detections, all_ground_truths, float(iou_t))
        maps_per_iou.append(float(np.mean(aps_t)) if aps_t else 0.0)
    mAP50_95 = float(np.mean(maps_per_iou))

    # Global P, R, F1
    precision = float(np.mean([v["precision"] for v in per_class.values()]))
    recall = float(np.mean([v["recall"] for v in per_class.values()]))
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "mAP50": round(mAP50, 4),
        "mAP50_95": round(mAP50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_detections": len(all_detections),
        "n_ground_truths": len(all_ground_truths),
        "per_class": per_class,
    }


def _compute_aps_at_iou(detections, ground_truths, iou_threshold):
    gt_by_img_cls: Dict[tuple, list] = defaultdict(list)
    for img_idx, cls_id, bbox in ground_truths:
        gt_by_img_cls[(img_idx, cls_id)].append({"bbox": bbox, "matched": False})

    aps = []
    for c in range(NC):
        dets_c = [(d[0], d[2], d[3]) for d in detections if d[1] == c]
        dets_c.sort(key=lambda x: x[1], reverse=True)
        tp = np.zeros(len(dets_c))
        fp = np.zeros(len(dets_c))
        n_gt_c = sum(1 for gt in ground_truths if gt[1] == c)

        for key in gt_by_img_cls:
            for gt in gt_by_img_cls[key]:
                gt["matched"] = False

        for i, (img_idx, conf, bbox) in enumerate(dets_c):
            gts = gt_by_img_cls.get((img_idx, c), [])
            best_iou, best_gt = 0.0, -1
            for g_idx, gt in enumerate(gts):
                iv = _iou(bbox, gt["bbox"])
                if iv > best_iou:
                    best_iou = iv
                    best_gt = g_idx
            if best_iou >= iou_threshold and best_gt >= 0 and not gts[best_gt]["matched"]:
                tp[i] = 1
                gts[best_gt]["matched"] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / (n_gt_c + 1e-8)
        prec = tp_cum / (tp_cum + fp_cum + 1e-8)
        aps.append(_ap_101(rec, prec))
    return aps


def _ap_101(recall, precision):
    """101-point AP interpolation (COCO style)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    points = np.linspace(0, 1, 101)
    ap = 0.0
    for t in points:
        idx = np.where(mrec >= t)[0]
        if len(idx) > 0:
            ap += mpre[idx[0]]
    return ap / 101.0


# =====================================================================
#  Comparison gate
# =====================================================================

def compare_and_gate(fp32_metrics: dict, int8_metrics: dict, model_name: str) -> dict:
    """Compare FP32 vs INT8 and classify degradation."""
    f_map50 = fp32_metrics["mAP50"]
    q_map50 = int8_metrics["mAP50"]

    if f_map50 > 0:
        degradation = (1 - q_map50 / f_map50) * 100
    else:
        degradation = 0.0

    if degradation < 25:
        verdict = "PASS"
        symbol = "✅"
    elif degradation < 50:
        verdict = "MARGINAL"
        symbol = "⚠️"
    else:
        verdict = "FAIL"
        symbol = "⛔"

    print(f"\n  {symbol} {model_name}: mAP50 degradation = {degradation:.1f}% → {verdict}")
    print(f"     FP32: mAP50={f_map50:.4f}  INT8: mAP50={q_map50:.4f}")

    return {
        "mAP50_degradation_pct": round(degradation, 2),
        "verdict": verdict,
    }


# =====================================================================
#  Visualization
# =====================================================================

def select_representative_images(
    images: list, gt_by_image: dict, n: int = 8
) -> list:
    """Select diverse images: at least 1 per class + multi-class scenes."""
    # Assign primary class to each image
    class_images: Dict[int, list] = defaultdict(list)
    multi_class = []

    for img in images:
        gts = gt_by_image.get(img["id"], [])
        if not gts:
            continue
        classes = set(g["class_id"] for g in gts)
        for c in classes:
            class_images[c].append(img)
        if len(classes) >= 2:
            multi_class.append(img)

    selected = []
    seen_ids = set()

    # 1 per class
    for c in range(NC):
        candidates = class_images.get(c, [])
        for img in candidates:
            if img["id"] not in seen_ids:
                selected.append(img)
                seen_ids.add(img["id"])
                break

    # Fill remaining with multi-class images
    for img in multi_class:
        if len(selected) >= n:
            break
        if img["id"] not in seen_ids:
            selected.append(img)
            seen_ids.add(img["id"])

    # Fill remaining with any remaining images
    for img in images:
        if len(selected) >= n:
            break
        if img["id"] not in seen_ids:
            selected.append(img)
            seen_ids.add(img["id"])

    return selected[:n]


def generate_comparison_grid(
    images_info: list,
    detections_fp32: Dict[int, list],
    detections_int8: Dict[int, list],
    gt_by_image: dict,
    model_label: str,
    save_path: str,
) -> None:
    """Generate a 3-row × 4-col grid: GT / FP32 / INT8 for 4 images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    COLORS = [
        (0/255, 114/255, 189/255),    # dog - blue
        (217/255, 83/255, 25/255),     # door - orange
        (237/255, 177/255, 32/255),    # obstacle - yellow
        (126/255, 47/255, 142/255),    # person - purple
        (119/255, 172/255, 48/255),    # stair - green
    ]

    n_imgs = min(len(images_info), 4)
    fig, axes = plt.subplots(3, n_imgs, figsize=(n_imgs * 4.5, 3 * 4))
    fig.suptitle(f"Comparación FP32 vs INT8 — {model_label}",
                 fontsize=14, fontweight="bold", y=0.98)

    row_labels = ["Ground Truth", "FP32 (ONNX)", "INT8 (esp-ppq)"]

    for col, img_info in enumerate(images_info[:n_imgs]):
        img_id = img_info["id"]
        img_path = str(TEST_IMAGES_DIR / img_info["file_name"])
        if not os.path.exists(img_path):
            # Try without images/ subfolder
            img_path = str(DATASET_DIR / "test" / img_info["file_name"])

        from PIL import Image
        raw_img = Image.open(img_path).convert("RGB")
        img_arr = np.array(raw_img)
        h_orig, w_orig = img_arr.shape[:2]

        # Row data: GT, FP32, INT8
        gts = gt_by_image.get(img_id, [])
        gt_dets = [(g["class_id"], 1.0, g["bbox"]) for g in gts]
        fp32_dets = detections_fp32.get(img_id, [])
        int8_dets = detections_int8.get(img_id, [])

        for row, (dets, label) in enumerate([
            (gt_dets, row_labels[0]),
            (fp32_dets, row_labels[1]),
            (int8_dets, row_labels[2]),
        ]):
            ax = axes[row][col] if n_imgs > 1 else axes[row]
            ax.imshow(img_arr)
            ax.axis("off")

            if col == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold",
                              rotation=90, labelpad=10)
                ax.yaxis.set_visible(True)
                ax.set_yticks([])

            for det in dets:
                cls_id, conf, bbox = det
                x1n, y1n, x2n, y2n = bbox
                rx = x1n * w_orig
                ry = y1n * h_orig
                rw = (x2n - x1n) * w_orig
                rh = (y2n - y1n) * h_orig

                color = COLORS[cls_id % len(COLORS)]
                ls = "--" if row == 0 else "-"
                lw = 1.5 if row == 0 else 2.0

                rect = patches.Rectangle(
                    (rx, ry), rw, rh, linewidth=lw,
                    edgecolor=color, facecolor="none", linestyle=ls
                )
                ax.add_patch(rect)

                label_text = CLASS_NAMES[cls_id]
                if row > 0:
                    label_text += f" {conf:.2f}"
                ax.text(
                    rx, max(ry - 3, 8), label_text,
                    fontsize=6, color="white",
                    bbox=dict(facecolor=color, alpha=0.75, pad=1, boxstyle="round,pad=0.2"),
                )

            n_dets = len(dets)
            ax.set_title(f"{label} ({n_dets})" if col == 0 and row > 0
                         else f"({n_dets} {'GT' if row==0 else 'dets'})",
                         fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    📊 Saved: {save_path}")


def generate_unified_grid(
    images_info: list,
    all_detections: Dict[str, Dict[int, list]],
    gt_by_image: dict,
    save_path: str,
) -> None:
    """Generate unified comparison: 3 images × 7 columns (GT + 3 models × FP32/INT8)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    COLORS = [
        (0/255, 114/255, 189/255),
        (217/255, 83/255, 25/255),
        (237/255, 177/255, 32/255),
        (126/255, 47/255, 142/255),
        (119/255, 172/255, 48/255),
    ]

    n_imgs = min(len(images_info), 3)
    model_names = list(all_detections.keys())
    # Columns: GT, then for each model: FP32, INT8
    n_cols = 1 + len(model_names) * 2
    col_labels = ["GT"]
    for mn in model_names:
        label = MODELS[mn]["label"]
        col_labels.extend([f"{label} FP32", f"{label} INT8"])

    fig, axes = plt.subplots(n_imgs, n_cols, figsize=(n_cols * 3.5, n_imgs * 3.5))
    fig.suptitle("Comparación unificada: GT vs FP32 vs INT8 — 3 modelos",
                 fontsize=13, fontweight="bold", y=0.99)

    if n_imgs == 1:
        axes = axes.reshape(1, -1)

    for row, img_info in enumerate(images_info[:n_imgs]):
        img_id = img_info["id"]
        img_path = str(TEST_IMAGES_DIR / img_info["file_name"])
        if not os.path.exists(img_path):
            img_path = str(DATASET_DIR / "test" / img_info["file_name"])

        from PIL import Image
        raw_img = Image.open(img_path).convert("RGB")
        img_arr = np.array(raw_img)
        h_orig, w_orig = img_arr.shape[:2]

        gts = gt_by_image.get(img_id, [])
        gt_dets = [(g["class_id"], 1.0, g["bbox"]) for g in gts]

        col = 0
        # GT column
        ax = axes[row][col]
        ax.imshow(img_arr); ax.axis("off")
        if row == 0:
            ax.set_title("GT", fontsize=9, fontweight="bold")
        _draw_dets_on_ax(ax, gt_dets, w_orig, h_orig, COLORS, linestyle="--")
        col += 1

        # Model columns
        for mn in model_names:
            for variant in ["fp32", "int8"]:
                ax = axes[row][col]
                ax.imshow(img_arr); ax.axis("off")
                if row == 0:
                    ax.set_title(col_labels[col], fontsize=8, fontweight="bold")

                dets = all_detections[mn].get(variant, {}).get(img_id, [])
                _draw_dets_on_ax(ax, dets, w_orig, h_orig, COLORS)
                col += 1

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    📊 Saved: {save_path}")


def _draw_dets_on_ax(ax, dets, w, h, colors, linestyle="-"):
    import matplotlib.patches as patches
    for det in dets:
        cls_id, conf, bbox = det
        x1n, y1n, x2n, y2n = bbox
        rx, ry = x1n * w, y1n * h
        rw, rh = (x2n - x1n) * w, (y2n - y1n) * h
        color = colors[cls_id % len(colors)]
        rect = patches.Rectangle(
            (rx, ry), rw, rh, linewidth=1.5,
            edgecolor=color, facecolor="none", linestyle=linestyle
        )
        ax.add_patch(rect)
        lbl = CLASS_NAMES[cls_id]
        if conf < 1.0:
            lbl += f" {conf:.2f}"
        ax.text(rx, max(ry - 2, 6), lbl, fontsize=5, color="white",
                bbox=dict(facecolor=color, alpha=0.7, pad=0.5,
                          boxstyle="round,pad=0.15"))


# =====================================================================
#  Main evaluation pipeline
# =====================================================================

def evaluate_model(
    model_name: str,
    config: dict,
    images: list,
    gt_by_image: dict,
    calib_data: Optional[list] = None,
    skip_int8: bool = False,
) -> dict:
    """Run FP32 and INT8 evaluation for a single model.

    Returns:
        {
            "fp32": metrics_dict,
            "int8": metrics_dict or None,
            "degradation": gate_dict or None,
            "detections_fp32": {img_id: [dets]},
            "detections_int8": {img_id: [dets]},
        }
    """
    import onnxruntime as ort

    label = config["label"]
    family = config["family"]
    onnx_path = config["onnx"]

    print(f"\n{'='*60}")
    print(f"  EVALUATING: {label} ({model_name})")
    print(f"{'='*60}")

    if not os.path.exists(onnx_path):
        print(f"  ❌ ONNX not found: {onnx_path}")
        return {}

    # === FP32 inference ===
    print(f"\n  --- FP32 Inference (onnxruntime) ---")
    sess = ort.InferenceSession(onnx_path)
    output_names_fp32 = [o.name for o in sess.get_outputs()]
    input_name = config["input_name"]

    all_dets_fp32 = []
    dets_by_img_fp32 = {}
    t0 = time.time()

    for idx, img_info in enumerate(images):
        img_path = str(TEST_IMAGES_DIR / img_info["file_name"])
        if not os.path.exists(img_path):
            img_path = str(DATASET_DIR / "test" / img_info["file_name"])

        img_np = preprocess_image(img_path)
        raw_outputs = sess.run(None, {input_name: img_np})
        outputs_dict = {n: v for n, v in zip(output_names_fp32, raw_outputs)}

        dets = run_inference_and_decode(outputs_dict, output_names_fp32, config, is_int8=False)

        for d in dets:
            all_dets_fp32.append((idx, d[0], d[1], d[2]))
        dets_by_img_fp32[img_info["id"]] = dets

        if (idx + 1) % 50 == 0:
            print(f"    ... {idx+1}/{len(images)}")

    fp32_time = time.time() - t0
    print(f"  FP32 done: {fp32_time:.1f}s, {len(all_dets_fp32)} total detections")

    # Build GT list
    all_gts = []
    for idx, img_info in enumerate(images):
        for gt in gt_by_image.get(img_info["id"], []):
            all_gts.append((idx, gt["class_id"], gt["bbox"]))

    fp32_metrics = compute_metrics(all_dets_fp32, all_gts)
    fp32_metrics["inference_time_s"] = round(fp32_time, 1)
    fp32_metrics["avg_ms_per_image"] = round(fp32_time / len(images) * 1000, 1)

    print(f"  FP32 mAP@50:    {fp32_metrics['mAP50']:.4f}")
    print(f"  FP32 mAP@50-95: {fp32_metrics['mAP50_95']:.4f}")
    print(f"  FP32 P/R/F1:    {fp32_metrics['precision']:.4f} / "
          f"{fp32_metrics['recall']:.4f} / {fp32_metrics['f1']:.4f}")

    result = {
        "fp32": fp32_metrics,
        "int8": None,
        "degradation": None,
        "detections_fp32": dets_by_img_fp32,
        "detections_int8": {},
    }

    # === INT8 inference ===
    if skip_int8:
        print(f"\n  --- INT8 skipped ---")
        return result

    print(f"\n  --- INT8 Quantization + Inference (esp-ppq TorchExecutor) ---")
    try:
        t0 = time.time()
        executor, output_names_int8 = quantize_for_evaluation(
            onnx_path, config, calib_data
        )
        quant_time = time.time() - t0
        print(f"  Quantization: {quant_time:.1f}s")
    except Exception as e:
        print(f"  ❌ Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return result

    all_dets_int8 = []
    dets_by_img_int8 = {}
    t0 = time.time()

    for idx, img_info in enumerate(images):
        img_path = str(TEST_IMAGES_DIR / img_info["file_name"])
        if not os.path.exists(img_path):
            img_path = str(DATASET_DIR / "test" / img_info["file_name"])

        img_np = preprocess_image(img_path)
        raw_outputs = infer_int8(executor, img_np)

        outputs_dict = {n: v for n, v in zip(output_names_int8, raw_outputs)}
        dets = run_inference_and_decode(outputs_dict, output_names_int8, config, is_int8=True)

        for d in dets:
            all_dets_int8.append((idx, d[0], d[1], d[2]))
        dets_by_img_int8[img_info["id"]] = dets

        if (idx + 1) % 50 == 0:
            print(f"    ... {idx+1}/{len(images)}")

    int8_time = time.time() - t0
    print(f"  INT8 done: {int8_time:.1f}s, {len(all_dets_int8)} total detections")

    int8_metrics = compute_metrics(all_dets_int8, all_gts)
    int8_metrics["inference_time_s"] = round(int8_time, 1)
    int8_metrics["avg_ms_per_image"] = round(int8_time / len(images) * 1000, 1)
    int8_metrics["quantization_time_s"] = round(quant_time, 1)

    print(f"  INT8 mAP@50:    {int8_metrics['mAP50']:.4f}")
    print(f"  INT8 mAP@50-95: {int8_metrics['mAP50_95']:.4f}")
    print(f"  INT8 P/R/F1:    {int8_metrics['precision']:.4f} / "
          f"{int8_metrics['recall']:.4f} / {int8_metrics['f1']:.4f}")

    gate = compare_and_gate(fp32_metrics, int8_metrics, config["label"])

    result["int8"] = int8_metrics
    result["degradation"] = gate
    result["detections_int8"] = dets_by_img_int8

    return result


# =====================================================================
#  Main
# =====================================================================

def main():
    global TEST_IMAGES_DIR

    parser = argparse.ArgumentParser(
        description="Evaluación FP32 vs INT8 — 3 modelos TinyML"
    )
    parser.add_argument("--models", nargs="*", default=None,
                        help="Models to evaluate (default: all)")
    parser.add_argument("--skip-int8", action="store_true",
                        help="Only run FP32 evaluation")
    parser.add_argument("--skip-viz", action="store_true",
                        help="Skip visualization generation")
    parser.add_argument("--n-calib", type=int, default=500,
                        help="Number of calibration images (default: 500)")
    parser.add_argument("--n-viz", type=int, default=8,
                        help="Number of visualization images (default: 8)")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"  eval_fp32_vs_int8.py — Post-Quantization Evaluation")
    print(f"  Test set: {TEST_ANNOTATIONS}")
    print(f"  Skip INT8: {args.skip_int8}")
    print(f"{'#'*60}")

    # Test images subfolder: check whether images exist directly or in images/
    if not TEST_IMAGES_DIR.exists():
        # Try test/ directly (images might be there without images/ subfolder)
        alt_dir = DATASET_DIR / "test"
        sample = None
        with open(str(TEST_ANNOTATIONS)) as f:
            coco = json.load(f)
        if coco["images"]:
            sample = coco["images"][0]["file_name"]
        if sample and (alt_dir / sample).exists():
            TEST_IMAGES_DIR = alt_dir
            print(f"  Images at: {TEST_IMAGES_DIR}")
        else:
            print(f"  ❌ Test images not found at {TEST_IMAGES_DIR}")
            sys.exit(1)

    # Load test set
    images, gt_by_image = load_test_set()

    # Calibration data (shared across models)
    calib_data = None
    if not args.skip_int8:
        print(f"\n  Loading calibration data ({args.n_calib} samples)...")
        calib_data = create_calibration_dataset(args.n_calib)

    # Setup output dirs
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # Select models
    models_to_eval = args.models or list(MODELS.keys())
    models_to_eval = [m for m in models_to_eval if m in MODELS]

    # Run evaluations
    all_results = {}
    all_viz_data = {}  # For unified grid

    for model_name in models_to_eval:
        config = MODELS[model_name]
        result = evaluate_model(
            model_name, config, images, gt_by_image,
            calib_data=calib_data,
            skip_int8=args.skip_int8,
        )
        all_results[model_name] = {
            "label": config["label"],
            "family": config["family"],
            "fp32": result.get("fp32"),
            "int8": result.get("int8"),
            "degradation": result.get("degradation"),
        }
        all_viz_data[model_name] = {
            "fp32": result.get("detections_fp32", {}),
            "int8": result.get("detections_int8", {}),
        }

    # === Save JSON results ===
    results_path = OUTPUT_DIR / "eval_fp32_vs_int8.json"
    with open(str(results_path), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Results saved: {results_path}")

    # === Visualizations ===
    if not args.skip_viz:
        print(f"\n  --- Generating visualizations ---")
        viz_images = select_representative_images(images, gt_by_image, n=args.n_viz)
        print(f"  Selected {len(viz_images)} representative images")

        for model_name in models_to_eval:
            config = MODELS[model_name]
            label = config["label"]
            dets_fp32 = all_viz_data[model_name]["fp32"]
            dets_int8 = all_viz_data[model_name]["int8"]

            save_path = str(VIZ_DIR / f"{model_name}_fp32_vs_int8.png")
            generate_comparison_grid(
                viz_images, dets_fp32, dets_int8, gt_by_image,
                model_label=label, save_path=save_path,
            )

        # Unified grid (if all 3 models evaluated)
        if len(models_to_eval) >= 2:
            save_path = str(VIZ_DIR / "comparison_3models_grid.png")
            generate_unified_grid(
                viz_images, all_viz_data, gt_by_image, save_path=save_path,
            )

    # === Print summary ===
    print(f"\n{'='*70}")
    print(f"  SUMMARY — FP32 vs INT8 Post-Quantization Evaluation")
    print(f"{'='*70}")
    print(f"\n  {'Model':<18} {'mAP50 FP32':>12} {'mAP50 INT8':>12} "
          f"{'Δ%':>8} {'Verdict':>10}")
    print(f"  {'─'*62}")

    for mn in models_to_eval:
        r = all_results[mn]
        fp32_map = r["fp32"]["mAP50"] if r["fp32"] else 0
        int8_map = r["int8"]["mAP50"] if r["int8"] else "—"
        deg = r["degradation"]["mAP50_degradation_pct"] if r["degradation"] else "—"
        verdict = r["degradation"]["verdict"] if r["degradation"] else "—"

        if isinstance(int8_map, float):
            print(f"  {r['label']:<18} {fp32_map:>12.4f} {int8_map:>12.4f} "
                  f"{deg:>7.1f}% {verdict:>10}")
        else:
            print(f"  {r['label']:<18} {fp32_map:>12.4f} {'—':>12} "
                  f"{'—':>8} {'—':>10}")

    print(f"\n{'#'*60}")
    print(f"  Evaluation complete")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()

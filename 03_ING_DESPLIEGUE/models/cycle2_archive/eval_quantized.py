"""
eval_quantized.py — Evaluación mAP de modelos cuantizados (simulación INT8)
=============================================================================
Calcula mAP50 y mAP50-95 sobre el dataset de validación usando TorchExecutor
de PPQ para simular la inferencia INT8, SIN necesidad de flashear al ESP32.

Basado en el enfoque de Espressif:
    esp-dl/models/coco_detect/tools/quantization/yolo11n_eval.py

Uso:
    python models/eval_quantized.py --model yolo11n
    python models/eval_quantized.py --model yolo26n
    python models/eval_quantized.py --model yolo11n --float-only
    python models/eval_quantized.py --model yolo11n --skip-float

Requisitos:
    pip install ultralytics esp-ppq onnxruntime pycocotools
"""

import os
import sys
import argparse
import pickle
import time
import copy
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Imports ESP-PPQ
# ---------------------------------------------------------------------------
try:
    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_onnx
    from esp_ppq.executor import TorchExecutor
except ImportError:
    print("[ERROR] esp-ppq no encontrado. pip install esp-ppq==1.2.4")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Imports Ultralytics
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import ops
except ImportError:
    print("[ERROR] ultralytics no encontrado. pip install ultralytics")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuración de modelos
# ---------------------------------------------------------------------------
DATASET_YAML = os.path.join(BASE_DIR, "data.yaml")

IMGSZ = 224
NC = 5
TARGET_CHIP = "esp32s3"

CLASS_NAMES = ["dog", "door", "obstacle", "person", "stair"]

MODELS_CONFIG = {
    "yolo11n": {
        "pt_file": "yolo11n_v1_best.pt",
        "onnx_file": "yolo11n_v1_best_esp.onnx",
        "espdl_file": "yolo11n_v1_best.espdl",
        "calib_file": "calib_set_nchw.pkl",
        "input_shape": [1, 3, IMGSZ, IMGSZ],
        "reg_max": 16,
        "strides": [8, 16, 32],
    },
    "yolo26n": {
        "pt_file": "yolo26n_v1_best.pt",
        "onnx_file": "yolo26n_v1_best_esp.onnx",
        "espdl_file": "yolo26n_v1_best.espdl",
        "calib_file": "calib_set_nchw.pkl",
        "input_shape": [1, 3, IMGSZ, IMGSZ],
        "reg_max": 1,
        "strides": [8, 16, 32],
    },
}


# ---------------------------------------------------------------------------
# Funciones de cuantización
# ---------------------------------------------------------------------------

def quantize_model(onnx_path, espdl_path, calib_pkl_path, input_shape):
    """Cuantiza el ONNX y retorna el grafo PPQ + TorchExecutor."""

    # Cargar calibración
    with open(calib_pkl_path, "rb") as f:
        np_data = pickle.load(f)
    calib_data = [torch.from_numpy(arr).float() for arr in np_data]
    print(f"  Calibración: {len(calib_data)} muestras")

    # Settings con equalization (Ciclo 2)
    setting = QuantizationSettingFactory.espdl_setting()
    setting.equalization = True
    setting.equalization_setting.iterations = 3
    setting.equalization_setting.value_threshold = 2.0

    print(f"  Cuantizando con equalization...")
    t0 = time.time()

    ppq_graph = espdl_quantize_onnx(
        onnx_import_file=onnx_path,
        espdl_export_file=espdl_path,
        calib_dataloader=calib_data,
        calib_steps=min(len(calib_data), 256),
        input_shape=input_shape,
        target=TARGET_CHIP,
        setting=setting,
        collate_fn=lambda batch: batch.float(),
        error_report=False,
    )
    elapsed = time.time() - t0
    size_kb = os.path.getsize(espdl_path) / 1024
    print(f"  Cuantización OK: {elapsed:.1f}s, {size_kb:.1f} KB")

    executor = TorchExecutor(ppq_graph, device="cpu")
    output_names = list(ppq_graph.outputs)
    print(f"  Outputs: {output_names}")

    return executor, output_names, ppq_graph


# ---------------------------------------------------------------------------
# Decodificador de 6 salidas crudas → [N, 6] (x1, y1, x2, y2, conf, cls)
# ---------------------------------------------------------------------------

def dfl_integral(box_raw, reg_max):
    """Aplica DFL (Distribution Focal Loss) integral sobre reg_max bins.

    box_raw: [N, reg_max*4]  → [N, 4] distancias (left, top, right, bottom)
    """
    N = box_raw.shape[0]
    box = box_raw.reshape(N, 4, reg_max)
    # Softmax sobre la dimensión de reg_max
    box = torch.softmax(box, dim=2)
    # Integral: sum(i * prob_i)
    arange = torch.arange(reg_max, dtype=box.dtype, device=box.device)
    box = (box * arange).sum(dim=2)  # [N, 4]
    return box


def decode_raw_outputs(outputs, output_names, reg_max, strides, imgsz):
    """Decodifica las 6 salidas crudas (box0-2, score0-2) en detecciones.

    Args:
        outputs: lista de tensores [box0, score0, box1, score1, box2, score2]
        output_names: nombres de las salidas
        reg_max: 16 para YOLO11n, 1 para YOLO26n
        strides: [8, 16, 32]
        imgsz: tamaño de imagen (224)

    Returns:
        preds: tensor [1, N_total, 4+nc] donde N_total = 1029 para 224x224
               4 = bbox (x1, y1, x2, y2 en píxeles), nc = scores por clase
    """
    all_boxes = []
    all_scores = []

    for level in range(3):
        # Buscar box y score por índice (box0/score0, box1/score1, box2/score2)
        box_idx = level * 2      # 0, 2, 4
        score_idx = level * 2 + 1  # 1, 3, 5

        box_tensor = outputs[box_idx]   # [1, reg_max*4, H, W]
        score_tensor = outputs[score_idx]  # [1, nc, H, W]

        if isinstance(box_tensor, torch.Tensor):
            box_t = box_tensor.detach().float()
            score_t = score_tensor.detach().float()
        else:
            box_t = torch.from_numpy(np.array(box_tensor)).float()
            score_t = torch.from_numpy(np.array(score_tensor)).float()

        B, _, H, W = box_t.shape
        stride = strides[level]

        # Reshape a [B, C, H*W] y transponer a [B, H*W, C]
        box_flat = box_t.reshape(B, -1, H * W).permute(0, 2, 1)    # [1, HW, reg_max*4]
        score_flat = score_t.reshape(B, NC, H * W).permute(0, 2, 1)  # [1, HW, nc]

        N = H * W

        # DFL o decodificación directa
        if reg_max > 1:
            boxes = dfl_integral(box_flat.reshape(-1, reg_max * 4), reg_max)  # [N, 4]
            boxes = boxes.unsqueeze(0)  # [1, N, 4]
        else:
            boxes = box_flat  # [1, N, 4] ya son distancias directas

        # Generar grid de centros
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij"
        )
        grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # [N, 2]
        grid = grid.unsqueeze(0)  # [1, N, 2]

        # dist2bbox: (cx, cy) ± (left, top, right, bottom)
        # boxes = [left, top, right, bottom]
        x1y1 = (grid + 0.5 - boxes[..., :2]) * stride
        x2y2 = (grid + 0.5 + boxes[..., 2:]) * stride
        bboxes = torch.cat([x1y1, x2y2], dim=-1)  # [1, N, 4]

        # Sigmoid sobre scores
        scores = score_flat.sigmoid()  # [1, N, nc]

        all_boxes.append(bboxes)
        all_scores.append(scores)

    # Concatenar todos los niveles
    all_boxes = torch.cat(all_boxes, dim=1)    # [1, 1029, 4]
    all_scores = torch.cat(all_scores, dim=1)  # [1, 1029, nc]

    return torch.cat([all_boxes, all_scores], dim=-1)  # [1, 1029, 4+nc]


# ---------------------------------------------------------------------------
# Post-procesamiento: NMS
# ---------------------------------------------------------------------------

def postprocess_preds(preds, conf_thres=0.001, iou_thres=0.65, max_det=300):
    """Aplica NMS sobre las predicciones decodificadas.

    Args:
        preds: [1, N, 4+nc] con boxes en xyxy y scores post-sigmoid
    Returns:
        Lista de dicts {"bboxes": [M,4], "conf": [M], "cls": [M], "extra": [M,0]}
        (formato requerido por Ultralytics 8.4.x DetectionValidator)
    """
    output = []
    for xi, pred in enumerate(preds):
        # pred: [N, 4+nc]
        boxes = pred[:, :4]
        scores = pred[:, 4:]

        # Máximo score por candidato
        max_scores, max_cls = scores.max(dim=1)
        mask = max_scores > conf_thres
        boxes = boxes[mask]
        max_scores = max_scores[mask]
        max_cls = max_cls[mask]

        if boxes.shape[0] == 0:
            output.append({
                "bboxes": torch.zeros((0, 4)),
                "conf": torch.zeros(0),
                "cls": torch.zeros(0),
                "extra": torch.zeros((0, 0)),
            })
            continue

        # NMS
        try:
            from torchvision.ops import nms
            keep = nms(boxes, max_scores, iou_thres)
            keep = keep[:max_det]
            boxes = boxes[keep]
            max_scores = max_scores[keep]
            max_cls = max_cls[keep]
        except ImportError:
            boxes = boxes[:max_det]
            max_scores = max_scores[:max_det]
            max_cls = max_cls[:max_det]

        output.append({
            "bboxes": boxes,
            "conf": max_scores,
            "cls": max_cls.float(),
            "extra": torch.zeros((boxes.shape[0], 0)),
        })

    return output


# ---------------------------------------------------------------------------
# Clase validador cuantizado (subclase de DetectionValidator de Ultralytics)
# ---------------------------------------------------------------------------

def make_quant_validator_class(executor, output_names, model_config):
    """Crea una subclase de DetectionValidator que usa TorchExecutor."""

    reg_max = model_config["reg_max"]
    strides = model_config["strides"]

    class QuantDetectionValidator(DetectionValidator):

        def __call__(self, trainer=None, model=None):
            """Override completo del loop de validación."""
            # Setup device (requerido por preprocess)
            self.device = torch.device("cpu")

            # Parsear data.yaml a dict si es string
            if isinstance(self.args.data, str):
                self.data = check_det_dataset(self.args.data)
            else:
                self.data = self.args.data
            self.dataloader = self.get_dataloader(
                self.data.get("val") or self.data.get("test"), self.args.batch
            )
            self.init_metrics(model)

            bar = enumerate(self.dataloader)
            n_batches = len(self.dataloader)
            desc = f"Evaluando INT8 cuantizado ({n_batches} batches)"
            print(f"\n  {desc}")

            for batch_i, batch in bar:
                # Preprocesar batch
                self.batch = batch
                self.preprocess(batch)
                imgs = batch["img"].float()  # [B, 3, H, W] normalizado 0-1

                # Inferencia cuantizada imagen por imagen
                all_preds = []
                for i in range(imgs.shape[0]):
                    img_single = imgs[i:i+1]  # [1, 3, H, W]

                    # Ejecutar TorchExecutor
                    raw_outputs = executor.forward(img_single)
                    if not isinstance(raw_outputs, (list, tuple)):
                        raw_outputs = [raw_outputs]

                    # Decodificar 6 salidas → [1, 1029, 4+nc]
                    decoded = decode_raw_outputs(
                        raw_outputs, output_names, reg_max, strides, IMGSZ
                    )
                    all_preds.append(decoded)

                # Concatenar batch
                preds_batch = torch.cat(all_preds, dim=0)  # [B, 1029, 4+nc]

                # Post-procesamiento: NMS
                preds_nms = postprocess_preds(preds_batch)

                # Actualizar métricas
                self.update_metrics(preds_nms, batch)

                if batch_i % 50 == 0:
                    print(f"    batch {batch_i}/{n_batches}")

            stats = self.get_stats()
            self.print_results()
            return stats

    return QuantDetectionValidator


# ---------------------------------------------------------------------------
# Evaluación float (referencia)
# ---------------------------------------------------------------------------

def eval_float(model_name, config, data_yaml, batch_size):
    """Evalúa el modelo float original (.pt) como referencia."""
    pt_path = os.path.join(BASE_DIR, config["pt_file"])
    if not os.path.isfile(pt_path):
        print(f"  [WARN] No encontrado: {pt_path}, omitiendo eval float")
        return None

    print(f"\n{'='*60}")
    print(f"  EVALUACIÓN FLOAT — {model_name}")
    print(f"{'='*60}")

    model = YOLO(pt_path)
    results = model.val(
        data=data_yaml,
        imgsz=IMGSZ,
        batch=batch_size,
        conf=0.001,
        iou=0.65,
        verbose=False,
    )

    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "per_class_ap50": {},
    }
    for i, cls_name in enumerate(CLASS_NAMES):
        if i < len(results.box.ap50):
            metrics["per_class_ap50"][cls_name] = float(results.box.ap50[i])

    print(f"\n  Float mAP50:    {metrics['mAP50']:.4f}")
    print(f"  Float mAP50-95: {metrics['mAP50-95']:.4f}")
    for cls_name, ap in metrics["per_class_ap50"].items():
        print(f"    {cls_name}: AP50={ap:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Evaluación cuantizada
# ---------------------------------------------------------------------------

def eval_quantized(model_name, config, data_yaml, batch_size):
    """Cuantiza y evalúa el modelo con simulación INT8."""
    onnx_path = os.path.join(BASE_DIR, config["onnx_file"])
    espdl_path = os.path.join(BASE_DIR, config["espdl_file"])
    calib_pkl = os.path.join(BASE_DIR, config["calib_file"])

    if not os.path.isfile(onnx_path):
        print(f"  [ERROR] ONNX no encontrado: {onnx_path}")
        return None

    print(f"\n{'='*60}")
    print(f"  EVALUACIÓN INT8 — {model_name}")
    print(f"{'='*60}")

    # Paso 1: Cuantizar
    executor, output_names, ppq_graph = quantize_model(
        onnx_path, espdl_path, calib_pkl, config["input_shape"]
    )

    # Diagnóstico rápido: inferencia sobre 1 imagen de calibración
    print(f"\n  --- Diagnóstico rápido (misma imagen) ---")
    with open(calib_pkl, "rb") as f:
        calib_np = pickle.load(f)
    test_img = torch.from_numpy(calib_np[0]).float()
    if test_img.ndim == 3:
        test_img = test_img.unsqueeze(0)  # [3,H,W] -> [1,3,H,W]

    # --- Float ONNX (referencia) ---
    print(f"  [FLOAT ONNX]")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        ort_inputs = {sess.get_inputs()[0].name: test_img.numpy()}
        ort_outputs = sess.run(None, ort_inputs)
        ort_names = [o.name for o in sess.get_outputs()]
        for name, arr in zip(ort_names, ort_outputs):
            if "score" in name:
                sig = 1.0 / (1.0 + np.exp(-np.clip(arr, -50, 50)))
                print(f"    {name}: logit[{arr.min():.2f}, {arr.max():.2f}] "
                      f"sigmoid_max={sig.max():.4f} >0.3={int((sig > 0.3).sum())}")
            else:
                print(f"    {name}: raw[{arr.min():.2f}, {arr.max():.2f}]")
    except Exception as e:
        print(f"    [WARN] No se pudo ejecutar ONNX float: {e}")

    # --- INT8 cuantizado ---
    print(f"  [INT8 CUANTIZADO]")
    raw = executor.forward(test_img)
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    for name, tensor in zip(output_names, raw):
        arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.array(tensor)
        if "score" in name:
            sig = 1.0 / (1.0 + np.exp(-np.clip(arr, -50, 50)))
            print(f"    {name}: logit[{arr.min():.2f}, {arr.max():.2f}] "
                  f"sigmoid_max={sig.max():.4f} >0.3={int((sig > 0.3).sum())}")
        else:
            print(f"    {name}: raw[{arr.min():.2f}, {arr.max():.2f}]")

    # Paso 2: Crear validador cuantizado
    pt_path = os.path.join(BASE_DIR, config["pt_file"])
    model = YOLO(pt_path)

    QuantValidator = make_quant_validator_class(executor, output_names, config)

    # Configurar validador
    cfg = get_cfg(overrides={
        "data": data_yaml,
        "imgsz": IMGSZ,
        "batch": batch_size,
        "conf": 0.001,
        "iou": 0.65,
        "verbose": False,
        "plots": False,
        "task": "detect",
    })

    validator = QuantValidator(args=cfg)
    validator.is_coco = False

    print(f"\n  Ejecutando validación INT8 simulada...")
    t0 = time.time()
    try:
        stats = validator(model=model.model)
        elapsed = time.time() - t0
    except Exception as e:
        print(f"  [ERROR] Validación falló: {e}")
        import traceback
        traceback.print_exc()
        return None

    metrics = {
        "mAP50": float(stats.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(stats.get("metrics/mAP50-95(B)", 0)),
        "elapsed": elapsed,
    }

    print(f"\n  INT8 mAP50:    {metrics['mAP50']:.4f}")
    print(f"  INT8 mAP50-95: {metrics['mAP50-95']:.4f}")
    print(f"  Tiempo eval:   {elapsed:.1f}s")

    return metrics


# ---------------------------------------------------------------------------
# Comparación y gate de validación
# ---------------------------------------------------------------------------

def compare_and_gate(float_metrics, quant_metrics, model_name):
    """Compara métricas float vs INT8 e imprime veredicto."""
    print(f"\n{'='*60}")
    print(f"  COMPARACIÓN — {model_name}")
    print(f"{'='*60}")

    if float_metrics is None or quant_metrics is None:
        print("  [WARN] No se pueden comparar (faltan métricas)")
        return

    f_map50 = float_metrics["mAP50"]
    q_map50 = quant_metrics["mAP50"]
    f_map95 = float_metrics["mAP50-95"]
    q_map95 = quant_metrics["mAP50-95"]

    degradation_50 = (1 - q_map50 / max(f_map50, 1e-6)) * 100
    degradation_95 = (1 - q_map95 / max(f_map95, 1e-6)) * 100

    print(f"  {'Métrica':<20} {'Float':>10} {'INT8':>10} {'Degradación':>12}")
    print(f"  {'─'*54}")
    print(f"  {'mAP50':<20} {f_map50:>10.4f} {q_map50:>10.4f} {degradation_50:>11.1f}%")
    print(f"  {'mAP50-95':<20} {f_map95:>10.4f} {q_map95:>10.4f} {degradation_95:>11.1f}%")

    # Gate de validación
    print()
    if f_map50 > 0 and q_map50 < f_map50 * 0.5:
        print(f"  ⛔ GATE FALLIDO: degradación mAP50 > 50%")
        print(f"     NO FLASHEAR — la cuantización es demasiado agresiva")
        print(f"     Investiga equalization/mixed-precision antes de flashear")
    elif f_map50 > 0 and q_map50 < f_map50 * 0.75:
        print(f"  ⚠️  GATE MARGINAL: degradación mAP50 entre 25-50%")
        print(f"     Flashear con precaución — evaluar on-device")
    else:
        print(f"  ✅ GATE OK: degradación aceptable (< 25%)")
        print(f"     Modelo listo para flashear al ESP32-S3")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluación mAP de modelos cuantizados (simulación INT8)")
    parser.add_argument("--model", choices=list(MODELS_CONFIG.keys()),
                        required=True, help="Modelo a evaluar")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size para validación (default: 16)")
    parser.add_argument("--data", type=str, default=None,
                        help="Ruta al data.yaml (default: dataset_maestro_aug)")
    parser.add_argument("--float-only", action="store_true",
                        help="Solo evaluar modelo float (referencia)")
    parser.add_argument("--skip-float", action="store_true",
                        help="Omitir evaluación float, solo INT8")
    args = parser.parse_args()

    config = MODELS_CONFIG[args.model]
    data_yaml = args.data or DATASET_YAML

    if not os.path.isfile(data_yaml):
        print(f"[ERROR] Dataset YAML no encontrado: {data_yaml}")
        print(f"  Usa --data para especificar la ruta")
        sys.exit(1)

    print(f"\n{'#'*60}")
    print(f"  eval_quantized.py — Ciclo 2")
    print(f"  Modelo:  {args.model}")
    print(f"  Dataset: {data_yaml}")
    print(f"  imgsz:   {IMGSZ}")
    print(f"  nc:      {NC}")
    print(f"{'#'*60}")

    float_metrics = None
    quant_metrics = None

    if not args.skip_float:
        float_metrics = eval_float(args.model, config, data_yaml, args.batch)

    if args.float_only:
        print("\n  (--float-only: omitiendo evaluación INT8)")
    else:
        quant_metrics = eval_quantized(args.model, config, data_yaml, args.batch)

    if float_metrics and quant_metrics:
        compare_and_gate(float_metrics, quant_metrics, args.model)

    print(f"\n{'#'*60}")
    print(f"  Evaluación completada")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()

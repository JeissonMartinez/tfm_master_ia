#!/usr/bin/env python3
"""
yolo26_comparative_plots.py
===========================
Genera gráficos comparativos de los 3 entrenamientos del modelo YOLO26n Custom
(Ultralytics 2-Phase) para el TFM.

Fuentes de datos:
  - experiment.json       → métricas resumen, configuración, latencia
  - test_evaluation.json  → métricas por clase en test
  - val_evaluation.json   → métricas por clase en validación
  - results_combined.csv  → curvas de pérdida época a época (V2, V3)
  - yolo_project/phase1/results.csv → curvas Phase 1 (V1 solo tiene este)

Uso:
  python 02_ING_MODELOS/Train_MLOps/scripts/yolo26_comparative_plots.py

Salida:
  02_ING_MODELOS/Train_MLOps/outputs/yolo26_comparative_plots/*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ─────────────────────────── Constantes ───────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
SAVE_DIR = OUTPUTS_DIR / "yolo26_comparative_plots"

# Mapeo ordenado: (folder, etiqueta, épocas completadas, tiempo min)
# Datos del Registro de Entrenamiento YOLO26
TRAININGS: list[dict] = [
    {"id": "yolo26n_custom_v1-run1", "label": "T1 · Baseline",     "epochs": 100, "time_min": 26.0},
    {"id": "yolo26n_custom_v2-run1", "label": "T2 · MuSGD",        "epochs":  98, "time_min": 32.6},
    {"id": "yolo26n_custom_v3-run1", "label": "T3 · DFL Removal",  "epochs":  92, "time_min": 30.5},
]

SHORT_LABELS = [f"T{i+1}" for i in range(len(TRAININGS))]
CLASS_NAMES = ["dog", "door", "obstacle", "person", "stair"]
N_TRAIN = len(TRAININGS)

# Índice del mejor modelo (T2 — mejor en todas las métricas)
IDX_BEST = 1

# Paleta magma — 3 colores bien separados
MAGMA_COLORS = plt.cm.magma(np.linspace(0.25, 0.80, N_TRAIN))

# Estilo global
DPI = 200
GRID_ALPHA = 0.3
FONT_TITLE = 14
FONT_LABEL = 12
FONT_TICK = 10
FONT_ANNOT = 8

# ─────────────────────────── Carga de datos ───────────────────────────


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_training_history(folder: Path) -> pd.DataFrame:
    """Carga el historial de entrenamiento.

    V2 y V3 tienen results_combined.csv con ambas fases.
    V1 solo tiene yolo_project/phase1/results.csv (30 epochs).
    """
    combined = folder / "results_combined.csv"
    if combined.exists():
        df = pd.read_csv(combined)
    else:
        # Fallback: solo Phase 1
        phase1 = folder / "yolo_project" / "phase1" / "results.csv"
        df = pd.read_csv(phase1)

    # Normalizar nombres de columnas (Ultralytics añade espacios)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_all_data() -> dict:
    """Carga todos los datos de los 3 entrenamientos."""
    data: dict = {
        "experiments": [],
        "test_evals": [],
        "val_evals": [],
        "histories": [],
    }
    for t in TRAININGS:
        folder = OUTPUTS_DIR / t["id"]
        data["experiments"].append(load_json(folder / "experiment.json"))
        data["test_evals"].append(load_json(folder / "test_evaluation.json"))
        data["val_evals"].append(load_json(folder / "val_evaluation.json"))
        data["histories"].append(load_training_history(folder))

    return data


# ─────────────────────────── Utilidades ───────────────────────────


def highlight_best(ax, x_pos: float, width: float = 0.8):
    """Fondo sutil en la columna del mejor modelo."""
    ax.axvspan(x_pos - width / 2, x_pos + width / 2, color="gold", alpha=0.08, zorder=0)


def save_fig(fig, name: str):
    path = SAVE_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ────────────────────── GRÁFICO 1: Barras agrupadas métricas globales ──────────────────────


def plot_01_global_metrics(data: dict):
    """Barras agrupadas: mAP@50, mAP@50:95, F1, Precision, Recall en test."""
    metrics = ["mAP@50", "mAP@50:95", "F1", "Precision", "Recall"]
    values = np.zeros((N_TRAIN, len(metrics)))
    for i, te in enumerate(data["test_evals"]):
        values[i] = [te["mAP50"], te.get("mAP50_95", 0), te["f1"], te["precision"], te["recall"]]

    x = np.arange(N_TRAIN)
    n_metrics = len(metrics)
    width = 0.14
    offsets = np.linspace(-(n_metrics - 1) * width / 2, (n_metrics - 1) * width / 2, n_metrics)

    metric_colors = plt.cm.magma(np.linspace(0.20, 0.85, n_metrics))

    fig, ax = plt.subplots(figsize=(12, 7))
    highlight_best(ax, IDX_BEST, width=1.0)

    for j, (metric, color) in enumerate(zip(metrics, metric_colors)):
        bars = ax.bar(x + offsets[j], values[:, j], width, label=metric,
                      color=color, edgecolor="white", linewidth=0.5, zorder=3)
        # Borde dorado para T2 (mejor)
        for k, bar in enumerate(bars):
            if k == IDX_BEST:
                bar.set_edgecolor("#D4AF37")
                bar.set_linewidth(2.0)
        # Valores sobre barras
        for k, v in enumerate(values[:, j]):
            ax.text(x[k] + offsets[j], v + 0.008, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", rotation=90, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([t["label"] for t in TRAININGS], fontsize=FONT_TICK + 1)
    ax.set_ylabel("Valor de la métrica", fontsize=FONT_LABEL)
    ax.set_ylim(0, 1.0)
    ax.set_title("Modelo 2: YOLO Family\nComparativa de métricas de evaluación — Test set",
                 fontsize=FONT_TITLE, fontweight="bold", pad=15)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=GRID_ALPHA, zorder=0)

    ax.annotate("★ Mejor modelo (T2 · MuSGD)", xy=(0.98, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="#D4AF37", fontweight="bold")

    fig.tight_layout()
    save_fig(fig, "01_metricas_globales_test")


# ────────────────────── GRÁFICO 2: Heatmap AP50 por clase ──────────────────────


def plot_02_heatmap_ap50(data: dict):
    """Heatmap 3×5: AP@50 por clase y entrenamiento."""
    matrix = np.zeros((N_TRAIN, len(CLASS_NAMES)))
    for i, te in enumerate(data["test_evals"]):
        for j, cls in enumerate(CLASS_NAMES):
            matrix[i, j] = te["per_class_ap50"].get(cls, 0.0)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, cmap="magma", aspect="auto", vmin=0.55, vmax=0.90)

    for i in range(N_TRAIN):
        for j in range(len(CLASS_NAMES)):
            val = matrix[i, j]
            text_color = "white" if val < 0.72 else "black"
            fontw = "bold" if val == matrix[:, j].max() else "normal"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=text_color, fontweight=fontw)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES], fontsize=FONT_TICK + 1)
    ax.set_yticks(range(N_TRAIN))
    ylabels = [t["label"] for t in TRAININGS]
    ylabels[IDX_BEST] += " ★"
    ax.set_yticklabels(ylabels, fontsize=FONT_TICK + 1)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("AP@50", fontsize=FONT_LABEL)

    ax.set_title("Modelo 2: YOLO Family\nAP@50 por clase y entrenamiento — Test set",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)

    fig.tight_layout()
    save_fig(fig, "02_heatmap_ap50_por_clase")


# ────────────────────── GRÁFICO 3: Heatmap F1 por clase ──────────────────────


def plot_03_heatmap_f1(data: dict):
    """Heatmap 3×5: F1-Score por clase y entrenamiento."""
    matrix = np.zeros((N_TRAIN, len(CLASS_NAMES)))
    for i, te in enumerate(data["test_evals"]):
        for j, cls in enumerate(CLASS_NAMES):
            matrix[i, j] = te["per_class_f1"].get(cls, 0.0)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, cmap="magma", aspect="auto", vmin=0.50, vmax=0.90)

    for i in range(N_TRAIN):
        for j in range(len(CLASS_NAMES)):
            val = matrix[i, j]
            text_color = "white" if val < 0.70 else "black"
            fontw = "bold" if val == matrix[:, j].max() else "normal"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=text_color, fontweight=fontw)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES], fontsize=FONT_TICK + 1)
    ax.set_yticks(range(N_TRAIN))
    ylabels = [t["label"] for t in TRAININGS]
    ylabels[IDX_BEST] += " ★"
    ax.set_yticklabels(ylabels, fontsize=FONT_TICK + 1)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("F1-Score", fontsize=FONT_LABEL)

    ax.set_title("Modelo 2: YOLO Family\nF1-Score por clase y entrenamiento — Test set",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)

    fig.tight_layout()
    save_fig(fig, "03_heatmap_f1_por_clase")


# ────────────────────── GRÁFICO 4: Curvas box_loss train ──────────────────────


def plot_04_train_box_loss(data: dict):
    """Overlay de box loss (train) para los 3 entrenamientos.

    Nota: V3 usa identity DFL (reg_max=1), por lo que su dfl_loss es ~0.01
    y el box_loss es comparable entre los tres.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (hist, t) in enumerate(zip(data["histories"], TRAININGS)):
        n_epochs = len(hist)
        epochs = hist["epoch"].values
        lw = 2.5 if i == IDX_BEST else 1.5
        ls = "-" if i == IDX_BEST else "--"
        ax.plot(epochs, hist["train/box_loss"], color=MAGMA_COLORS[i],
                linewidth=lw, linestyle=ls, label=t["label"], zorder=3 + i,
                marker="o", markersize=2, alpha=0.9)

    # Phase transition line (epoch 30 in all)
    ax.axvline(30, color="gray", linewidth=1.2, linestyle=":", alpha=0.6, zorder=2)
    ax.text(30.5, ax.get_ylim()[1] * 0.95, "Phase 1 → 2", fontsize=9,
            color="gray", style="italic", va="top")

    ax.set_xlabel("Época", fontsize=FONT_LABEL)
    ax.set_ylabel("Box Loss (CIoU) — Train", fontsize=FONT_LABEL)
    ax.set_title("Modelo 2: YOLO Family\nEvolución de Box Loss (Train) — 3 entrenamientos",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=GRID_ALPHA)

    fig.tight_layout()
    save_fig(fig, "04_train_box_loss_overlay")


# ────────────────────── GRÁFICO 5a: Curvas cls_loss train ──────────────────────


def plot_05a_cls_loss_train(data: dict):
    """Overlay de classification loss (train) para los 3 entrenamientos."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (hist, t) in enumerate(zip(data["histories"], TRAININGS)):
        epochs = hist["epoch"].values
        lw = 2.5 if i == IDX_BEST else 1.5
        ls = "-" if i == IDX_BEST else "--"

        ax.plot(epochs, hist["train/cls_loss"], color=MAGMA_COLORS[i],
                linewidth=lw, linestyle=ls, label=t["label"], zorder=3 + i,
                marker="o", markersize=2, alpha=0.9)

    ax.axvline(30, color="gray", linewidth=1.0, linestyle=":", alpha=0.5)
    ax.set_xlabel("Época", fontsize=FONT_LABEL)
    ax.set_ylabel("Classification Loss (BCE)", fontsize=FONT_LABEL)
    ax.set_title("Modelo 2: YOLO Family\nEvolución de Classification Loss (Train) — 3 entrenamientos",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=GRID_ALPHA)

    fig.tight_layout()
    save_fig(fig, "05a_cls_loss_train")


# ────────────────────── GRÁFICO 5b: Curvas cls_loss validación ──────────────────────


def plot_05b_cls_loss_val(data: dict):
    """Overlay de classification loss (val) para los 3 entrenamientos."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (hist, t) in enumerate(zip(data["histories"], TRAININGS)):
        epochs = hist["epoch"].values
        lw = 2.5 if i == IDX_BEST else 1.5
        ls = "-" if i == IDX_BEST else "--"

        ax.plot(epochs, hist["val/cls_loss"], color=MAGMA_COLORS[i],
                linewidth=lw, linestyle=ls, label=t["label"], zorder=3 + i,
                marker="s", markersize=2, alpha=0.9)

    ax.axvline(30, color="gray", linewidth=1.0, linestyle=":", alpha=0.5)
    ax.set_xlabel("Época", fontsize=FONT_LABEL)
    ax.set_ylabel("Classification Loss (BCE)", fontsize=FONT_LABEL)
    ax.set_title("Modelo 2: YOLO Family\nEvolución de Classification Loss (Val) — 3 entrenamientos",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=GRID_ALPHA)

    fig.tight_layout()
    save_fig(fig, "05b_cls_loss_val")


# ────────────────────── GRÁFICO 6: Scatter Precision vs Recall ──────────────────────


def plot_06_precision_recall_scatter(data: dict):
    """Scatterplot: Precision (X) vs Recall (Y) con iso-F1."""
    precisions = [te["precision"] for te in data["test_evals"]]
    recalls = [te["recall"] for te in data["test_evals"]]
    map50s = [te["mAP50"] for te in data["test_evals"]]
    f1s = [te["f1"] for te in data["test_evals"]]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Límites
    p_min, p_max = min(precisions) - 0.06, max(precisions) + 0.06
    r_min, r_max = min(recalls) - 0.06, max(recalls) + 0.06
    ax.set_xlim(p_min, p_max)
    ax.set_ylim(r_min, r_max)

    # Curvas iso-F1
    for f1_val in [0.65, 0.70, 0.75, 0.80]:
        p_range = np.linspace(max(p_min, 0.01), p_max, 300)
        denom = 2 * p_range - f1_val
        with np.errstate(divide="ignore", invalid="ignore"):
            r_iso = np.where(np.abs(denom) > 1e-12, (f1_val * p_range) / denom, np.nan)
        mask = (r_iso >= r_min) & (r_iso <= r_max) & np.isfinite(r_iso)
        if mask.any():
            ax.plot(p_range[mask], r_iso[mask], color="gray", alpha=0.25,
                    linewidth=0.8, linestyle="--")
            idx_label = np.argmax(p_range[mask])
            ax.text(p_range[mask][idx_label] - 0.01, r_iso[mask][idx_label],
                    f"F1={f1_val:.2f}", fontsize=7, color="gray", alpha=0.6,
                    ha="right", va="bottom")

    # Puntos — tamaño proporcional a mAP@50
    sizes = [200 + 400 * (m - min(map50s)) / (max(map50s) - min(map50s) + 1e-9) for m in map50s]

    for i in range(N_TRAIN):
        edgecolor = "#D4AF37" if i == IDX_BEST else "white"
        lw = 2.5 if i == IDX_BEST else 1.0
        ax.scatter(precisions[i], recalls[i], s=sizes[i], c=[MAGMA_COLORS[i]],
                   marker="o", edgecolors=edgecolor, linewidths=lw, zorder=5 + i)

    # Etiquetas
    offsets_manual = {
        0: (15, -15),   # T1
        1: (15, 10),    # T2
        2: (-15, -15),  # T3
    }
    for i in range(N_TRAIN):
        oxy = offsets_manual[i]
        ha = "left" if oxy[0] > 0 else "right"
        ax.annotate(f"{SHORT_LABELS[i]}  F1={f1s[i]:.3f}\nmAP@50={map50s[i]:.3f}",
                    (precisions[i], recalls[i]),
                    textcoords="offset points", xytext=oxy,
                    fontsize=9, fontweight="bold", ha=ha,
                    color=MAGMA_COLORS[i],
                    arrowprops=dict(arrowstyle="-", color=MAGMA_COLORS[i], alpha=0.5, lw=0.8))

    ax.set_xlabel("Precision", fontsize=FONT_LABEL)
    ax.set_ylabel("Recall", fontsize=FONT_LABEL)
    ax.set_title("Modelo 2: YOLO Family\nTrade-off Precision–Recall — Test set  (tamaño ∝ mAP@50)",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.grid(True, alpha=GRID_ALPHA)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D4AF37",
               markeredgecolor="#D4AF37", markersize=12, label="★ Mejor modelo (T2)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower left", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "06_precision_recall_scatter")


# ────────────────────── GRÁFICO 7: Radar chart T1 vs T2 vs T3 ──────────────────────


def plot_07_radar_chart(data: dict):
    """Radar chart: mAP@50, mAP@50:95, Precision, Recall, F1 para los 3 entrenamientos."""
    labels_radar = ["mAP@50", "mAP@50:95", "Precision", "Recall", "F1"]

    values_all = []
    for te in data["test_evals"]:
        values_all.append([te["mAP50"], te.get("mAP50_95", 0), te["precision"], te["recall"], te["f1"]])

    n_axes = len(labels_radar)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    line_styles = ["--", "-", "-."]
    markers = ["o", "*", "D"]

    for k, (vals, t) in enumerate(zip(values_all, TRAININGS)):
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, color=MAGMA_COLORS[k], linewidth=2.5,
                linestyle=line_styles[k], label=t["label"], zorder=5)
        ax.fill(angles, vals_closed, color=MAGMA_COLORS[k], alpha=0.12)
        ax.scatter(angles[:-1], vals, color=MAGMA_COLORS[k], s=60,
                   marker=markers[k], zorder=6, edgecolors="white", linewidths=0.5)

        for a, v in zip(angles[:-1], vals):
            ax.text(a, v + 0.035, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=MAGMA_COLORS[k])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_radar, fontsize=FONT_TICK + 1, fontweight="bold")
    ax.set_ylim(0, 0.95)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=FONT_TICK - 1, color="gray")
    ax.set_title("Modelo 2: YOLO Family\nPerfil de métricas — 3 entrenamientos YOLO26n",
                 fontsize=FONT_TITLE, fontweight="bold", pad=30)
    ax.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.25, 1.12), framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "07_radar_chart")


# ────────────────────── GRÁFICO 8: Detecciones TP vs FP ──────────────────────


def plot_08_detections_tp_fp(data: dict):
    """Barras horizontales apiladas: TP vs FP por entrenamiento (de confusion matrix)."""
    # Extraer TP y FP de la confusion matrix
    # Filas 0..4 = GT classes, última fila = FP (background predictions)
    # Diagonal de las primeras 5 filas = TP
    # Última fila sums = total FP
    tps = []
    fps = []
    for te in data["test_evals"]:
        cm = np.array(te["confusion_matrix"])
        tp = int(sum(cm[i][i] for i in range(5)))
        fp = int(sum(cm[5]))  # FP row
        tps.append(tp)
        fps.append(fp)

    n_gt = sum(int(sum(row[:6])) for row in np.array(data["test_evals"][0]["confusion_matrix"])[:5])
    totals = [tp + fp for tp, fp in zip(tps, fps)]

    y = np.arange(N_TRAIN)
    labels = [t["label"] for t in TRAININGS]

    fig, ax = plt.subplots(figsize=(12, 5))

    bars_tp = ax.barh(y, tps, color=plt.cm.magma(0.65), edgecolor="white",
                      linewidth=0.5, label="True Positives (TP)", zorder=3)
    bars_fp = ax.barh(y, fps, left=tps, color=plt.cm.magma(0.25), edgecolor="white",
                      linewidth=0.5, label="False Positives (FP)", zorder=3)

    # Borde dorado para T2
    for bar_set in [bars_tp, bars_fp]:
        bar_set[IDX_BEST].set_edgecolor("#D4AF37")
        bar_set[IDX_BEST].set_linewidth(2.0)

    for k in range(N_TRAIN):
        ax.text(tps[k] / 2, y[k], f"TP={tps[k]}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        fp_center = tps[k] + fps[k] / 2
        ax.text(fp_center, y[k], f"FP={fps[k]}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white" if fps[k] > 80 else "black")
        ax.text(totals[k] + 5, y[k], f"Total: {totals[k]}", ha="left", va="center",
                fontsize=9, color="gray")

    # Línea referencia: ground truths
    ax.axvline(n_gt, color="red", linewidth=1.5, linestyle="--", alpha=0.7, zorder=5)
    ax.text(n_gt + 3, N_TRAIN - 0.3, f"Ground Truths = {n_gt}", fontsize=9,
            color="red", alpha=0.8, va="bottom")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FONT_TICK + 1)
    ax.set_xlabel("Número de detecciones", fontsize=FONT_LABEL)
    ax.set_title("Modelo 2: YOLO Family\nDistribución de detecciones: TP vs FP — Test set",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.grid(axis="x", alpha=GRID_ALPHA, zorder=0)
    ax.invert_yaxis()

    fig.tight_layout()
    save_fig(fig, "08_detecciones_tp_vs_fp")


# ────────────────────── GRÁFICO 9: Eficiencia tiempo vs mAP ──────────────────────


def plot_09_efficiency(data: dict):
    """Scatterplot: Tiempo (min) vs mAP@50, con anotaciones de épocas y ONNX size."""
    times = [t["time_min"] for t in TRAININGS]
    map50s = [te["mAP50"] for te in data["test_evals"]]
    onnx_sizes = [exp["onnx_size_mb"] for exp in data["experiments"]]

    fig, ax = plt.subplots(figsize=(11, 7))

    for i in range(N_TRAIN):
        edgecolor = "#D4AF37" if i == IDX_BEST else "white"
        lw = 2.5 if i == IDX_BEST else 1.0
        ax.scatter(times[i], map50s[i], s=250, c=[MAGMA_COLORS[i]],
                   marker="o", edgecolors=edgecolor, linewidths=lw, zorder=5 + i)

        ax.annotate(f"{SHORT_LABELS[i]}\n{TRAININGS[i]['epochs']}ep\nONNX:{onnx_sizes[i]:.1f}MB",
                    (times[i], map50s[i]),
                    textcoords="offset points", xytext=(15, -8),
                    fontsize=9, fontweight="bold", color=MAGMA_COLORS[i],
                    arrowprops=dict(arrowstyle="-", color=MAGMA_COLORS[i], alpha=0.4, lw=0.8))

    ax.set_xlabel("Tiempo de entrenamiento (min)", fontsize=FONT_LABEL)
    ax.set_ylabel("mAP@50 (Test)", fontsize=FONT_LABEL)
    ax.set_title("Modelo 2: YOLO Family\nEficiencia — Tiempo de entrenamiento vs mAP@50",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.grid(True, alpha=GRID_ALPHA)

    ax.annotate("★ Mejor modelo (T2 · MuSGD)", xy=(0.02, 0.97), xycoords="axes fraction",
                fontsize=9, color="#D4AF37", fontweight="bold", va="top")

    fig.tight_layout()
    save_fig(fig, "09_eficiencia_tiempo_vs_map")


# ────────────────────── GRÁFICO 10: Evolución iterativa ──────────────────────


def plot_10_iterative_evolution(data: dict):
    """Evolución iterativa: mAP@50 y F1 a lo largo de los 3 entrenamientos."""
    x = np.arange(N_TRAIN)
    test_map50 = [te["mAP50"] for te in data["test_evals"]]
    val_map50 = [exp["val_map50"] for exp in data["experiments"]]
    test_f1 = [te["f1"] for te in data["test_evals"]]

    fig, ax1 = plt.subplots(figsize=(12, 7))

    color_test = plt.cm.magma(0.75)
    color_val = plt.cm.magma(0.45)
    color_f1 = plt.cm.magma(0.25)

    line_test = ax1.plot(x, test_map50, "o-", color=color_test, linewidth=2.5, markersize=12,
                         label="mAP@50 (Test)", zorder=5)
    line_val = ax1.plot(x, val_map50, "s--", color=color_val, linewidth=2.0, markersize=10,
                        label="mAP@50 (Val)", zorder=4, alpha=0.8)

    # Marcar T2
    ax1.scatter([IDX_BEST], [test_map50[IDX_BEST]], s=250, marker="o",
                color="#D4AF37", edgecolors="black", linewidths=1.5, zorder=10)

    ax1.set_ylabel("mAP@50", fontsize=FONT_LABEL, color=color_test)
    ax1.tick_params(axis="y", labelcolor=color_test)

    # Eje secundario
    ax2 = ax1.twinx()
    line_f1 = ax2.plot(x, test_f1, "^-.", color=color_f1, linewidth=2.0, markersize=10,
                       label="F1 (Test)", zorder=3, alpha=0.85)
    ax2.set_ylabel("F1-Score", fontsize=FONT_LABEL, color=color_f1)
    ax2.tick_params(axis="y", labelcolor=color_f1)

    # Valores
    for i in range(N_TRAIN):
        ax1.text(i, test_map50[i] + 0.006, f"{test_map50[i]:.4f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color=color_test)
        ax2.text(i, test_f1[i] - 0.006, f"{test_f1[i]:.4f}", ha="center", va="top",
                 fontsize=9, color=color_f1)

    ax1.set_xticks(x)
    xlabels = [t["label"].replace(" · ", "\n") for t in TRAININGS]
    ax1.set_xticklabels(xlabels, fontsize=FONT_TICK + 1, ha="center")

    ax1.set_title("Modelo 2: YOLO Family\nEvolución iterativa de métricas entre entrenamientos",
                  fontsize=FONT_TITLE, fontweight="bold", pad=15)
    ax1.grid(True, alpha=GRID_ALPHA)

    # Leyenda combinada
    lines = line_test + line_val + line_f1
    labels_leg = [str(l.get_label()) for l in lines]
    ax1.legend(lines, labels_leg, fontsize=10, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.15), framealpha=0.9)

    # Anotaciones de cambios
    changes = [
        (0, 1, "MuSGD\noptimizer"),
        (1, 2, "DFL removal\nyolo26n.pt"),
    ]
    for i_from, i_to, txt in changes:
        mid_x = (i_from + i_to) / 2
        mid_y = (test_map50[i_from] + test_map50[i_to]) / 2
        ax1.annotate(txt, xy=(mid_x, mid_y + 0.012), fontsize=8, ha="center",
                     va="bottom", color="gray", style="italic")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    save_fig(fig, "10_evolucion_iterativa")


# ────────────────────── MAIN ──────────────────────


def main():
    print("=" * 60)
    print("YOLO26n — Gráficos Comparativos (3 entrenamientos)")
    print("=" * 60)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDirectorio de salida: {SAVE_DIR}\n")

    print("Cargando datos...")
    data = load_all_data()
    print(f"  {N_TRAIN} entrenamientos cargados.\n")

    print("Generando gráficos:\n")

    plot_01_global_metrics(data)
    plot_02_heatmap_ap50(data)
    plot_03_heatmap_f1(data)
    plot_04_train_box_loss(data)
    plot_05a_cls_loss_train(data)
    plot_05b_cls_loss_val(data)
    plot_06_precision_recall_scatter(data)
    plot_07_radar_chart(data)
    plot_08_detections_tp_fp(data)
    plot_09_efficiency(data)
    plot_10_iterative_evolution(data)

    print(f"\n{'=' * 60}")
    print(f"✅ 11 gráficos generados en: {SAVE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

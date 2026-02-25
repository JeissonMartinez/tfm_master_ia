#!/usr/bin/env python3
"""
fcos_comparative_plots.py
=========================
Genera gráficos comparativos de los 8 entrenamientos del modelo FCOS
(MobileNetV3-Small + SimpleFPN) para el TFM.

Fuentes de datos:
  - experiment.json      → métricas resumen, configuración, latencia
  - test_evaluation.json → métricas por clase en test
  - val_evaluation.json  → métricas por clase en validación
  - training_history.csv → curvas de pérdida época a época

Uso:
  python 02_ING_MODELOS/Train_MLOps/scripts/fcos_comparative_plots.py

Salida:
  02_ING_MODELOS/Train_MLOps/outputs/fcos_comparative_plots/*.png
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
SAVE_DIR = OUTPUTS_DIR / "fcos_comparative_plots"

# Mapeo ordenado: (Job ID folder, etiqueta corta, épocas totales, tiempo min)
# Épocas y tiempo tomados del Registro de Entrenamiento
TRAININGS: list[dict] = [
    {"id": "fcos_v3s_v1-1771683868", "label": "T1 · Baseline",          "epochs": 52,  "time_min": 12.1},
    {"id": "fcos_v3s_v1-1771687747", "label": "T2 · Stride Norm",       "epochs": 74,  "time_min": 15.9},
    {"id": "fcos_v3s_v1-1771690809", "label": "T3 · GIoU Loss",         "epochs": 101, "time_min": 23.6},
    {"id": "fcos_v3s_v1-1771695807", "label": "T4 · Scoring Ref.",      "epochs": 77,  "time_min": 17.9},
    {"id": "fcos_v3s_v1-1771710798", "label": "T5 · Hybrid Loss",       "epochs": 76,  "time_min": 17.9},
    {"id": "fcos_v3s_v1-1771715459", "label": "T6 · Phase1 Ext.",       "epochs": 86,  "time_min": 19.5},
    {"id": "fcos_v3s_v1-1771726575", "label": "T7 · Config Final",      "epochs": 98,  "time_min": 23.0},
    {"id": "fcos_v3s_v1-1771751066", "label": "T8 · Focal Loss",        "epochs": 66,  "time_min": 17.0},
]

SHORT_LABELS = [f"T{i+1}" for i in range(len(TRAININGS))]
CLASS_NAMES = ["dog", "door", "obstacle", "person", "stair"]
N_TRAIN = len(TRAININGS)

# Índices destacados
IDX_SELECTED = 2  # T3 — modelo seleccionado
IDX_BENCHMARK = 6  # T7 — benchmark

# Paleta magma — 8 colores equiespaciados evitando extremos
MAGMA_COLORS = plt.cm.magma(np.linspace(0.15, 0.92, N_TRAIN))

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


def load_all_data() -> dict:
    """Carga todos los datos de los 8 entrenamientos."""
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
        data["histories"].append(pd.read_csv(folder / "training_history.csv"))

    return data


# ─────────────────────────── Utilidades ───────────────────────────


def highlight_selected(ax, x_pos: float, width: float = 0.8):
    """Añade un fondo sutil en la columna del modelo seleccionado."""
    ax.axvspan(x_pos - width / 2, x_pos + width / 2, color="gold", alpha=0.08, zorder=0)


def save_fig(fig, name: str):
    path = SAVE_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {path.name}")


def add_star_annotation(ax, x, y, offset_y=0.015):
    """Marca el modelo seleccionado con una estrella."""
    ax.annotate("★", (x, y + offset_y), fontsize=11, ha="center", va="bottom",
                color="#D4AF37", fontweight="bold", zorder=10)


# ────────────────────── GRÁFICO 1: Barras agrupadas métricas globales ──────────────────────


def plot_01_global_metrics(data: dict):
    """Barras agrupadas: mAP@50, F1, Precision, Recall en test."""
    metrics = ["mAP@50", "F1", "Precision", "Recall"]
    values = np.zeros((N_TRAIN, 4))
    for i, te in enumerate(data["test_evals"]):
        values[i] = [te["mAP50"], te["f1"], te["precision"], te["recall"]]

    x = np.arange(N_TRAIN)
    n_metrics = len(metrics)
    width = 0.18
    offsets = np.linspace(-(n_metrics - 1) * width / 2, (n_metrics - 1) * width / 2, n_metrics)

    # Colores para las métricas sacados de magma
    metric_colors = plt.cm.magma([0.25, 0.45, 0.65, 0.85])

    fig, ax = plt.subplots(figsize=(16, 8))
    highlight_selected(ax, IDX_SELECTED, width=1.0)

    for j, (metric, color) in enumerate(zip(metrics, metric_colors)):
        bars = ax.bar(x + offsets[j], values[:, j], width, label=metric,
                      color=color, edgecolor="white", linewidth=0.5, zorder=3)
        # Bordes especiales para T3 y T7
        for k, bar in enumerate(bars):
            if k == IDX_SELECTED:
                bar.set_edgecolor("#D4AF37")
                bar.set_linewidth(2.0)
            elif k == IDX_BENCHMARK:
                bar.set_edgecolor("#4A90D9")
                bar.set_linewidth(1.5)
        # Valores sobre barras
        for k, v in enumerate(values[:, j]):
            ax.text(x[k] + offsets[j], v + 0.008, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold", rotation=90, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([t["label"] for t in TRAININGS], fontsize=FONT_TICK, rotation=20, ha="right")
    ax.set_ylabel("Valor de la métrica", fontsize=FONT_LABEL)
    ax.set_ylim(0, 0.85)
    ax.set_title("Comparativa de métricas de evaluación — Test set",
                 fontsize=FONT_TITLE, fontweight="bold", pad=15)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=GRID_ALPHA, zorder=0)

    # Leyenda de marcas
    ax.annotate("★ Modelo seleccionado (T3)", xy=(0.98, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="#D4AF37", fontweight="bold")
    ax.annotate("◆ Benchmark (T7)", xy=(0.98, 0.93), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="#4A90D9")

    fig.tight_layout()
    save_fig(fig, "01_metricas_globales_test")


# ────────────────────── GRÁFICO 2: Heatmap AP50 por clase ──────────────────────


def plot_02_heatmap_ap50(data: dict):
    """Heatmap 8×5: AP@50 por clase y entrenamiento."""
    matrix = np.zeros((N_TRAIN, len(CLASS_NAMES)))
    for i, te in enumerate(data["test_evals"]):
        for j, cls in enumerate(CLASS_NAMES):
            matrix[i, j] = te["per_class_ap50"].get(cls, 0.0)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(matrix, cmap="magma", aspect="auto", vmin=0.3, vmax=0.85)

    # Texto en celdas
    for i in range(N_TRAIN):
        for j in range(len(CLASS_NAMES)):
            val = matrix[i, j]
            text_color = "white" if val < 0.55 else "black"
            fontw = "bold" if val == matrix[:, j].max() else "normal"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight=fontw)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES], fontsize=FONT_TICK)
    ax.set_yticks(range(N_TRAIN))
    ylabels = [t["label"] for t in TRAININGS]
    ylabels[IDX_SELECTED] += " ★"
    ylabels[IDX_BENCHMARK] += " ◆"
    ax.set_yticklabels(ylabels, fontsize=FONT_TICK)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("AP@50", fontsize=FONT_LABEL)

    ax.set_title("AP@50 por clase y entrenamiento — Test set",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)

    fig.tight_layout()
    save_fig(fig, "02_heatmap_ap50_por_clase")


# ────────────────────── GRÁFICO 3: Heatmap F1 por clase ──────────────────────


def plot_03_heatmap_f1(data: dict):
    """Heatmap 8×5: F1-Score por clase y entrenamiento."""
    matrix = np.zeros((N_TRAIN, len(CLASS_NAMES)))
    for i, te in enumerate(data["test_evals"]):
        for j, cls in enumerate(CLASS_NAMES):
            matrix[i, j] = te["per_class_f1"].get(cls, 0.0)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(matrix, cmap="magma", aspect="auto", vmin=0.2, vmax=0.8)

    for i in range(N_TRAIN):
        for j in range(len(CLASS_NAMES)):
            val = matrix[i, j]
            text_color = "white" if val < 0.50 else "black"
            fontw = "bold" if val == matrix[:, j].max() else "normal"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight=fontw)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES], fontsize=FONT_TICK)
    ax.set_yticks(range(N_TRAIN))
    ylabels = [t["label"] for t in TRAININGS]
    ylabels[IDX_SELECTED] += " ★"
    ylabels[IDX_BENCHMARK] += " ◆"
    ax.set_yticklabels(ylabels, fontsize=FONT_TICK)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("F1-Score", fontsize=FONT_LABEL)

    ax.set_title("F1-Score por clase y entrenamiento — Test set",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)

    fig.tight_layout()
    save_fig(fig, "03_heatmap_f1_por_clase")


# ────────────────────── GRÁFICO 4: Curvas cls_loss train ──────────────────────


def plot_04_train_cls_loss(data: dict):
    """Overlay de classification loss (train) para los 8 entrenamientos.

    Se usa cls_loss porque reg_loss no es comparable entre T1-T2 (SL1 absoluto)
    y T3-T8 (GIoU stride-normalized). La cls_loss sí es comparable.
    Eje X normalizado a % de épocas para comparabilidad.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (hist, t) in enumerate(zip(data["histories"], TRAININGS)):
        n_epochs = len(hist)
        x_pct = np.linspace(0, 100, n_epochs)
        lw = 2.5 if i == IDX_SELECTED else (2.0 if i == IDX_BENCHMARK else 1.2)
        alpha = 1.0 if i in (IDX_SELECTED, IDX_BENCHMARK) else 0.75
        ls = "-" if i == IDX_SELECTED else ("--" if i == IDX_BENCHMARK else "-")
        ax.plot(x_pct, hist["train_cls_loss"], color=MAGMA_COLORS[i],
                linewidth=lw, alpha=alpha, linestyle=ls, label=t["label"], zorder=3 + i)

    # Marca transición Phase 1→2 (época 30 = ~30/N * 100 %)
    for i, t in enumerate(TRAININGS):
        pct_phase2 = 30 / t["epochs"] * 100
        ax.axvline(pct_phase2, color=MAGMA_COLORS[i], alpha=0.15, linewidth=0.8, linestyle=":")

    ax.set_xlabel("Progreso del entrenamiento (%)", fontsize=FONT_LABEL)
    ax.set_ylabel("Classification Loss (Train)", fontsize=FONT_LABEL)
    ax.set_title("Evolución de Classification Loss (Train) — 8 entrenamientos",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xlim(0, 100)

    # Anotación de fases
    ax.annotate("← Phase 1 | Phase 2 →", xy=(0.38, 0.95), xycoords="axes fraction",
                fontsize=9, ha="center", va="top", color="gray", style="italic")

    fig.tight_layout()
    save_fig(fig, "04_train_cls_loss_overlay")


# ────────────────────── GRÁFICO 5: Curvas cls_loss validación ──────────────────────


def plot_05_val_cls_loss(data: dict):
    """Overlay de classification loss (val) para los 8 entrenamientos."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (hist, t) in enumerate(zip(data["histories"], TRAININGS)):
        n_epochs = len(hist)
        x_pct = np.linspace(0, 100, n_epochs)
        lw = 2.5 if i == IDX_SELECTED else (2.0 if i == IDX_BENCHMARK else 1.2)
        alpha = 1.0 if i in (IDX_SELECTED, IDX_BENCHMARK) else 0.75
        ls = "-" if i == IDX_SELECTED else ("--" if i == IDX_BENCHMARK else "-")
        ax.plot(x_pct, hist["val_cls_loss"], color=MAGMA_COLORS[i],
                linewidth=lw, alpha=alpha, linestyle=ls, label=t["label"], zorder=3 + i)

    ax.set_xlabel("Progreso del entrenamiento (%)", fontsize=FONT_LABEL)
    ax.set_ylabel("Classification Loss (Val)", fontsize=FONT_LABEL)
    ax.set_title("Evolución de Classification Loss (Val) — 8 entrenamientos",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xlim(0, 100)

    fig.tight_layout()
    save_fig(fig, "05_val_cls_loss_overlay")


# ────────────────────── GRÁFICO 6: Scatter Precision vs Recall ──────────────────────


def plot_06_precision_recall_scatter(data: dict):
    """Scatterplot: Precision (X) vs Recall (Y) con iso-F1 y tamaño ∝ mAP@50."""
    precisions = [te["precision"] for te in data["test_evals"]]
    recalls = [te["recall"] for te in data["test_evals"]]
    map50s = [te["mAP50"] for te in data["test_evals"]]
    f1s = [te["f1"] for te in data["test_evals"]]

    fig, ax = plt.subplots(figsize=(14, 7))

    # Límites del gráfico — ajustados al rango real de los datos con margen
    p_min, p_max = min(precisions) - 0.05, max(precisions) + 0.08
    r_min, r_max = min(recalls) - 0.05, max(recalls) + 0.05
    ax.set_xlim(p_min, p_max)
    ax.set_ylim(r_min, r_max)

    # Curvas iso-F1 (rango adaptado a los límites del gráfico)
    for f1_val in [0.3, 0.4, 0.5, 0.6, 0.7]:
        p_range = np.linspace(max(p_min, 0.01), p_max, 300)
        denom = 2 * p_range - f1_val
        with np.errstate(divide="ignore", invalid="ignore"):
            r_iso = np.where(np.abs(denom) > 1e-12, (f1_val * p_range) / denom, np.nan)
        mask = (r_iso >= r_min) & (r_iso <= r_max) & np.isfinite(r_iso)
        if mask.any():
            ax.plot(p_range[mask], r_iso[mask], color="gray", alpha=0.25,
                    linewidth=0.8, linestyle="--")
            # Etiqueta iso-F1 — colocada en el borde derecho visible del area
            idx_label = np.argmax(p_range[mask])
            ax.text(p_range[mask][idx_label] - 0.01, r_iso[mask][idx_label],
                    f"F1={f1_val:.1f}", fontsize=7, color="gray", alpha=0.6,
                    ha="right", va="bottom")

    # Puntos — todos círculos, tamaño proporcional a mAP@50
    sizes = [120 + 500 * (m - min(map50s)) / (max(map50s) - min(map50s) + 1e-9) for m in map50s]

    for i in range(N_TRAIN):
        edgecolor = "#D4AF37" if i == IDX_SELECTED else ("#4A90D9" if i == IDX_BENCHMARK else "white")
        lw = 2.5 if i in (IDX_SELECTED, IDX_BENCHMARK) else 1.0

        ax.scatter(precisions[i], recalls[i], s=sizes[i], c=[MAGMA_COLORS[i]],
                   marker="o", edgecolors=edgecolor, linewidths=lw, zorder=5 + i)

    # Etiquetas con desplazamientos manuales para evitar solapamiento
    # Datos: T1(0.54,0.53) T2(0.60,0.63) T3(0.66,0.63) T4(0.35,0.69)
    #        T5(0.35,0.68) T6(0.33,0.66) T7(0.37,0.69) T8(0.21,0.71)
    manual_offsets = {
        0: (12, -18),   # T1 — abajo-derecha (aislado)
        1: (-15, -18),  # T2 — abajo-izquierda
        2: (14, -15),   # T3 — derecha-abajo
        3: (18, 10),    # T4 — arriba-derecha (cluster)
        4: (18, -14),   # T5 — abajo-derecha (cluster)
        5: (-50, -14),  # T6 — izquierda (cluster)
        6: (-15, 14),   # T7 — arriba-izquierda (cluster)
        7: (14, 8),     # T8 — arriba-derecha (aislado)
    }

    for i in range(N_TRAIN):
        oxy = manual_offsets[i]
        ha = "left" if oxy[0] > 0 else "right"
        ax.annotate(f"{SHORT_LABELS[i]}  F1={f1s[i]:.2f}",
                    (precisions[i], recalls[i]),
                    textcoords="offset points", xytext=oxy,
                    fontsize=8.5, fontweight="bold", ha=ha,
                    color=MAGMA_COLORS[i],
                    arrowprops=dict(arrowstyle="-", color=MAGMA_COLORS[i],
                                   alpha=0.5, lw=0.8))

    ax.set_xlabel("Precision", fontsize=FONT_LABEL)
    ax.set_ylabel("Recall", fontsize=FONT_LABEL)
    ax.set_title("Trade-off Precision–Recall — Test set  (tamaño ∝ mAP@50)",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_aspect("auto")

    # Leyenda
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D4AF37",
               markeredgecolor="#D4AF37", markersize=12, label="Modelo seleccionado (T3)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4A90D9",
               markeredgecolor="#4A90D9", markersize=10, label="Benchmark (T7)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower left", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "06_precision_recall_scatter")


# ────────────────────── GRÁFICO 7: Radar chart T1 vs T3 vs T7 ──────────────────────


def plot_07_radar_chart(data: dict):
    """Radar chart: mAP@50, mAP@50:95, Precision, Recall, F1 para T1, T3, T7."""
    indices = [0, IDX_SELECTED, IDX_BENCHMARK]  # T1, T3, T7
    labels_radar = ["mAP@50", "mAP@50:95", "Precision", "Recall", "F1"]

    values_all = []
    for idx in indices:
        te = data["test_evals"][idx]
        values_all.append([te["mAP50"], te.get("mAP50_95", 0), te["precision"], te["recall"], te["f1"]])

    n_axes = len(labels_radar)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # Cerrar polígono

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    colors_radar = [MAGMA_COLORS[0], MAGMA_COLORS[IDX_SELECTED], MAGMA_COLORS[IDX_BENCHMARK]]
    labels_legend = [
        "T1 · Baseline",
        "T3 · GIoU Loss ★",
        "T7 · Config Final ◆",
    ]
    line_styles = ["--", "-", "-."]
    markers = ["o", "*", "D"]

    for k, (vals, color, lbl, ls, marker) in enumerate(
        zip(values_all, colors_radar, labels_legend, line_styles, markers)
    ):
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, color=color, linewidth=2.5, linestyle=ls, label=lbl, zorder=5)
        ax.fill(angles, vals_closed, color=color, alpha=0.12)
        ax.scatter(angles[:-1], vals, color=color, s=60, marker=marker, zorder=6, edgecolors="white", linewidths=0.5)

        # Valores junto a los puntos
        for a, v in zip(angles[:-1], vals):
            offset_r = 0.04
            ax.text(a, v + offset_r, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_radar, fontsize=FONT_TICK + 1, fontweight="bold")
    ax.set_ylim(0, 0.80)
    ax.set_yticks([0.2, 0.4, 0.6])
    ax.set_yticklabels(["0.2", "0.4", "0.6"], fontsize=FONT_TICK - 1, color="gray")
    ax.set_title("Perfil de métricas — Baseline vs Seleccionado vs Benchmark",
                 fontsize=FONT_TITLE, fontweight="bold", pad=30)
    ax.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.25, 1.12), framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "07_radar_chart_t1_t3_t7")


# ────────────────────── GRÁFICO 8: TP vs FP barras horizontales ──────────────────────


def plot_08_detections_tp_fp(data: dict):
    """Barras horizontales apiladas: TP vs FP por entrenamiento."""
    n_gt = data["test_evals"][0]["n_ground_truths"]  # 576 para todos

    tps = []
    fps = []
    n_dets = []
    for te in data["test_evals"]:
        tp = round(te["recall"] * n_gt)
        n_det = te["n_detections"]
        fp = n_det - tp
        tps.append(tp)
        fps.append(fp)
        n_dets.append(n_det)

    y = np.arange(N_TRAIN)
    labels = [t["label"] for t in TRAININGS]

    fig, ax = plt.subplots(figsize=(14, 7))

    bars_tp = ax.barh(y, tps, color=plt.cm.magma(0.65), edgecolor="white",
                      linewidth=0.5, label="True Positives (TP)", zorder=3)
    bars_fp = ax.barh(y, fps, left=tps, color=plt.cm.magma(0.25), edgecolor="white",
                      linewidth=0.5, label="False Positives (FP)", zorder=3)

    # Bordes especiales
    for k in range(N_TRAIN):
        if k == IDX_SELECTED:
            bars_tp[k].set_edgecolor("#D4AF37")
            bars_tp[k].set_linewidth(2.0)
            bars_fp[k].set_edgecolor("#D4AF37")
            bars_fp[k].set_linewidth(2.0)
        elif k == IDX_BENCHMARK:
            bars_tp[k].set_edgecolor("#4A90D9")
            bars_tp[k].set_linewidth(1.5)
            bars_fp[k].set_edgecolor("#4A90D9")
            bars_fp[k].set_linewidth(1.5)

    # Anotaciones
    for k in range(N_TRAIN):
        # TP centrado
        ax.text(tps[k] / 2, y[k], f"TP={tps[k]}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")
        # FP centrado
        fp_center = tps[k] + fps[k] / 2
        ax.text(fp_center, y[k], f"FP={fps[k]}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white" if fps[k] > 80 else "black")
        # Total a la derecha
        ax.text(n_dets[k] + 8, y[k], f"Total: {n_dets[k]}", ha="left", va="center",
                fontsize=8, color="gray")

    # Línea de referencia en n_gt
    ax.axvline(n_gt, color="red", linewidth=1.5, linestyle="--", alpha=0.7, zorder=5)
    ax.text(n_gt + 5, N_TRAIN - 0.3, f"Ground Truths = {n_gt}", fontsize=9,
            color="red", alpha=0.8, va="bottom")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FONT_TICK)
    ax.set_xlabel("Número de detecciones", fontsize=FONT_LABEL)
    ax.set_title("Distribución de detecciones: TP vs FP — Test set",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.grid(axis="x", alpha=GRID_ALPHA, zorder=0)
    ax.invert_yaxis()

    fig.tight_layout()
    save_fig(fig, "08_detecciones_tp_vs_fp")


# ────────────────────── GRÁFICO 9: Eficiencia tiempo vs mAP ──────────────────────


def plot_09_efficiency(data: dict):
    """Scatterplot: Tiempo entrenamiento (min) vs mAP@50."""
    times = [t["time_min"] for t in TRAININGS]
    map50s = [te["mAP50"] for te in data["test_evals"]]

    fig, ax = plt.subplots(figsize=(12, 8))

    for i in range(N_TRAIN):
        marker = "*" if i == IDX_SELECTED else ("D" if i == IDX_BENCHMARK else "o")
        ms = 250 if i in (IDX_SELECTED, IDX_BENCHMARK) else 150
        edgecolor = "#D4AF37" if i == IDX_SELECTED else ("#4A90D9" if i == IDX_BENCHMARK else "white")
        lw = 2.5 if i in (IDX_SELECTED, IDX_BENCHMARK) else 1.0

        ax.scatter(times[i], map50s[i], s=ms, c=[MAGMA_COLORS[i]],
                   marker=marker, edgecolors=edgecolor, linewidths=lw, zorder=5 + i)

        # Etiqueta
        offset_x = 0.4
        ax.annotate(f"{SHORT_LABELS[i]}\n{TRAININGS[i]['epochs']}ep",
                    (times[i], map50s[i]),
                    textcoords="offset points", xytext=(12, -5),
                    fontsize=9, fontweight="bold", color=MAGMA_COLORS[i],
                    arrowprops=dict(arrowstyle="-", color=MAGMA_COLORS[i], alpha=0.4, lw=0.8))

    ax.set_xlabel("Tiempo de entrenamiento (min)", fontsize=FONT_LABEL)
    ax.set_ylabel("mAP@50 (Test)", fontsize=FONT_LABEL)
    ax.set_title("Eficiencia — Tiempo de entrenamiento vs mAP@50",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.grid(True, alpha=GRID_ALPHA)

    # Tendencia (regresión lineal para referencia visual)
    z = np.polyfit(times, map50s, 1)
    p = np.poly1d(z)
    t_range = np.linspace(min(times) - 1, max(times) + 1, 50)
    ax.plot(t_range, p(t_range), color="gray", alpha=0.3, linewidth=1.5, linestyle=":", label="Tendencia lineal")

    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.annotate("★ Modelo seleccionado (T3)", xy=(0.02, 0.97), xycoords="axes fraction",
                fontsize=9, color="#D4AF37", fontweight="bold", va="top")
    ax.annotate("◆ Benchmark (T7)", xy=(0.02, 0.93), xycoords="axes fraction",
                fontsize=9, color="#4A90D9", va="top")

    fig.tight_layout()
    save_fig(fig, "09_eficiencia_tiempo_vs_map")


# ────────────────────── GRÁFICO 10: Evolución iterativa ──────────────────────


def plot_10_iterative_evolution(data: dict):
    """Evolución iterativa: mAP@50 y F1 a lo largo de los 8 entrenamientos."""
    x = np.arange(N_TRAIN)
    test_map50 = [te["mAP50"] for te in data["test_evals"]]
    val_map50 = [exp["val_map50"] for exp in data["experiments"]]
    test_f1 = [te["f1"] for te in data["test_evals"]]

    fig, ax1 = plt.subplots(figsize=(15, 7))

    # mAP@50 test y val
    color_test = plt.cm.magma(0.75)
    color_val = plt.cm.magma(0.45)
    color_f1 = plt.cm.magma(0.25)

    line_test = ax1.plot(x, test_map50, "o-", color=color_test, linewidth=2.5, markersize=10,
                         label="mAP@50 (Test)", zorder=5)
    line_val = ax1.plot(x, val_map50, "s--", color=color_val, linewidth=2.0, markersize=8,
                        label="mAP@50 (Val)", zorder=4, alpha=0.8)

    # Marcar T3 y T7
    ax1.scatter([IDX_SELECTED], [test_map50[IDX_SELECTED]], s=200, marker="*",
                color="#D4AF37", edgecolors="black", linewidths=1, zorder=10)
    ax1.scatter([IDX_BENCHMARK], [test_map50[IDX_BENCHMARK]], s=150, marker="D",
                color="#4A90D9", edgecolors="black", linewidths=1, zorder=10)

    ax1.set_ylabel("mAP@50", fontsize=FONT_LABEL, color=color_test)
    ax1.tick_params(axis="y", labelcolor=color_test)

    # Eje secundario para F1
    ax2 = ax1.twinx()
    line_f1 = ax2.plot(x, test_f1, "^-.", color=color_f1, linewidth=2.0, markersize=9,
                       label="F1 (Test)", zorder=3, alpha=0.85)
    ax2.set_ylabel("F1-Score", fontsize=FONT_LABEL, color=color_f1)
    ax2.tick_params(axis="y", labelcolor=color_f1)

    # Valores sobre puntos
    for i in range(N_TRAIN):
        ax1.text(i, test_map50[i] + 0.010, f"{test_map50[i]:.3f}", ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color=color_test)
        ax2.text(i, test_f1[i] - 0.015, f"{test_f1[i]:.3f}", ha="center", va="top",
                 fontsize=8, color=color_f1)

    ax1.set_xticks(x)
    xlabels = [t["label"].replace(" · ", "\n") for t in TRAININGS]
    ax1.set_xticklabels(xlabels, fontsize=FONT_TICK - 1, ha="center")

    ax1.set_title("Evolución iterativa de métricas entre entrenamientos",
                  fontsize=FONT_TITLE, fontweight="bold", pad=15)
    ax1.grid(True, alpha=GRID_ALPHA)

    # Leyenda combinada
    lines = line_test + line_val + line_f1
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=9, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.18), framealpha=0.9)

    # Anotaciones de cambios clave (flechas entre entrenamientos)
    changes = [
        (0, 1, "Stride\nNorm"),
        (1, 2, "GIoU +\népocas"),
        (2, 3, "Scoring\nrefinements"),
        (6, 7, "Focal\nLoss"),
    ]
    for i_from, i_to, txt in changes:
        mid_x = (i_from + i_to) / 2
        mid_y = (test_map50[i_from] + test_map50[i_to]) / 2
        ax1.annotate(txt, xy=(mid_x, mid_y + 0.025), fontsize=7, ha="center",
                     va="bottom", color="gray", style="italic")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    save_fig(fig, "10_evolucion_iterativa")


# ────────────────────── MAIN ──────────────────────


def main():
    print("=" * 60)
    print("FCOS — Gráficos Comparativos (8 entrenamientos)")
    print("=" * 60)

    # Crear directorio de salida
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDirectorio de salida: {SAVE_DIR}\n")

    # Cargar datos
    print("Cargando datos...")
    data = load_all_data()
    print(f"  {N_TRAIN} entrenamientos cargados.\n")

    # Generar gráficos
    print("Generando gráficos:\n")

    plot_01_global_metrics(data)
    plot_02_heatmap_ap50(data)
    plot_03_heatmap_f1(data)
    plot_04_train_cls_loss(data)
    plot_05_val_cls_loss(data)
    plot_06_precision_recall_scatter(data)
    plot_07_radar_chart(data)
    plot_08_detections_tp_fp(data)
    plot_09_efficiency(data)
    plot_10_iterative_evolution(data)

    print(f"\n{'=' * 60}")
    print(f"✅ 10 gráficos generados en: {SAVE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

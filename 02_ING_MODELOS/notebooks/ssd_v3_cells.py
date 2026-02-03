"""
SSD V3 Cells - Copy these into notebook after cell 27 (Paso 3c markdown)
Each block separated by ### CELL ### is a separate code cell
"""

### CELL 1 - Configuration & Model Setup ###
# 🔄 Recarga de módulos actualizados para SSD V3
import importlib
import utils_ssd_model as _ssd_model
import utils_ssd_data as _ssd_data
import utils_ssd_losses as _ssd_losses

importlib.reload(_ssd_model)
importlib.reload(_ssd_data)
importlib.reload(_ssd_losses)

from utils_ssd_model import build_mobilenet_ssd_anchor_head
from utils_ssd_losses import (
    focal_loss_with_ignore_mask,
    masked_bbox_smooth_l1_loss,
)
from utils_ssd_data import SSDAnchorDataGeneratorV3

# Configuración V3
SSD_MODEL_NAME_V3 = "ssd_anchor_v3"
SSD_V3_EPOCHS = 80

# Crear modelo desde cero
ssd_anchor_model_v3 = build_mobilenet_ssd_anchor_head(
    input_shape=(BASELINE_CONFIG["ssd_v6"]["input_size"],) * 2 + (3,),
    num_classes=len(CLASS_NAMES),
    anchors_per_cell=ANCHORS_PER_CELL,
    alpha=BASELINE_CONFIG["ssd_v6"]["alpha"],
    feature_channels=256,
    use_batchnorm=True,
    model_name="MobileNetV2_SSD_Anchors_V3",
)

# Focal Loss con HNM
focal_cls_loss_v3 = focal_loss_with_ignore_mask(
    alpha=0.25,
    gamma=2.0,
    neg_pos_ratio=3.0,
)

ssd_anchor_model_v3.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, clipnorm=1.0),
    loss={
        "class_out": focal_cls_loss_v3,
        "bbox_out_sigmoid": masked_bbox_smooth_l1_loss,
    },
    loss_weights={"class_out": 1.5, "bbox_out_sigmoid": 1.0},
    metrics={"class_out": "accuracy"},
)

# Generadores V3 con matching conservador
train_gen_ssd_v3 = SSDAnchorDataGeneratorV3(
    image_dir=os.path.join(config.dirs["yolo_dataset"], "images", "train"),
    label_dir=os.path.join(config.dirs["yolo_dataset"], "labels", "train"),
    anchors=ANCHORS,
    batch_size=BASELINE_CONFIG["ssd_v6"]["batch"],
    img_size=BASELINE_CONFIG["ssd_v6"]["input_size"],
    num_classes=len(CLASS_NAMES),
    iou_threshold=0.45,
    iou_ignore_threshold=0.30,
    use_center_matching=True,
    center_radius=1.0,
    augment=True,
    shuffle=True,
)

val_gen_ssd_v3 = SSDAnchorDataGeneratorV3(
    image_dir=os.path.join(config.dirs["yolo_dataset"], "images", "val"),
    label_dir=os.path.join(config.dirs["yolo_dataset"], "labels", "val"),
    anchors=ANCHORS,
    batch_size=BASELINE_CONFIG["ssd_v6"]["batch"],
    img_size=BASELINE_CONFIG["ssd_v6"]["input_size"],
    num_classes=len(CLASS_NAMES),
    iou_threshold=0.45,
    iou_ignore_threshold=0.30,
    use_center_matching=True,
    center_radius=1.0,
    augment=False,
    shuffle=False,
)

# Verificar proporción de positivos
x_test, y_test = train_gen_ssd_v3[0]
bg_count = (y_test["class_out"][:, :, 0] == 1.0).sum()
fg_count = (y_test["class_out"][:, :, 0] == 0.0).sum()
ignore_count = (y_test["ignore_mask"] == 0.0).sum()
total = bg_count + fg_count
print(f"✅ Balance V3: {fg_count} positivos ({100*fg_count/total:.1f}%) vs {bg_count} negativos ({100*bg_count/total:.1f}%)")
print(f"   Anchors ignorados: {ignore_count} ({100*ignore_count/(total):.1f}%)")
print(f"   (V2 tenía ~21.5% positivos, V3 debería tener ~10-15%)")
print(f"✅ Modelo V3 listo con matching conservador")


### CELL 2 - Training ###
# Entrenamiento SSD Anchor V3
from utils_training import create_callbacks
import pandas as pd

ssd_v3_callbacks = create_callbacks(
    model_name=SSD_MODEL_NAME_V3,
    logs_dir=config.dirs["logs"],
    checkpoints_dir=config.dirs["models_chk"],
    monitor="val_loss",
    mode="min",
    patience_early=10,
    patience_lr=5,
)

print("\n--- ENTRENANDO SSD ANCHOR V3 (BALANCED MATCHING) ---")
print(f"   Épocas: {SSD_V3_EPOCHS}")
print(f"   IoU threshold: 0.45 (vs 0.35 en V2)")
print(f"   Center radius: 1.0 (vs 1.5 en V2)")
print(f"   Ignore zone: IoU ∈ [0.30, 0.45)\n")

history_ssd_v3 = ssd_anchor_model_v3.fit(
    train_gen_ssd_v3,
    validation_data=val_gen_ssd_v3,
    epochs=SSD_V3_EPOCHS,
    callbacks=ssd_v3_callbacks,
    verbose=1,
)

# Guardar modelo final
final_path_v3 = os.path.join(config.dirs["models_final"], f"{SSD_MODEL_NAME_V3}.keras")
ssd_anchor_model_v3.save(final_path_v3)
print(f"✅ Modelo SSD V3 guardado en: {final_path_v3}")

# Guardar historial en CSV
hist_df_v3 = pd.DataFrame(history_ssd_v3.history)
hist_path_v3 = os.path.join(config.dirs["logs"], f"{SSD_MODEL_NAME_V3}_history.csv")
hist_df_v3.to_csv(hist_path_v3, index=False)
print(f"✅ Historial guardado en: {hist_path_v3}")


### CELL 3 - Plot Training Curves ###
# Visualizar curvas de entrenamiento V3
from utils_viz import plot_ssd_v2_history

plot_ssd_v2_history(hist_df_v3, title="SSD Anchor V3 - Balanced Matching Training")


### CELL 4 - MARKDOWN: ### Paso 3.1c Evaluación SSD V3 en test


### CELL 5 - Evaluation ###
# Evaluación en test set - SSD V3 (con umbrales ajustados)
import importlib
import utils_ssd_infer as _ssd_infer

importlib.reload(_ssd_infer)
from utils_ssd_infer import run_ssd_inference
from utils_eval import evaluate_model

# Predicciones con umbrales más estrictos
ssd_v3_results = []
for images, gts, ids in test_loader.iter_batches(batch_size=8):
    images_norm = images.astype(np.float32) / 255.0
    batch_results = run_ssd_inference(
        model=ssd_anchor_model_v3,
        image_batch=images_norm,
        class_names=CLASS_NAMES,
        image_ids=ids,  # type: ignore
        ground_truths=gts,  # type: ignore
        model_name=SSD_MODEL_NAME_V3,
        conf_threshold=0.45,  # MÁS ALTO que V2 (0.25)
        nms_iou=0.30,         # MÁS ESTRICTO que V2 (0.50)
        image_size=(TEST_IMG_SIZE, TEST_IMG_SIZE),
    )
    ssd_v3_results.extend(batch_results)

print(f"📊 Total predicciones SSD V3: {sum(len(r.predictions) for r in ssd_v3_results)}")

# Evaluar métricas
ssd_v3_metrics = evaluate_model(SSD_MODEL_NAME_V3, ssd_v3_results, CLASS_NAMES)

print(f"\n=== Métricas SSD Anchor V3 (test) ===")
print(f"mAP@50:    {ssd_v3_metrics.map_50:.4f}")
print(f"Precision: {ssd_v3_metrics.precision:.4f}")
print(f"Recall:    {ssd_v3_metrics.recall:.4f}")
print(f"F1-Score:  {ssd_v3_metrics.f1_score:.4f}")
print(f"TP: {ssd_v3_metrics.total_tp}, FP: {ssd_v3_metrics.total_fp}, FN: {ssd_v3_metrics.total_fn}")

# Comparación V1 vs V2 vs V3
print(f"\n📈 Comparación V1 → V2 → V3:")
print(f"   mAP:       V1={ssd_metrics.map_50:.3f} → V2={ssd_v2_metrics.map_50:.3f} → V3={ssd_v3_metrics.map_50:.3f}")
print(f"   Precision: V1={ssd_metrics.precision:.3f} → V2={ssd_v2_metrics.precision:.3f} → V3={ssd_v3_metrics.precision:.3f}")
print(f"   Recall:    V1={ssd_metrics.recall:.3f} → V2={ssd_v2_metrics.recall:.3f} → V3={ssd_v3_metrics.recall:.3f}")
print(f"   FP:        V1={ssd_metrics.total_fp} → V2={ssd_v2_metrics.total_fp} → V3={ssd_v3_metrics.total_fp}")


### CELL 6 - MARKDOWN: ### Paso 3.2c Visualización de detecciones SSD V3


### CELL 7 - Visualization ###
# Visualizar detecciones SSD V3
from utils_viz import show_random_samples

sample_images_v3 = []
sample_gts_v3 = []
sample_preds_v3 = []
total_preds_v3 = 0

for images, gts, ids in test_loader.iter_batches(batch_size=4):
    images_norm = images.astype(np.float32) / 255.0
    batch_results = run_ssd_inference(
        model=ssd_anchor_model_v3, 
        image_batch=images_norm,
        class_names=CLASS_NAMES,
        image_ids=ids,  # type: ignore
        ground_truths=gts,  # type: ignore
        model_name=SSD_MODEL_NAME_V3,
        conf_threshold=0.45,
        nms_iou=0.30,
        image_size=(TEST_IMG_SIZE, TEST_IMG_SIZE),
        top_k=10,
    )
    for i in range(len(images)):  # type: ignore
        sample_images_v3.append(images[i])  # type: ignore
        sample_gts_v3.append(gts[i])  # type: ignore
        sample_preds_v3.append(batch_results[i].predictions)
        total_preds_v3 += len(batch_results[i].predictions)
    if len(sample_images_v3) >= 4:
        break

print(f"Predicciones totales V3 en muestras: {total_preds_v3}")
show_random_samples(sample_images_v3, sample_gts_v3, sample_preds_v3, max_samples=4, title="SSD V3 Test Samples")

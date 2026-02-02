import json
import os
import random
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO


def visualize_coco_samples(
    json_path: str,
    images_dir: str,
    num_images: int = 3,
    seed: int = 42,
    figsize: int = 8,
    class_filter: Optional[List[str]] = None,
) -> None:
    """Visualiza imágenes COCO con bboxes y etiquetas.

    Args:
        json_path: Ruta al JSON COCO.
        images_dir: Carpeta con imágenes.
        num_images: Número de imágenes a mostrar.
        seed: Semilla para muestreo reproducible.
        figsize: Tamaño de figura para cada imagen.
        class_filter: Si se pasa, solo muestra imágenes que contengan
            alguna de estas clases (por nombre).
    """

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No existe el JSON: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    name_to_cat_id = {cat["name"]: cat["id"] for cat in data["categories"]}

    coco = COCO(json_path)
    img_ids = coco.getImgIds()

    if class_filter:
        allowed_ids = {name_to_cat_id[n] for n in class_filter if n in name_to_cat_id}
        filtered_img_ids = []
        for img_id in img_ids:
            ann_ids = coco.getAnnIds(imgIds=int(img_id))
            anns = coco.loadAnns(ann_ids)
            if any(a["category_id"] in allowed_ids for a in anns):
                filtered_img_ids.append(img_id)
        img_ids = filtered_img_ids

    if not img_ids:
        print("⚠️ No hay imágenes que cumplan el filtro.")
        return

    random.seed(seed)
    sampled_ids = random.sample(img_ids, min(num_images, len(img_ids)))

    for img_id in sampled_ids:
        img_info = coco.loadImgs(int(img_id))[0]
        img_path = os.path.join(images_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ No se pudo leer: {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = coco.getAnnIds(imgIds=int(img_id))
        anns = coco.loadAnns(ann_ids)

        plt.figure(figsize=(figsize, figsize))
        plt.imshow(image)

        for ann in anns:
            x, y, w, h = ann["bbox"]
            rect = plt.Rectangle((x, y), w, h, fill=False, color="red", linewidth=2)
            plt.gca().add_patch(rect)
            cat_name = cat_id_to_name.get(ann["category_id"], f"ID:{ann['category_id']}")
            plt.text(
                x,
                max(y - 10, 0),
                cat_name,
                color="yellow",
                fontsize=12,
                weight="bold",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1),
            )

        plt.axis("off")
        plt.title(f"Image ID: {img_id} - {img_info['file_name']}")
        plt.show()

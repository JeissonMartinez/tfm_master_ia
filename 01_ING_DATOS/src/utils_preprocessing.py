"""
Utilidades de preprocesamiento para datasets de detección en formato COCO.

Este módulo centraliza funciones modulares para normalización geométrica
de imágenes y anotaciones, incluyendo redimensionamiento tipo Letterbox
con preservación de aspect ratio.
"""

from __future__ import annotations

import copy
import json
import os
import random
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

try:
	import tensorflow as tf
except Exception:
	tf = None


@dataclass
class LetterboxTransform:
	"""
	Parámetros de transformación Letterbox para una imagen.

	Attributes:
		scale: Factor de escala uniforme aplicado a ancho y alto.
		pad_left: Padding horizontal izquierdo en píxeles.
		pad_right: Padding horizontal derecho en píxeles.
		pad_top: Padding vertical superior en píxeles.
		pad_bottom: Padding vertical inferior en píxeles.
		original_width: Ancho original de la imagen.
		original_height: Alto original de la imagen.
		resized_width: Ancho resultante tras escalar (sin padding).
		resized_height: Alto resultante tras escalar (sin padding).
		target_width: Ancho objetivo final.
		target_height: Alto objetivo final.
	"""

	scale: float
	pad_left: int
	pad_right: int
	pad_top: int
	pad_bottom: int
	original_width: int
	original_height: int
	resized_width: int
	resized_height: int
	target_width: int
	target_height: int


@dataclass
class CocoSplitConfig:
	"""
	Configuración para split estratificado de dataset COCO.

	Attributes:
		train_ratio: Proporción de imágenes para entrenamiento.
		valid_ratio: Proporción de imágenes para validación.
		test_ratio: Proporción de imágenes para prueba.
		seed: Semilla para reproducibilidad.
		copy_images: Si True, copia imágenes a cada split.
	"""

	train_ratio: float = 0.70
	valid_ratio: float = 0.15
	test_ratio: float = 0.15
	seed: int = 42
	copy_images: bool = True


def ensure_dir(directory: str) -> None:
	"""
	Crea un directorio si no existe.

	Args:
		directory: Ruta del directorio a crear.
	"""
	if not os.path.exists(directory):
		os.makedirs(directory)
		print(f"📁 Directorio creado: {directory}")


def compute_letterbox_transform(
	original_width: int,
	original_height: int,
	target_size: Tuple[int, int] = (640, 640),
) -> LetterboxTransform:
	"""
	Calcula la transformación Letterbox para una dimensión origen.

	Args:
		original_width: Ancho original de la imagen.
		original_height: Alto original de la imagen.
		target_size: Tamaño objetivo final (ancho, alto).

	Returns:
		Objeto LetterboxTransform con escala y paddings.

	Raises:
		ValueError: Si las dimensiones de entrada son inválidas.
	"""
	if original_width <= 0 or original_height <= 0:
		raise ValueError(
			f"Dimensiones originales inválidas: {original_width}x{original_height}"
		)

	target_width, target_height = target_size
	if target_width <= 0 or target_height <= 0:
		raise ValueError(f"target_size inválido: {target_size}")

	scale = min(target_width / original_width, target_height / original_height)
	resized_width = int(round(original_width * scale))
	resized_height = int(round(original_height * scale))

	pad_w = max(0, target_width - resized_width)
	pad_h = max(0, target_height - resized_height)

	pad_left = pad_w // 2
	pad_right = pad_w - pad_left
	pad_top = pad_h // 2
	pad_bottom = pad_h - pad_top

	return LetterboxTransform(
		scale=scale,
		pad_left=pad_left,
		pad_right=pad_right,
		pad_top=pad_top,
		pad_bottom=pad_bottom,
		original_width=original_width,
		original_height=original_height,
		resized_width=resized_width,
		resized_height=resized_height,
		target_width=target_width,
		target_height=target_height,
	)


def apply_letterbox_to_image(
	image,
	transform: LetterboxTransform,
	padding_color: Tuple[int, int, int] = (114, 114, 114),
	interpolation: int = cv2.INTER_LINEAR,
):
	"""
	Aplica letterbox a una imagen usando una transformación precomputada.

	Args:
		image: Imagen en formato OpenCV (H, W, C) en BGR.
		transform: Parámetros de transformación Letterbox.
		padding_color: Color BGR del padding.
		interpolation: Método de interpolación OpenCV para resize.

	Returns:
		Imagen final con tamaño target_width x target_height.
	"""
	if image is None:
		raise ValueError("La imagen recibida es None")

	resized = cv2.resize(
		image,
		(transform.resized_width, transform.resized_height),
		interpolation=interpolation,
	)

	letterboxed = cv2.copyMakeBorder(
		resized,
		transform.pad_top,
		transform.pad_bottom,
		transform.pad_left,
		transform.pad_right,
		borderType=cv2.BORDER_CONSTANT,
		value=padding_color,
	)

	return letterboxed


def transform_coco_bbox_letterbox(
	bbox: Sequence[float],
	transform: LetterboxTransform,
	clip: bool = True,
) -> Optional[List[float]]:
	"""
	Transforma una bbox COCO [x, y, w, h] con parámetros de letterbox.

	Args:
		bbox: Bounding box en formato COCO absoluto [x, y, w, h].
		transform: Transformación letterbox aplicada a la imagen.
		clip: Si True, recorta la bbox para quedar dentro del marco objetivo.

	Returns:
		Nueva bbox [x, y, w, h] en float, o None si queda inválida.
	"""
	if len(bbox) != 4:
		return None

	x, y, w, h = [float(v) for v in bbox]
	if w <= 0 or h <= 0:
		return None

	x1 = x * transform.scale + transform.pad_left
	y1 = y * transform.scale + transform.pad_top
	x2 = (x + w) * transform.scale + transform.pad_left
	y2 = (y + h) * transform.scale + transform.pad_top

	if clip:
		x1 = max(0.0, min(float(transform.target_width), x1))
		y1 = max(0.0, min(float(transform.target_height), y1))
		x2 = max(0.0, min(float(transform.target_width), x2))
		y2 = max(0.0, min(float(transform.target_height), y2))

	new_w = x2 - x1
	new_h = y2 - y1
	if new_w <= 0.0 or new_h <= 0.0:
		return None

	return [float(x1), float(y1), float(new_w), float(new_h)]


def transform_coco_annotations_for_image(
	annotations: List[Dict],
	transform: LetterboxTransform,
) -> Tuple[List[Dict], int]:
	"""
	Transforma todas las anotaciones de una imagen y recalcula el área.

	Args:
		annotations: Lista de anotaciones COCO de una imagen.
		transform: Transformación letterbox aplicada.

	Returns:
		Tupla (anotaciones_transformadas, total_descartadas).
	"""
	transformed_annotations: List[Dict] = []
	removed = 0

	for ann in annotations:
		ann_new = copy.deepcopy(ann)
		new_bbox = transform_coco_bbox_letterbox(ann_new.get("bbox", []), transform)

		if new_bbox is None:
			removed += 1
			continue

		ann_new["bbox"] = new_bbox
		ann_new["area"] = float(new_bbox[2] * new_bbox[3])
		transformed_annotations.append(ann_new)

	return transformed_annotations, removed


def _index_annotations_by_image(annotations: List[Dict]) -> Dict[int, List[Dict]]:
	"""Crea índice image_id -> lista de anotaciones."""
	ann_index: Dict[int, List[Dict]] = {}
	for ann in annotations:
		image_id = ann.get("image_id")
		if image_id is None:
			continue
		ann_index.setdefault(int(image_id), []).append(ann)
	return ann_index


def letterbox_coco_dataset_offline(
	coco_json_path: str,
	images_dir: str,
	output_dir: str,
	output_json_name: str = "_annotations.letterbox.coco.json",
	target_size: Tuple[int, int] = (640, 640),
	padding_color: Tuple[int, int, int] = (114, 114, 114),
	interpolation: int = cv2.INTER_LINEAR,
) -> Dict[str, object]:
	"""
	Procesa un dataset COCO completo aplicando letterbox offline.

	Flujo:
	  1) Lee JSON COCO.
	  2) Recorre imágenes y aplica letterbox 640x640.
	  3) Actualiza `images.width` y `images.height`.
	  4) Transforma bbox COCO y recalcula `area`.
	  5) Guarda imágenes resultantes y JSON COCO actualizado.

	Args:
		coco_json_path: Ruta al archivo de anotaciones COCO de entrada.
		images_dir: Directorio de imágenes de entrada.
		output_dir: Directorio raíz de salida.
		output_json_name: Nombre del JSON COCO transformado.
		target_size: Resolución final deseada (ancho, alto).
		padding_color: Color BGR para el padding letterbox.
		interpolation: Interpolación OpenCV en el resize.

	Returns:
		Diccionario con rutas y métricas de procesamiento.
	"""
	ensure_dir(output_dir)
	output_images_dir = os.path.join(output_dir, "data", "images")
	ensure_dir(output_images_dir)

	with open(coco_json_path, "r", encoding="utf-8") as f:
		coco_data = json.load(f)

	images = coco_data.get("images", [])
	annotations = coco_data.get("annotations", [])
	ann_index = _index_annotations_by_image(annotations)

	processed_images = 0
	missing_images = 0
	unreadable_images = 0
	removed_annotations = 0
	transformed_annotations: List[Dict] = []

	print("\n🧩 Iniciando normalización Letterbox COCO...")
	print(f"   📄 JSON entrada: {coco_json_path}")
	print(f"   🖼️  Imágenes entrada: {images_dir}")
	print(f"   📦 Salida: {output_dir}")

	for idx, img_info in enumerate(images, start=1):
		file_name = img_info.get("file_name")
		if not file_name:
			continue

		src_img_path = os.path.join(images_dir, file_name)
		dst_img_path = os.path.join(output_images_dir, file_name)

		if not os.path.exists(src_img_path):
			missing_images += 1
			continue

		image = cv2.imread(src_img_path)
		if image is None:
			unreadable_images += 1
			continue

		original_h, original_w = image.shape[:2]
		transform = compute_letterbox_transform(
			original_width=original_w,
			original_height=original_h,
			target_size=target_size,
		)

		image_letterbox = apply_letterbox_to_image(
			image=image,
			transform=transform,
			padding_color=padding_color,
			interpolation=interpolation,
		)

		dst_parent = os.path.dirname(dst_img_path)
		if dst_parent:
			ensure_dir(dst_parent)
		cv2.imwrite(dst_img_path, image_letterbox)

		# Actualizar metadatos de imagen a 640x640
		img_info["width"] = int(transform.target_width)
		img_info["height"] = int(transform.target_height)

		# Transformar anotaciones de esta imagen
		img_id = int(img_info["id"])
		anns_for_img = ann_index.get(img_id, [])
		anns_transformed, removed_count = transform_coco_annotations_for_image(
			anns_for_img,
			transform,
		)
		transformed_annotations.extend(anns_transformed)
		removed_annotations += removed_count

		processed_images += 1
		if idx % 200 == 0:
			print(f"   🔄 Progreso: {idx}/{len(images)} imágenes revisadas")

	coco_data["annotations"] = transformed_annotations

	output_json_path = os.path.join(output_dir, output_json_name)
	with open(output_json_path, "w", encoding="utf-8") as f:
		json.dump(coco_data, f, ensure_ascii=False)

	summary = {
		"output_json_path": output_json_path,
		"output_images_dir": output_images_dir,
		"processed_images": processed_images,
		"missing_images": missing_images,
		"unreadable_images": unreadable_images,
		"input_annotations": len(annotations),
		"output_annotations": len(transformed_annotations),
		"removed_annotations": removed_annotations,
		"target_size": target_size,
		"padding_color": padding_color,
	}

	print("\n✅ Letterbox offline completado")
	print(f"   🖼️  Imágenes procesadas: {processed_images}")
	print(f"   ⚠️  Imágenes faltantes: {missing_images}")
	print(f"   ⚠️  Imágenes no legibles: {unreadable_images}")
	print(f"   📝 Anotaciones entrada: {len(annotations)}")
	print(f"   📝 Anotaciones salida: {len(transformed_annotations)}")
	print(f"   ❌ Anotaciones descartadas: {removed_annotations}")
	print(f"   💾 JSON salida: {output_json_path}")

	return summary


def summarize_image_dimensions_from_coco(coco_json_path: str) -> Dict[str, int]:
	"""
	Resume resolución única de imágenes en un JSON COCO.

	Args:
		coco_json_path: Ruta al archivo JSON COCO.

	Returns:
		Diccionario con número de imágenes y número de resoluciones únicas.
	"""
	with open(coco_json_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	images = data.get("images", [])
	resolutions = {(img.get("width"), img.get("height")) for img in images}

	return {
		"num_images": len(images),
		"num_unique_resolutions": len(resolutions),
	}


def _validate_split_config(config: CocoSplitConfig) -> None:
	"""Valida consistencia de ratios de split."""
	total = config.train_ratio + config.valid_ratio + config.test_ratio
	if abs(total - 1.0) > 1e-6:
		raise ValueError(
			f"Los ratios deben sumar 1.0 y actualmente suman {total:.6f}"
		)

	for name, ratio in [
		("train_ratio", config.train_ratio),
		("valid_ratio", config.valid_ratio),
		("test_ratio", config.test_ratio),
	]:
		if ratio <= 0:
			raise ValueError(f"{name} debe ser > 0. Valor recibido: {ratio}")


def _compute_split_image_counts(total_images: int, config: CocoSplitConfig) -> Dict[str, int]:
	"""
	Calcula número de imágenes por split con redondeo determinista.
	"""
	if total_images <= 0:
		return {"train": 0, "valid": 0, "test": 0}

	raw = {
		"train": total_images * config.train_ratio,
		"valid": total_images * config.valid_ratio,
		"test": total_images * config.test_ratio,
	}

	floor_counts = {k: int(v) for k, v in raw.items()}
	remaining = total_images - sum(floor_counts.values())

	# Repartir residuo por mayor parte fraccionaria
	fractions = sorted(
		raw.keys(),
		key=lambda k: (raw[k] - floor_counts[k]),
		reverse=True,
	)

	for i in range(remaining):
		floor_counts[fractions[i % len(fractions)]] += 1

	return floor_counts


def _get_category_id_to_index(coco_data: Dict) -> Dict[int, int]:
	"""Construye mapeo category_id -> índice vectorial."""
	cat_ids = [int(cat["id"]) for cat in coco_data.get("categories", [])]
	cat_ids_sorted = sorted(cat_ids)
	return {cat_id: idx for idx, cat_id in enumerate(cat_ids_sorted)}


def _build_annotations_index(coco_data: Dict) -> Dict[int, List[Dict]]:
	"""Índice de anotaciones por image_id."""
	return _index_annotations_by_image(coco_data.get("annotations", []))


def _compute_image_instance_vectors(
	images: List[Dict],
	anns_index: Dict[int, List[Dict]],
	cat_id_to_idx: Dict[int, int],
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
	"""
	Calcula vector de instancias por clase para cada imagen y bandera background.

	Returns:
		Tupla (image_vectors, image_background_flag)
	"""
	n_classes = len(cat_id_to_idx)
	image_vectors: Dict[int, List[int]] = {}
	image_background_flag: Dict[int, int] = {}

	for img in images:
		img_id = int(img["id"])
		vec = [0] * n_classes
		anns = anns_index.get(img_id, [])

		for ann in anns:
			cat_id = int(ann.get("category_id", -1))
			if cat_id not in cat_id_to_idx:
				continue
			vec[cat_id_to_idx[cat_id]] += 1

		image_vectors[img_id] = vec
		image_background_flag[img_id] = 1 if sum(vec) == 0 else 0

	return image_vectors, image_background_flag


def _sum_class_vectors(vectors: List[List[int]]) -> List[int]:
	"""Suma componente a componente una lista de vectores."""
	if not vectors:
		return []
	n = len(vectors[0])
	acc = [0] * n
	for vec in vectors:
		for i in range(n):
			acc[i] += vec[i]
	return acc


def _split_assignment_cost(
	projected_vec: List[int],
	projected_bg: int,
	projected_imgs: int,
	target_vec: List[float],
	target_bg: float,
	target_imgs: int,
) -> float:
	"""Calcula costo de asignación de una imagen a un split."""
	eps = 1e-6
	class_cost = 0.0
	for value, target in zip(projected_vec, target_vec):
		class_cost += abs(value - target) / max(target, 1.0)

	bg_cost = abs(projected_bg - target_bg) / max(target_bg, 1.0)
	img_cost = abs(projected_imgs - target_imgs) / max(target_imgs, 1.0)

	# Más peso a distribución de instancias de clase, luego background y cupo
	return class_cost + (0.5 * bg_cost) + (0.2 * img_cost)


def _assign_images_stratified_by_instances(
	images: List[Dict],
	image_vectors: Dict[int, List[int]],
	image_background_flag: Dict[int, int],
	image_counts_target: Dict[str, int],
	config: CocoSplitConfig,
) -> Dict[str, List[int]]:
	"""
	Asigna imágenes a splits minimizando desviación de instancias por clase.

	Incluye explícitamente el balance de imágenes background (sin anotaciones).
	"""
	split_names = ["train", "valid", "test"]
	rng = random.Random(config.seed)

	image_ids = [int(img["id"]) for img in images]
	bg_ids = [i for i in image_ids if image_background_flag[i] == 1]
	fg_ids = [i for i in image_ids if image_background_flag[i] == 0]
	n_classes = len(next(iter(image_vectors.values()))) if image_vectors else 0

	global_vec = _sum_class_vectors([image_vectors[i] for i in fg_ids])
	global_bg = len(bg_ids)
	global_n = max(len(image_ids), 1)

	target_vec: Dict[str, List[float]] = {}
	target_bg: Dict[str, float] = {}
	for split in split_names:
		ratio = image_counts_target[split] / global_n
		target_vec[split] = [v * ratio for v in global_vec]
		target_bg[split] = global_bg * ratio

	assignments: Dict[str, List[int]] = {s: [] for s in split_names}
	current_vec: Dict[str, List[int]] = {s: [0] * n_classes for s in split_names}
	current_bg: Dict[str, int] = {s: 0 for s in split_names}

	# 1) Asignar backgrounds primero para forzar su distribución 70/15/15
	bg_cfg = CocoSplitConfig(
		train_ratio=config.train_ratio,
		valid_ratio=config.valid_ratio,
		test_ratio=config.test_ratio,
		seed=config.seed,
		copy_images=config.copy_images,
	)
	bg_targets = _compute_split_image_counts(len(bg_ids), bg_cfg)
	rng.shuffle(bg_ids)
	bg_cursor = 0
	for split in split_names:
		count = bg_targets[split]
		selected = bg_ids[bg_cursor:bg_cursor + count]
		bg_cursor += count
		assignments[split].extend(selected)
		current_bg[split] += len(selected)

	# Ordenar por dificultad (más instancias primero, luego no-background)
	sorted_ids = sorted(
		fg_ids,
		key=lambda i: sum(image_vectors[i]),
		reverse=True,
	)

	for img_id in sorted_ids:
		vec = image_vectors[img_id]
		bg = image_background_flag[img_id]

		eligible = [
			s for s in split_names
			if len(assignments[s]) < image_counts_target[s]
		]
		if not eligible:
			break

		costs: List[Tuple[str, float]] = []
		for split in eligible:
			proj_vec = [
				current_vec[split][j] + vec[j]
				for j in range(n_classes)
			]
			proj_bg = current_bg[split] + bg
			proj_imgs = len(assignments[split]) + 1

			cost = _split_assignment_cost(
				projected_vec=proj_vec,
				projected_bg=proj_bg,
				projected_imgs=proj_imgs,
				target_vec=target_vec[split],
				target_bg=target_bg[split],
				target_imgs=image_counts_target[split],
			)
			costs.append((split, cost))

		min_cost = min(cost for _, cost in costs)
		best_splits = [s for s, c in costs if abs(c - min_cost) <= 1e-12]
		chosen = rng.choice(best_splits)

		assignments[chosen].append(img_id)
		for j in range(n_classes):
			current_vec[chosen][j] += vec[j]
		current_bg[chosen] += bg

	# Fallback defensivo: completar si quedó algún hueco por edge case
	assigned = set(assignments["train"] + assignments["valid"] + assignments["test"])
	leftover = [i for i in image_ids if i not in assigned]
	for img_id in leftover:
		for split in split_names:
			if len(assignments[split]) < image_counts_target[split]:
				assignments[split].append(img_id)
				break

	return assignments


def _build_coco_subset(coco_data: Dict, selected_image_ids: set[int]) -> Dict:
	"""Construye un subconjunto COCO preservando estructura e IDs originales."""
	images = [
		img for img in coco_data.get("images", [])
		if int(img.get("id", -1)) in selected_image_ids
	]
	annotations = [
		ann for ann in coco_data.get("annotations", [])
		if int(ann.get("image_id", -1)) in selected_image_ids
	]

	return {
		"info": coco_data.get("info", {}),
		"licenses": coco_data.get("licenses", []),
		"categories": coco_data.get("categories", []),
		"images": images,
		"annotations": annotations,
	}


def _copy_images_for_split(
	images: List[Dict],
	source_images_dir: str,
	target_images_dir: str,
) -> Dict[str, int]:
	"""Copia imágenes de un split al directorio destino."""
	ensure_dir(target_images_dir)

	copied = 0
	missing = 0
	errors = 0

	for img in images:
		file_name = img.get("file_name")
		if not file_name:
			errors += 1
			continue

		src = os.path.join(source_images_dir, file_name)
		dst = os.path.join(target_images_dir, file_name)

		if not os.path.exists(src):
			missing += 1
			continue

		dst_parent = os.path.dirname(dst)
		if dst_parent:
			ensure_dir(dst_parent)

		try:
			shutil.copy2(src, dst)
			copied += 1
		except Exception:
			errors += 1

	return {
		"copied": copied,
		"missing": missing,
		"errors": errors,
	}


def _count_background_images(
	images: List[Dict],
	anns_index: Dict[int, List[Dict]],
) -> int:
	"""Cuenta imágenes sin anotaciones en un subconjunto."""
	count = 0
	for img in images:
		img_id = int(img["id"])
		if len(anns_index.get(img_id, [])) == 0:
			count += 1
	return count


def split_coco_dataset_stratified(
	coco_json_path: str,
	images_dir: str,
	output_dir: str,
	config: Optional[CocoSplitConfig] = None,
) -> Dict[str, object]:
	"""
	Genera split Train/Val/Test estratificado por instancias de clase.

	Características:
	  - Ratios por defecto 70/15/15.
	  - Incluye balance explícito de imágenes background (sin bbox).
	  - Mantiene estructura COCO en cada split.
	  - Copia imágenes físicas por split (opcional).

	Args:
		coco_json_path: Ruta al JSON COCO de entrada.
		images_dir: Directorio de imágenes fuente.
		output_dir: Directorio raíz donde crear `train/valid/test`.
		config: Configuración opcional de split.

	Returns:
		Diccionario resumen con rutas, métricas y distribución resultante.
	"""
	cfg = config or CocoSplitConfig()
	_validate_split_config(cfg)

	with open(coco_json_path, "r", encoding="utf-8") as f:
		coco_data = json.load(f)

	images = coco_data.get("images", [])
	if not images:
		raise ValueError("El JSON COCO no contiene imágenes")

	anns_index = _build_annotations_index(coco_data)
	cat_id_to_idx = _get_category_id_to_index(coco_data)
	image_vectors, image_bg = _compute_image_instance_vectors(images, anns_index, cat_id_to_idx)

	total_images = len(images)
	image_counts_target = _compute_split_image_counts(total_images, cfg)

	print("\n🧪 Iniciando split estratificado COCO...")
	print(f"   📄 JSON entrada: {coco_json_path}")
	print(f"   🖼️  Imágenes fuente: {images_dir}")
	print(f"   📦 Salida: {output_dir}")
	print(f"   🎯 Objetivo imágenes: {image_counts_target}")

	assignments = _assign_images_stratified_by_instances(
		images=images,
		image_vectors=image_vectors,
		image_background_flag=image_bg,
		image_counts_target=image_counts_target,
		config=cfg,
	)

	ensure_dir(output_dir)
	summary_splits: Dict[str, Dict[str, object]] = {}

	for split in ["train", "valid", "test"]:
		split_ids = set(assignments[split])
		split_data = _build_coco_subset(coco_data, split_ids)

		split_dir = os.path.join(output_dir, split)
		split_images_dir = os.path.join(split_dir, "images")
		ensure_dir(split_dir)

		split_json_path = os.path.join(split_dir, "_annotations.coco.json")
		with open(split_json_path, "w", encoding="utf-8") as f:
			json.dump(split_data, f, ensure_ascii=False)

		copy_stats = {"copied": 0, "missing": 0, "errors": 0}
		if cfg.copy_images:
			copy_stats = _copy_images_for_split(
				images=split_data["images"],
				source_images_dir=images_dir,
				target_images_dir=split_images_dir,
			)

		split_anns_index = _index_annotations_by_image(split_data.get("annotations", []))
		background_count = _count_background_images(split_data.get("images", []), split_anns_index)

		summary_splits[split] = {
			"split_dir": split_dir,
			"json_path": split_json_path,
			"images_dir": split_images_dir,
			"num_images": len(split_data.get("images", [])),
			"num_annotations": len(split_data.get("annotations", [])),
			"num_background_images": background_count,
			"copy_stats": copy_stats,
		}

	assigned_total = sum(len(v) for v in assignments.values())
	unique_assigned = len(set(assignments["train"] + assignments["valid"] + assignments["test"]))

	summary = {
		"output_dir": output_dir,
		"source_json": coco_json_path,
		"source_images_dir": images_dir,
		"seed": cfg.seed,
		"ratios": {
			"train": cfg.train_ratio,
			"valid": cfg.valid_ratio,
			"test": cfg.test_ratio,
		},
		"target_image_counts": image_counts_target,
		"source_num_images": len(coco_data.get("images", [])),
		"source_num_annotations": len(coco_data.get("annotations", [])),
		"assigned_total_images": assigned_total,
		"assigned_unique_images": unique_assigned,
		"splits": summary_splits,
	}

	print("\n✅ Split COCO completado")
	for split in ["train", "valid", "test"]:
		info = summary_splits[split]
		print(
			f"   {split.upper()}: {info['num_images']} imgs | "
			f"{info['num_annotations']} anns | "
			f"{info['num_background_images']} background"
		)

	return summary


def analyze_split_class_distribution(
	split_json_paths: Dict[str, str],
	class_names: Optional[List[str]] = None,
	palette: str = "magma",
	figsize: Tuple[int, int] = (10, 5),
	title: str = "Distribución de Clases por Split",
	output_dir: Optional[str] = None,
	filename: str = "split_class_distribution",
	show_plot: bool = True,
) -> pd.DataFrame:
	"""
	Analiza la distribución de clases por split y genera tabla + figuras.

	Incluye explícitamente la clase `background`, definida como imágenes sin
	anotaciones en cada split.

	Args:
		split_json_paths: Diccionario con rutas JSON COCO por split.
		                 Ejemplo: {'train': '...json', 'valid': '...json', 'test': '...json'}
		class_names: Lista de clases a reportar. Si es None usa:
		             ['dog', 'door', 'person', 'obstacle', 'stair', 'background']
		palette: Paleta para las figuras (ej. 'magma').
		figsize: Tamaño de figura.
		title: Título de la figura.
		output_dir: Directorio para guardar CSV y PNG (None = no guardar).
		filename: Nombre base para archivos de salida.
		show_plot: Si True, muestra figuras en notebook.

	Returns:
		DataFrame con conteos por split y clase, incluyendo columnas resumen:
		['split', <clases...>, 'total_annotations', 'total_images']
	"""
	if not split_json_paths:
		raise ValueError("split_json_paths no puede estar vacío")

	if class_names is None:
		class_names = ["dog", "door", "person", "obstacle", "stair", "background"]

	# Normalizar nombres de clase para matching robusto
	class_names = [c.strip().lower() for c in class_names]
	include_background = "background" in class_names
	class_names_no_bg = [c for c in class_names if c != "background"]

	rows: List[Dict[str, object]] = []

	for split_name, json_path in split_json_paths.items():
		if not os.path.exists(json_path):
			raise FileNotFoundError(f"No se encontró JSON de split '{split_name}': {json_path}")

		coco = COCO(json_path)
		img_ids = coco.getImgIds()
		cat_ids = coco.getCatIds()
		cats = coco.loadCats(cat_ids)

		# name_lower -> list[cat_id] (por si hay nombres repetidos)
		cat_name_to_ids: Dict[str, List[int]] = {}
		for cat in cats:
			name_lower = str(cat.get("name", "")).strip().lower()
			cat_name_to_ids.setdefault(name_lower, []).append(int(cat["id"]))

		row: Dict[str, object] = {"split": split_name}

		# Conteo por clase (anotaciones)
		for class_name in class_names_no_bg:
			ids_for_name = cat_name_to_ids.get(class_name, [])
			if not ids_for_name:
				row[class_name] = 0
				continue

			ann_count = len(coco.getAnnIds(catIds=ids_for_name))
			row[class_name] = int(ann_count)

		# Conteo de background como imágenes sin anotaciones
		if include_background:
			bg_count = 0
			for img_id in img_ids:
				if len(coco.getAnnIds(imgIds=[img_id])) == 0:
					bg_count += 1
			row["background"] = int(bg_count)

		row["total_annotations"] = int(len(coco.getAnnIds()))
		row["total_images"] = int(len(img_ids))
		rows.append(row)

	df = pd.DataFrame(rows)
	if df.empty:
		return df

	# Reordenar columnas
	ordered_cols = ["split"] + class_names + ["total_annotations", "total_images"]
	for col in ordered_cols:
		if col not in df.columns:
			df[col] = 0
	df = df[ordered_cols]

	print("\n📊 Distribución de clases por split:")
	print(df.to_markdown(index=False))

	# Guardar tabla
	if output_dir:
		ensure_dir(output_dir)
		csv_path = os.path.join(output_dir, f"{filename}.csv")
		df.to_csv(csv_path, index=False)
		print(f"💾 Tabla guardada: {csv_path}")

	# Para gráficos evitamos mezclar unidades: excluimos background
	# (background = imágenes sin anotaciones, clases = anotaciones).
	plot_classes = class_names_no_bg if class_names_no_bg else class_names
	plot_df_counts = df.set_index("split")[plot_classes].astype(float)

	# Gráfico principal: barras apiladas al 100%
	row_sums = plot_df_counts.sum(axis=1)
	plot_df_pct = plot_df_counts.div(row_sums.replace(0.0, 1.0), axis=0) * 100.0
	plot_df_pct.loc[row_sums == 0.0, :] = 0.0

	colors = sns.color_palette(palette, n_colors=max(len(plot_classes), 1))
	fig_pct, ax_pct = plt.subplots(figsize=figsize)
	bottom = [0.0] * len(plot_df_pct.index)
	for idx, class_name in enumerate(plot_classes):
		prev_bottom = bottom.copy()
		values = plot_df_pct[class_name].tolist()
		bars = ax_pct.bar(
			plot_df_pct.index,
			values,
			bottom=bottom,
			label=class_name,
			color=colors[idx],
		)
		for bar, value, btm in zip(bars, values, prev_bottom):
			if value <= 0.0:
				continue
			x_txt = bar.get_x() + (bar.get_width() / 2)
			y_txt = btm + (value / 2.0)
			ax_pct.text(
				x_txt,
				y_txt,
				f"{value:.1f}%",
				ha="center",
				va="center",
				fontsize=9,
				fontweight="bold",
				color="white",
			)
		bottom = [b + v for b, v in zip(bottom, values)]

	ax_pct.set_title(f"{title} (100% apilado)", fontsize=13, fontweight="bold")
	ax_pct.set_xlabel("Split")
	ax_pct.set_ylabel("Porcentaje (%)")
	ax_pct.set_ylim(0, 100)
	ax_pct.legend(title="Clase", bbox_to_anchor=(1.02, 1), loc="upper left")
	plt.tight_layout()

	if output_dir:
		fig_pct_path = os.path.join(output_dir, f"{filename}_pct.png")
		fig_pct.savefig(fig_pct_path, dpi=300, bbox_inches="tight")
		print(f"💾 Figura porcentual guardada: {fig_pct_path}")

	# Gráfico secundario: barras agrupadas con conteos absolutos
	fig_abs, ax_abs = plt.subplots(figsize=figsize)
	x = list(range(len(plot_df_counts.index)))
	n_classes = max(len(plot_classes), 1)
	group_width = 0.8
	bar_width = group_width / n_classes
	left_offset = -group_width / 2 + bar_width / 2

	for idx, class_name in enumerate(plot_classes):
		x_positions = [xi + left_offset + (idx * bar_width) for xi in x]
		bars = ax_abs.bar(
			x_positions,
			plot_df_counts[class_name].tolist(),
			width=bar_width,
			label=class_name,
			color=colors[idx],
		)
		for bar in bars:
			height = float(bar.get_height())
			if height <= 0.0:
				continue
			x_txt = bar.get_x() + (bar.get_width() / 2)
			y_txt = height
			ax_abs.annotate(
				f"{int(round(height))}",
				xy=(x_txt, y_txt),
				xytext=(0, 3),
				textcoords="offset points",
				ha="center",
				va="bottom",
				fontsize=9,
				fontweight="bold",
			)

	ax_abs.set_title(f"{title} (conteos absolutos)", fontsize=13, fontweight="bold")
	ax_abs.set_xlabel("Split")
	ax_abs.set_ylabel("Conteo")
	ax_abs.set_xticks(x)
	ax_abs.set_xticklabels(plot_df_counts.index.tolist())
	max_count = float(plot_df_counts.to_numpy().max()) if not plot_df_counts.empty else 0.0
	if max_count > 0.0:
		ax_abs.set_ylim(0, max_count * 1.12)
	ax_abs.legend(title="Clase", bbox_to_anchor=(1.02, 1), loc="upper left")
	plt.tight_layout()

	if output_dir:
		fig_abs_path = os.path.join(output_dir, f"{filename}_abs.png")
		fig_abs.savefig(fig_abs_path, dpi=300, bbox_inches="tight")
		print(f"💾 Figura absoluta guardada: {fig_abs_path}")

	if show_plot:
		plt.show()
	else:
		plt.close(fig_pct)
		plt.close(fig_abs)

	return df


@dataclass
class OfflineAugPolicy:
	"""
	Política de augmentación offline para una clase objetivo.

	Attributes:
		class_name: Nombre de la clase objetivo.
		target_count: Conteo objetivo de instancias (anotaciones bbox).
		use_horizontal_flip: Si True, aplica flip horizontal (obligatorio en esta fase).
		transform_kind: Transformación específica adicional.
		gaussian_sigma: Sigma normalizada para ruido gaussiano (0-1).
		perspective_ratio: Intensidad máxima de skew para perspective warp.
		zoom_factor: Factor de zoom para crop/zoom.
		blur_kernel_size: Tamaño kernel impar para desenfoque.
	"""

	class_name: str
	target_count: int = 545
	use_horizontal_flip: bool = True
	transform_kind: str = "none"  # none | gaussian_noise | perspective | crop_zoom | blur_sharpen
	gaussian_sigma: float = 0.02
	perspective_ratio: float = 0.05
	zoom_factor: float = 1.1
	blur_kernel_size: int = 3


@dataclass
class CascadeControlConfig:
	"""
	Configuración de control de cascada por co-ocurrencia.

	Attributes:
		enabled: Activa/desactiva el control de cascada.
		collateral_tolerance_ratio: Tolerancia proporcional sobre el tope colateral.
		target_overshoot_tolerance: Máx. sobrepaso permitido en la clase objetivo.
		enforce_for_all_classes: Si True, limita también clases no objetivo.
	"""

	enabled: bool = True
	collateral_tolerance_ratio: float = 0.03
	target_overshoot_tolerance: int = 2
	enforce_for_all_classes: bool = True


@dataclass
class OfflineAugRunConfig:
	"""
	Configuración general de ejecución para augmentación offline.

	Attributes:
		seed: Semilla reproducible.
		min_bbox_area: Área mínima de bbox para conservar anotación.
		min_visibility: Fracción mínima de área conservada tras transformación.
		max_generated_images: Límite de imágenes aumentadas a generar.
		output_json_name: Nombre del JSON COCO de salida.
		output_images_subdir: Carpeta de imágenes dentro de output_dir.
		verbose: Muestra logs de progreso.
	"""

	seed: int = 42
	min_bbox_area: float = 4.0
	min_visibility: float = 0.20
	max_generated_images: int = 10000
	output_json_name: str = "_annotations.coco.json"
	output_images_subdir: str = "images"
	verbose: bool = True


def build_default_offline_balance_policies(target_count: int = 545) -> Dict[str, OfflineAugPolicy]:
	"""
	Construye la matriz de políticas de augmentación offline definida en notebook.

	Args:
		target_count: Conteo objetivo de instancias por clase.

	Returns:
		Diccionario class_name -> OfflineAugPolicy.
	"""
	return {
		"dog": OfflineAugPolicy(
			class_name="dog",
			target_count=target_count,
			use_horizontal_flip=True,
			transform_kind="gaussian_noise",
			gaussian_sigma=0.02,
		),
		"door": OfflineAugPolicy(
			class_name="door",
			target_count=target_count,
			use_horizontal_flip=True,
			transform_kind="perspective",
			perspective_ratio=0.05,
		),
		"stair": OfflineAugPolicy(
			class_name="stair",
			target_count=target_count,
			use_horizontal_flip=True,
			transform_kind="crop_zoom",
			zoom_factor=1.1,
		),
		"person": OfflineAugPolicy(
			class_name="person",
			target_count=target_count,
			use_horizontal_flip=True,
			transform_kind="blur_sharpen",
		),
	}


def _count_instances_by_category(annotations: List[Dict]) -> Dict[int, int]:
	"""Cuenta instancias bbox por category_id."""
	counts: Dict[int, int] = {}
	for ann in annotations:
		cat_id = ann.get("category_id")
		if cat_id is None:
			continue
		cid = int(cat_id)
		counts[cid] = counts.get(cid, 0) + 1
	return counts


def _build_coco_indexes(coco_data: Dict) -> Tuple[Dict[int, Dict], Dict[int, List[Dict]], Dict[int, Set[int]]]:
	"""
	Construye índices útiles para procesamiento offline.

	Returns:
		(image_by_id, annotations_by_image_id, class_to_image_ids)
	"""
	image_by_id: Dict[int, Dict] = {}
	for img in coco_data.get("images", []):
		if "id" not in img:
			continue
		image_by_id[int(img["id"])] = img

	anns_by_image = _index_annotations_by_image(coco_data.get("annotations", []))
	class_to_image_ids: Dict[int, Set[int]] = {}
	for image_id, anns in anns_by_image.items():
		for ann in anns:
			cat_id = ann.get("category_id")
			if cat_id is None:
				continue
			cid = int(cat_id)
			class_to_image_ids.setdefault(cid, set()).add(int(image_id))

	return image_by_id, anns_by_image, class_to_image_ids


def _clip_bbox_xywh(
	bbox: Sequence[float],
	img_w: int,
	img_h: int,
	min_area: float,
) -> Optional[List[float]]:
	"""Recorta bbox al marco de imagen y valida área."""
	if len(bbox) != 4:
		return None
	x, y, w, h = [float(v) for v in bbox]
	if w <= 0 or h <= 0:
		return None

	x1 = max(0.0, min(float(img_w), x))
	y1 = max(0.0, min(float(img_h), y))
	x2 = max(0.0, min(float(img_w), x + w))
	y2 = max(0.0, min(float(img_h), y + h))

	new_w = x2 - x1
	new_h = y2 - y1
	if new_w <= 0.0 or new_h <= 0.0:
		return None

	if (new_w * new_h) < float(min_area):
		return None

	return [float(x1), float(y1), float(new_w), float(new_h)]


def _flip_bboxes_horizontally(bboxes: List[List[float]], img_w: int) -> List[List[float]]:
	"""
	Aplica flip horizontal en formato COCO [x, y, w, h].

	Equivalente al ajuste en centro normalizado: x_center_new = 1 - x_center_old.
	"""
	flipped: List[List[float]] = []
	for bbox in bboxes:
		x, y, w, h = bbox
		new_x = float(img_w) - (x + w)
		flipped.append([float(new_x), float(y), float(w), float(h)])
	return flipped


def _transform_bboxes_perspective(
	bboxes: List[List[float]],
	mat: np.ndarray,
	img_w: int,
	img_h: int,
	min_area: float,
	min_visibility: float,
) -> List[Optional[List[float]]]:
	"""Transforma bboxes por perspectiva y aplica validación tight."""
	transformed: List[Optional[List[float]]] = []
	for bbox in bboxes:
		x, y, w, h = bbox
		orig_area = max(w * h, 1e-6)
		pts = np.array(
			[[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
			dtype=np.float32,
		).reshape(-1, 1, 2)
		pts_t = cv2.perspectiveTransform(pts, mat).reshape(-1, 2)

		x1 = float(np.min(pts_t[:, 0]))
		y1 = float(np.min(pts_t[:, 1]))
		x2 = float(np.max(pts_t[:, 0]))
		y2 = float(np.max(pts_t[:, 1]))

		clipped = _clip_bbox_xywh([x1, y1, x2 - x1, y2 - y1], img_w, img_h, min_area)
		if clipped is None:
			transformed.append(None)
			continue

		new_area = clipped[2] * clipped[3]
		if (new_area / orig_area) < float(min_visibility):
			transformed.append(None)
			continue

		transformed.append(clipped)
	return transformed


def _transform_bboxes_crop_zoom(
	bboxes: List[List[float]],
	img_w: int,
	img_h: int,
	zoom_factor: float,
	min_area: float,
	min_visibility: float,
) -> Tuple[List[Optional[List[float]]], Dict[str, Any]]:
	"""Transforma bboxes para crop central + resize (zoom)."""
	zoom = max(float(zoom_factor), 1.0)
	crop_w = max(1, int(round(float(img_w) / zoom)))
	crop_h = max(1, int(round(float(img_h) / zoom)))
	x0 = max(0, (img_w - crop_w) // 2)
	y0 = max(0, (img_h - crop_h) // 2)
	x_scale = float(img_w) / float(crop_w)
	y_scale = float(img_h) / float(crop_h)

	transformed: List[Optional[List[float]]] = []
	for bbox in bboxes:
		x, y, w, h = bbox
		orig_area = max(w * h, 1e-6)

		x1 = (x - x0) * x_scale
		y1 = (y - y0) * y_scale
		x2 = ((x + w) - x0) * x_scale
		y2 = ((y + h) - y0) * y_scale

		clipped = _clip_bbox_xywh([x1, y1, x2 - x1, y2 - y1], img_w, img_h, min_area)
		if clipped is None:
			transformed.append(None)
			continue

		new_area = clipped[2] * clipped[3]
		if (new_area / orig_area) < float(min_visibility):
			transformed.append(None)
			continue

		transformed.append(clipped)

	meta = {
		"crop_x": int(x0),
		"crop_y": int(y0),
		"crop_w": int(crop_w),
		"crop_h": int(crop_h),
		"zoom_factor": float(zoom),
	}
	return transformed, meta


def _apply_policy_transform(
	image: np.ndarray,
	bboxes: List[List[float]],
	policy: OfflineAugPolicy,
	rng: random.Random,
	np_rng: np.random.Generator,
	run_cfg: OfflineAugRunConfig,
) -> Tuple[np.ndarray, List[Optional[List[float]]], Dict[str, Any]]:
	"""
	Aplica pipeline de augmentación de una política a imagen + bboxes.

	Nota: Vertical flip está explícitamente prohibido por diseño.
	"""
	aug_img = image.copy()
	aug_boxes = [list(map(float, b)) for b in bboxes]
	img_h, img_w = aug_img.shape[:2]
	meta: Dict[str, Any] = {
		"horizontal_flip": bool(policy.use_horizontal_flip),
		"transform_kind": str(policy.transform_kind),
	}

	# 1) Flip horizontal (prohibido flip vertical en esta fase)
	if policy.use_horizontal_flip:
		aug_img = cv2.flip(aug_img, 1)
		aug_boxes = _flip_bboxes_horizontally(aug_boxes, img_w)

	# 2) Transformación específica por clase
	kind = str(policy.transform_kind).strip().lower()
	if kind == "gaussian_noise":
		sigma_px = max(0.0, float(policy.gaussian_sigma)) * 255.0
		if sigma_px > 0.0:
			noise = np_rng.normal(0.0, sigma_px, aug_img.shape)
			aug_img = np.clip(aug_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
		meta["gaussian_sigma"] = float(policy.gaussian_sigma)
		return aug_img, [list(map(float, b)) for b in aug_boxes], meta

	if kind == "perspective":
		ratio = max(0.0, float(policy.perspective_ratio))
		jitter_x = ratio * float(img_w)
		jitter_y = ratio * float(img_h)
		src = np.array(
			[[0.0, 0.0], [img_w - 1.0, 0.0], [img_w - 1.0, img_h - 1.0], [0.0, img_h - 1.0]],
			dtype=np.float32,
		)
		dst = src.copy()
		for i in range(4):
			dst[i, 0] += float(rng.uniform(-jitter_x, jitter_x))
			dst[i, 1] += float(rng.uniform(-jitter_y, jitter_y))
		mat = cv2.getPerspectiveTransform(src, dst)
		aug_img = cv2.warpPerspective(
			aug_img,
			mat,
			(img_w, img_h),
			flags=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_REPLICATE,
		)
		aug_boxes_opt = _transform_bboxes_perspective(
			bboxes=aug_boxes,
			mat=mat,
			img_w=img_w,
			img_h=img_h,
			min_area=run_cfg.min_bbox_area,
			min_visibility=run_cfg.min_visibility,
		)
		meta["perspective_ratio"] = ratio
		return aug_img, aug_boxes_opt, meta

	if kind == "crop_zoom":
		zoom_factor = max(1.0, float(policy.zoom_factor))
		crop_w = max(1, int(round(float(img_w) / zoom_factor)))
		crop_h = max(1, int(round(float(img_h) / zoom_factor)))
		x0 = max(0, (img_w - crop_w) // 2)
		y0 = max(0, (img_h - crop_h) // 2)
		crop = aug_img[y0:y0 + crop_h, x0:x0 + crop_w]
		aug_img = cv2.resize(crop, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
		aug_boxes_opt, zoom_meta = _transform_bboxes_crop_zoom(
			bboxes=aug_boxes,
			img_w=img_w,
			img_h=img_h,
			zoom_factor=zoom_factor,
			min_area=run_cfg.min_bbox_area,
			min_visibility=run_cfg.min_visibility,
		)
		meta.update(zoom_meta)
		return aug_img, aug_boxes_opt, meta

	if kind == "blur_sharpen":
		k = int(policy.blur_kernel_size)
		if k < 3:
			k = 3
		if k % 2 == 0:
			k += 1
		if rng.random() < 0.5:
			aug_img = cv2.GaussianBlur(aug_img, (k, k), sigmaX=0)
			meta["mode"] = "blur"
		else:
			kernel = np.array(
				[[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
				dtype=np.float32,
			)
			aug_img = cv2.filter2D(aug_img, ddepth=-1, kernel=kernel)
			meta["mode"] = "sharpen"
		meta["blur_kernel_size"] = int(k)
		return aug_img, [list(map(float, b)) for b in aug_boxes], meta

	# Sin transformación adicional
	return aug_img, [list(map(float, b)) for b in aug_boxes], meta


def _build_image_purity_rank(
	image_ids: Sequence[int],
	anns_by_image: Dict[int, List[Dict]],
	target_class_id: int,
	obstacle_class_id: Optional[int],
) -> List[int]:
	"""
	Ordena candidatas por pureza para mitigar cascada de co-ocurrencia.

	Criterio (asc):
	  1) Menor número de clases diferentes.
	  2) Menor número total de instancias no objetivo.
	  3) Menor presencia de obstacle (si existe).
	  4) Menor total de anotaciones en imagen.
	"""
	def score(img_id: int) -> Tuple[int, int, int, int]:
		anns = anns_by_image.get(int(img_id), [])
		by_class: Dict[int, int] = {}
		for ann in anns:
			cid = int(ann.get("category_id", -1))
			by_class[cid] = by_class.get(cid, 0) + 1

		n_classes = len(by_class)
		n_non_target = sum(v for cid, v in by_class.items() if cid != int(target_class_id))
		obstacle_count = int(by_class.get(int(obstacle_class_id), 0)) if obstacle_class_id is not None else 0
		total_anns = len(anns)
		return (n_classes, n_non_target, obstacle_count, total_anns)

	return sorted([int(i) for i in image_ids], key=score)


def _compute_delta_counts(annotations: List[Dict]) -> Dict[int, int]:
	"""Calcula delta de instancias por clase para un lote de anotaciones nuevas."""
	delta: Dict[int, int] = {}
	for ann in annotations:
		cid = int(ann.get("category_id", -1))
		delta[cid] = delta.get(cid, 0) + 1
	return delta


def _should_accept_augmented_sample(
	delta_counts: Dict[int, int],
	target_class_id: int,
	current_counts: Dict[int, int],
	target_by_class: Dict[int, int],
	reference_cap: int,
	cascade_cfg: CascadeControlConfig,
) -> Tuple[bool, str]:
	"""Evalúa si una muestra aumentada se acepta según control de cascada."""
	target_id = int(target_class_id)
	delta_target = int(delta_counts.get(target_id, 0))
	if delta_target <= 0:
		return False, "missing_target_after_transform"

	target_goal = int(target_by_class[target_id])
	projected_target = int(current_counts.get(target_id, 0)) + delta_target
	if projected_target > (target_goal + int(cascade_cfg.target_overshoot_tolerance)):
		return False, "target_overshoot"

	if not cascade_cfg.enabled:
		return True, "accepted"

	for cid, delta in delta_counts.items():
		if cid == target_id or delta <= 0:
			continue

		cid_current = int(current_counts.get(cid, 0))
		cid_projected = cid_current + int(delta)

		if cid in target_by_class:
			base_cap = int(target_by_class[cid])
		elif cascade_cfg.enforce_for_all_classes:
			base_cap = int(reference_cap)
		else:
			continue

		cap = int(round(base_cap * (1.0 + float(cascade_cfg.collateral_tolerance_ratio))))
		if cid_projected > cap:
			return False, f"collateral_cap_exceeded:{cid}"

	return True, "accepted"


def augment_coco_train_offline_balanced(
	train_coco_json_path: str,
	train_images_dir: str,
	output_dir: str,
	policies: Optional[Dict[str, OfflineAugPolicy]] = None,
	cascade_cfg: Optional[CascadeControlConfig] = None,
	run_cfg: Optional[OfflineAugRunConfig] = None,
) -> Dict[str, Any]:
	"""
	Ejecuta augmentación offline para balancear clases del split train (COCO).

	La función implementa:
	  - políticas por clase objetivo (según matriz definida en notebook),
	  - priorización por pureza de imagen para mitigar co-ocurrencia,
	  - contador dinámico de inventario de clases,
	  - control de cascada con tolerancia configurable,
	  - validaciones de bbox y exportación COCO consistente.

	Args:
		train_coco_json_path: Ruta al JSON COCO del split train.
		train_images_dir: Carpeta de imágenes del split train.
		output_dir: Directorio de salida con dataset balanceado.
		policies: Políticas de augment por clase. Si None usa defaults.
		cascade_cfg: Configuración de cascada. Si None usa defaults.
		run_cfg: Configuración general. Si None usa defaults.

	Returns:
		Resumen detallado del proceso y rutas de salida.
	"""
	if not os.path.exists(train_coco_json_path):
		raise FileNotFoundError(f"No se encontró JSON train: {train_coco_json_path}")
	if not os.path.isdir(train_images_dir):
		raise FileNotFoundError(f"No se encontró directorio de imágenes train: {train_images_dir}")

	policy_map = policies or build_default_offline_balance_policies()
	cascade = cascade_cfg or CascadeControlConfig()
	cfg = run_cfg or OfflineAugRunConfig()

	rng = random.Random(cfg.seed)
	np_rng = np.random.default_rng(cfg.seed)

	with open(train_coco_json_path, "r", encoding="utf-8") as f:
		coco_data = json.load(f)

	categories = coco_data.get("categories", [])
	if not categories:
		raise ValueError("El JSON COCO no contiene categorías")

	cat_name_to_id: Dict[str, int] = {
		str(cat.get("name", "")).strip().lower(): int(cat["id"])
		for cat in categories
		if "id" in cat
	}
	cat_id_to_name: Dict[int, str] = {
		int(cat["id"]): str(cat.get("name", ""))
		for cat in categories
		if "id" in cat
	}

	target_by_class: Dict[int, int] = {}
	policy_by_class_id: Dict[int, OfflineAugPolicy] = {}
	for class_name, policy in policy_map.items():
		key = str(class_name).strip().lower()
		if key not in cat_name_to_id:
			raise ValueError(f"Clase objetivo no existe en categories: '{class_name}'")
		cid = int(cat_name_to_id[key])
		target_by_class[cid] = int(policy.target_count)
		policy_by_class_id[cid] = policy

	obstacle_id = cat_name_to_id.get("obstacle")

	image_by_id, anns_by_image, class_to_image_ids = _build_coco_indexes(coco_data)
	if not image_by_id:
		raise ValueError("El JSON COCO no contiene imágenes válidas")

	initial_counts = _count_instances_by_category(coco_data.get("annotations", []))
	current_counts = dict(initial_counts)

	reference_cap = max(target_by_class.values()) if target_by_class else 0

	# Construir ranking de pureza por clase objetivo
	ranked_candidates_by_class: Dict[int, List[int]] = {}
	for target_cid in target_by_class:
		img_ids = list(class_to_image_ids.get(int(target_cid), set()))
		ranked_candidates_by_class[target_cid] = _build_image_purity_rank(
			image_ids=img_ids,
			anns_by_image=anns_by_image,
			target_class_id=target_cid,
			obstacle_class_id=obstacle_id,
		)

	ensure_dir(output_dir)
	output_images_dir = os.path.join(output_dir, cfg.output_images_subdir)
	ensure_dir(output_images_dir)

	# Copiar imágenes originales al dataset de salida
	missing_original_images = 0
	copied_original_images = 0
	for img in coco_data.get("images", []):
		file_name = img.get("file_name")
		if not file_name:
			continue
		src = os.path.join(train_images_dir, file_name)
		dst = os.path.join(output_images_dir, file_name)
		if not os.path.exists(src):
			missing_original_images += 1
			continue
		parent = os.path.dirname(dst)
		if parent:
			ensure_dir(parent)
		shutil.copy2(src, dst)
		copied_original_images += 1

	out_data = {
		"info": copy.deepcopy(coco_data.get("info", {})),
		"licenses": copy.deepcopy(coco_data.get("licenses", [])),
		"categories": copy.deepcopy(categories),
		"images": copy.deepcopy(coco_data.get("images", [])),
		"annotations": copy.deepcopy(coco_data.get("annotations", [])),
	}

	next_image_id = (
		max([int(img.get("id", 0)) for img in out_data["images"]], default=0) + 1
	)
	next_ann_id = (
		max([int(ann.get("id", 0)) for ann in out_data["annotations"]], default=0) + 1
	)

	class_cursors: Dict[int, int] = {cid: 0 for cid in target_by_class}
	rejection_reasons: Dict[str, int] = {}
	accepted_by_class: Dict[int, int] = {cid: 0 for cid in target_by_class}

	def _class_deficit(cid: int) -> int:
		return max(0, int(target_by_class[cid]) - int(current_counts.get(cid, 0)))

	if cfg.verbose:
		print("\n🚀 Iniciando augmentación offline balanceada (TRAIN)")
		print(f"   📄 JSON train: {train_coco_json_path}")
		print(f"   🖼️  Imágenes train: {train_images_dir}")
		print(f"   📦 Output: {output_dir}")
		print("   🎯 Targets:")
		for cid in target_by_class:
			print(
				f"      - {cat_id_to_name.get(cid, str(cid))}: "
				f"{current_counts.get(cid, 0)} -> {target_by_class[cid]}"
			)

	iterations_without_progress = 0
	max_stall = max(20, len(target_by_class) * 5)
	generated_images = 0

	while generated_images < int(cfg.max_generated_images):
		deficits = {cid: _class_deficit(cid) for cid in target_by_class}
		pending = {cid: d for cid, d in deficits.items() if d > 0}
		if not pending:
			break

		# Clase con mayor déficit actual
		target_cid = max(pending.keys(), key=lambda c: pending[c])
		policy = policy_by_class_id[target_cid]
		candidates = ranked_candidates_by_class.get(target_cid, [])

		if not candidates:
			rejection_reasons["no_candidates_for_class"] = rejection_reasons.get("no_candidates_for_class", 0) + 1
			iterations_without_progress += 1
			if iterations_without_progress >= max_stall:
				break
			continue

		accepted_this_round = False
		attempts = len(candidates)
		for _ in range(attempts):
			cursor = class_cursors[target_cid] % len(candidates)
			class_cursors[target_cid] += 1
			img_id = int(candidates[cursor])

			img_info = image_by_id.get(img_id)
			anns_src = anns_by_image.get(img_id, [])
			if img_info is None or not anns_src:
				rejection_reasons["invalid_candidate_image"] = rejection_reasons.get("invalid_candidate_image", 0) + 1
				continue

			file_name = img_info.get("file_name")
			if not file_name:
				rejection_reasons["missing_filename"] = rejection_reasons.get("missing_filename", 0) + 1
				continue

			src_img_path = os.path.join(train_images_dir, file_name)
			img = cv2.imread(src_img_path)
			if img is None:
				rejection_reasons["unreadable_source_image"] = rejection_reasons.get("unreadable_source_image", 0) + 1
				continue

			img_h, img_w = img.shape[:2]
			base_bboxes: List[List[float]] = []
			base_meta: List[Dict[str, Any]] = []
			for ann in anns_src:
				bbox = ann.get("bbox", [])
				if len(bbox) != 4:
					continue
				clipped = _clip_bbox_xywh(bbox, img_w, img_h, cfg.min_bbox_area)
				if clipped is None:
					continue
				base_bboxes.append(clipped)
				base_meta.append(ann)

			if not base_bboxes:
				rejection_reasons["candidate_without_valid_bboxes"] = rejection_reasons.get("candidate_without_valid_bboxes", 0) + 1
				continue

			aug_img, aug_boxes_opt, _ = _apply_policy_transform(
				image=img,
				bboxes=base_bboxes,
				policy=policy,
				rng=rng,
				np_rng=np_rng,
				run_cfg=cfg,
			)

			new_annotations_batch: List[Dict] = []
			for src_ann, aug_bbox_opt in zip(base_meta, aug_boxes_opt):
				if aug_bbox_opt is None:
					continue
				new_annotations_batch.append(
					{
						"id": -1,  # se asigna al aceptar
						"image_id": -1,  # se asigna al aceptar
						"category_id": int(src_ann.get("category_id", -1)),
						"bbox": [float(v) for v in aug_bbox_opt],
						"area": float(aug_bbox_opt[2] * aug_bbox_opt[3]),
						"iscrowd": int(src_ann.get("iscrowd", 0)),
						"segmentation": [],
					}
				)

			if not new_annotations_batch:
				rejection_reasons["all_bboxes_dropped"] = rejection_reasons.get("all_bboxes_dropped", 0) + 1
				continue

			delta_counts = _compute_delta_counts(new_annotations_batch)
			accept, reason = _should_accept_augmented_sample(
				delta_counts=delta_counts,
				target_class_id=target_cid,
				current_counts=current_counts,
				target_by_class=target_by_class,
				reference_cap=reference_cap,
				cascade_cfg=cascade,
			)
			if not accept:
				rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
				continue

			# Persistir imagen aumentada
			stem, ext = os.path.splitext(str(file_name))
			if not ext:
				ext = ".jpg"
			class_tag = str(policy.class_name).strip().lower()
			aug_file_name = f"{stem}__aug_{class_tag}_{generated_images + 1:06d}{ext}"
			dst_aug_path = os.path.join(output_images_dir, aug_file_name)

			parent = os.path.dirname(dst_aug_path)
			if parent:
				ensure_dir(parent)
			ok_write = cv2.imwrite(dst_aug_path, aug_img)
			if not ok_write:
				rejection_reasons["failed_to_write_aug_image"] = rejection_reasons.get("failed_to_write_aug_image", 0) + 1
				continue

			new_image_id = int(next_image_id)
			next_image_id += 1
			out_data["images"].append(
				{
					"id": new_image_id,
					"file_name": aug_file_name,
					"width": int(img_w),
					"height": int(img_h),
				}
			)

			for ann in new_annotations_batch:
				ann["id"] = int(next_ann_id)
				ann["image_id"] = int(new_image_id)
				next_ann_id += 1
				out_data["annotations"].append(ann)
				cid = int(ann["category_id"])
				current_counts[cid] = current_counts.get(cid, 0) + 1

			generated_images += 1
			accepted_by_class[target_cid] = accepted_by_class.get(target_cid, 0) + 1
			accepted_this_round = True
			break

		if accepted_this_round:
			iterations_without_progress = 0
		else:
			iterations_without_progress += 1
			if iterations_without_progress >= max_stall:
				break

	output_json_path = os.path.join(output_dir, cfg.output_json_name)
	with open(output_json_path, "w", encoding="utf-8") as f:
		json.dump(out_data, f, ensure_ascii=False)

	final_counts = _count_instances_by_category(out_data.get("annotations", []))
	target_report: List[Dict[str, Any]] = []
	for cid, target in target_by_class.items():
		before = int(initial_counts.get(cid, 0))
		after = int(final_counts.get(cid, 0))
		target_report.append(
			{
				"class_id": int(cid),
				"class_name": cat_id_to_name.get(cid, str(cid)),
				"before": before,
				"after": after,
				"delta": after - before,
				"target": int(target),
				"target_hit": bool(after >= int(target)),
			}
		)

	summary: Dict[str, Any] = {
		"output_dir": output_dir,
		"output_json_path": output_json_path,
		"output_images_dir": output_images_dir,
		"copied_original_images": int(copied_original_images),
		"missing_original_images": int(missing_original_images),
		"generated_images": int(generated_images),
		"initial_annotation_count": int(len(coco_data.get("annotations", []))),
		"final_annotation_count": int(len(out_data.get("annotations", []))),
		"accepted_by_class": {
			cat_id_to_name.get(cid, str(cid)): int(val)
			for cid, val in accepted_by_class.items()
		},
		"target_report": target_report,
		"rejection_reasons": rejection_reasons,
		"seed": int(cfg.seed),
		"cascade": {
			"enabled": bool(cascade.enabled),
			"collateral_tolerance_ratio": float(cascade.collateral_tolerance_ratio),
			"target_overshoot_tolerance": int(cascade.target_overshoot_tolerance),
		},
	}

	if cfg.verbose:
		print("\n✅ Augmentación offline completada")
		for row in target_report:
			status = "OK" if row["target_hit"] else "PENDIENTE"
			print(
				f"   {row['class_name']}: {row['before']} -> {row['after']} "
				f"(target={row['target']}, {status})"
			)
		print(f"   🖼️  Imágenes generadas: {generated_images}")
		print(f"   💾 JSON salida: {output_json_path}")

	return summary


def augment_dataset_split_train_offline(
	dataset_split_dir: str,
	output_dir: Optional[str] = None,
	policies: Optional[Dict[str, OfflineAugPolicy]] = None,
	cascade_cfg: Optional[CascadeControlConfig] = None,
	run_cfg: Optional[OfflineAugRunConfig] = None,
) -> Dict[str, Any]:
	"""
	Wrapper para aplicar augmentación offline sobre `dataset_split/train`.

	Args:
		dataset_split_dir: Ruta al directorio `dataset_split`.
		output_dir: Ruta de salida. Si None usa `dataset_split/train_augmented_offline`.
		policies: Políticas por clase objetivo.
		cascade_cfg: Configuración de control de cascada.
		run_cfg: Configuración de ejecución.

	Returns:
		Resumen de ejecución de augmentación offline.
	"""
	if not os.path.isdir(dataset_split_dir):
		raise FileNotFoundError(f"No existe dataset_split_dir: {dataset_split_dir}")

	train_dir = os.path.join(dataset_split_dir, "train")
	train_json = os.path.join(train_dir, "_annotations.coco.json")
	train_images = os.path.join(train_dir, "images")

	out_dir = output_dir or os.path.join(dataset_split_dir, "train_augmented_offline")

	return augment_coco_train_offline_balanced(
		train_coco_json_path=train_json,
		train_images_dir=train_images,
		output_dir=out_dir,
		policies=policies,
		cascade_cfg=cascade_cfg,
		run_cfg=run_cfg,
	)


def load_train_before_after_coco(
	before_json_path: str,
	after_json_path: str,
) -> Dict[str, Any]:
	"""
	Carga COCO before/after de train y valida compatibilidad de categorías.

	Args:
		before_json_path: Ruta JSON COCO train antes de augmentación.
		after_json_path: Ruta JSON COCO train después de augmentación.

	Returns:
		Diccionario con objetos COCO, mapeos de clase y tabla resumen.
	"""
	if not os.path.exists(before_json_path):
		raise FileNotFoundError(f"No se encontró JSON before: {before_json_path}")
	if not os.path.exists(after_json_path):
		raise FileNotFoundError(f"No se encontró JSON after: {after_json_path}")

	coco_before = COCO(before_json_path)
	coco_after = COCO(after_json_path)

	before_cats = coco_before.loadCats(coco_before.getCatIds())
	after_cats = coco_after.loadCats(coco_after.getCatIds())

	before_name_to_id = {
		str(cat["name"]).strip().lower(): int(cat["id"])
		for cat in before_cats
	}
	after_name_to_id = {
		str(cat["name"]).strip().lower(): int(cat["id"])
		for cat in after_cats
	}

	if set(before_name_to_id.keys()) != set(after_name_to_id.keys()):
		missing_in_after = sorted(set(before_name_to_id.keys()) - set(after_name_to_id.keys()))
		missing_in_before = sorted(set(after_name_to_id.keys()) - set(before_name_to_id.keys()))
		raise ValueError(
			"Categorías inconsistentes entre before/after. "
			f"Faltan en after={missing_in_after}; faltan en before={missing_in_before}"
		)

	cat_names_sorted = sorted(before_name_to_id.keys())
	cat_ids_before = [before_name_to_id[name] for name in cat_names_sorted]
	cat_ids_after = [after_name_to_id[name] for name in cat_names_sorted]

	id_to_name_before = {
		before_name_to_id[name]: name
		for name in cat_names_sorted
	}
	id_to_name_after = {
		after_name_to_id[name]: name
		for name in cat_names_sorted
	}

	summary_df = pd.DataFrame(
		[
			{
				"version": "before",
				"num_images": len(coco_before.getImgIds()),
				"num_annotations": len(coco_before.getAnnIds()),
				"num_categories": len(cat_ids_before),
			},
			{
				"version": "after",
				"num_images": len(coco_after.getImgIds()),
				"num_annotations": len(coco_after.getAnnIds()),
				"num_categories": len(cat_ids_after),
			},
		]
	)

	return {
		"coco_before": coco_before,
		"coco_after": coco_after,
		"cat_ids_before": cat_ids_before,
		"cat_ids_after": cat_ids_after,
		"id_to_name_before": id_to_name_before,
		"id_to_name_after": id_to_name_after,
		"summary_df": summary_df,
	}


def validate_coco_annotation_integrity(
	coco_json_path: str,
	images_dir: str,
	min_bbox_area: float = 4.0,
	check_image_readability: bool = True,
	max_samples_per_error: int = 20,
) -> Dict[str, Any]:
	"""
	Valida integridad estructural de un dataset COCO.

	Checks:
	  - IDs únicos de imágenes y anotaciones
	  - Referencias image_id/category_id válidas
	  - bbox con w/h > 0 y área mínima
	  - bbox dentro de límites de imagen
	  - área consistente con w*h
	  - imágenes existentes (y legibles opcional)
	"""
	if not os.path.exists(coco_json_path):
		raise FileNotFoundError(f"No se encontró JSON COCO: {coco_json_path}")
	if not os.path.isdir(images_dir):
		raise FileNotFoundError(f"No se encontró images_dir: {images_dir}")

	with open(coco_json_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	images = data.get("images", [])
	annotations = data.get("annotations", [])
	categories = data.get("categories", [])

	image_ids = [int(img.get("id", -1)) for img in images if "id" in img]
	ann_ids = [int(ann.get("id", -1)) for ann in annotations if "id" in ann]
	valid_cat_ids = {int(cat.get("id", -1)) for cat in categories if "id" in cat}
	image_id_set = set(image_ids)

	error_counts: Dict[str, int] = {
		"duplicate_image_ids": 0,
		"duplicate_annotation_ids": 0,
		"missing_image_files": 0,
		"unreadable_images": 0,
		"invalid_annotation_image_ref": 0,
		"invalid_annotation_category_ref": 0,
		"invalid_bbox_format": 0,
		"invalid_bbox_size": 0,
		"bbox_out_of_bounds": 0,
		"bbox_area_mismatch": 0,
	}

	examples: Dict[str, List[Any]] = {k: [] for k in error_counts.keys()}

	if len(image_ids) != len(set(image_ids)):
		error_counts["duplicate_image_ids"] = len(image_ids) - len(set(image_ids))

	if len(ann_ids) != len(set(ann_ids)):
		error_counts["duplicate_annotation_ids"] = len(ann_ids) - len(set(ann_ids))

	image_size_by_id: Dict[int, Tuple[int, int]] = {}
	for img in images:
		if "id" not in img:
			continue
		img_id = int(img["id"])
		w = int(img.get("width", 0))
		h = int(img.get("height", 0))
		image_size_by_id[img_id] = (w, h)

		file_name = str(img.get("file_name", "")).strip()
		if not file_name:
			error_counts["missing_image_files"] += 1
			if len(examples["missing_image_files"]) < max_samples_per_error:
				examples["missing_image_files"].append({"image_id": img_id, "file_name": file_name})
			continue

		img_path = os.path.join(images_dir, file_name)
		if not os.path.exists(img_path):
			error_counts["missing_image_files"] += 1
			if len(examples["missing_image_files"]) < max_samples_per_error:
				examples["missing_image_files"].append({"image_id": img_id, "file_name": file_name})
			continue

		if check_image_readability:
			im = cv2.imread(img_path)
			if im is None:
				error_counts["unreadable_images"] += 1
				if len(examples["unreadable_images"]) < max_samples_per_error:
					examples["unreadable_images"].append({"image_id": img_id, "file_name": file_name})

	for ann in annotations:
		ann_id = int(ann.get("id", -1))
		image_id = int(ann.get("image_id", -1))
		cat_id = int(ann.get("category_id", -1))
		bbox = ann.get("bbox", [])

		if image_id not in image_id_set:
			error_counts["invalid_annotation_image_ref"] += 1
			if len(examples["invalid_annotation_image_ref"]) < max_samples_per_error:
				examples["invalid_annotation_image_ref"].append({"ann_id": ann_id, "image_id": image_id})
			continue

		if cat_id not in valid_cat_ids:
			error_counts["invalid_annotation_category_ref"] += 1
			if len(examples["invalid_annotation_category_ref"]) < max_samples_per_error:
				examples["invalid_annotation_category_ref"].append({"ann_id": ann_id, "category_id": cat_id})

		if not isinstance(bbox, list) or len(bbox) != 4:
			error_counts["invalid_bbox_format"] += 1
			if len(examples["invalid_bbox_format"]) < max_samples_per_error:
				examples["invalid_bbox_format"].append({"ann_id": ann_id, "bbox": bbox})
			continue

		x, y, w, h = [float(v) for v in bbox]
		if w <= 0 or h <= 0 or (w * h) < float(min_bbox_area):
			error_counts["invalid_bbox_size"] += 1
			if len(examples["invalid_bbox_size"]) < max_samples_per_error:
				examples["invalid_bbox_size"].append({"ann_id": ann_id, "bbox": bbox})
			continue

		img_w, img_h = image_size_by_id.get(image_id, (0, 0))
		if img_w > 0 and img_h > 0:
			x2 = x + w
			y2 = y + h
			if x < 0 or y < 0 or x2 > img_w or y2 > img_h:
				error_counts["bbox_out_of_bounds"] += 1
				if len(examples["bbox_out_of_bounds"]) < max_samples_per_error:
					examples["bbox_out_of_bounds"].append(
						{"ann_id": ann_id, "bbox": bbox, "img_w": img_w, "img_h": img_h}
					)

		area = ann.get("area")
		if area is not None:
			ann_area = float(area)
			bbox_area = float(w * h)
			if abs(ann_area - bbox_area) > max(1.0, 0.01 * bbox_area):
				error_counts["bbox_area_mismatch"] += 1
				if len(examples["bbox_area_mismatch"]) < max_samples_per_error:
					examples["bbox_area_mismatch"].append(
						{"ann_id": ann_id, "area": ann_area, "bbox_area": bbox_area}
					)

	critical_keys = [
		"duplicate_image_ids",
		"duplicate_annotation_ids",
		"missing_image_files",
		"unreadable_images",
		"invalid_annotation_image_ref",
		"invalid_annotation_category_ref",
		"invalid_bbox_format",
		"invalid_bbox_size",
	]
	critical_errors = sum(error_counts[k] for k in critical_keys)
	total_errors = sum(error_counts.values())

	report_df = pd.DataFrame(
		[
			{"check": key, "count": int(value)}
			for key, value in error_counts.items()
		]
	).sort_values("count", ascending=False).reset_index(drop=True)

	return {
		"coco_json_path": coco_json_path,
		"images_dir": images_dir,
		"num_images": len(images),
		"num_annotations": len(annotations),
		"num_categories": len(categories),
		"total_errors": int(total_errors),
		"critical_errors": int(critical_errors),
		"status": "PASS" if critical_errors == 0 else "FAIL",
		"report_df": report_df,
		"examples": examples,
	}


def get_augmented_image_ids(
	before_coco: COCO,
	after_coco: COCO,
	filename_token: str = "__aug_",
) -> Dict[str, List[int]]:
	"""
	Obtiene IDs de imágenes nuevas creadas tras augmentación.

	Usa tres criterios y devuelve su intersección/union útil:
	  - diferencia por file_name
	  - diferencia por image_id
	  - convención de nombre con token (ej. '__aug_')
	"""
	before_imgs = before_coco.loadImgs(before_coco.getImgIds())
	after_imgs = after_coco.loadImgs(after_coco.getImgIds())

	before_by_id = {int(img["id"]): img for img in before_imgs}
	after_by_id = {int(img["id"]): img for img in after_imgs}

	before_file_set = {str(img.get("file_name", "")) for img in before_imgs}

	new_by_filename = sorted(
		[
			int(img_id)
			for img_id, img in after_by_id.items()
			if str(img.get("file_name", "")) not in before_file_set
		]
	)
	new_by_id = sorted(list(set(after_by_id.keys()) - set(before_by_id.keys())))
	token = str(filename_token)
	token_ids = sorted(
		[
			int(img_id)
			for img_id, img in after_by_id.items()
			if token in str(img.get("file_name", ""))
		]
	)

	combined = sorted(list(set(new_by_filename) | set(new_by_id) | set(token_ids)))

	return {
		"new_by_filename": new_by_filename,
		"new_by_id": new_by_id,
		"token_ids": token_ids,
		"augmented_ids": combined,
	}


def sample_augmented_image_ids_for_review(
	after_coco: COCO,
	augmented_img_ids: List[int],
	per_class: int = 3,
	random_k: int = 12,
	seed: int = 42,
) -> Dict[str, List[int]]:
	"""
	Genera muestras reproducibles de imágenes aumentadas para revisión visual.

	Estrategia:
	  - Muestra estratificada por clase (hasta per_class por clase)
	  - Muestra aleatoria global (random_k) sobre el remanente
	"""
	rng = random.Random(seed)
	pool = sorted(list({int(i) for i in augmented_img_ids}))
	if not pool:
		return {
			"stratified_ids": [],
			"random_ids": [],
			"final_ids": [],
		}

	pool_set = set(pool)
	stratified: List[int] = []
	cat_ids = after_coco.getCatIds()

	for cat_id in cat_ids:
		img_ids_cat = set(after_coco.getImgIds(catIds=[cat_id]))
		candidates = sorted(list(img_ids_cat & pool_set))
		if not candidates:
			continue
		k = min(int(per_class), len(candidates))
		if k <= 0:
			continue
		stratified.extend(rng.sample(candidates, k))

	stratified = sorted(list(set(stratified)))
	remaining = [img_id for img_id in pool if img_id not in set(stratified)]
	if remaining:
		random_sample = rng.sample(remaining, min(int(random_k), len(remaining)))
	else:
		random_sample = []

	final_ids = sorted(list(set(stratified + random_sample)))

	return {
		"stratified_ids": stratified,
		"random_ids": sorted(random_sample),
		"final_ids": final_ids,
	}


def compare_train_before_after_reference_metrics(
	coco_before: COCO,
	coco_after: COCO,
	output_dir: Optional[str] = None,
	prefix: str = "09_before_after_train",
	palette: str = "magma",
) -> Dict[str, Any]:
	"""
	Compara métricas base de train before vs after (equivalentes a celdas 20-23).

	Incluye:
	  1) distribución de clases,
	  2) densidad imágenes vs objetos,
	  3) ratio objetos/imagen,
	  4) co-ocurrencia de clases.
	"""
	from utils_eda import (
		calculate_and_plot_density_ratio,
		calculate_cooccurrence_matrix,
		calculate_image_object_density,
		get_category_distribution,
		plot_category_distribution,
		plot_cooccurrence_matrix,
		plot_image_object_density,
	)

	if output_dir:
		ensure_dir(output_dir)

	# 1) Distribución de clases
	df_before_counts = get_category_distribution(coco_before)
	df_after_counts = get_category_distribution(coco_after)

	plot_category_distribution(
		df_counts=df_before_counts.rename(columns={"Category": "Category", "Count": "Count"})[["Category", "Count"]],
		scale_type="linear",
		palette=palette,
		output_dir=output_dir,
		filename=f"{prefix}_01_class_distribution_before",
		title="Distribución de Clases TRAIN (Before)",
	)
	plot_category_distribution(
		df_counts=df_after_counts.rename(columns={"Category": "Category", "Count": "Count"})[["Category", "Count"]],
		scale_type="linear",
		palette=palette,
		output_dir=output_dir,
		filename=f"{prefix}_02_class_distribution_after",
		title="Distribución de Clases TRAIN (After Offline Aug)",
	)

	df_class_compare = (
		df_before_counts[["Category", "Count"]]
		.rename(columns={"Count": "Count_before"})
		.merge(
			df_after_counts[["Category", "Count"]].rename(columns={"Count": "Count_after"}),
			on="Category",
			how="outer",
		)
		.fillna(0)
	)
	df_class_compare["Delta"] = df_class_compare["Count_after"] - df_class_compare["Count_before"]

	# 2) Densidad imágenes vs objetos
	cat_ids_before = coco_before.getCatIds()
	cat_ids_after = coco_after.getCatIds()
	df_density_before, active_ids_before = calculate_image_object_density(coco_before, cat_ids_before)
	df_density_after, active_ids_after = calculate_image_object_density(coco_after, cat_ids_after)

	plot_image_object_density(
		df_density=df_density_before,
		palette=palette,
		figsize=(12, 6),
		output_dir=output_dir,
		filename=f"{prefix}_03_density_before",
		title="Densidad Imágenes vs Objetos (TRAIN Before)",
	)
	plot_image_object_density(
		df_density=df_density_after,
		palette=palette,
		figsize=(12, 6),
		output_dir=output_dir,
		filename=f"{prefix}_04_density_after",
		title="Densidad Imágenes vs Objetos (TRAIN After)",
	)

	df_density_compare = (
		df_density_before.rename(columns={"Images": "Images_before", "Objects": "Objects_before"})
		.merge(
			df_density_after.rename(columns={"Images": "Images_after", "Objects": "Objects_after"}),
			on="Category",
			how="outer",
		)
		.fillna(0)
	)
	df_density_compare["Delta_images"] = df_density_compare["Images_after"] - df_density_compare["Images_before"]
	df_density_compare["Delta_objects"] = df_density_compare["Objects_after"] - df_density_compare["Objects_before"]

	# 3) Ratio objetos/imagen
	df_ratio_before = calculate_and_plot_density_ratio(
		df_density_before,
		palette=palette,
		output_dir=output_dir,
		filename=f"{prefix}_05_ratio_before",
		title="Ratio Objetos/Imagen (TRAIN Before)",
	)
	df_ratio_after = calculate_and_plot_density_ratio(
		df_density_after,
		palette=palette,
		output_dir=output_dir,
		filename=f"{prefix}_06_ratio_after",
		title="Ratio Objetos/Imagen (TRAIN After)",
	)

	df_ratio_compare = (
		df_ratio_before[["Category", "Ratio"]]
		.rename(columns={"Ratio": "Ratio_before"})
		.merge(
			df_ratio_after[["Category", "Ratio"]].rename(columns={"Ratio": "Ratio_after"}),
			on="Category",
			how="outer",
		)
		.fillna(0)
	)
	df_ratio_compare["Delta_ratio"] = df_ratio_compare["Ratio_after"] - df_ratio_compare["Ratio_before"]

	# 4) Co-ocurrencia
	co_before, labels_before = calculate_cooccurrence_matrix(coco_before, active_ids_before)
	co_after, labels_after = calculate_cooccurrence_matrix(coco_after, active_ids_after)

	plot_cooccurrence_matrix(
		co_occurrence_matrix=co_before,
		cat_labels=labels_before,
		cmap="YlGnBu",
		output_dir=output_dir,
		filename=f"{prefix}_07_cooccurrence_before",
		title="Matriz de Co-ocurrencia TRAIN (Before)",
	)
	plot_cooccurrence_matrix(
		co_occurrence_matrix=co_after,
		cat_labels=labels_after,
		cmap="YlGnBu",
		output_dir=output_dir,
		filename=f"{prefix}_08_cooccurrence_after",
		title="Matriz de Co-ocurrencia TRAIN (After)",
	)

	df_co_before = pd.DataFrame(co_before, index=labels_before, columns=labels_before)
	df_co_after = pd.DataFrame(co_after, index=labels_after, columns=labels_after)
	all_labels = sorted(list(set(labels_before) | set(labels_after)))
	df_co_before = df_co_before.reindex(index=all_labels, columns=all_labels, fill_value=0)
	df_co_after = df_co_after.reindex(index=all_labels, columns=all_labels, fill_value=0)
	df_co_delta = df_co_after - df_co_before

	if output_dir:
		df_class_compare.to_csv(os.path.join(output_dir, f"{prefix}_class_compare.csv"), index=False)
		df_density_compare.to_csv(os.path.join(output_dir, f"{prefix}_density_compare.csv"), index=False)
		df_ratio_compare.to_csv(os.path.join(output_dir, f"{prefix}_ratio_compare.csv"), index=False)
		df_co_delta.to_csv(os.path.join(output_dir, f"{prefix}_cooccurrence_delta.csv"))

	return {
		"class_compare": df_class_compare,
		"density_before": df_density_before,
		"density_after": df_density_after,
		"density_compare": df_density_compare,
		"ratio_before": df_ratio_before,
		"ratio_after": df_ratio_after,
		"ratio_compare": df_ratio_compare,
		"cooccurrence_before": df_co_before,
		"cooccurrence_after": df_co_after,
		"cooccurrence_delta": df_co_delta,
	}


def compare_train_before_after_additional_metrics(
	coco_before: COCO,
	coco_after: COCO,
	output_dir: Optional[str] = None,
	prefix: str = "09_before_after_train_ext",
) -> Dict[str, Any]:
	"""
	Comparaciones complementarias beyond celdas 20-23.

	Incluye:
	  - geometría bbox (área/ratio/tamaño),
	  - complejidad de escena (densidad de objetos),
	  - oclusión por IoU entre bboxes vecinas.
	"""
	from utils_eda import (
		calculate_density_and_iou,
		extract_bbox_geometry,
		plot_aspect_ratio,
		plot_density_histogram,
		plot_iou_histogram,
		plot_size_distribution,
		plot_spatial_heatmap,
	)

	if output_dir:
		ensure_dir(output_dir)

	cat_ids_before = coco_before.getCatIds()
	cat_ids_after = coco_after.getCatIds()

	df_geo_before = extract_bbox_geometry(
		coco=coco_before,
		target_cat_ids=cat_ids_before,
		output_dir=output_dir,
		filename=f"{prefix}_01_geo_before",
	)
	df_geo_after = extract_bbox_geometry(
		coco=coco_after,
		target_cat_ids=cat_ids_after,
		output_dir=output_dir,
		filename=f"{prefix}_02_geo_after",
	)

	plot_size_distribution(
		df_geo_before,
		output_dir=output_dir,
		filename=f"{prefix}_03_size_before",
		title="Distribución de Tamaños COCO (Before)",
	)
	plot_size_distribution(
		df_geo_after,
		output_dir=output_dir,
		filename=f"{prefix}_04_size_after",
		title="Distribución de Tamaños COCO (After)",
	)

	plot_aspect_ratio(
		df_geo_before,
		output_dir=output_dir,
		filename=f"{prefix}_05_aspect_before",
		title="Aspect Ratio BBox (Before)",
	)
	plot_aspect_ratio(
		df_geo_after,
		output_dir=output_dir,
		filename=f"{prefix}_06_aspect_after",
		title="Aspect Ratio BBox (After)",
	)

	plot_spatial_heatmap(
		df_geo_before,
		output_dir=output_dir,
		filename=f"{prefix}_07_spatial_before",
		title="Mapa espacial de objetos (Before)",
	)
	plot_spatial_heatmap(
		df_geo_after,
		output_dir=output_dir,
		filename=f"{prefix}_08_spatial_after",
		title="Mapa espacial de objetos (After)",
	)

	density_before, iou_before = calculate_density_and_iou(coco_before, cat_ids_before)
	density_after, iou_after = calculate_density_and_iou(coco_after, cat_ids_after)

	plot_density_histogram(
		density_before,
		output_dir=output_dir,
		filename=f"{prefix}_09_density_hist_before",
		title="Densidad de escena (Before)",
	)
	plot_density_histogram(
		density_after,
		output_dir=output_dir,
		filename=f"{prefix}_10_density_hist_after",
		title="Densidad de escena (After)",
	)

	if iou_before:
		plot_iou_histogram(
			iou_before,
			output_dir=output_dir,
			filename=f"{prefix}_11_iou_hist_before",
			title="IoU vecinas (Before)",
		)
	if iou_after:
		plot_iou_histogram(
			iou_after,
			output_dir=output_dir,
			filename=f"{prefix}_12_iou_hist_after",
			title="IoU vecinas (After)",
		)

	summary_df = pd.DataFrame(
		[
			{
				"metric": "mean_objects_per_image",
				"before": float(np.mean(density_before)) if density_before else 0.0,
				"after": float(np.mean(density_after)) if density_after else 0.0,
			},
			{
				"metric": "median_objects_per_image",
				"before": float(np.median(density_before)) if density_before else 0.0,
				"after": float(np.median(density_after)) if density_after else 0.0,
			},
			{
				"metric": "mean_iou_neighbors",
				"before": float(np.mean(iou_before)) if iou_before else 0.0,
				"after": float(np.mean(iou_after)) if iou_after else 0.0,
			},
		]
	)
	summary_df["delta"] = summary_df["after"] - summary_df["before"]

	if output_dir:
		summary_df.to_csv(os.path.join(output_dir, f"{prefix}_summary_delta.csv"), index=False)

	return {
		"geo_before": df_geo_before,
		"geo_after": df_geo_after,
		"density_before": density_before,
		"density_after": density_after,
		"iou_before": iou_before,
		"iou_after": iou_after,
		"summary_df": summary_df,
	}


@dataclass
class DatasetExportPathsConfig:
	"""
	Configuración de rutas para exportación multi-formato desde splits COCO.

	Attributes:
		train_split_dir: Ruta al split train (con _annotations.coco.json + images/).
		valid_split_dir: Ruta al split valid.
		test_split_dir: Ruta al split test.
		output_root_dir: Directorio raíz de salida (creará coco/, yolo/, tfrecord/).
		include_empty_yolo_labels: Si True, crea .txt vacío para imágenes sin objetos.
		tfrecord_shards: Número de shards TFRecord por split.
	"""

	train_split_dir: str
	valid_split_dir: str
	test_split_dir: str
	output_root_dir: str
	include_empty_yolo_labels: bool = True
	tfrecord_shards: int = 1


@dataclass
class DatasetClassMappingConfig:
	"""
	Política de mapeo de IDs de clase por formato de salida.

	Attributes:
		yolo_zero_based: Si True, YOLO usa IDs 0..N-1.
		tfrecord_one_based: Si True, TFRecord usa IDs 1..N.
	"""

	yolo_zero_based: bool = True
	tfrecord_one_based: bool = True


def _load_coco_split_from_dir(split_dir: str) -> Tuple[Dict[str, Any], str, str]:
	"""Carga un split COCO desde un directorio estándar con JSON + images/."""
	if not os.path.isdir(split_dir):
		raise FileNotFoundError(f"No existe split_dir: {split_dir}")

	json_path = os.path.join(split_dir, "_annotations.coco.json")
	images_dir = os.path.join(split_dir, "images")

	if not os.path.exists(json_path):
		raise FileNotFoundError(f"No se encontró JSON COCO: {json_path}")
	if not os.path.isdir(images_dir):
		raise FileNotFoundError(f"No se encontró carpeta images: {images_dir}")

	with open(json_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	return data, json_path, images_dir


def _normalize_categories(categories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Ordena y valida categorías COCO por id ascendente."""
	if not categories:
		raise ValueError("El dataset no contiene categorías")

	normalized = sorted(
		[copy.deepcopy(cat) for cat in categories if "id" in cat and "name" in cat],
		key=lambda c: int(c["id"]),
	)

	ids = [int(cat["id"]) for cat in normalized]
	names = [str(cat["name"]).strip() for cat in normalized]

	if len(ids) != len(set(ids)):
		raise ValueError("Hay category_id duplicados en categories")
	if len(names) != len(set([n.lower() for n in names])):
		raise ValueError("Hay nombres de categoría duplicados en categories")

	return normalized


def _validate_categories_consistency(
	reference_categories: List[Dict[str, Any]],
	candidate_categories: List[Dict[str, Any]],
	split_name: str,
) -> None:
	"""Valida que train/valid/test tengan exactamente el mismo catálogo de clases."""
	ref = [(int(cat["id"]), str(cat["name"]).strip().lower()) for cat in reference_categories]
	cnd = [(int(cat["id"]), str(cat["name"]).strip().lower()) for cat in candidate_categories]
	if ref != cnd:
		raise ValueError(
			"Categorías inconsistentes entre splits. "
			f"Split conflictivo: {split_name}. "
			f"reference={ref}, candidate={cnd}"
		)


def _build_category_mappings(
	categories: List[Dict[str, Any]],
	mapping_cfg: DatasetClassMappingConfig,
) -> Dict[str, Any]:
	"""Construye mapeos category_id -> {name, yolo_id, tfrecord_id}."""
	cat_id_to_name: Dict[int, str] = {}
	cat_id_to_yolo_id: Dict[int, int] = {}
	cat_id_to_tfrecord_id: Dict[int, int] = {}

	sorted_categories = _normalize_categories(categories)
	for idx, cat in enumerate(sorted_categories):
		cat_id = int(cat["id"])
		name = str(cat["name"]).strip()
		cat_id_to_name[cat_id] = name

		if mapping_cfg.yolo_zero_based:
			yolo_id = idx
		else:
			yolo_id = idx + 1

		if mapping_cfg.tfrecord_one_based:
			tfrecord_id = idx + 1
		else:
			tfrecord_id = idx

		cat_id_to_yolo_id[cat_id] = int(yolo_id)
		cat_id_to_tfrecord_id[cat_id] = int(tfrecord_id)

	max_yolo_id = max(cat_id_to_yolo_id.values())
	yolo_names = [""] * (max_yolo_id + 1)
	for cat_id, yolo_id in cat_id_to_yolo_id.items():
		yolo_names[yolo_id] = cat_id_to_name[cat_id]

	return {
		"cat_id_to_name": cat_id_to_name,
		"cat_id_to_yolo_id": cat_id_to_yolo_id,
		"cat_id_to_tfrecord_id": cat_id_to_tfrecord_id,
		"yolo_names": yolo_names,
	}


def _sanitize_coco_for_export(coco_data: Dict[str, Any]) -> Dict[str, Any]:
	"""Devuelve una copia COCO con orden estable de imágenes/anotaciones/categorías."""
	clean = {
		"info": copy.deepcopy(coco_data.get("info", {})),
		"licenses": copy.deepcopy(coco_data.get("licenses", [])),
		"categories": _normalize_categories(coco_data.get("categories", [])),
		"images": sorted(
			[copy.deepcopy(img) for img in coco_data.get("images", []) if "id" in img],
			key=lambda i: int(i["id"]),
		),
		"annotations": sorted(
			[copy.deepcopy(ann) for ann in coco_data.get("annotations", []) if "id" in ann],
			key=lambda a: int(a["id"]),
		),
	}
	return clean


def _export_coco_split(
	coco_data: Dict[str, Any],
	source_images_dir: str,
	dest_split_dir: str,
) -> Dict[str, Any]:
	"""Exporta un split en formato COCO estándar (JSON + images/)."""
	ensure_dir(dest_split_dir)
	dest_images_dir = os.path.join(dest_split_dir, "images")
	ensure_dir(dest_images_dir)

	clean = _sanitize_coco_for_export(coco_data)
	copy_stats = _copy_images_for_split(
		images=clean.get("images", []),
		source_images_dir=source_images_dir,
		target_images_dir=dest_images_dir,
	)

	json_path = os.path.join(dest_split_dir, "_annotations.coco.json")
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(clean, f, ensure_ascii=False)

	return {
		"json_path": json_path,
		"images_dir": dest_images_dir,
		"num_images": len(clean.get("images", [])),
		"num_annotations": len(clean.get("annotations", [])),
		"copy_stats": copy_stats,
	}


def _convert_coco_bbox_to_yolo_xywh(
	bbox: Sequence[float],
	img_w: int,
	img_h: int,
) -> Optional[List[float]]:
	"""Convierte bbox COCO [x,y,w,h] a YOLO normalizado [xc,yc,w,h]."""
	if len(bbox) != 4:
		return None
	if img_w <= 0 or img_h <= 0:
		return None

	x, y, w, h = [float(v) for v in bbox]
	if w <= 0 or h <= 0:
		return None

	x1 = max(0.0, min(float(img_w), x))
	y1 = max(0.0, min(float(img_h), y))
	x2 = max(0.0, min(float(img_w), x + w))
	y2 = max(0.0, min(float(img_h), y + h))

	new_w = x2 - x1
	new_h = y2 - y1
	if new_w <= 0.0 or new_h <= 0.0:
		return None

	xc = (x1 + (new_w / 2.0)) / float(img_w)
	yc = (y1 + (new_h / 2.0)) / float(img_h)
	w_n = new_w / float(img_w)
	h_n = new_h / float(img_h)

	vals = [xc, yc, w_n, h_n]
	vals = [max(0.0, min(1.0, float(v))) for v in vals]
	return vals


def _export_yolo_split(
	coco_data: Dict[str, Any],
	source_images_dir: str,
	dest_split_dir: str,
	cat_id_to_yolo_id: Dict[int, int],
	include_empty_labels: bool,
) -> Dict[str, Any]:
	"""Exporta un split COCO a estructura YOLO (images/ + labels/)."""
	ensure_dir(dest_split_dir)
	dest_images_dir = os.path.join(dest_split_dir, "images")
	dest_labels_dir = os.path.join(dest_split_dir, "labels")
	ensure_dir(dest_images_dir)
	ensure_dir(dest_labels_dir)

	clean = _sanitize_coco_for_export(coco_data)
	copy_stats = _copy_images_for_split(
		images=clean.get("images", []),
		source_images_dir=source_images_dir,
		target_images_dir=dest_images_dir,
	)

	anns_by_image = _index_annotations_by_image(clean.get("annotations", []))

	labels_written = 0
	objects_written = 0
	objects_dropped = 0

	for img in clean.get("images", []):
		img_id = int(img.get("id", -1))
		img_w = int(img.get("width", 0))
		img_h = int(img.get("height", 0))
		file_name = str(img.get("file_name", ""))
		stem, _ = os.path.splitext(file_name)
		label_path = os.path.join(dest_labels_dir, f"{stem}.txt")
		label_parent = os.path.dirname(label_path)
		if label_parent:
			ensure_dir(label_parent)

		lines: List[str] = []
		for ann in anns_by_image.get(img_id, []):
			cat_id = int(ann.get("category_id", -1))
			if cat_id not in cat_id_to_yolo_id:
				objects_dropped += 1
				continue

			yolo_bbox = _convert_coco_bbox_to_yolo_xywh(
				bbox=ann.get("bbox", []),
				img_w=img_w,
				img_h=img_h,
			)
			if yolo_bbox is None:
				objects_dropped += 1
				continue

			yolo_id = int(cat_id_to_yolo_id[cat_id])
			lines.append(
				f"{yolo_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
			)
			objects_written += 1

		if lines or include_empty_labels:
			with open(label_path, "w", encoding="utf-8") as f:
				f.write("\n".join(lines))
			labels_written += 1

	return {
		"images_dir": dest_images_dir,
		"labels_dir": dest_labels_dir,
		"num_images": len(clean.get("images", [])),
		"num_annotations": len(clean.get("annotations", [])),
		"labels_written": int(labels_written),
		"objects_written": int(objects_written),
		"objects_dropped": int(objects_dropped),
		"copy_stats": copy_stats,
	}


def _write_yolo_data_yaml(
	yolo_root_dir: str,
	class_names: List[str],
) -> str:
	"""Escribe data.yaml para entrenamiento YOLO con splits train/valid/test."""
	yaml_path = os.path.join(yolo_root_dir, "data.yaml")
	lines = [
		"path: .",
		"train: train/images",
		"val: valid/images",
		"test: test/images",
		f"nc: {len(class_names)}",
		"names:",
	]
	for idx, name in enumerate(class_names):
		lines.append(f"  {idx}: {name}")

	with open(yaml_path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")

	return yaml_path


def _require_tensorflow() -> Any:
	"""Obtiene el módulo tensorflow o lanza ImportError descriptivo."""
	global tf
	if tf is None:
		try:
			import tensorflow as tf_runtime
			tf = tf_runtime
		except Exception as exc:
			raise ImportError(
				"TensorFlow no está disponible en el entorno actual. "
				"Instálalo o activa el entorno correcto antes de exportar TFRecord. "
				f"Detalle: {exc}"
			) from exc
	return tf


def _tf_bytes_feature(value: bytes):
	tf_mod = _require_tensorflow()
	return tf_mod.train.Feature(bytes_list=tf_mod.train.BytesList(value=[value]))


def _tf_bytes_list_feature(values: List[bytes]):
	tf_mod = _require_tensorflow()
	return tf_mod.train.Feature(bytes_list=tf_mod.train.BytesList(value=values))


def _tf_float_list_feature(values: List[float]):
	tf_mod = _require_tensorflow()
	return tf_mod.train.Feature(float_list=tf_mod.train.FloatList(value=values))


def _tf_int64_feature(value: int):
	tf_mod = _require_tensorflow()
	return tf_mod.train.Feature(int64_list=tf_mod.train.Int64List(value=[int(value)]))


def _tf_int64_list_feature(values: List[int]):
	tf_mod = _require_tensorflow()
	return tf_mod.train.Feature(int64_list=tf_mod.train.Int64List(value=[int(v) for v in values]))


def _infer_image_format(file_name: str) -> bytes:
	"""Infiere formato de imagen en bytes para TFExample."""
	ext = os.path.splitext(str(file_name).lower())[1]
	if ext in [".jpg", ".jpeg"]:
		return b"jpeg"
	if ext == ".png":
		return b"png"
	return b"jpeg"


def _convert_coco_bbox_to_xyxy_norm(
	bbox: Sequence[float],
	img_w: int,
	img_h: int,
) -> Optional[Tuple[float, float, float, float]]:
	"""Convierte bbox COCO a coordenadas normalizadas xmin,ymin,xmax,ymax."""
	if len(bbox) != 4 or img_w <= 0 or img_h <= 0:
		return None

	x, y, w, h = [float(v) for v in bbox]
	if w <= 0 or h <= 0:
		return None

	xmin = max(0.0, min(float(img_w), x))
	ymin = max(0.0, min(float(img_h), y))
	xmax = max(0.0, min(float(img_w), x + w))
	ymax = max(0.0, min(float(img_h), y + h))

	if xmax <= xmin or ymax <= ymin:
		return None

	return (
		float(xmin / float(img_w)),
		float(ymin / float(img_h)),
		float(xmax / float(img_w)),
		float(ymax / float(img_h)),
	)


def _build_tf_example_from_coco_image(
	image_info: Dict[str, Any],
	image_annotations: List[Dict[str, Any]],
	image_path: str,
	cat_id_to_name: Dict[int, str],
	cat_id_to_tfrecord_id: Dict[int, int],
) -> Optional[Any]:
	"""Construye un tf.train.Example para una imagen COCO."""
	tf_mod = _require_tensorflow()

	if not os.path.exists(image_path):
		return None

	img_w = int(image_info.get("width", 0))
	img_h = int(image_info.get("height", 0))
	if img_w <= 0 or img_h <= 0:
		return None

	with open(image_path, "rb") as f:
		encoded = f.read()

	file_name = str(image_info.get("file_name", ""))
	image_format = _infer_image_format(file_name)

	xmins: List[float] = []
	ymins: List[float] = []
	xmaxs: List[float] = []
	ymaxs: List[float] = []
	class_texts: List[bytes] = []
	class_labels: List[int] = []
	iscrowd: List[int] = []

	for ann in image_annotations:
		cat_id = int(ann.get("category_id", -1))
		if cat_id not in cat_id_to_tfrecord_id or cat_id not in cat_id_to_name:
			continue

		norm = _convert_coco_bbox_to_xyxy_norm(
			bbox=ann.get("bbox", []),
			img_w=img_w,
			img_h=img_h,
		)
		if norm is None:
			continue

		xmin, ymin, xmax, ymax = norm
		xmins.append(xmin)
		ymins.append(ymin)
		xmaxs.append(xmax)
		ymaxs.append(ymax)
		class_texts.append(str(cat_id_to_name[cat_id]).encode("utf-8"))
		class_labels.append(int(cat_id_to_tfrecord_id[cat_id]))
		iscrowd.append(int(ann.get("iscrowd", 0)))

	features = {
		"image/height": _tf_int64_feature(img_h),
		"image/width": _tf_int64_feature(img_w),
		"image/filename": _tf_bytes_feature(file_name.encode("utf-8")),
		"image/source_id": _tf_bytes_feature(str(image_info.get("id", "")).encode("utf-8")),
		"image/encoded": _tf_bytes_feature(encoded),
		"image/format": _tf_bytes_feature(image_format),
		"image/object/bbox/xmin": _tf_float_list_feature(xmins),
		"image/object/bbox/xmax": _tf_float_list_feature(xmaxs),
		"image/object/bbox/ymin": _tf_float_list_feature(ymins),
		"image/object/bbox/ymax": _tf_float_list_feature(ymaxs),
		"image/object/class/text": _tf_bytes_list_feature(class_texts),
		"image/object/class/label": _tf_int64_list_feature(class_labels),
		"image/object/is_crowd": _tf_int64_list_feature(iscrowd),
	}

	return tf_mod.train.Example(features=tf_mod.train.Features(feature=features))


def _export_tfrecord_split(
	coco_data: Dict[str, Any],
	source_images_dir: str,
	dest_split_dir: str,
	cat_id_to_name: Dict[int, str],
	cat_id_to_tfrecord_id: Dict[int, int],
	num_shards: int,
) -> Dict[str, Any]:
	"""Exporta un split COCO a TFRecord (uno o más shards)."""
	tf_mod = _require_tensorflow()

	ensure_dir(dest_split_dir)
	clean = _sanitize_coco_for_export(coco_data)
	anns_by_image = _index_annotations_by_image(clean.get("annotations", []))

	shards = max(1, int(num_shards))
	writer_paths: List[str] = []
	writers = []
	for shard_idx in range(shards):
		path = os.path.join(dest_split_dir, f"data-{shard_idx:05d}-of-{shards:05d}.tfrecord")
		writer_paths.append(path)
		writers.append(tf_mod.io.TFRecordWriter(path))

	examples_written = 0
	images_missing = 0
	examples_skipped = 0

	try:
		for idx, img in enumerate(clean.get("images", [])):
			file_name = str(img.get("file_name", ""))
			img_path = os.path.join(source_images_dir, file_name)
			if not os.path.exists(img_path):
				images_missing += 1
				continue

			example = _build_tf_example_from_coco_image(
				image_info=img,
				image_annotations=anns_by_image.get(int(img.get("id", -1)), []),
				image_path=img_path,
				cat_id_to_name=cat_id_to_name,
				cat_id_to_tfrecord_id=cat_id_to_tfrecord_id,
			)
			if example is None:
				examples_skipped += 1
				continue

			writer = writers[idx % shards]
			writer.write(example.SerializeToString())
			examples_written += 1
	finally:
		for writer in writers:
			writer.close()

	return {
		"split_dir": dest_split_dir,
		"shards": int(shards),
		"tfrecord_files": writer_paths,
		"num_images": len(clean.get("images", [])),
		"num_annotations": len(clean.get("annotations", [])),
		"examples_written": int(examples_written),
		"examples_skipped": int(examples_skipped),
		"images_missing": int(images_missing),
	}


def _write_label_map_pbtxt(
	label_map_path: str,
	class_names: List[str],
) -> str:
	"""Escribe label_map.pbtxt con IDs 1..N para pipelines TF de detección."""
	lines: List[str] = []
	for idx, name in enumerate(class_names, start=1):
		lines.extend(
			[
				"item {",
				f"  id: {idx}",
				f"  name: '{name}'",
				"}",
			]
		)

	with open(label_map_path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")

	return label_map_path


def export_dataset_all_formats(
	paths_cfg: DatasetExportPathsConfig,
	mapping_cfg: Optional[DatasetClassMappingConfig] = None,
	export_coco: bool = True,
	export_yolo: bool = True,
	export_tfrecord: bool = True,
) -> Dict[str, Any]:
	"""
	Exporta splits COCO (train/valid/test) a COCO, YOLO y TFRecord.

	Crea estructura final:
	  - <output_root>/coco/{train,valid,test}
	  - <output_root>/yolo/{train,valid,test} + data.yaml
	  - <output_root>/tfrecord/{train,valid,test} + label_map.pbtxt + metadata.json
	"""
	map_cfg = mapping_cfg or DatasetClassMappingConfig()

	split_source_dirs = {
		"train": paths_cfg.train_split_dir,
		"valid": paths_cfg.valid_split_dir,
		"test": paths_cfg.test_split_dir,
	}

	loaded_splits: Dict[str, Dict[str, Any]] = {}
	for split_name, split_dir in split_source_dirs.items():
		data, json_path, images_dir = _load_coco_split_from_dir(split_dir)
		loaded_splits[split_name] = {
			"data": data,
			"json_path": json_path,
			"images_dir": images_dir,
		}

	ref_categories = _normalize_categories(loaded_splits["train"]["data"].get("categories", []))
	for split_name in ["valid", "test"]:
		candidate = _normalize_categories(loaded_splits[split_name]["data"].get("categories", []))
		_validate_categories_consistency(ref_categories, candidate, split_name=split_name)

	mappings = _build_category_mappings(ref_categories, map_cfg)
	cat_id_to_name = mappings["cat_id_to_name"]
	cat_id_to_yolo_id = mappings["cat_id_to_yolo_id"]
	cat_id_to_tfrecord_id = mappings["cat_id_to_tfrecord_id"]
	yolo_names = mappings["yolo_names"]

	ensure_dir(paths_cfg.output_root_dir)

	summary: Dict[str, Any] = {
		"output_root_dir": paths_cfg.output_root_dir,
		"source_splits": split_source_dirs,
		"class_mapping": {
			"categories": [
				{
					"coco_id": int(cat_id),
					"name": str(cat_id_to_name[cat_id]),
					"yolo_id": int(cat_id_to_yolo_id[cat_id]),
					"tfrecord_id": int(cat_id_to_tfrecord_id[cat_id]),
				}
				for cat_id in sorted(cat_id_to_name.keys())
			]
		},
		"formats": {},
	}

	if export_coco:
		coco_root = os.path.join(paths_cfg.output_root_dir, "coco")
		ensure_dir(coco_root)
		coco_summary: Dict[str, Any] = {"root_dir": coco_root, "splits": {}}

		for split_name in ["train", "valid", "test"]:
			dest_split_dir = os.path.join(coco_root, split_name)
			split_info = loaded_splits[split_name]
			coco_summary["splits"][split_name] = _export_coco_split(
				coco_data=split_info["data"],
				source_images_dir=split_info["images_dir"],
				dest_split_dir=dest_split_dir,
			)

		summary["formats"]["coco"] = coco_summary

	if export_yolo:
		yolo_root = os.path.join(paths_cfg.output_root_dir, "yolo")
		ensure_dir(yolo_root)
		yolo_summary: Dict[str, Any] = {"root_dir": yolo_root, "splits": {}}

		for split_name in ["train", "valid", "test"]:
			dest_split_dir = os.path.join(yolo_root, split_name)
			split_info = loaded_splits[split_name]
			yolo_summary["splits"][split_name] = _export_yolo_split(
				coco_data=split_info["data"],
				source_images_dir=split_info["images_dir"],
				dest_split_dir=dest_split_dir,
				cat_id_to_yolo_id=cat_id_to_yolo_id,
				include_empty_labels=bool(paths_cfg.include_empty_yolo_labels),
			)

		yolo_summary["data_yaml"] = _write_yolo_data_yaml(
			yolo_root_dir=yolo_root,
			class_names=yolo_names,
		)
		summary["formats"]["yolo"] = yolo_summary

	if export_tfrecord:
		tfrecord_root = os.path.join(paths_cfg.output_root_dir, "tfrecord")
		ensure_dir(tfrecord_root)
		tfrecord_summary: Dict[str, Any] = {"root_dir": tfrecord_root, "splits": {}}

		for split_name in ["train", "valid", "test"]:
			dest_split_dir = os.path.join(tfrecord_root, split_name)
			split_info = loaded_splits[split_name]
			tfrecord_summary["splits"][split_name] = _export_tfrecord_split(
				coco_data=split_info["data"],
				source_images_dir=split_info["images_dir"],
				dest_split_dir=dest_split_dir,
				cat_id_to_name=cat_id_to_name,
				cat_id_to_tfrecord_id=cat_id_to_tfrecord_id,
				num_shards=int(paths_cfg.tfrecord_shards),
			)

		label_map_path = os.path.join(tfrecord_root, "label_map.pbtxt")
		tfrecord_summary["label_map_path"] = _write_label_map_pbtxt(
			label_map_path=label_map_path,
			class_names=yolo_names,
		)

		metadata_path = os.path.join(tfrecord_root, "metadata.json")
		metadata_payload = {
			"format": "tfrecord",
			"splits": tfrecord_summary["splits"],
			"class_mapping": summary["class_mapping"],
		}
		with open(metadata_path, "w", encoding="utf-8") as f:
			json.dump(metadata_payload, f, ensure_ascii=False, indent=2)
		tfrecord_summary["metadata_path"] = metadata_path

		summary["formats"]["tfrecord"] = tfrecord_summary

	return summary


def print_dataset_export_summary(summary: Dict[str, Any]) -> None:
	"""Imprime resumen legible de exportación multi-formato."""
	print("\n✅ Exportación de dataset completada")
	print(f"   📦 Output root: {summary.get('output_root_dir')}")

	class_mapping = summary.get("class_mapping", {}).get("categories", [])
	if class_mapping:
		print("   🧭 Mapping de clases:")
		for row in class_mapping:
			print(
				"      "
				f"COCO[{row['coco_id']}]={row['name']} -> "
				f"YOLO[{row['yolo_id']}], TF[{row['tfrecord_id']}]"
			)

	formats = summary.get("formats", {})
	for fmt_name in ["coco", "yolo", "tfrecord"]:
		if fmt_name not in formats:
			continue
		fmt = formats[fmt_name]
		print(f"\n   📁 Formato {fmt_name.upper()}: {fmt.get('root_dir')}")
		splits = fmt.get("splits", {})
		for split_name in ["train", "valid", "test"]:
			if split_name not in splits:
				continue
			info = splits[split_name]
			if fmt_name == "tfrecord":
				print(
					f"      - {split_name}: "
					f"examples={info.get('examples_written', 0)}, "
					f"shards={info.get('shards', 0)}"
				)
			elif fmt_name == "yolo":
				print(
					f"      - {split_name}: "
					f"images={info.get('num_images', 0)}, "
					f"labels={info.get('labels_written', 0)}, "
					f"objects={info.get('objects_written', 0)}"
				)
			else:
				print(
					f"      - {split_name}: "
					f"images={info.get('num_images', 0)}, "
					f"annotations={info.get('num_annotations', 0)}"
				)

	if "yolo" in formats and formats["yolo"].get("data_yaml"):
		print(f"\n   📝 YOLO data.yaml: {formats['yolo']['data_yaml']}")
	if "tfrecord" in formats:
		label_map = formats["tfrecord"].get("label_map_path")
		metadata = formats["tfrecord"].get("metadata_path")
		if label_map:
			print(f"   📝 TF label_map: {label_map}")
		if metadata:
			print(f"   📝 TF metadata: {metadata}")


def _safe_read_json(path: str) -> Optional[Dict[str, Any]]:
	"""Lee JSON de forma segura devolviendo None si falla."""
	if not os.path.exists(path):
		return None
	try:
		with open(path, "r", encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return None


def _list_files_recursive(root_dir: str, suffixes: Tuple[str, ...]) -> List[str]:
	"""Lista archivos recursivamente filtrando por sufijo."""
	if not os.path.isdir(root_dir):
		return []
	acc: List[str] = []
	for current_root, _, files in os.walk(root_dir):
		for name in files:
			if name.lower().endswith(suffixes):
				acc.append(os.path.join(current_root, name))
	return sorted(acc)


def _count_dir_images(images_dir: str) -> int:
	"""Cuenta imágenes en un directorio recursivo."""
	return len(_list_files_recursive(images_dir, (".jpg", ".jpeg", ".png", ".bmp", ".webp")))


def _validate_coco_export_block(coco_block: Dict[str, Any]) -> Dict[str, Any]:
	"""Sanity checks del formato COCO exportado."""
	result: Dict[str, Any] = {"status": "PASS", "splits": {}, "errors": []}

	for split_name in ["train", "valid", "test"]:
		split_info = coco_block.get("splits", {}).get(split_name, {})
		json_path = split_info.get("json_path")
		images_dir = split_info.get("images_dir")

		split_result = {
			"json_exists": bool(json_path and os.path.exists(json_path)),
			"images_dir_exists": bool(images_dir and os.path.isdir(images_dir)),
			"json_num_images": 0,
			"json_num_annotations": 0,
			"disk_num_images": 0,
			"expected_num_images": int(split_info.get("num_images", 0)),
			"expected_num_annotations": int(split_info.get("num_annotations", 0)),
			"checks": {},
		}

		data = _safe_read_json(str(json_path)) if json_path else None
		if data is not None:
			split_result["json_num_images"] = int(len(data.get("images", [])))
			split_result["json_num_annotations"] = int(len(data.get("annotations", [])))

		split_result["disk_num_images"] = _count_dir_images(str(images_dir)) if images_dir else 0

		split_result["checks"] = {
			"json_readable": data is not None,
			"images_count_match_json_vs_expected": split_result["json_num_images"] == split_result["expected_num_images"],
			"annotations_count_match_json_vs_expected": split_result["json_num_annotations"] == split_result["expected_num_annotations"],
			"images_count_match_disk_vs_expected": split_result["disk_num_images"] == split_result["expected_num_images"],
		}

		if not all(split_result["checks"].values()):
			result["status"] = "FAIL"
			result["errors"].append(f"COCO split '{split_name}' con inconsistencias")

		result["splits"][split_name] = split_result

	return result


def _parse_yolo_label_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
	"""Parsea una línea YOLO: class xc yc w h."""
	parts = line.strip().split()
	if len(parts) != 5:
		return None
	try:
		cls = int(parts[0])
		xc = float(parts[1])
		yc = float(parts[2])
		w = float(parts[3])
		h = float(parts[4])
		return cls, xc, yc, w, h
	except Exception:
		return None


def _validate_yolo_export_block(yolo_block: Dict[str, Any], class_count: int) -> Dict[str, Any]:
	"""Sanity checks del formato YOLO exportado."""
	result: Dict[str, Any] = {"status": "PASS", "splits": {}, "errors": []}

	data_yaml = yolo_block.get("data_yaml")
	result["data_yaml_exists"] = bool(data_yaml and os.path.exists(data_yaml))
	if not result["data_yaml_exists"]:
		result["status"] = "FAIL"
		result["errors"].append("Falta data.yaml en export YOLO")

	for split_name in ["train", "valid", "test"]:
		split_info = yolo_block.get("splits", {}).get(split_name, {})
		images_dir = split_info.get("images_dir")
		labels_dir = split_info.get("labels_dir")

		image_files = _list_files_recursive(str(images_dir), (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
		label_files = _list_files_recursive(str(labels_dir), (".txt",))

		expected_num_images = int(split_info.get("num_images", 0))
		expected_labels = int(split_info.get("labels_written", 0))

		invalid_lines = 0
		out_of_range_values = 0
		invalid_class_ids = 0
		parsed_objects = 0

		for label_path in label_files:
			try:
				with open(label_path, "r", encoding="utf-8") as f:
					lines = [ln.strip() for ln in f.readlines() if ln.strip()]
			except Exception:
				invalid_lines += 1
				continue

			for line in lines:
				parsed = _parse_yolo_label_line(line)
				if parsed is None:
					invalid_lines += 1
					continue

				cls, xc, yc, w, h = parsed
				parsed_objects += 1
				if cls < 0 or cls >= int(class_count):
					invalid_class_ids += 1
				if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
					out_of_range_values += 1

		split_result = {
			"expected_num_images": expected_num_images,
			"expected_num_labels": expected_labels,
			"disk_num_images": len(image_files),
			"disk_num_labels": len(label_files),
			"parsed_objects": int(parsed_objects),
			"invalid_lines": int(invalid_lines),
			"invalid_class_ids": int(invalid_class_ids),
			"out_of_range_values": int(out_of_range_values),
			"checks": {
				"images_count_match": len(image_files) == expected_num_images,
				"labels_count_match": len(label_files) == expected_labels,
				"all_label_lines_valid": invalid_lines == 0,
				"class_ids_valid": invalid_class_ids == 0,
				"bbox_values_in_range": out_of_range_values == 0,
			},
		}

		if not all(split_result["checks"].values()):
			result["status"] = "FAIL"
			result["errors"].append(f"YOLO split '{split_name}' con inconsistencias")

		result["splits"][split_name] = split_result

	return result


def _validate_tfrecord_export_block(tfrecord_block: Dict[str, Any]) -> Dict[str, Any]:
	"""Sanity checks del formato TFRecord exportado."""
	result: Dict[str, Any] = {"status": "PASS", "splits": {}, "errors": []}

	label_map_path = tfrecord_block.get("label_map_path")
	metadata_path = tfrecord_block.get("metadata_path")
	result["label_map_exists"] = bool(label_map_path and os.path.exists(label_map_path))
	result["metadata_exists"] = bool(metadata_path and os.path.exists(metadata_path))

	if not result["label_map_exists"]:
		result["status"] = "FAIL"
		result["errors"].append("Falta label_map.pbtxt en export TFRecord")
	if not result["metadata_exists"]:
		result["status"] = "FAIL"
		result["errors"].append("Falta metadata.json en export TFRecord")

	tf_import_error: Optional[str] = None
	tf_mod = None
	try:
		tf_mod = _require_tensorflow()
	except Exception as exc:
		tf_import_error = str(exc)

	for split_name in ["train", "valid", "test"]:
		split_info = tfrecord_block.get("splits", {}).get(split_name, {})
		tfrecord_files = [p for p in split_info.get("tfrecord_files", []) if os.path.exists(p)]
		expected_examples = int(split_info.get("examples_written", 0))

		examples_probe = 0
		parse_errors = 0
		probed_files = 0

		if tf_mod is not None:
			for tf_path in tfrecord_files:
				probed_files += 1
				try:
					ds = tf_mod.data.TFRecordDataset([tf_path])
					for raw in ds.take(3):
						ex = tf_mod.train.Example()
						try:
							ex.ParseFromString(raw.numpy())
							examples_probe += 1
						except Exception:
							parse_errors += 1
				except Exception:
					parse_errors += 1

		split_result = {
			"expected_examples": expected_examples,
			"num_tfrecord_files": len(tfrecord_files),
			"probed_files": int(probed_files),
			"probe_examples_parsed": int(examples_probe),
			"probe_parse_errors": int(parse_errors),
			"tf_available_for_probe": tf_mod is not None,
			"checks": {
				"has_tfrecord_files": len(tfrecord_files) > 0,
				"files_non_empty": all(os.path.getsize(p) > 0 for p in tfrecord_files) if tfrecord_files else False,
				"probe_no_parse_errors": parse_errors == 0 if tf_mod is not None else True,
			},
		}

		if not all(split_result["checks"].values()):
			result["status"] = "FAIL"
			result["errors"].append(f"TFRecord split '{split_name}' con inconsistencias")

		result["splits"][split_name] = split_result

	if tf_import_error is not None:
		result["warnings"] = [
			"No fue posible hacer probe de parseo TFRecord con TensorFlow en este entorno.",
			tf_import_error,
		]

	return result


def _get_notebook_consistent_color_pool() -> List[Tuple[int, int, int]]:
	"""Paleta BGR consistente con celdas de visualización del notebook."""
	return [
		(255, 0, 0),
		(0, 255, 0),
		(0, 0, 255),
		(255, 255, 0),
		(255, 0, 255),
		(0, 255, 255),
	]


def _build_export_color_maps(class_mapping: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""Construye mapeos de color y nombre por ID de clase en cada formato."""
	pool = _get_notebook_consistent_color_pool()

	coco_id_to_color: Dict[int, Tuple[int, int, int]] = {}
	yolo_id_to_color: Dict[int, Tuple[int, int, int]] = {}
	tf_id_to_color: Dict[int, Tuple[int, int, int]] = {}
	coco_id_to_name: Dict[int, str] = {}
	yolo_id_to_name: Dict[int, str] = {}
	tf_id_to_name: Dict[int, str] = {}

	for idx, row in enumerate(class_mapping):
		color = pool[idx % len(pool)]
		name = str(row.get("name", "unknown"))
		coco_id = int(row.get("coco_id", -1))
		yolo_id = int(row.get("yolo_id", -1))
		tf_id = int(row.get("tfrecord_id", -1))

		if coco_id >= 0:
			coco_id_to_color[coco_id] = color
			coco_id_to_name[coco_id] = name
		if yolo_id >= 0:
			yolo_id_to_color[yolo_id] = color
			yolo_id_to_name[yolo_id] = name
		if tf_id >= 0:
			tf_id_to_color[tf_id] = color
			tf_id_to_name[tf_id] = name

	return {
		"coco_id_to_color": coco_id_to_color,
		"yolo_id_to_color": yolo_id_to_color,
		"tf_id_to_color": tf_id_to_color,
		"coco_id_to_name": coco_id_to_name,
		"yolo_id_to_name": yolo_id_to_name,
		"tf_id_to_name": tf_id_to_name,
	}


def _draw_boxes_with_labels(
	image_bgr: np.ndarray,
	boxes: List[Dict[str, Any]],
	default_color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
	"""Dibuja bboxes con etiqueta de clase sobre imagen BGR."""
	out = image_bgr.copy()
	for item in boxes:
		x1 = int(round(float(item.get("x1", 0))))
		y1 = int(round(float(item.get("y1", 0))))
		x2 = int(round(float(item.get("x2", 0))))
		y2 = int(round(float(item.get("y2", 0))))
		if x2 <= x1 or y2 <= y1:
			continue

		color = item.get("color", default_color)
		label = str(item.get("label", "obj"))

		cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

		(text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
		text_y1 = max(0, y1 - text_h - baseline - 4)
		text_y2 = max(0, y1)
		text_x2 = min(out.shape[1] - 1, x1 + text_w + 6)
		cv2.rectangle(out, (x1, text_y1), (text_x2, text_y2), color, -1)
		cv2.putText(
			out,
			label,
			(x1 + 3, max(text_h + 1, text_y2 - 4)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			(255, 255, 255),
			1,
			cv2.LINE_AA,
		)

	return out


def _choose_common_eval_images(
	coco_block: Dict[str, Any],
	seed: Optional[int],
) -> Dict[str, Dict[str, Any]]:
	"""Selecciona 1 imagen por split (train/valid/test) para evaluación comparada."""
	rng = random.Random(seed) if seed is not None else random.SystemRandom()
	chosen: Dict[str, Dict[str, Any]] = {}

	for split_name in ["train", "valid", "test"]:
		split_info = coco_block.get("splits", {}).get(split_name, {})
		json_path = split_info.get("json_path")
		data = _safe_read_json(str(json_path)) if json_path else None
		if data is None:
			continue

		pool = [img for img in data.get("images", []) if str(img.get("file_name", "")).strip()]
		if not pool:
			continue

		selected = rng.choice(pool)
		chosen[split_name] = {
			"id": int(selected.get("id", -1)),
			"file_name": str(selected.get("file_name", "")),
			"width": int(selected.get("width", 0)),
			"height": int(selected.get("height", 0)),
		}

	return chosen


def _coco_boxes_for_image(
	json_path: str,
	image_id: int,
	color_maps: Dict[str, Any],
) -> List[Dict[str, Any]]:
	"""Obtiene cajas desde COCO para una imagen dada."""
	data = _safe_read_json(json_path)
	if data is None:
		return []

	anns = [ann for ann in data.get("annotations", []) if int(ann.get("image_id", -1)) == int(image_id)]
	boxes: List[Dict[str, Any]] = []
	for ann in anns:
		bbox = ann.get("bbox", [])
		if not isinstance(bbox, list) or len(bbox) != 4:
			continue
		x, y, w, h = [float(v) for v in bbox]
		if w <= 0 or h <= 0:
			continue
		cid = int(ann.get("category_id", -1))
		boxes.append(
			{
				"x1": x,
				"y1": y,
				"x2": x + w,
				"y2": y + h,
				"label": str(color_maps["coco_id_to_name"].get(cid, f"class_{cid}")),
				"color": color_maps["coco_id_to_color"].get(cid, (0, 255, 0)),
			}
		)
	return boxes


def _yolo_boxes_for_image(
	labels_dir: str,
	file_name: str,
	img_w: int,
	img_h: int,
	color_maps: Dict[str, Any],
) -> List[Dict[str, Any]]:
	"""Obtiene cajas desde archivo YOLO .txt para una imagen dada."""
	stem, _ = os.path.splitext(file_name)
	label_path = os.path.join(labels_dir, f"{stem}.txt")
	if not os.path.exists(label_path):
		return []

	boxes: List[Dict[str, Any]] = []
	with open(label_path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			parsed = _parse_yolo_label_line(line)
			if parsed is None:
				continue
			cls, xc, yc, w_n, h_n = parsed
			x1 = (xc - (w_n / 2.0)) * img_w
			y1 = (yc - (h_n / 2.0)) * img_h
			x2 = (xc + (w_n / 2.0)) * img_w
			y2 = (yc + (h_n / 2.0)) * img_h

			boxes.append(
				{
					"x1": x1,
					"y1": y1,
					"x2": x2,
					"y2": y2,
					"label": str(color_maps["yolo_id_to_name"].get(int(cls), f"class_{cls}")),
					"color": color_maps["yolo_id_to_color"].get(int(cls), (0, 255, 0)),
				}
			)

	return boxes


def _tfrecord_examples_by_filename(
	tfrecord_files: List[str],
	selected_filenames: Set[str],
) -> Dict[str, Dict[str, Any]]:
	"""Extrae ejemplos TFRecord por filename para un subconjunto objetivo."""
	tf_mod = _require_tensorflow()
	selected = {str(x) for x in selected_filenames}
	result: Dict[str, Dict[str, Any]] = {}
	if not selected:
		return result

	for tf_path in tfrecord_files:
		if not os.path.exists(tf_path):
			continue
		try:
			ds = tf_mod.data.TFRecordDataset([tf_path])
			for raw in ds:
				ex = tf_mod.train.Example()
				ex.ParseFromString(raw.numpy())
				feat = ex.features.feature

				fname_bytes = feat.get("image/filename").bytes_list.value
				if not fname_bytes:
					continue
				file_name = fname_bytes[0].decode("utf-8")
				if file_name not in selected:
					continue

				encoded_list = feat.get("image/encoded").bytes_list.value
				image_bytes = encoded_list[0] if encoded_list else b""

				result[file_name] = {
					"width": int(feat.get("image/width").int64_list.value[0]) if feat.get("image/width").int64_list.value else 0,
					"height": int(feat.get("image/height").int64_list.value[0]) if feat.get("image/height").int64_list.value else 0,
					"xmins": [float(v) for v in feat.get("image/object/bbox/xmin").float_list.value],
					"xmaxs": [float(v) for v in feat.get("image/object/bbox/xmax").float_list.value],
					"ymins": [float(v) for v in feat.get("image/object/bbox/ymin").float_list.value],
					"ymaxs": [float(v) for v in feat.get("image/object/bbox/ymax").float_list.value],
					"class_ids": [int(v) for v in feat.get("image/object/class/label").int64_list.value],
					"image_bytes": image_bytes,
				}

				if len(result) == len(selected):
					return result
		except Exception:
			continue

	return result


def _decode_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
	"""Decodifica bytes de imagen a BGR (OpenCV)."""
	if not image_bytes:
		return None
	arr = np.frombuffer(image_bytes, dtype=np.uint8)
	img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
	return img


def _tfrecord_boxes_for_image(
	tf_example: Dict[str, Any],
	color_maps: Dict[str, Any],
) -> List[Dict[str, Any]]:
	"""Convierte cajas normalizadas TFRecord a coordenadas absolutas."""
	img_w = int(tf_example.get("width", 0))
	img_h = int(tf_example.get("height", 0))
	xmins = tf_example.get("xmins", [])
	xmaxs = tf_example.get("xmaxs", [])
	ymins = tf_example.get("ymins", [])
	ymaxs = tf_example.get("ymaxs", [])
	class_ids = tf_example.get("class_ids", [])

	n = min(len(xmins), len(xmaxs), len(ymins), len(ymaxs), len(class_ids))
	boxes: List[Dict[str, Any]] = []
	for i in range(n):
		cid = int(class_ids[i])
		x1 = float(xmins[i]) * img_w
		x2 = float(xmaxs[i]) * img_w
		y1 = float(ymins[i]) * img_h
		y2 = float(ymaxs[i]) * img_h
		boxes.append(
			{
				"x1": x1,
				"y1": y1,
				"x2": x2,
				"y2": y2,
				"label": str(color_maps["tf_id_to_name"].get(cid, f"class_{cid}")),
				"color": color_maps["tf_id_to_color"].get(cid, (0, 255, 0)),
			}
		)

	return boxes


def _render_cross_format_visual_check(
	export_summary: Dict[str, Any],
	selected_images: Dict[str, Dict[str, Any]],
	color_maps: Dict[str, Any],
	show_plot: bool,
	save_visual_samples: bool,
	visual_output_dir: Optional[str],
) -> Dict[str, Any]:
	"""Renderiza figura comparativa: filas=formato, columnas=split (misma imagen por split)."""
	formats = export_summary.get("formats", {})
	available_formats = [fmt for fmt in ["coco", "yolo", "tfrecord"] if fmt in formats]
	if not available_formats:
		return {"status": "SKIP", "reason": "No hay formatos disponibles para render visual"}

	splits = [s for s in ["train", "valid", "test"] if s in selected_images]
	if not splits:
		return {"status": "SKIP", "reason": "No hay imágenes seleccionadas para visualización"}

	rows = len(available_formats)
	cols = len(splits)

	tf_examples_by_split: Dict[str, Dict[str, Dict[str, Any]]] = {}
	if "tfrecord" in available_formats:
		tf_block = formats.get("tfrecord", {})
		for split_name in splits:
			tf_files = tf_block.get("splits", {}).get(split_name, {}).get("tfrecord_files", [])
			fname = selected_images[split_name]["file_name"]
			tf_examples_by_split[split_name] = _tfrecord_examples_by_filename(tf_files, {fname})

	if show_plot:
		fig = plt.figure(figsize=(5 * cols, 5.5 * rows), constrained_layout=True)
		fig.suptitle(
			"Validación visual cruzada por formato (mismas imágenes train/valid/test)",
			fontsize=14,
			fontweight="bold",
		)
		subfigs = fig.subfigures(rows, 1, hspace=0.06)
		if rows == 1:
			subfigs = [subfigs]
		axes = np.empty((rows, cols), dtype=object)
		for r, fmt_name in enumerate(available_formats):
			subfigs[r].suptitle(f"Formato: {fmt_name.upper()}", fontsize=12, fontweight="bold")
			row_axes = subfigs[r].subplots(1, cols)
			if cols == 1:
				row_axes = np.array([row_axes])
			for c in range(cols):
				axes[r, c] = row_axes[c]

	saved_paths: List[str] = []

	for r, fmt_name in enumerate(available_formats):
		for c, split_name in enumerate(splits):
			ax = axes[r, c] if show_plot else None
			selected = selected_images[split_name]
			image_id = int(selected["id"])
			file_name = str(selected["file_name"])

			image_bgr = None
			boxes: List[Dict[str, Any]] = []

			if fmt_name == "coco":
				split_info = formats["coco"]["splits"][split_name]
				img_path = os.path.join(split_info["images_dir"], file_name)
				image_bgr = cv2.imread(img_path)
				boxes = _coco_boxes_for_image(split_info["json_path"], image_id, color_maps)

			elif fmt_name == "yolo":
				split_info = formats["yolo"]["splits"][split_name]
				img_path = os.path.join(split_info["images_dir"], file_name)
				image_bgr = cv2.imread(img_path)
				if image_bgr is not None:
					img_h, img_w = image_bgr.shape[:2]
					boxes = _yolo_boxes_for_image(
						labels_dir=split_info["labels_dir"],
						file_name=file_name,
						img_w=img_w,
						img_h=img_h,
						color_maps=color_maps,
					)

			elif fmt_name == "tfrecord":
				tf_example = tf_examples_by_split.get(split_name, {}).get(file_name)
				if tf_example is not None:
					image_bgr = _decode_image_from_bytes(tf_example.get("image_bytes", b""))
					boxes = _tfrecord_boxes_for_image(tf_example, color_maps)
				if image_bgr is None:
					# fallback visual a imagen COCO del mismo split
					coco_split_info = formats.get("coco", {}).get("splits", {}).get(split_name, {})
					if coco_split_info:
						image_bgr = cv2.imread(os.path.join(coco_split_info.get("images_dir", ""), file_name))

			if image_bgr is None:
				if show_plot:
					ax.set_title(f"{split_name} | missing")
					ax.axis("off")
				continue

			overlay = _draw_boxes_with_labels(image_bgr=image_bgr, boxes=boxes)
			overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

			if show_plot:
				ax.imshow(overlay_rgb)
				ax.set_title(f"{split_name} | id={image_id} | {file_name} | anns={len(boxes)}", fontsize=10)
				ax.axis("off")

			if save_visual_samples:
				root = visual_output_dir or os.path.join(export_summary.get("output_root_dir", "."), "validation_samples")
				fmt_dir = os.path.join(root, fmt_name)
				ensure_dir(fmt_dir)
				save_path = os.path.join(fmt_dir, f"{split_name}_id{image_id}_{file_name}")
				cv2.imwrite(save_path, overlay)
				saved_paths.append(save_path)

	if show_plot:
		plt.show()

	return {
		"status": "PASS",
		"selected_images": selected_images,
		"formats_displayed": available_formats,
		"splits_displayed": splits,
		"saved_paths": saved_paths,
	}


def validate_exported_dataset_postcheck(
	export_summary: Dict[str, Any],
	samples_per_split: int = 1,
	seed: Optional[int] = None,
	show_visual_samples: bool = True,
	save_visual_samples: bool = False,
	visual_output_dir: Optional[str] = None,
) -> Dict[str, Any]:
	"""
	Valida automáticamente el dataset exportado en COCO/YOLO/TFRecord.

	Incluye:
	  - sanity checks estructurales por formato,
	  - consistencia de conteos vs resumen de exportación,
	  - validación de contenido YOLO,
	  - probe de lectura TFRecord,
	  - muestreo visual con bboxes desde COCO exportado.
	"""
	if not export_summary:
		raise ValueError("export_summary está vacío. Ejecuta primero export_dataset_all_formats(...)")

	class_mapping = export_summary.get("class_mapping", {}).get("categories", [])
	class_count = len(class_mapping)

	formats = export_summary.get("formats", {})
	report: Dict[str, Any] = {
		"status": "PASS",
		"formats": {},
		"visual_sampling": {},
		"errors": [],
	}

	if "coco" in formats:
		coco_report = _validate_coco_export_block(formats["coco"])
		report["formats"]["coco"] = coco_report
		if coco_report.get("status") != "PASS":
			report["status"] = "FAIL"
			report["errors"].extend(coco_report.get("errors", []))

	if "yolo" in formats:
		yolo_report = _validate_yolo_export_block(formats["yolo"], class_count=class_count)
		report["formats"]["yolo"] = yolo_report
		if yolo_report.get("status") != "PASS":
			report["status"] = "FAIL"
			report["errors"].extend(yolo_report.get("errors", []))

	if "tfrecord" in formats:
		tf_report = _validate_tfrecord_export_block(formats["tfrecord"])
		report["formats"]["tfrecord"] = tf_report
		if tf_report.get("status") != "PASS":
			report["status"] = "FAIL"
			report["errors"].extend(tf_report.get("errors", []))

	if show_visual_samples or save_visual_samples:
		if int(samples_per_split) != 1:
			print("   ℹ️ samples_per_split distinto de 1 detectado; se fuerza a 1 para comparación cruzada por formato.")

		coco_block = formats.get("coco", {})
		selected_images = _choose_common_eval_images(
			coco_block=coco_block,
			seed=seed,
		)
		color_maps = _build_export_color_maps(class_mapping=class_mapping)

		report["visual_sampling"] = _render_cross_format_visual_check(
			export_summary=export_summary,
			selected_images=selected_images,
			color_maps=color_maps,
			show_plot=bool(show_visual_samples),
			save_visual_samples=bool(save_visual_samples),
			visual_output_dir=visual_output_dir,
		)

	print("\n🧪 Validación post-export")
	print(f"   Estado global: {report['status']}")
	for fmt_name, fmt_report in report.get("formats", {}).items():
		print(f"   - {fmt_name.upper()}: {fmt_report.get('status', 'UNKNOWN')}")

	if report.get("errors"):
		print("   ⚠️ Errores detectados:")
		for err in report["errors"]:
			print(f"      • {err}")
	else:
		print("   ✅ Sin inconsistencias críticas en los checks ejecutados")

	return report



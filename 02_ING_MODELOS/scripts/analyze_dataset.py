#!/usr/bin/env python3
"""Analyze dataset statistics for footpath and obstacle classes."""

import json
from pathlib import Path
from collections import Counter
import statistics

target_classes = {'footpath', 'obstacle'}

train_path = Path('../01_ING_DATOS/Dataset/train/augmented2_images/train_final2.json')
val_path = Path('../01_ING_DATOS/Dataset/valid/val_final.json')
test_path = Path('../01_ING_DATOS/Dataset/test/test_final.json')

for split, path in [('train', train_path), ('val', val_path), ('test', test_path)]:
    with open(path) as f:
        data = json.load(f)
    
    cat_id_to_name = {c['id']: c['name'] for c in data['categories']}
    target_cat_ids = {cid for cid, name in cat_id_to_name.items() if name in target_classes}
    
    filtered_anns = [ann for ann in data['annotations'] if ann['category_id'] in target_cat_ids]
    
    per_class_areas = {name: [] for name in target_classes}
    per_class_ratios = {name: [] for name in target_classes}
    
    for ann in filtered_anns:
        name = cat_id_to_name[ann['category_id']]
        x, y, w, h = ann['bbox']
        per_class_areas[name].append(w * h)
        if h > 0:
            per_class_ratios[name].append(w / h)
    
    print(f'\n=== {split.upper()} SET (footpath + obstacle) ===')
    print(f'Total annotations: {len(filtered_anns)}')
    
    for cls_name in ['footpath', 'obstacle']:
        areas = per_class_areas[cls_name]
        ratios = per_class_ratios[cls_name]
        if areas:
            small = sum(1 for a in areas if a < 32*32)
            medium = sum(1 for a in areas if 32*32 <= a < 96*96)
            large = sum(1 for a in areas if a >= 96*96)
            print(f'  {cls_name}: {len(areas)} annotations')
            print(f'    Sizes: S={small}({100*small/len(areas):.0f}%), M={medium}({100*medium/len(areas):.0f}%), L={large}({100*large/len(areas):.0f}%)')
            print(f'    Area: min={min(areas):.0f}, max={max(areas):.0f}, med={statistics.median(areas):.0f}')
            print(f'    Aspect(w/h): min={min(ratios):.2f}, max={max(ratios):.2f}, med={statistics.median(ratios):.2f}')

# Class imbalance ratio
train_data = json.load(open(train_path))
cat_map = {c['id']: c['name'] for c in train_data['categories']}
counts = Counter(cat_map[a['category_id']] for a in train_data['annotations'] if cat_map[a['category_id']] in target_classes)
print(f'\n=== CLASS IMBALANCE ===')
print(f'footpath: {counts["footpath"]}')
print(f'obstacle: {counts["obstacle"]}')
print(f'Ratio obstacle/footpath: {counts["obstacle"]/counts["footpath"]:.2f}x')

import json, os

base = os.path.dirname(__file__)

# MobileNet
with open(os.path.join(base, 'MBNTv3S_ssdlite_v1/test_evaluation.json')) as f:
    te = json.load(f)
cm = te['confusion_matrix']
classes = te['class_names'] + ['background']

print("=" * 60)
print("MBNTv3S_ssdlite_v1 — Test Evaluation Analysis")
print("=" * 60)
print(f"n_ground_truths: {te['n_ground_truths']}")
print(f"n_detections:    {te['n_detections']}")
print(f"n_images:        {te['n_images']}")
print(f"GT per image:    {te['n_ground_truths'] / max(te['n_images'],1):.1f}")
print()

print("Confusion matrix row sums (GT per class):")
for i, row in enumerate(cm):
    print(f"  {classes[i]:12s}: {sum(row):.0f}")

print()
print("Confusion matrix col sums (pred per class):")
for j in range(len(cm[0])):
    col_sum = sum(cm[i][j] for i in range(len(cm)))
    print(f"  {classes[j]:12s}: {col_sum:.0f}")

print()
print("True positives (diagonal):")
for i in range(5):
    total_gt = sum(cm[i])
    print(f"  {classes[i]:12s}: TP={cm[i][i]:.0f}  FN(bg)={cm[i][5]:.0f}  GT_row={total_gt:.0f}")

# YOLO comparison
print()
print("=" * 60)
print("YOLO26n_v1 — Test Evaluation (for comparison)")
print("=" * 60)
with open(os.path.join(base, 'yolo26n_v1/test_evaluation.json')) as f:
    ye = json.load(f)
print(f"n_ground_truths: {ye['n_ground_truths']}")
print(f"n_detections:    {ye['n_detections']}")

print()
print("=" * 60)
print("Side-by-side comparison")
print("=" * 60)
print(f"{'Metric':<25s} {'YOLO26n':>10s} {'MBNTv3S':>10s}")
print("-" * 47)
me = json.load(open(os.path.join(base, 'MBNTv3S_ssdlite_v1/experiment.json')))['results']
ye2 = json.load(open(os.path.join(base, 'yolo26n_v1/experiment.json')))['results']
for k in ['test_mAP50','test_mAP50_95','test_precision','test_recall','test_f1',
          'training_time_min','tflite_size_mb','tflite_esp32_ok']:
    yv = ye2.get(k, 'N/A')
    mv = me.get(k, 'N/A')
    if isinstance(yv, float):
        print(f"  {k:<23s} {yv:>10.4f} {mv:>10.4f}")
    else:
        print(f"  {k:<23s} {str(yv):>10s} {str(mv):>10s}")

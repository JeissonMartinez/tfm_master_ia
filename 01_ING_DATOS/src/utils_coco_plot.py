import json
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns


def plot_class_distribution(json_path: str, title: str = "Distribución de Clases") -> Dict[str, int]:
    with open(json_path, "r") as f:
        data = json.load(f)

    cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    counts = {name: 0 for name in cat_id_to_name.values()}
    for ann in data["annotations"]:
        cat_name = cat_id_to_name.get(ann["category_id"])
        if cat_name in counts:
            counts[cat_name] += 1

    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    total = sum(counts.values())

    colors = sns.color_palette("magma", len(sorted_counts))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(list(sorted_counts.keys()), list(sorted_counts.values()), color=colors, edgecolor="black", alpha=0.8)

    for bar, (name, count) in zip(bars, sorted_counts.items()):
        height = bar.get_height()
        percentage = (count / total * 100) if total > 0 else 0
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.xlabel("Clase", fontsize=12, fontweight="bold")
    plt.ylabel("Número de Anotaciones", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()

    return counts

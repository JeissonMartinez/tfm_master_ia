import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data Setup
data = {
    'Category': ['Real', 'Real', 'Real', 'Sintético', 'Sintético'],
    'Subset': ['Dataset_Custom_1\n(QVGA - Noche)', 'Dataset_Custom_2\n(SVGA - Día)', 'Dataset_Custom_3\n(SVGA - Día Variable)', 
               'Dataset_Custom_4\n(Modificaciones Leves)', 'Dataset_Custom_5\n(Desde Cero)'],
    'Count': [512, 265, 213, 32, 102]
}
df = pd.DataFrame(data)

# Colors using magma
# We need 5 distinct colors from magma for the subsets
colors = plt.cm.magma(np.linspace(0.1, 0.85, len(df['Subset'])))

# --- Option 1: Nested Donut Chart (Sunburst) ---
fig1, ax1 = plt.subplots(figsize=(10, 7))

# Data for Outer Ring (Subsets)
sizes_subsets = df['Count'].tolist()
labels_subsets = df['Subset'].tolist()

# Data for Inner Ring (Categories)
grp = df.groupby('Category')['Count'].sum()
sizes_cats = [grp['Real'], grp['Sintético']]
labels_cats = [f'Real\n({grp["Real"]})', f'Sintético\n({grp["Sintético"]})']

# Colors for inner ring (average of outer colors roughly, or specific magma points)
colors_cats = [plt.cm.magma(0.4), plt.cm.magma(0.8)] 

# Plot Outer Ring
wedges1, texts1, autotexts1 = ax1.pie(sizes_subsets, radius=1.2, 
                                      labels=None, # Labels added manually or via legend to avoid clutter
                                      autopct='%1.1f%%', pctdistance=0.85,
                                      colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))

# Plot Inner Ring
wedges2, texts2, autotexts2 = ax1.pie(sizes_cats, radius=0.8, 
                                      labels=labels_cats, labeldistance=0.4,
                                      autopct='', pctdistance=0.4,
                                      colors=colors_cats, wedgeprops=dict(width=0.4, edgecolor='w'))

# Style adjustments
plt.setp(autotexts1, size=9, weight="bold", color="white")
plt.setp(texts2, size=12, weight="bold", color="white")
# Add a circle at the center to make it a donut (optional, but requested structure implies composition)
centre_circle = plt.Circle((0,0),0.40,fc='white')
fig1.gca().add_artist(centre_circle)

# Add Legend for Subsets to keep chart clean
ax1.legend(wedges1, labels_subsets,
          title="Subconjuntos",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.title('Distribución del Dataset Egocéntrico (Real vs Sintético)', fontsize=14, pad=20)
plt.text(0, 0, f"Total\n{sum(df['Count'])}", ha='center', va='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('grafico_opcion_1_sunburst.png')

# --- Option 2: Stacked Bar Chart ---
fig2, ax2 = plt.subplots(figsize=(10, 7))

# Prepare data for stacking
# Pivot: Index=Category, Columns=Subset, Values=Count
# We need to preserve the order and color mapping
# Since subsets are unique to categories, this is a bit tricky for standard stacked bar if we want distinct colors per subset.
# We will plot "Real" bar and "Synthetic" bar by stacking the specific subsets manually.

# Base bottoms
bottom_real = 0
bottom_synth = 0

real_subsets = df[df['Category'] == 'Real']
synth_subsets = df[df['Category'] == 'Sintético']

# Plot Real Stack
for i, (idx, row) in enumerate(real_subsets.iterrows()):
    p = ax2.bar('Real', row['Count'], bottom=bottom_real, color=colors[i], label=row['Subset'], width=0.5, edgecolor='white')
    # Add label in center of bar section
    ax2.bar_label(p, label_type='center', color='white', fontweight='bold', fmt='%d')
    bottom_real += row['Count']

# Plot Synthetic Stack (continue color index)
for i, (idx, row) in enumerate(synth_subsets.iterrows()):
    color_idx = len(real_subsets) + i
    p = ax2.bar('Sintético', row['Count'], bottom=bottom_synth, color=colors[color_idx], label=row['Subset'], width=0.5, edgecolor='white')
    ax2.bar_label(p, label_type='center', color='white', fontweight='bold', fmt='%d')
    bottom_synth += row['Count']

# Add totals on top
ax2.text('Real', bottom_real + 10, f"Total: {bottom_real}", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.text('Sintético', bottom_synth + 10, f"Total: {bottom_synth}", ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Cantidad de Imágenes')
ax2.set_title('Distribución del Dataset Egocéntrico (Real vs Sintético)', fontsize=14)
ax2.legend(title="Subconjuntos", bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('grafico_opcion_2_barras.png')
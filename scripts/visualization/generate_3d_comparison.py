import os
import re
import json
import gzip
import base64
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# Configuration
# =========================================================
PER_CLASS_CAP = 3000       # max cells per annotation per patient
COORD_PRECISION = 1        # decimal places for x/y/z coordinates

v2_csv = '/stornext/Bioinf/data/lab_brain_cancer/projects/tme_spatial/transcriptomics/venture/Ven2_23BCRL059T/data/out/aligned_annotations_240620.csv'
v3_csv = '/stornext/Bioinf/data/lab_brain_cancer/projects/tme_spatial/venture_multi_omics/venture_pt3/annotated_polygons/ven3_annotations_centroids.csv'
pt5_base = '/vast/projects/BCRL_Multi_Omics/venture_pt5/'
pt5_z_planes = [80, 70, 60, 50, 40, 30, 20, 10]

out_path = '/vast/projects/BCRL_Multi_Omics/SMINT/docs/bcrl_patient_3d_comparison.html'

# =========================================================
# Helpers
# =========================================================
def normalize_label(x):
    if pd.isna(x):
        return 'Unknown'
    s = ' '.join(str(x).strip().split())
    return s if s else 'Unknown'

def round_cols(df, cols, ndigits=1):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').round(ndigits)
    return df

def subsample(df, anno_col, cap, random_state=42):
    parts = []
    for ann, grp in df.groupby(anno_col):
        parts.append(grp.sample(n=min(len(grp), cap), random_state=random_state))
    return pd.concat(parts, ignore_index=True)

def minify_html(s):
    s = re.sub(r'>\s+<', '><', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()

# =========================================================
# Colors
# =========================================================
cell_type_to_rgb = {
    'AC-like': (0.949, 0.518, 0.509), 'prolif_AC-like': (0.949, 0.518, 0.509),
    'MES-like': (0.616, 0.306, 0.867), 'OPC-like': (0.416, 0.6, 0.306),
    'NPC-like': (0.227, 0.486, 0.647), 'Progenitor': (0.937, 0.137, 0.235),
    'Astrocyte': (0.976, 0.784, 0.780), 'Reactive Astrocyte': (1.0, 0.502, 0.961),
    'Oligodendrocyte': (0.804, 0.871, 0.725), 'OPC': (0.655, 0.788, 0.341),
    'Neuron': (0.388, 0.635, 0.784), 'Excitatory': (0.063, 0.957, 0.725),
    'Inhibitory': (0.612, 0.992, 1.0), 'Vasculature': (0.0, 0.0, 0.0),
    'Immune': (1.0, 0.918, 0.0), 'Unknown': (0.957, 0.953, 0.933),
    'T Cell': (0.980, 0.639, 0.027), 'NK Cell': (1.0, 0.729, 0.031),
    'B Cell': (1.0, 0.839, 0.039), 'CD8 T Cell': (0.910, 0.365, 0.027),
    'CD4 T Cell': (0.957, 0.549, 0.023), 'T Reg Cell': (0.949, 0.800, 0.561),
    'Dendritic Cell': (0.396, 0.427, 0.290), 'Neutrophil': (0.643, 0.675, 0.525),
    'Monocyte': (0.714, 0.678, 0.565), 'Microglia': (0.498, 0.310, 0.145),
    'Macrophage': (0.651, 0.541, 0.392), 'Mast Cell': (0.839, 0.820, 0.757),
    'Activated Microglia': (0.737, 0.349, 0.0), 'Immunosuppressive Microglia': (0.596, 0.165, 0.0),
    'Endothelial': (0.0, 0.0, 0.0), 'Mural': (0.678, 0.710, 0.741),
    'Fibroblast': (0.690, 0.408, 0.667), 'Death': (0.4, 0.027, 0.031),
    'Mixed-AC-like': (0.3, 0.027, 0.031),
}
cell_type_to_rgb_plotly = {k: f'rgb({int(v[0]*255)}, {int(v[1]*255)}, {int(v[2]*255)})' for k, v in cell_type_to_rgb.items()}
default_color_tuple = (0.5, 0.5, 0.5)

# =========================================================
# Load data
# =========================================================
print('Loading PT2...')
v2_all = pd.read_csv(v2_csv)
v2_all['Anno'] = v2_all['Anno'].apply(normalize_label)
v2_all = round_cols(v2_all, ['x_centroid', 'y_centroid', 'z_centroid'], COORD_PRECISION)
v2_all = subsample(v2_all, 'Anno', PER_CLASS_CAP)
print(f'  PT2 after subsampling: {len(v2_all)}')

print('Loading PT5...')
pt5_parts = []
for i, z_val in enumerate(pt5_z_planes):
    df = pd.read_csv(f'{pt5_base}/Z{i+1}_final_aligned_annos.csv')
    df['z_plane'] = z_val
    pt5_parts.append(df)
plot_df = pd.concat(pt5_parts, ignore_index=True)
plot_df['Anno'] = plot_df['Anno'].apply(normalize_label)
plot_df = round_cols(plot_df, ['x_final', 'y_final'], COORD_PRECISION)
plot_df = subsample(plot_df, 'Anno', PER_CLASS_CAP)
print(f'  PT5 after subsampling: {len(plot_df)}')

print('Loading PT3...')
v3_all = pd.read_csv(v3_csv)
v3_all = v3_all.rename(columns={'Anno2': 'Anno', 'new_x_centroid': 'x_centroid', 'new_y_centroid': 'y_centroid'})
v3_all['Anno'] = v3_all['Anno'].apply(normalize_label)
layer_to_z = {'z1': 10, 'z2': 20, 'z3': 30, 'z4': 40, 'z5': 50, 'z6': 60, 'z7': 70, 'z8': 80}
v3_all['z_centroid'] = v3_all['layer'].map(layer_to_z).fillna(0.0).astype(float)
v3_all = round_cols(v3_all, ['x_centroid', 'y_centroid', 'z_centroid'], COORD_PRECISION)
v3_all = subsample(v3_all, 'Anno', PER_CLASS_CAP)
print(f'  PT3 after subsampling: {len(v3_all)}')

# =========================================================
# Unified annotation list
# =========================================================
all_annotations = sorted(
    pd.Index(v2_all['Anno']).union(pd.Index(plot_df['Anno'])).union(pd.Index(v3_all['Anno'])).unique().tolist()
)

# =========================================================
# Build figures
# =========================================================
print('Building figures...')

fig_pt2 = px.scatter_3d(
    v2_all, x='x_centroid', y='y_centroid', z='z_centroid',
    color='Anno', color_discrete_map=cell_type_to_rgb_plotly,
    category_orders={'Anno': all_annotations}, title=None,
    labels={'x_centroid': 'X', 'y_centroid': 'Y', 'z_centroid': 'Z', 'Anno': 'Annotation'}
)
fig_pt2.update_traces(marker=dict(size=2), hoverinfo='skip', hovertemplate=None)

def build_scatter_fig(df, x_col, y_col, z_col):
    traces = []
    for ann in all_annotations:
        sub = df[df['Anno'] == ann]
        if sub.empty:
            continue
        r, g, b = cell_type_to_rgb.get(ann, default_color_tuple)
        traces.append(go.Scatter3d(
            x=sub[x_col], y=sub[y_col], z=sub[z_col],
            mode='markers', name=str(ann),
            marker=dict(size=2, color=f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'),
            hoverinfo='skip', legendgroup=str(ann), showlegend=True
        ))
    return go.Figure(data=traces)

fig_pt5 = build_scatter_fig(plot_df, 'x_final', 'y_final', 'z_plane')
fig_pt3 = build_scatter_fig(v3_all, 'x_centroid', 'y_centroid', 'z_centroid')

legend_style = dict(
    x=0.99, y=0.01, xanchor='right', yanchor='bottom',
    bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0.25)', borderwidth=1,
    font=dict(family='Arial, sans-serif', size=11), itemwidth=30,
)
for fig in (fig_pt2, fig_pt5, fig_pt3):
    fig.update_layout(
        title=None, margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True, legend=legend_style, legend_title_text='',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube'),
    )

# =========================================================
# Export
# =========================================================
print('Exporting HTML...')
div1 = fig_pt2.to_html(full_html=False, include_plotlyjs='cdn')
div2 = fig_pt5.to_html(full_html=False, include_plotlyjs=False)
div3 = fig_pt3.to_html(full_html=False, include_plotlyjs=False)

html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>
<title>Venture PT2, PT5, PT3 - Side-by-Side 3D Plots</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<style>
.grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;height:100vh;box-sizing:border-box;padding:8px;align-items:stretch;}}
.panel{{min-height:0;display:flex;flex-direction:column;}}
.panel>.title{{flex:0 0 auto;font-family:sans-serif;font-size:16px;font-weight:600;margin:4px 0 6px 4px;}}
.panel>.plot{{flex:1 1 0;min-height:0;}}
.panel>.plot>div{{height:100% !important;}}
body,html{{margin:0;height:100%;}}
</style></head><body>
<div class="grid">
<div class="panel"><div class="title">Venture Patient 2 - Transcriptomics</div><div class="plot">{div1}</div></div>
<div class="panel"><div class="title">Venture Patient 5 - Transcriptomics</div><div class="plot">{div2}</div></div>
<div class="panel"><div class="title">Venture Patient 3 - Transcriptomics</div><div class="plot">{div3}</div></div>
</div>
</body></html>"""

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(minify_html(html))

size_mb = os.path.getsize(out_path) / 1e6
print(f'Written: {out_path}  ({size_mb:.1f} MB)')

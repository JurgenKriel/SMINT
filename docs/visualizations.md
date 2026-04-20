# Interactive Visualizations

This page showcases interactive 3D multi-omics data from the Brain Cancer Research Lab (BCRL) generated using SMINT.

## Venture Patient Cohort — 3D Spatial Transcriptomics

Side-by-side interactive 3D plots comparing spatial transcriptomics across three Venture patients (PT2, PT5, PT3). Cell type annotations and spatial coordinates are rendered using Plotly, enabling rotation, zoom, and hover-based exploration of tumour microenvironment composition.

<div style="text-align: right; margin-bottom: 8px;">
  <a href="bcrl_patient_3d_comparison.html" target="_blank" style="font-size: 0.9em;">
    Open fullscreen ↗
  </a>
</div>

<div style="position: relative; width: 100%; padding-top: 62%; border: 1px solid #ddd; border-radius: 4px; overflow: hidden;">
  <iframe
    src="bcrl_patient_3d_comparison.html"
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;"
    loading="lazy"
    title="BCRL Venture Patients PT2, PT5, PT3 — 3D Spatial Transcriptomics Comparison">
  </iframe>
</div>

!!! tip "Interaction tips"
    - **Rotate**: click and drag within a panel
    - **Zoom**: scroll wheel or pinch
    - **Pan**: right-click and drag
    - **Toggle cell types**: click legend entries to show/hide groups
    - **Isolate cell type**: double-click a legend entry

## Data Details

| Patient | Modality | Description |
|---------|----------|-------------|
| Venture PT2 | Spatial Transcriptomics | Primary tumour section, 3D Z-stack alignment |
| Venture PT5 | Spatial Transcriptomics | Primary tumour section, 3D Z-stack alignment |
| Venture PT3 | Spatial Transcriptomics | Primary tumour section, 3D Z-stack alignment |

Cell type labels follow the glioblastoma tumour microenvironment taxonomy (AC-like, MES-like, OPC-like, NPC-like, and immune populations).

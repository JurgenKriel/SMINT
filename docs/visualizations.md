# Interactive Visualizations

This page showcases interactive 3D multi-omics data from the Brain Cancer Research Lab (BCRL) generated using SMINT.

## Patient Cohort — 3D Spatial Transcriptomics

Interactive 3D plots comparing spatial transcriptomics across three patients (Patient 1, Patient 5, Patient 3). Use the selector to view one patient at a time or all side by side. Cell type annotations and spatial coordinates are rendered using Plotly, enabling rotation, zoom, and hover-based exploration of tumour microenvironment composition.

<div style="text-align: right; margin-bottom: 8px;">
  <a href="../bcrl_patient_3d_comparison.html" target="_blank" style="font-size: 0.9em;">
    Open fullscreen ↗
  </a>
</div>

<div style="position: relative; width: 100%; padding-top: 62%; border: 1px solid #ddd; border-radius: 4px; overflow: hidden;">
  <iframe
    src="../bcrl_patient_3d_comparison.html"
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;"
    loading="lazy"
    title="BCRL Patients — 3D Spatial Transcriptomics Comparison">
  </iframe>
</div>

!!! tip "Interaction tips"
    - **Select patient**: use the buttons at the top of the plot
    - **Rotate**: click and drag within a panel
    - **Zoom**: scroll wheel or pinch
    - **Pan**: right-click and drag
    - **Toggle cell types**: click legend entries to show/hide groups
    - **Isolate cell type**: double-click a legend entry

## Data Details

| Patient | Modality | Description |
|---------|----------|-------------|
| Patient 1 | Spatial Transcriptomics | Primary tumour section, 3D Z-stack alignment |
| Patient 5 | Spatial Transcriptomics | Primary tumour section, 3D Z-stack alignment |
| Patient 3 | Spatial Transcriptomics | Primary tumour section, 3D Z-stack alignment |

Cell type labels follow the glioblastoma tumour microenvironment taxonomy (AC-like, MES-like, OPC-like, NPC-like, and immune populations).

# Reference

Look-up material for when you know what you want to do and need the exact name,
option, column or default. The command-line page is generated from SMINT's own
argument parsers and the API pages from its docstrings, so they match the
installed code.

<div class="grid cards" markdown>

-   :material-console:{ .lg .middle } **Command line**

    ---

    Every option of `smint-alignment`, `smint-register`, `run_segmentation.py`,
    the distributed segmentation entry point and `smint-viewer`, with defaults
    and a worked example per command.

    [:octicons-arrow-right-24: Command line](cli.md)

-   :material-table:{ .lg .middle } **Data formats**

    ---

    Every file SMINT reads or writes: centroid, matrix and annotation CSVs,
    landmark files, transforms, `job_spec.json` / `status.json`, and the
    segmentation and registration outputs, with example rows.

    [:octicons-arrow-right-24: Data formats](data_formats.md)

-   :material-cog-outline:{ .lg .middle } **Configuration**

    ---

    Every YAML key the segmentation pipelines read, with type, default and
    effect, and a complete example configuration.

    [:octicons-arrow-right-24: Configuration](configuration.md)

-   :material-vector-polyline:{ .lg .middle } **Alignment API**

    ---

    Pre-registration, same-section centroid registration and refinement, and
    STalign LDDMM for sequential sections.

    [:octicons-arrow-right-24: `smint.alignment`](api/alignment.md)

-   :material-grid:{ .lg .middle } **Segmentation API**

    ---

    Distributed and single-process segmentation, stitching and cell / nucleus
    reconciliation, and the Cellpose 3 / 4 compatibility layer.

    [:octicons-arrow-right-24: `smint.segmentation`](api/segmentation.md)

-   :material-server-network:{ .lg .middle } **Jobs API**

    ---

    Run registrations out of process, locally or on SLURM, and follow them
    through `status.json`.

    [:octicons-arrow-right-24: `smint.alignment.jobs`](api/jobs.md)

</div>

## Which page answers my question?

| Question | Go to |
| --- | --- |
| What does `--snap-radius` do, and what is its default? | [Command line](cli.md#smint-alignment-centroid) |
| Which columns must my input CSV have? | [Data formats → Inputs](data_formats.md#inputs) |
| Which coordinate columns will SMINT pick? | [Data formats → Coordinate columns](data_formats.md#coordinate-columns) |
| What does `registration_status = assigned` mean? | [Data formats → Registration outputs](data_formats.md#centroids_transformedcsv) |
| How do I read `status.json` after a run? | [Data formats → Registration jobs](data_formats.md#statusjson) |
| How do I set the tile overlap or run on SLURM GPUs? | [Configuration → distributed](configuration.md#the-distributed-section) |
| What does `register_centroids` return? | [Alignment API](api/alignment.md#smint.alignment.register_centroids) |
| How do I submit a registration from Python? | [Jobs API](api/jobs.md) |

# R Integration

SMINT provides seamless integration with R for spatial omics data analysis, allowing you to leverage existing R packages within your Python workflow.

## Quick Start

### Running an R Script

```python
from smint.r_integration import run_r_script

# Run an R script with arguments
return_code = run_r_script(
    script_path="path/to/analysis.R",
    args=["--input", "data.csv", "--output", "results/"]
)

print(f"Script completed with return code: {return_code}")

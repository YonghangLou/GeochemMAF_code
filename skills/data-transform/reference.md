## Common Transformation Choices
- Log transform is suitable for strongly right-skewed positive geochemical concentrations after zeros or negatives are handled safely.
- Box-Cox requires positive values and estimates a power parameter automatically.
- CLR, ALR, and ILR are appropriate for compositional data affected by closure and should use a pseudocount when zeros are present.

## Reproducibility
- Record the selected transformation and any applied shifts or pseudocounts.
- Keep transformed column names unchanged, or provide an explicit mapping if renaming is required downstream.

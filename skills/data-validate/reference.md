## Minimal Checklist
- Confirm shape, column names, and field types, especially whether numeric columns contain string artifacts.
- Quantify missing-value ratios and duplicate rows.
- Confirm that coordinate columns exist and are numerically plausible for either longitude and latitude or projected coordinates.

## Output Notes
- Group issues by severity when possible so the workflow can decide whether to continue.
- Return reproducible statistics such as top missing columns and duplicate counts.

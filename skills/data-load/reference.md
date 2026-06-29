## Common File Types
- CSV or TSV: confirm delimiter, encoding, and missing-value markers such as empty strings, `NA`, or `-9999`.
- Excel: confirm the target sheet and watch for merged cells or multi-row headers.
- JSON: confirm whether the file uses record-style rows or a nested structure that should be flattened first.

## Minimal Validation
- Required fields depend on the workflow, but sample ID and coordinates are commonly needed.
- Numeric geochemical columns should use consistent units and should not contain string artifacts.
- Check whether zeros, negative values, or extreme values represent real measurements, censoring limits, or missing codes.

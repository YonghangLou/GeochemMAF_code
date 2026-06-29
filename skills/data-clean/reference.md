## Missing Values
- Clarify how missing values are encoded, including empty strings, `NA`, `-9999`, or assay-limit substitutions.
- Median imputation is usually safer for numeric geochemical variables, while categorical fields can use mode imputation or an explicit unknown label.

## Duplicates And Outliers
- Confirm whether duplicate rows are true duplicates or repeat samples before removing them.
- Interpret outliers against geological context and detection limits before applying IQR clipping or percentile trimming.

## Output Consistency
- Preserve original indices or traceable identifiers so cleaned data can still be aligned with predictions and maps.

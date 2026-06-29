## Suitable Use Cases
- Distance-based models, clustering, PCA, SVM, and neural networks are often sensitive to feature scale.
- Scaling is also useful when element concentrations use different units or very different numeric ranges.

## Cautions
- Do not scale categorical fields, identifiers, or coordinate columns unless the downstream model explicitly requires it.
- Reuse the same fitted scaler between training and prediction to avoid leakage.

## Output Notes
- Record the selected method, such as `StandardScaler` or `MinMaxScaler`, together with the affected columns for reproducibility.

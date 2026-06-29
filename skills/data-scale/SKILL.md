---
name: data-scale
description: Standardize or normalize numeric features with StandardScaler or MinMaxScaler before scale-sensitive modeling
id: data.scale
---

## Skill ID
- data.scale

## When To Use
- Use after transformation and before PCA, clustering, SVM, neural networks, or other scale-sensitive methods.

## Inputs
- df: transformed DataFrame

## Outputs
- df: scaled DataFrame

## Execution Notes
- Scale only numeric feature columns and avoid altering the meaning of labels, coordinates, or identifiers.
- Keep original column names unchanged so downstream interpretation stays aligned.

## References
- [reference.md](reference.md)
- [tool.py](tool.py)

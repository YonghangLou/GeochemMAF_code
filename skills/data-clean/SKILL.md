---
name: data-clean
description: Clean geochemical tabular data by handling missing values, duplicates, and outliers before downstream modeling
id: data.clean
---

## Skill ID
- data.clean

## When To Use
- Use after loading raw geochemical tables when missing-value handling, deduplication, and outlier treatment are needed before transformation, scaling, or modeling.

## Inputs
- df: raw input DataFrame

## Outputs
- df: cleaned DataFrame

## Execution Notes
- Preserve key coordinate and label fields such as `Longitude`, `Latitude`, `Ore`, and `FID`.
- Keep row alignment traceable unless duplicate or invalid rows are intentionally removed.

## References
- [reference.md](reference.md)
- [tool.py](tool.py)

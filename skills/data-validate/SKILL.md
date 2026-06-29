---
name: data-validate
description: Check missing values, duplicates, numeric columns, and coordinate fields and return a diagnostic summary
id: data.validate
---

## Skill ID
- data.validate

## When To Use
- Use immediately after loading data when you want a fast diagnostic of data quality, duplicate rows, missing values, and coordinate readiness for mapping or spatial analysis.

## Inputs
- df: input DataFrame

## Outputs
- results: validation summary dictionary

## References
- [reference.md](reference.md)
- [tool.py](tool.py)

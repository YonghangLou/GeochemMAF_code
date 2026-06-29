---
name: data-transform
description: Apply distribution-aware geochemical transformations such as log, Box-Cox, CLR, ALR, or ILR before scale-sensitive modeling
id: data.transform
---

## Skill ID
- data.transform

## When To Use
- Use after cleaning when feature distributions remain highly skewed or when compositional effects should be handled with log-ratio transforms.
- Run this step before scaling or model fitting.

## Inputs
- df: cleaned DataFrame

## Outputs
- df: transformed DataFrame

## Execution Notes
- Transform only numeric geochemical feature columns and preserve labels, coordinates, and identifiers.
- Prefer strategies that remain reproducible under the same input and configuration.

## References
- [reference.md](reference.md)
- [tool.py](tool.py)

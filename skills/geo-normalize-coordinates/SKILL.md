---
name: geo-normalize-coordinates
description: Detect coordinate columns automatically and generate standardized longitude and latitude fields, including projected-coordinate conversion when possible
id: geo.normalize_coordinates
---

## Skill ID
- geo.normalize_coordinates

## When To Use
- Use when raw data uses inconsistent coordinate names such as `lon`, `lat`, `X`, or `Y`, or when the coordinates may be projected and should be normalized to standard `Longitude` and `Latitude` fields for mapping and spatial analysis.

## Inputs
- df: input DataFrame

## Outputs
- df: normalized DataFrame
- meta: normalization metadata

## References
- [reference.md](reference.md)
- [tool.py](tool.py)

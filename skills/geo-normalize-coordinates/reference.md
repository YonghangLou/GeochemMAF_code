## Common Coordinate Names
- Longitude-like names often include `lon`, `longitude`, `x`, or `long`.
- Latitude-like names often include `lat`, `latitude`, or `y`.

## Quick Projected-Coordinate Check
- If X and Y values are around `1e5` to `1e7` and fall outside valid longitude and latitude ranges, they are often projected coordinates such as UTM or Gauss-Kruger.
- If EPSG or projection-zone metadata exists, prefer that information instead of guessing from value ranges alone.

## Output Notes
- Record the original source columns, whether projected coordinates were suspected, attempted conversion parameters, and any failure reason in `meta`.

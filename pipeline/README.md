# Analog Terrax — Offline Data & Feature Pipeline (`pipeline/`)

This folder contains the **offline data ingestion, harmonization, and index-building pipeline** described in the Data Strategy Analysis. It is excluded from the Vercel serverless runtime via `.vercelignore` so heavy scientific/HDF5/GIS dependencies never bloat the FastAPI deployment.

## What It Generates (`../app/data/`)
Running `build_index.py` produces compact, pre-normalized artifacts ready for sub-20ms kNN queries at runtime:
- `taxonomy.json`: Shared 7-class cross-planet lithology taxonomy + $7 \times 7$ similarity matrix + 9 shared mineral dimensions.
- `feature_stats.json`: Pooled median & IQR normalization statistics for the 6 topographic fingerprint dimensions.
- `sites.json`: Curated Moon & Mars candidate base sites and Earth benchmark + newly discovered analog locations.
- `earth_index.npz`, `moon_index.npz`, `mars_index.npz`: Compressed NumPy feature matrices (`float32`) for instant vectorized similarity scoring.
- `profiles/laser_profiles.json`: Laser-vs-laser altimetry elevation transects comparing Earth **ICESat-2 ATLAS (ATL03)** with Moon **LRO LOLA** and Mars **MGS MOLA**.

## Usage

```bash
# Rebuild all runtime data artifacts in ../app/data/
uv run python pipeline/build_index.py
```

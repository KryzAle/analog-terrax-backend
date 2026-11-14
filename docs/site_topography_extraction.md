# Topographic extraction for lunar and Martian sites

## Purpose

[`scripts/extract_site_topography.py`](../scripts/extract_site_topography.py)
adapts the PDS3 reader and topographic workflow from
[`notebooks/ODE_REST_Moon_Mars_Extraction.ipynb`](../notebooks/ODE_REST_Moon_Mars_Extraction.ipynb).
It always downloads the LOLA `LDEM_64` and MOLA `MEGT90N000EB` DEMs, even when
local copies already exist. It selects the first five Moon rows and first five
Mars rows from `app/data/input/sites.csv`, then summarizes each DEM in a 25 ×
25 km square centered on its site. The output contains ten rows with only these
columns: `name`, `body`, `relief_m`, `slope_mean_deg`, `slope_p90_deg`,
`ruggedness_tri_m`, `roughness_rms_m`, and `hypsometric_integral`.

## Run

From the repository root, with the project environment active:

```powershell
python scripts/extract_site_topography.py
```

The default output is `app/data/processed/topography_sites_25km.csv`. Every run
downloads and replaces both DEMs and their PDS3 labels. Files are stored under
`app/data/raw/moon/topography/LUN-TOP-01/` and
`app/data/raw/mars/topography/MAR-TOP-01/`. Internet access is required. Window
size, site count, input, and output can be overridden:

```powershell
python scripts/extract_site_topography.py `
  --side-km 25 `
  --per-body 5 `
  --sites app/data/input/sites.csv `
  --output app/data/processed/topography_sites_25km.csv
```

This is the reproducible from-scratch mode. To process local DEMs without
downloading or overwriting them, use `--no-download` and pass local PDS3 labels
with `--moon-dem` and `--mars-dem`. The `.IMG` referenced by `^IMAGE` must be in
the same directory as its label.

## Features and definitions

Elevations are converted to meters using `SCALING_FACTOR`. A planetary-radius
`OFFSET` is ignored so the data remain relative heights. Pixel dimensions are
computed locally from the planetary radius and cylindrical projection
resolution. The following metrics are aggregated over valid pixels in each
window:

| Column | Definition | Unit |
|---|---|---|
| `relief_m` | Maximum elevation minus minimum elevation | m |
| `slope_mean_deg` | Mean slope calculated from central gradients | degrees |
| `slope_p90_deg` | 90th percentile of slope | degrees |
| `ruggedness_tri_m` | Mean Riley TRI: square root of the sum of squared differences to eight neighbors | m |
| `roughness_rms_m` | Mean local RMS elevation deviation in 3 × 3 neighborhoods | m |
| `hypsometric_integral` | `(mean elevation - minimum elevation) / relief`; 0 when relief is zero | unitless |

One pixel of margin is read for edge derivatives; summaries use the nominal
rectangle. Near a pole, the longitude span grows to preserve an approximately
25 km east-west width, capped at 359°.

## Site selection

Selection is deterministic: the first five rows with `body=moon` and first five
with `body=mars`, in CSV order. In the current file they are:

- Moon: `shackleton_crater`, `aristarchus_plateau`, `aristarchus_crater`, `marius_hills`, `reiner_gamma_anomaly`.
- Mars: `jezero_crater_center`, `jezero_crater_delta`, `gale_crater_center`, `gale_crater_curiosity`, `oxia_planum_center`.

Changing row order in `sites.csv` changes the selected locations.

## Data and known limitations

The default lunar DEM is LOLA LDEM 64 at
`app/data/raw/moon/topography/LUN-TOP-01/ldem_64.lbl`. The previous local MOLA
file `megc90n000cb.lbl` declares `NAME=COUNTS_PER_BIN` (observation counts); the
script replaces it with `MEGT90N000EB`, the topography DEM. In MOLA filenames,
`T` denotes topography and `C` denotes counts. The Mars DEM has 16 pixels per
degree (about 3.7 km per pixel at the equator), so metrics reflect terrain at
that sampling scale. Source URLs are defined in `DEM_URLS` in the script.

The reader supports single-band PDS3 rasters in cylindrical or equirectangular
projections. It reads only the rows needed for each window instead of mapping
the full lunar DEM into memory. To retry processing after both DEMs have already
downloaded, run with `--no-download`. Null cells are excluded from elevation
statistics; extraction stops if valid coverage is below 80%. NumPy is already
listed in the project dependencies.

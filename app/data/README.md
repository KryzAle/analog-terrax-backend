# `app/data/` layout

```text
app/data/
├── profiles/    Compact profiles used at runtime
├── *.json       Backend taxonomy, sites, and feature statistics (including moon_mars_full_real_data.json)
├── *.npz        Compact similarity indexes
├── raw/         Downloaded source rasters and files (ignored by Git)
├── processed/   Versioned engine_features.csv summary (other outputs are ignored)
├── catalogs/    Catalogs, plans, and validation reports (ignored by Git)
└── reference/   External source datasets (ignored by Git)
```

The JSON and NPZ files at the root of `app/data/` are compact artifacts used by
the API and remain versioned. `processed/engine_features.csv` tracks the
extracted Kaguya MI, MGS TES, and USGS GIS summary features, while the full
`engine_features` and `engine_features_provenance` payloads are merged into
`moon_mars_full_real_data.json`. Source rasters can be hundreds of megabytes, so
`raw/`, `extras/`, `catalogs/`, `input/`, and `reference/` are ignored by Git
and created by offline workflows when needed.

# `app/data/` layout

```text
app/data/
├── profiles/    Compact profiles used at runtime
├── *.json       Backend taxonomy, sites, and feature statistics
├── *.npz        Compact similarity indexes
├── input/       Small versioned inputs, including sites.csv
├── raw/         Downloaded source rasters and files (ignored by Git)
├── processed/   Versioned feature CSVs and enriched Moon/Mars Full Real JSON
├── catalogs/    Catalogs, plans, and validation reports (ignored by Git)
└── reference/   External source datasets (ignored by Git)
```

The JSON and NPZ files at the root of `app/data/` are compact artifacts used by
the API and remain versioned. `input/sites.csv` configures the sites used by
the science workflows. Only `processed/topography_sites_25km.csv` is versioned
for deployment, together with `geology_sites.csv` and `mineralogy_sites.csv`;
other processed outputs are ignored. Source rasters can be
hundreds of megabytes, so `raw/`, `catalogs/`, and `reference/` are ignored by
Git and created by the workflows when needed.

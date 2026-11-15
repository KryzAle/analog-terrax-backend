#!/usr/bin/env python3
"""Build engine-ready site rows and Moon/Mars NumPy indices from full-data JSON.

The input JSON must contain an ``engine_features`` object per Moon/Mars site:

    {
      "topography": {... six topography feature ids ...},
      "geology_fractions": {... seven lithology ids ...},
      "mineralogy": {... nine mineral ids; null is allowed ...},
      "flags": {"ndvi": ..., "is_impact": ..., "has_ice": ..., "is_volcanic": ...}
    }

Missing mineral values remain ``null`` in sites.json and become -1 with a false
mask in the NumPy index, matching the current engine representation. Missing
geology and optional flags remain ``null`` in sites.json. NumPy indices use
neutral zero values and include availability masks for those missing fields.

Examples
--------
  python scripts/build_planetary_indices.py --input app/data/moon_mars_full_real_data.json \
      --output-dir app/data/generated

  python scripts/build_planetary_indices.py --input app/data/moon_mars_full_real_data.json \
      --output-dir app/data --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "app" / "data" / "processed"
DEFAULT_INPUT = DATA_DIR / "moon_mars_full_real_data_features.json"
DEFAULT_EXISTING_SITES = DATA_DIR / "sites.json"

TOPOGRAPHY_FEATURES = (
    "relief_m",
    "slope_mean_deg",
    "slope_p90_deg",
    "ruggedness_tri_m",
    "roughness_rms_m",
    "hypsometric_integral",
)
LITHOLOGY_FEATURES = (
    "basaltic_volcanic",
    "feldspathic_plutonic",
    "pyroclastic",
    "impact_breccia_regolith",
    "clastic_sedimentary",
    "evaporite_chemical",
    "ice_polar",
)
MINERAL_FEATURES = (
    "plagioclase_feldspar",
    "pyroxene",
    "olivine",
    "fe_oxides",
    "phyllosilicates_clays",
    "sulfates",
    "carbonates",
    "silica_glass",
    "h_water_ice",
)
FLAG_FEATURES = ("ndvi", "is_impact", "has_ice", "is_volcanic")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Could not read JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _as_site_mapping(raw: Any) -> dict[str, dict[str, Any]]:
    if isinstance(raw, dict):
        sites = raw
    elif isinstance(raw, list):
        sites = {str(item.get("site_id", "")): item for item in raw if isinstance(item, dict)}
    else:
        raise ValueError("Input JSON must be an object keyed by site id or a list of site objects.")
    if not sites or any(not site_id or not isinstance(site, dict) for site_id, site in sites.items()):
        raise ValueError("Input JSON contains an invalid or empty site record.")
    return sites


def _require_number(value: Any, path: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"Required numeric feature is missing: {path}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Required numeric feature is not numeric: {path}") from exc
    if not np.isfinite(result):
        raise ValueError(f"Required numeric feature is not finite: {path}")
    return result


def _optional_number(value: Any, path: str, *, lower: float | None = None, upper: float | None = None) -> float | None:
    if value is None:
        return None
    result = _require_number(value, path)
    if lower is not None and result < lower or upper is not None and result > upper:
        raise ValueError(f"Feature is outside its allowed range: {path}={result}")
    return result


def _feature_group(site: dict[str, Any], key: str, aliases: tuple[str, ...] = ()) -> dict[str, Any]:
    features = site.get("engine_features")
    if not isinstance(features, dict):
        raise ValueError(f"Site {site.get('site_id', '<unknown>')} is missing engine_features.")
    for candidate in (key, *aliases):
        group = features.get(candidate)
        if isinstance(group, dict):
            return group
    raise ValueError(f"Site {site.get('site_id', '<unknown>')} is missing engine_features.{key}.")


def _validate_group(site_id: str, group: dict[str, Any], fields: tuple[str, ...], group_name: str, *, optional: bool) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for field in fields:
        value = group.get(field)
        if optional:
            result[field] = _optional_number(value, f"{site_id}.engine_features.{group_name}.{field}", lower=0.0, upper=1.0)
        else:
            result[field] = _require_number(value, f"{site_id}.engine_features.{group_name}.{field}")
            if group_name == "geology_fractions" and not 0.0 <= result[field] <= 1.0:
                raise ValueError(f"Geology fraction is outside [0, 1]: {site_id}.{field}")
            if group_name == "topography" and field == "hypsometric_integral" and not 0.0 <= result[field] <= 1.0:
                raise ValueError(f"Hypsometric integral is outside [0, 1]: {site_id}.{field}")
            if group_name == "topography" and field != "hypsometric_integral" and result[field] < 0:
                raise ValueError(f"Topography feature is negative: {site_id}.{field}")
    if group_name == "geology_fractions":
        observed = [value for value in result.values() if value is not None]
        if observed and not np.isclose(sum(observed), 1.0, atol=0.02):
            raise ValueError(f"Available geology fractions for {site_id} must sum to approximately 1.0.")
    return result


def _build_row(site_id: str, site: dict[str, Any], ordinal: int) -> tuple[dict[str, Any], dict[str, Any]]:
    body = str(site.get("body", "")).lower()
    if body not in {"moon", "mars"}:
        raise ValueError(f"{site_id}.body must be 'moon' or 'mars'.")
    if site.get("site_id") and site["site_id"] != site_id:
        raise ValueError(f"Top-level site id {site_id!r} does not match site_id {site['site_id']!r}.")

    topo = _validate_group(site_id, _feature_group(site, "topography"), TOPOGRAPHY_FEATURES, "topography", optional=False)
    geo = _validate_group(
        site_id,
        _feature_group(site, "geology_fractions", ("geology",)),
        LITHOLOGY_FEATURES,
        "geology_fractions",
        optional=True,
    )
    minerals = _validate_group(site_id, _feature_group(site, "mineralogy"), MINERAL_FEATURES, "mineralogy", optional=True)
    flags = _feature_group(site, "flags")
    ndvi = _optional_number(flags.get("ndvi"), f"{site_id}.engine_features.flags.ndvi", lower=0.0, upper=1.0)
    bool_flags = {
        key: None if flags.get(key) is None else bool(flags[key])
        for key in FLAG_FEATURES
        if key != "ndvi"
    }

    cell_id = str(site.get("cell_id") or f"{body}_c{ordinal:03d}")
    topography_source = site.get("topography_2d_dem") or site.get("planetary_provenance", {}).get("topography_2d_dem")
    elevation = site.get("elevation_m")
    if elevation is None and isinstance(topography_source, dict):
        elevation = topography_source.get("mean_elevation_m")
    elevation = _require_number(elevation, f"{site_id}.elevation_m")

    row = {
        "site_id": site_id,
        "cell_id": cell_id,
        "body": body,
        "name": site.get("name", site_id),
        "region": site.get("region", body.title()),
        "lat": _require_number(site.get("lat"), f"{site_id}.lat"),
        "lon": _require_number(site.get("lon"), f"{site_id}.lon"),
        "elevation_m": elevation,
        "is_curated_site": bool(site.get("is_curated_site", True)),
        "is_known_analog": bool(site.get("is_known_analog", False)),
        "mission_relevance": site.get("mission_relevance", ""),
        "topo": [topo[field] for field in TOPOGRAPHY_FEATURES],
        "geo": [geo[field] for field in LITHOLOGY_FEATURES],
        "geo_mask": [geo[field] is not None for field in LITHOLOGY_FEATURES],
        "min": [minerals[field] for field in MINERAL_FEATURES],
        "flags": {"ndvi": ndvi, **bool_flags},
        "flag_mask": {
            "ndvi": ndvi is not None,
            **{key: value is not None for key, value in bool_flags.items()},
        },
    }
    if site.get("analog_category") is not None:
        row["analog_category"] = site["analog_category"]
    for provenance_key in (
        "planetary_provenance",
        "earth_provenance",
        "earthdata_granule",
        "engine_features_provenance",
    ):
        if provenance_key in site:
            row[provenance_key] = site[provenance_key]

    arrays = {
        "cell_id": cell_id,
        "site_id": site_id,
        "lat": row["lat"],
        "lon": row["lon"],
        "elevation": elevation,
        "is_curated": row["is_curated_site"],
        "is_known_analog": row["is_known_analog"],
        "topo": row["topo"],
        "geo": [value if value is not None else 0.0 for value in row["geo"]],
        "geo_mask": row["geo_mask"],
        "min": [value if value is not None else -1.0 for value in row["min"]],
        "min_mask": [value is not None for value in row["min"]],
        "ndvi": 0.0 if ndvi is None else ndvi,
        "ndvi_mask": ndvi is not None,
        "is_impact": False if bool_flags["is_impact"] is None else bool_flags["is_impact"],
        "is_impact_mask": bool_flags["is_impact"] is not None,
        "has_ice": False if bool_flags["has_ice"] is None else bool_flags["has_ice"],
        "has_ice_mask": bool_flags["has_ice"] is not None,
        "is_volcanic": False if bool_flags["is_volcanic"] is None else bool_flags["is_volcanic"],
        "is_volcanic_mask": bool_flags["is_volcanic"] is not None,
    }
    return row, arrays


def _write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cell_ids=np.asarray([row["cell_id"] for row in rows], dtype="<U64"),
        site_ids=np.asarray([row["site_id"] for row in rows], dtype="<U128"),
        lats=np.asarray([row["lat"] for row in rows], dtype=np.float32),
        lons=np.asarray([row["lon"] for row in rows], dtype=np.float32),
        elevations=np.asarray([row["elevation"] for row in rows], dtype=np.float32),
        is_curated=np.asarray([row["is_curated"] for row in rows], dtype=np.bool_),
        is_known_analog=np.asarray([row["is_known_analog"] for row in rows], dtype=np.bool_),
        topo=np.asarray([row["topo"] for row in rows], dtype=np.float32),
        geo=np.asarray([row["geo"] for row in rows], dtype=np.float32),
        geo_mask=np.asarray([row["geo_mask"] for row in rows], dtype=np.bool_),
        min_vec=np.asarray([row["min"] for row in rows], dtype=np.float32),
        min_mask=np.asarray([row["min_mask"] for row in rows], dtype=np.bool_),
        ndvi=np.asarray([row["ndvi"] for row in rows], dtype=np.float32),
        ndvi_mask=np.asarray([row["ndvi_mask"] for row in rows], dtype=np.bool_),
        is_impact=np.asarray([row["is_impact"] for row in rows], dtype=np.bool_),
        is_impact_mask=np.asarray([row["is_impact_mask"] for row in rows], dtype=np.bool_),
        has_ice=np.asarray([row["has_ice"] for row in rows], dtype=np.bool_),
        has_ice_mask=np.asarray([row["has_ice_mask"] for row in rows], dtype=np.bool_),
        is_volcanic=np.asarray([row["is_volcanic"] for row in rows], dtype=np.bool_),
        is_volcanic_mask=np.asarray([row["is_volcanic_mask"] for row in rows], dtype=np.bool_),
    )


def build(input_path: Path, output_dir: Path, existing_sites_path: Path | None, overwrite: bool) -> dict[str, Path]:
    input_sites = _as_site_mapping(_read_json(input_path))
    grouped_rows: dict[str, list[dict[str, Any]]] = {"moon": [], "mars": []}
    grouped_arrays: dict[str, list[dict[str, Any]]] = {"moon": [], "mars": []}
    seen_ids: set[str] = set()
    for site_id, site in input_sites.items():
        if site.get("body") not in {"moon", "mars"}:
            continue
        if site_id in seen_ids:
            raise ValueError(f"Duplicate site id: {site_id}")
        seen_ids.add(site_id)
        row, arrays = _build_row(site_id, site, len(grouped_rows[site["body"]]) + 1)
        grouped_rows[site["body"]].append(row)
        grouped_arrays[site["body"]].append(arrays)
    if not grouped_rows["moon"] or not grouped_rows["mars"]:
        raise ValueError("Input must contain at least one Moon site and one Mars site.")

    output_dir.mkdir(parents=True, exist_ok=True)
    sites_path = output_dir / "sites.json"
    if sites_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {sites_path}")
    merged_sites: dict[str, Any] = {}
    if existing_sites_path and existing_sites_path.exists():
        merged_sites = _read_json(existing_sites_path)
        if not isinstance(merged_sites, dict):
            raise ValueError(f"Existing sites file must be a JSON object: {existing_sites_path}")
    merged_sites["moon"] = grouped_rows["moon"]
    merged_sites["mars"] = grouped_rows["mars"]
    sites_path.write_text(json.dumps(merged_sites, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    index_paths = {}
    for body in ("moon", "mars"):
        index_path = output_dir / f"{body}_index.npz"
        if index_path.exists() and not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {index_path}")
        _write_index(index_path, grouped_arrays[body])
        index_paths[body] = index_path
    return {"sites": sites_path, **index_paths}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Full-data JSON containing engine_features.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "generated", help="Directory for sites.json and Moon/Mars indices.")
    parser.add_argument("--existing-sites", type=Path, default=DEFAULT_EXISTING_SITES, help="Existing sites.json used to preserve Earth rows.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in --output-dir.")
    args = parser.parse_args()
    try:
        outputs = build(args.input, args.output_dir, args.existing_sites, args.overwrite)
    except (FileExistsError, OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

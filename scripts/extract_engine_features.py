"""Build the 12 Moon/Mars engine feature records from local source datasets.

Reads only site coordinates and IDs from Full Real Data. It calculates geology
from map polygons and mineralogy from raster maps, then writes a feature CSV and
a copy of the JSON with the requested ``engine_features`` object.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.windows import Window
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import extract_similarity_features as source  # noqa: E402

DATA_DIR = ROOT / "app" / "data"
EXTRAS_DIR = DATA_DIR / "extras"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
SOURCE_MAPS_DIR = PROCESSED_DIR / "engine_source_maps"
INPUT_JSON = DATA_DIR / "moon_mars_full_real_data.json"
OUTPUT_JSON = PROCESSED_DIR / "moon_mars_full_real_data_features.json"
OUTPUT_CSV = PROCESSED_DIR / "engine_features.csv"
MOON_CRATERS = REFERENCE_DIR / "craters_moon" / "lunar_crater_database_robbins_2018.csv"

TOPOGRAPHY_KEYS = (
    "relief_m",
    "slope_mean_deg",
    "slope_p90_deg",
    "ruggedness_tri_m",
    "roughness_rms_m",
    "hypsometric_integral",
)
MINERAL_KEYS = tuple(source.MINERALS)
GEOLOGY_KEYS = tuple(source.LITHOLOGY_CLASSES)
TES_BANDS = {
    "plagioclase": 2,
    "low_ca_pyroxene": 4,
    "high_ca_pyroxene": 5,
    "olivine": 6,
    "hematite": 7,
    "sulfates": 8,
    "carbonates": 9,
    "sheet_silicates_high_si_glass": 10,
}
TES_MAX_ABUNDANCE = 0.2

POLAR_MAPS = {
    "plagioclase": "gridded_plagioclase_mosaic_{hemisphere}_pole.tiff.zip",
    "low_ca_pyroxene": "gridded_low_calcium_pyroxene_mosaic_{hemisphere}_pole.tiff.zip",
    "high_ca_pyroxene": "gridded_high_calcium_pyroxene_mosaic_{hemisphere}_pole.tiff.zip",
    "olivine": "gridded_olivine_mosaic_{hemisphere}_pole.tiff.zip",
    "feo": "gridded_feo_mosaic_{hemisphere}_pole.tif.zip",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=INPUT_JSON)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument(
        "--skip-live-moon-mi",
        action="store_true",
        help="Do not sample the configured equatorial Kaguya MI source URLs.",
    )
    parser.add_argument(
        "--skip-mars-craters",
        action="store_true",
        help="Leave Mars impact flags null instead of querying the USGS crater service.",
    )
    parser.add_argument(
        "--cached-moon-mi-csv",
        type=Path,
        default=PROCESSED_DIR / "mineralogy_sites.csv",
        help="Use previously extracted equatorial Kaguya values if remote maps are unavailable.",
    )
    args = parser.parse_args()

    catalog = json.loads(args.input_json.read_text(encoding="utf-8"))
    sites = _get_sites(catalog)
    if sites["body"].str.lower().value_counts().to_dict() != {"moon": 6, "mars": 6}:
        raise ValueError("Expected exactly six Moon and six Mars sites in the input JSON.")

    _extract_source_archives()
    geology, coverage = _extract_geology(sites)
    minerals = _extract_mineralogy(
        sites, not args.skip_live_moon_mi, args.cached_moon_mi_csv
    )
    rows: list[dict[str, Any]] = []

    lunar_craters = source.craters_csv(MOON_CRATERS) if MOON_CRATERS.exists() else None
    for site in sites.to_dict(orient="records"):
        site_id = site["site_id"]
        geo = geology[site_id]
        min_features = minerals[site_id]
        topo = _topography(catalog[site_id])
        flags = _flags(
            site,
            geo,
            coverage[site_id],
            lunar_craters,
            not args.skip_mars_craters,
        )
        engine = {
            "topography": topo,
            "geology_fractions": geo,
            "mineralogy": min_features,
            "flags": flags,
        }
        catalog[site_id]["engine_features"] = engine
        catalog[site_id]["engine_features_provenance"] = _provenance(
            catalog[site_id], site, coverage[site_id], min_features
        )
        rows.append(
            {
                **{key: site[key] for key in ("site_id", "cell_id", "body", "name", "lat", "lon")},
                **topo,
                **{f"geo_{key}": geo[key] for key in GEOLOGY_KEYS},
                **{f"min_{key}": min_features[key] for key in MINERAL_KEYS},
                **flags,
                "geology_coverage": coverage[site_id],
                "geo_vector": _json_cell([geo[key] for key in GEOLOGY_KEYS]),
                "min_vector": _json_cell([min_features[key] for key in MINERAL_KEYS]),
                "engine_features_json": _json_cell(engine),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    args.output_json.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} site rows to {args.output_csv}")
    print(f"Added engine_features for {len(rows)} sites to {args.output_json}")
    missing_geo = [site_id for site_id, value in coverage.items() if value < 0.999]
    missing_minerals = [key for key in MINERAL_KEYS if all(pd.isna(v[key]) for v in minerals.values())]
    if missing_geo:
        print("Geology map coverage is incomplete for:", ", ".join(missing_geo))
    if missing_minerals:
        print("Mineral dimensions without source data:", ", ".join(missing_minerals))


def _get_sites(catalog: dict[str, dict[str, Any]]) -> pd.DataFrame:
    records = []
    for key, record in catalog.items():
        if str(record.get("body", "")).lower() not in {"moon", "mars"}:
            continue
        records.append(
            {
                "site_id": record.get("site_id", key),
                "cell_id": record.get("cell_id", ""),
                "body": str(record["body"]).lower(),
                "name": record.get("name", key),
                "lat": float(record["lat"]),
                "lon": float(record["lon"]),
            }
        )
    return pd.DataFrame(records)


def _extract_source_archives() -> None:
    """Expand GIS/rasters once; GeoTIFF polar georeferencing uses sidecar files."""
    SOURCE_MAPS_DIR.mkdir(parents=True, exist_ok=True)
    archive_names = (
        "jezero_database.zip",
        "Mars_Elysium_5M_I2147_Viking_GIS.zip",
        "GIS_Arcadia.zip",
        "GIS_Mela.zip",
    )
    for archive_name in archive_names:
        archive = EXTRAS_DIR / archive_name
        if not archive.exists():
            continue
        destination = SOURCE_MAPS_DIR / archive.stem
        if archive_name == "jezero_database.zip":
            needed = destination / "sim3464_JezeroNili_FINAL" / "Geodatabase" / "Nili.gdb" / "gdb"

            def selector(name: str) -> bool:
                return "/Geodatabase/Nili.gdb/" in f"/{name}"
        else:
            target_shp = {
                "Mars_Elysium_5M_I2147_Viking_GIS.zip": "I2147_GeoUnits",
                "GIS_Arcadia.zip": "MC03_Arcadia_GeoUnits",
                "GIS_Mela.zip": "MC18_Coprates_GeoUnits",
            }[archive_name]
            needed = next(destination.rglob(f"{target_shp}.shp"), destination / f"{target_shp}.shp")
            def selector(name: str, target_shp: str = target_shp) -> bool:
                return Path(name).name.startswith(target_shp) and Path(name).suffix.lower() in {
                    ".shp",
                    ".dbf",
                    ".shx",
                    ".prj",
                    ".cpg",
                }
        if not needed.exists():
            _safe_extract(archive, destination, selector)

    polar_dir = SOURCE_MAPS_DIR / "lunar_polar"
    polar_dir.mkdir(parents=True, exist_ok=True)
    for filename in POLAR_MAPS.values():
        for hemisphere in ("north", "south"):
            archive = EXTRAS_DIR / filename.format(hemisphere=hemisphere)
            if not archive.exists():
                continue
            extracted = polar_dir / archive.stem
            if not any(extracted.glob("*.tif")) and not any(extracted.glob("*.tiff")):
                _safe_extract(
                    archive,
                    extracted,
                    lambda name: (
                        Path(name).suffix.lower() in {".tif", ".tiff"}
                        or name.lower().endswith(".tfwx")
                        or name.lower().endswith(".aux.xml")
                    ),
                )


def _safe_extract(archive: Path, destination: Path, selector: Any = None) -> None:
    """Extract a trusted local archive without allowing paths outside destination."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        members = [member for member in zipped.infolist() if selector is None or selector(member.filename)]
        for member in members:
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in archive {archive.name}: {member.filename}")
        for member in members:
            zipped.extract(member, destination)


def _extract_geology(sites: pd.DataFrame) -> tuple[dict[str, dict[str, float | None]], dict[str, float]]:
    map_frames: dict[str, tuple[gpd.GeoDataFrame, str, str, dict[str, str]]] = {}
    for body in ("moon", "mars"):
        map_dir = next(
            (
                path
                for path in (REFERENCE_DIR / f"{body}_geology", REFERENCE_DIR / f"{body}_geologia")
                if path.exists()
            ),
            REFERENCE_DIR / f"{body}_geologia",
        )
        frame, symbol_col, name_col = source.load_geology(body, map_dir)
        crosswalk_file = PROCESSED_DIR / f"harmonization_{body}.csv"
        if crosswalk_file.exists():
            table = pd.read_csv(crosswalk_file, dtype=str).fillna("")
            if {"unit", "class_id"} <= set(table.columns):
                crosswalk = dict(zip(table["unit"], table["class_id"], strict=True))
            elif {"unidad", "clase"} <= set(table.columns):
                crosswalk = dict(zip(table["unidad"], table["clase"], strict=True))
            else:
                raise ValueError(f"Crosswalk must contain unit,class_id: {crosswalk_file}")
        else:
            crosswalk = {}
        map_frames[body] = (frame, symbol_col, name_col, crosswalk)

    mars_maps = _load_regional_maps()
    result: dict[str, dict[str, float | None]] = {}
    coverages: dict[str, float] = {}
    for site in sites.to_dict(orient="records"):
        site_id = site["site_id"]
        body = site["body"]
        chosen = None
        if body == "mars":
            chosen = _regional_geology(site, mars_maps)
        if chosen is None or chosen[1] <= 0:
            frame, symbol_col, _name_col, crosswalk = map_frames[body]
            result[site_id], coverages[site_id] = _polygon_fractions(
                frame,
                symbol_col,
                crosswalk,
                float(site["lat"]),
                float(site["lon"]),
                body,
            )
        else:
            result[site_id], coverages[site_id] = chosen
        result[site_id] = _normalize_classified_fractions(result[site_id])
    return result, coverages


def _normalize_classified_fractions(
    fractions: dict[str, float | None],
) -> dict[str, float | None]:
    """Normalize mapped classes to one while keeping missing classes null."""
    total = sum(value for value in fractions.values() if value is not None)
    if total <= 0:
        return fractions
    return {
        key: (value / total if value is not None else None)
        for key, value in fractions.items()
    }


def _load_regional_maps() -> dict[str, gpd.GeoDataFrame]:
    maps: dict[str, gpd.GeoDataFrame] = {}
    candidates = {
        "jezero": next(SOURCE_MAPS_DIR.glob("jezero_database/**/Nili.gdb"), None),
        "elysium": next(SOURCE_MAPS_DIR.glob("Mars_Elysium*/**/I2147_GeoUnits.shp"), None),
        "melas": next(SOURCE_MAPS_DIR.glob("GIS_Mela/**/MC18_Coprates_GeoUnits.shp"), None),
        "arcadia": next(SOURCE_MAPS_DIR.glob("GIS_Arcadia/**/MC03_Arcadia_GeoUnits.shp"), None),
    }
    for key, path in candidates.items():
        if path is None:
            continue
        if key == "jezero":
            maps[key] = gpd.read_file(path, layer="GeoUnits_v1")
        else:
            maps[key] = gpd.read_file(path)
    return maps


def _regional_geology(
    site: dict[str, Any], maps: dict[str, gpd.GeoDataFrame]
) -> tuple[dict[str, float | None], float] | None:
    site_id = site["site_id"]
    key = {
        "mars_jezero_crater": "jezero",
        "mars_arcadia_planitia": "arcadia",
        "mars_elysium_planitia": "elysium",
        "mars_melas_chasma": "melas",
    }.get(site_id)
    frame = maps.get(key or "")
    if frame is None:
        return None

    lat, lon = float(site["lat"]), float(source._wrap180(float(site["lon"])))
    sphere = CRS.from_proj4("+proj=longlat +R=3396190 +no_defs")
    x, y = Transformer.from_crs(sphere, frame.crs, always_xy=True).transform(lon, lat)
    if not np.isfinite([x, y]).all():
        return None
    search = box(x - 30000, y - 30000, x + 30000, y + 30000)
    nearby = frame[frame.geometry.intersects(search)].copy()
    if nearby.empty:
        return None
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +R=3396190 +units=m +no_defs"
    )
    nearby = nearby.to_crs(local_crs)
    half = source.WINDOW_KM * 500.0
    window = box(-half, -half, half, half)
    nearby["window_area"] = nearby.geometry.intersection(window).area
    nearby = nearby[nearby["window_area"] > 0]
    if nearby.empty:
        return None

    unit_column = next(
        (name for name in ("name", "U_Name", "UName", "U_Symbol", "unit", "Unit") if name in nearby),
        None,
    )
    if unit_column is None:
        return None
    descriptions = (
        nearby["name"].fillna("").astype(str)
        if "name" in nearby
        else nearby[unit_column].fillna("").astype(str)
    )
    symbols = (
        nearby["symbol"].fillna("").astype(str)
        if "symbol" in nearby
        else nearby["U_Symbol"].fillna("").astype(str)
        if "U_Symbol" in nearby
        else nearby[unit_column].fillna("").astype(str)
    )
    mapped = np.zeros(len(GEOLOGY_KEYS), dtype=float)
    for area, name, symbol in zip(nearby["window_area"], descriptions, symbols, strict=True):
        class_id = _regional_class(key or "", name, symbol)
        if class_id in GEOLOGY_KEYS:
            mapped[GEOLOGY_KEYS.index(class_id)] += float(area)
    total_window = (source.WINDOW_KM * 1000.0) ** 2
    coverage = min(float(mapped.sum() / total_window), 1.0)
    if coverage == 0:
        return {class_id: None for class_id in GEOLOGY_KEYS}, 0.0
    return {
        class_id: float(mapped[index] / total_window)
        for index, class_id in enumerate(GEOLOGY_KEYS)
    }, coverage


def _regional_class(map_key: str, description: str, symbol: str) -> str | None:
    text = f"{description} {symbol}".lower().replace("_", " ")
    if map_key == "jezero":
        if re.search(r"fan unit|eolian bedform", text):
            return "clastic_sedimentary"
        if re.search(r"crater rim", text):
            return "impact_breccia_regolith"
        if re.search(r"jezero floor", text):
            return "basaltic_volcanic"
        return None
    if map_key == "elysium" and "ridged plains material" in text:
        return "basaltic_volcanic"
    return None


def _polygon_fractions(
    frame: gpd.GeoDataFrame,
    symbol_col: str,
    crosswalk: dict[str, str],
    lat: float,
    lon: float,
    body: str,
) -> tuple[dict[str, float | None], float]:
    radius_m = source.RADIO_KM[body] * 1000.0
    geographic = source._a_esfera(frame, radius_m)
    uses_negative_longitudes = geographic.total_bounds[0] < -1.0
    center_lon = float(source._wrap180(lon)) if uses_negative_longitudes else lon % 360.0
    bounds = source.bbox_km(lat, center_lon, source.WINDOW_KM, source.RADIO_KM[body])
    margin = 0.05
    nearby = geographic.cx[
        center_lon - bounds["delta_lon_deg"] / 2 - margin : center_lon
        + bounds["delta_lon_deg"] / 2
        + margin,
        bounds["minlat"] - margin : bounds["maxlat"] + margin,
    ]
    window_area = (source.WINDOW_KM * 1000.0) ** 2
    if nearby.empty:
        return {class_id: None for class_id in GEOLOGY_KEYS}, 0.0
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={center_lon} +R={radius_m} +units=m +no_defs"
    )
    nearby = nearby.to_crs(local_crs)
    half = source.WINDOW_KM * 500.0
    window = box(-half, -half, half, half)
    mapped = np.zeros(len(GEOLOGY_KEYS), dtype=float)
    for geometry, unit in zip(nearby.geometry, nearby[symbol_col].astype(str), strict=True):
        class_id = crosswalk.get(unit)
        if class_id not in GEOLOGY_KEYS or geometry is None or geometry.is_empty:
            continue
        mapped[GEOLOGY_KEYS.index(class_id)] += geometry.intersection(window).area
    coverage = min(float(mapped.sum() / window_area), 1.0)
    if coverage == 0:
        return {class_id: None for class_id in GEOLOGY_KEYS}, 0.0
    return {
        class_id: float(mapped[index] / window_area)
        for index, class_id in enumerate(GEOLOGY_KEYS)
    }, coverage


def _extract_mineralogy(
    sites: pd.DataFrame, sample_live_mi: bool, cached_mi_csv: Path
) -> dict[str, dict[str, float | None]]:
    mars = _sample_tes_stack(sites[sites["body"] == "mars"])
    polar_maps = _get_polar_map_paths()
    cached_mi = {}
    if cached_mi_csv.exists():
        cache = pd.read_csv(cached_mi_csv)
        cached_mi = {
            row["site_id"]: {key: _finite_number(row.get(key)) for key in MINERAL_KEYS}
            for row in cache[cache["body"].str.lower() == "moon"].to_dict(orient="records")
        }
    result: dict[str, dict[str, float | None]] = {}
    for site in sites.to_dict(orient="records"):
        features = {key: None for key in MINERAL_KEYS}
        if site["body"] == "mars":
            raw = mars[site["site_id"]]
            features["plagioclase_feldspar"] = _scale_tes(raw.get("plagioclase"))
            low = _scale_tes(raw.get("low_ca_pyroxene"))
            high = _scale_tes(raw.get("high_ca_pyroxene"))
            if low is not None and high is not None:
                features["pyroxene"] = float(np.clip((low + high) / 2, 0, 1))
            features["olivine"] = _scale_tes(raw.get("olivine"))
            features["fe_oxides"] = _scale_tes(raw.get("hematite"))
            features["sulfates"] = _scale_tes(raw.get("sulfates"))
            features["carbonates"] = _scale_tes(raw.get("carbonates"))
            # TES combines sheet silicates and high-Si glass in one layer; it
            # cannot supply these two separate fields without inventing a split.
        else:
            lat = float(site["lat"])
            if abs(lat) >= 50.0:
                hemisphere = "north" if lat > 0 else "south"
                for feature, raw_key in (
                    ("plagioclase_feldspar", "plagioclase"),
                    ("olivine", "olivine"),
                ):
                    path = polar_maps.get((raw_key, hemisphere))
                    raw = _sample_polar(path, lat, float(site["lon"])) if path else None
                    features[feature] = _bounded(raw)
                low_path = polar_maps.get(("low_ca_pyroxene", hemisphere))
                high_path = polar_maps.get(("high_ca_pyroxene", hemisphere))
                low = _sample_polar(low_path, lat, float(site["lon"])) if low_path else None
                high = _sample_polar(high_path, lat, float(site["lon"])) if high_path else None
                if low is not None and high is not None:
                    features["pyroxene"] = float(np.clip(low + high, 0.0, 1.0))
                feo_path = polar_maps.get(("feo", hemisphere))
                feo = _sample_polar(feo_path, lat, float(site["lon"])) if feo_path else None
                features["fe_oxides"] = (
                    float(np.clip(feo / 25.0, 0.0, 1.0)) if feo is not None else None
                )
            elif sample_live_mi:
                raw = _sample_moon_mi(lat, float(site["lon"]))
                cache = cached_mi.get(site["site_id"], {})
                for key in ("plagioclase_feldspar", "pyroxene", "olivine", "fe_oxides"):
                    if raw.get(key) is None:
                        raw[key] = cache.get(key)
                features.update(raw)
        result[site["site_id"]] = features
    return result


def _get_polar_map_paths() -> dict[tuple[str, str], Path]:
    maps: dict[tuple[str, str], Path] = {}
    root = SOURCE_MAPS_DIR / "lunar_polar"
    for raw_key, pattern in POLAR_MAPS.items():
        for hemisphere in ("north", "south"):
            archive = EXTRAS_DIR / pattern.format(hemisphere=hemisphere)
            if not archive.exists():
                continue
            files = list((root / archive.stem).glob("*.tif")) + list(
                (root / archive.stem).glob("*.tiff")
            )
            if files:
                maps[(raw_key, hemisphere)] = files[0]
    return maps


def _sample_polar(path: Path, lat: float, lon: float) -> float | None:
    """Sample polar maps whose WKT/transform are carried in ArcGIS sidecars."""
    hemisphere = "north" if lat > 0 else "south"
    center_crs = CRS.from_proj4(
        f"+proj=stere +lat_0={90 if hemisphere == 'north' else -90} "
        f"+lat_ts={90 if hemisphere == 'north' else -90} +lon_0=0 "
        "+R=1737400 +units=m +no_defs"
    )
    moon_geo = CRS.from_proj4("+proj=longlat +R=1737400 +no_defs")
    x, y = Transformer.from_crs(moon_geo, center_crs, always_xy=True).transform(
        float(source._wrap180(lon)), lat
    )
    pixel_size = (2.0 * 1_267_472.0) / 2529.0
    transform = Affine(pixel_size, 0, -1_267_472.0, 0, -pixel_size, 1_267_472.0)
    half = source.WINDOW_KM * 500.0
    col0, row0 = (~transform) * (x - half, y + half)
    col1, row1 = (~transform) * (x + half, y - half)
    c0, r0 = max(0, math.floor(col0)), max(0, math.floor(row0))
    c1, r1 = min(2529, math.ceil(col1)), min(2529, math.ceil(row1))
    if c1 <= c0 or r1 <= r0:
        return None
    with rasterio.open(path) as dataset:
        values = dataset.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(float)
        if dataset.nodata is not None:
            values[values == dataset.nodata] = np.nan
    if "feo" in path.name.lower():
        values[values < 0] = np.nan
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def _sample_moon_mi(lat: float, lon: float) -> dict[str, float | None]:
    raw = {key: None for key in MINERAL_KEYS}
    try:
        values = {
            key: source.sample_raster(raster, lat, lon, "moon") if raster else None
            for key, raster in source.MI_RASTERS.items()
        }
    except Exception:
        # A single unavailable map must not discard the other mineral maps.
        values = {}
        for key, raster in source.MI_RASTERS.items():
            try:
                values[key] = source.sample_raster(raster, lat, lon, "moon") if raster else None
            except Exception:
                values[key] = None
    raw["plagioclase_feldspar"] = _bounded(values.get("plagioclase"))
    raw["olivine"] = _bounded(values.get("olivine"))
    low, high = values.get("lcp"), values.get("hcp")
    if low is not None and high is not None and np.isfinite([low, high]).all():
        raw["pyroxene"] = float(np.clip(float(low) + float(high), 0.0, 1.0))
    feo = values.get("feo")
    if feo is not None and np.isfinite(feo):
        raw["fe_oxides"] = float(np.clip(float(feo) / 25.0, 0.0, 1.0))
    return raw


def _sample_tes_stack(sites: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    path = EXTRAS_DIR / "TES_mineral_maps_4_ppd.ascii.gz"
    if not path.exists():
        return {site_id: {} for site_id in sites["site_id"]}

    # ASU's archive stores 13 720 x 1440 maps in order documented at
    # https://tes.mars.asu.edu/products/. Stream row windows to keep memory low.
    windows: dict[str, tuple[int, int, int, int]] = {}
    for site in sites.to_dict(orient="records"):
        lat, lon = float(site["lat"]), float(source._wrap180(float(site["lon"])))
        bounds = source.bbox_km(lat, lon, source.WINDOW_KM, source.RADIO_KM["mars"])
        row0 = max(0, int(math.floor((90.0 - bounds["maxlat"]) * 4)))
        row1 = min(720, int(math.ceil((90.0 - bounds["minlat"]) * 4)))
        col0 = max(0, int(math.floor((lon - bounds["delta_lon_deg"] / 2 + 180.0) * 4)))
        col1 = min(1440, int(math.ceil((lon + bounds["delta_lon_deg"] / 2 + 180.0) * 4)))
        windows[site["site_id"]] = (row0, row1, col0, col1)

    sums = {site_id: np.zeros(13, dtype=float) for site_id in windows}
    counts = {site_id: np.zeros(13, dtype=np.int64) for site_id in windows}
    row_index = 0
    with gzip.open(path, "rt", encoding="ascii") as stream:
        for line in stream:
            if not line.strip():
                continue
            band = row_index // 720
            raster_row = row_index % 720
            if band >= 13:
                raise ValueError("TES mineral stack has more than 13 layers.")
            for site_id, (row0, row1, col0, col1) in windows.items():
                if row0 <= raster_row < row1:
                    row = np.fromstring(line, sep=" ", dtype=np.float32)
                    if row.size != 1440:
                        raise ValueError("Unexpected width in TES 4 ppd ASCII map.")
                    values = row[col0:col1]
                    valid = np.isfinite(values) & (values >= 0)
                    sums[site_id][band] += float(values[valid].sum())
                    counts[site_id][band] += int(valid.sum())
            row_index += 1
    if row_index != 13 * 720:
        raise ValueError(f"Expected 9360 TES rows; found {row_index}.")

    result: dict[str, dict[str, float | None]] = {}
    for site_id in windows:
        result[site_id] = {
            name: (float(sums[site_id][band] / counts[site_id][band]) if counts[site_id][band] else None)
            for name, band in TES_BANDS.items()
        }
    return result


def _scale_tes(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(np.clip(value / TES_MAX_ABUNDANCE, 0.0, 1.0))


def _bounded(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        if not np.isfinite(float(value)):
            return None
    except (TypeError, ValueError):
        return None
    return float(np.clip(float(value), 0.0, 1.0))


def _topography(record: dict[str, Any]) -> dict[str, float | None]:
    existing = record.get("topography_2d_dem", {})
    return {
        key: _finite_number(existing.get(key))
        for key in TOPOGRAPHY_KEYS
    }


def _provenance(
    record: dict[str, Any],
    site: dict[str, Any],
    geology_coverage: float,
    minerals: dict[str, float | None],
) -> dict[str, Any]:
    """Describe the source and method behind each engine feature group."""
    body = site["body"]
    topography_source = record.get("topography_2d_dem", {})
    topography = {
        key: topography_source[key]
        for key in (
            "source",
            "granule_id",
            "img_url",
            "lbl_url",
            "native_gsd_m",
            "window_size_km",
            "resampled_grid_shape",
            "resampled_gsd_m",
        )
        if key in topography_source
    }

    regional_source = {
        "mars_jezero_crater": "jezero_database.zip / USGS SIM 3464 Nili.gdb",
        "mars_elysium_planitia": "Mars_Elysium_5M_I2147_Viking_GIS.zip",
        "mars_arcadia_planitia": "GIS_Arcadia.zip",
        "mars_melas_chasma": "GIS_Mela.zip",
    }.get(site["site_id"])
    if body == "moon":
        geology_sources = [
            "USGS Unified Geologic Map of the Moon GIS",
            "app/data/processed/harmonization_moon.csv, when present",
        ]
    else:
        geology_sources = [
            "USGS global Mars geology map in app/data/reference/mars_geologia or mars_geology",
            "app/data/processed/harmonization_mars.csv, when present",
        ]
        if regional_source and geology_coverage > 0:
            geology_sources.insert(0, regional_source)

    if body == "mars":
        mineral_sources = ["app/data/extras/TES_mineral_maps_4_ppd.ascii.gz (TES, 4 pixels/degree)"]
        mineral_method = (
            "25 km window mean; abundance divided by 0.2 and clipped to [0, 1]. "
            "TES combines sheet silicates and high-silica glass; it has no water-ice layer."
        )
    elif abs(float(site["lat"])) >= 50:
        hemisphere = "north" if float(site["lat"]) > 0 else "south"
        mineral_sources = [
            f"app/data/extras/{pattern.format(hemisphere=hemisphere)}"
            for pattern in POLAR_MAPS.values()
        ]
        mineral_method = "Mean of valid pixels in the 25 km window; FeO divided by 25."
    else:
        mineral_sources = [
            "Kaguya MI mineral mosaics configured in scripts/extract_similarity_features.py",
            "app/data/processed/mineralogy_sites.csv may supply values missing from the live maps",
        ]
        mineral_method = "Remote Kaguya MI sampling with cached CSV fallback for missing values."

    if body == "moon":
        impact_source = "USGS Robbins lunar crater catalog (local reference CSV)"
    else:
        impact_source = "USGS Astrogeology Robbins Mars crater API"
    return {
        "topography": topography,
        "geology_fractions": {
            "sources": geology_sources,
            "window_size_km": source.WINDOW_KM,
            "classified_window_fraction": geology_coverage,
            "method": "Fractions normalized within the classifiable area; coverage reports the classifiable share of the full window.",
        },
        "mineralogy": {
            "sources": mineral_sources,
            "window_size_km": source.WINDOW_KM,
            "method": mineral_method,
            "unavailable_fields": [key for key, value in minerals.items() if value is None],
        },
        "flags": {
            "ndvi": "Placeholder value 0.0; no NDVI raster was sampled.",
            "is_impact": impact_source,
            "has_ice": "Derived from geology_fractions.ice_polar when geology coverage is complete.",
            "is_volcanic": "Derived from basaltic_volcanic + pyroclastic geology fractions when coverage is complete.",
        },
    }


def _flags(
    site: dict[str, Any],
    geology: dict[str, float | None],
    coverage: float,
    lunar_craters: pd.DataFrame | None,
    query_mars_craters: bool,
) -> dict[str, float | bool | None]:
    body = site["body"]
    impact: bool | None = None
    try:
        if body == "moon" and lunar_craters is not None:
            impact = source.craterized_fraction(
                lunar_craters, float(site["lat"]), float(site["lon"]), "moon"
            ) >= source.IMPACT_THRESHOLD
        elif body == "mars" and query_mars_craters:
            crater_data = source.craters_mars_api(float(site["lat"]), float(site["lon"]))
            impact = source.craterized_fraction(
                crater_data, float(site["lat"]), float(site["lon"]), "mars"
            ) >= source.IMPACT_THRESHOLD
    except Exception:
        impact = None

    if coverage >= 0.999:
        ice = bool((geology["ice_polar"] or 0.0) >= source.ICE_THRESHOLD)
        volcanic = bool(
            (geology["basaltic_volcanic"] or 0.0) + (geology["pyroclastic"] or 0.0)
            >= source.VOLCANIC_THRESHOLD
        )
    else:
        ice = volcanic = None
    return {"ndvi": 0.0, "is_impact": impact, "has_ice": ice, "is_volcanic": volcanic}


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


if __name__ == "__main__":
    main()

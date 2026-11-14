"""Extrae seis métricas topográficas de ventanas 25 x 25 km en sitios.csv.

Lee DEM PDS3 locales del flujo ODE del notebook. Solo usa dependencias ya
presentes en el proyecto: NumPy (y biblioteca estándar).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "app" / "data"
SITES_CSV = DATA_DIR / "input" / "sitios.csv"
DEM_LABELS = {
    "moon": DATA_DIR / "raw/moon/topography/LUN-TOP-01/ldem_64.lbl",
    "mars": DATA_DIR / "raw/mars/topography/MAR-TOP-01/megt90n000eb.lbl",
}
DEM_URLS = {
    "moon": "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img",
    "mars": "https://pds-geosciences.wustl.edu/mgs/urn-nasa-pds-mgs_mola_topography_derived/meg016",
}
DEM_FILES = {
    "moon": ("ldem_64.lbl", "ldem_64.img"),
    "mars": ("megt90n000eb.lbl", "megt90n000eb.img"),
}
PLANET_RADIUS_KM = {"moon": 1737.4, "mars": 3396.0}
METRICS = (
    "relief_m",
    "slope_mean_deg",
    "slope_p90_deg",
    "ruggedness_tri_m",
    "roughness_rms_m",
    "hypsometric_integral",
)


def _label(path: Path) -> tuple[dict[str, dict[str, str]], Path]:
    objects: dict[str, dict[str, str]] = {"": {}}
    stack = [""]
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("/*"):
            continue
        match = re.match(r"^OBJECT\s*=\s*(\S+)", line, re.I)
        if match:
            name = match.group(1).upper()
            stack.append(name)
            objects.setdefault(name, {})
        elif re.match(r"^END_OBJECT", line, re.I):
            if len(stack) > 1:
                stack.pop()
        elif line.upper() == "END":
            break
        elif "=" in line:
            key, value = line.split("=", 1)
            objects[stack[-1]].setdefault(key.strip().upper(), value.strip())

    def get(key: str, default: str | None = None) -> str | None:
        for scope in ("IMAGE", "IMAGE_MAP_PROJECTION", ""):
            if key in objects.get(scope, {}):
                return objects[scope][key]
        for values in objects.values():
            if key in values:
                return values[key]
        return default

    def number(key: str, default: float | None = None) -> float:
        value = get(key)
        found = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", value or "")
        if found:
            return float(found.group())
        if default is None:
            raise ValueError(f"Falta {key} en la etiqueta {path}")
        return default

    projection = (get("MAP_PROJECTION_TYPE", "") or "").upper()
    if not any(kind in projection for kind in ("CYLINDRICAL", "EQUIRECTANGULAR")):
        raise ValueError(f"Proyección PDS3 no soportada: {projection or 'desconocida'}")
    try:
        image_obj = objects["IMAGE"]
    except KeyError as exc:
        raise ValueError(f"La etiqueta no contiene OBJECT=IMAGE: {path}") from exc
    description = " ".join(image_obj.values()).upper()
    if not any(term in description for term in ("HEIGHT", "ELEVATION", "TOPOGRAPH", "RADIUS")):
        raise ValueError(
            f"{path.name} no parece un DEM (IMAGE={image_obj.get('NAME', 'sin nombre')}). "
            "El insumo debe ser un producto de elevación MOLA/MEGDR, no COUNTS_PER_BIN."
        )

    pointer = objects[""].get("^IMAGE") or image_obj.get("^IMAGE")
    record_bytes = int(number("RECORD_BYTES", 1))
    offset = 0
    filename = None
    if pointer:
        match = re.search(r'"([^"]+)"', pointer)
        if match:
            filename = match.group(1)
        pos = re.search(r"(?:,\s*)?(\d+)\s*(<BYTES>)?", pointer, re.I)
        if pos:
            unit_bytes = bool(pos.group(2))
            offset = (int(pos.group(1)) - 1) * (1 if unit_bytes else record_bytes)
    image_path = path.parent / filename if filename else path.with_suffix(".img")
    if not image_path.exists():
        image_path = path.with_suffix(".IMG")
    if not image_path.exists():
        raise FileNotFoundError(f"No se encuentra el raster asociado a {path}")

    bits = int(number("SAMPLE_BITS", 16))
    sample_type = (get("SAMPLE_TYPE", "") or "").upper()
    if bits not in (8, 16, 32, 64):
        raise ValueError(f"SAMPLE_BITS no soportado: {bits}")
    endian = "<" if sample_type.startswith(("PC", "LSB", "VAX")) or "LSB" in sample_type else ">"
    kind = "f" if "REAL" in sample_type or "FLOAT" in sample_type else ("u" if "UNSIGNED" in sample_type else "i")
    dtype = np.dtype("u1" if bits == 8 and kind == "u" else "i1" if bits == 8 else f"{endian}{kind}{bits // 8}")
    lines, samples = int(number("LINES")), int(number("LINE_SAMPLES"))
    def text_num(key: str, default: float = 0.0) -> float:
        return number(key, default)

    meta: dict[str, Any] = {
        "dtype": dtype,
        "data_offset": offset,
        "lines": lines,
        "samples": samples,
        "resolution": number("MAP_RESOLUTION"),
        "line_offset": number("LINE_PROJECTION_OFFSET"),
        "sample_offset": number("SAMPLE_PROJECTION_OFFSET"),
        "center_lon": text_num("CENTER_LONGITUDE"),
        "center_lat": text_num("CENTER_LATITUDE"),
        "radius_km": number("A_AXIS_RADIUS", 0.0),
        "scale": number("SCALING_FACTOR", 1.0),
        "offset": number("OFFSET", 0.0),
        "image_path": image_path,
        "nulls": [number(k) for k in ("MISSING_CONSTANT", "CORE_NULL", "NULL_CONSTANT", "INVALID_CONSTANT") if get(k)],
    }
    if meta["radius_km"] < 100:
        meta["radius_km"] = 0.0
    return meta, image_path


def _wrap180(value: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(value) + 180.0) % 360.0 - 180.0


def _download_dem(body: str, label_path: Path) -> None:
    """Always fetch the pinned DEM and label, replacing any local copies."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_name, image_name = DEM_FILES[body]
    for name in (label_name, image_name):
        destination = label_path.parent / name
        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"Descargando DEM {body}: {name}")
        try:
            urllib.request.urlretrieve(f"{DEM_URLS[body]}/{name}", temporary)
            if temporary.stat().st_size == 0:
                raise OSError(f"Descarga vacía: {name}")
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def _read_window(meta: dict[str, Any], lat: float, lon: float, body: str, side_km: float) -> tuple[np.ndarray, float, float]:
    radius_km = meta["radius_km"] or PLANET_RADIUS_KM[body]
    res = meta["resolution"]
    center_lat, center_lon = meta["center_lat"], meta["center_lon"]
    dlat = side_km / (radius_km * math.pi / 180)
    cos_lat = max(math.cos(math.radians(lat)), 1e-6)
    dlon = min(side_km / (radius_km * cos_lat * math.pi / 180), 359.0)
    lat_min, lat_max = max(-90, lat - dlat / 2), min(90, lat + dlat / 2)
    fac = res * math.cos(math.radians(center_lat))
    row0 = max(0, int(math.floor(meta["line_offset"] - lat_max * res)) - 1)
    row1 = min(meta["lines"] - 1, int(math.ceil(meta["line_offset"] - lat_min * res)) + 1)
    delta = float(_wrap180(lon - center_lon))
    col0 = int(math.floor(meta["sample_offset"] + (delta - dlon / 2) * fac)) - 1
    col1 = int(math.ceil(meta["sample_offset"] + (delta + dlon / 2) * fac)) + 1
    raw_cols = np.arange(col0, col1 + 1)
    if meta["samples"] / fac >= 359:
        cols = raw_cols % meta["samples"]
    else:
        valid = (raw_cols >= 0) & (raw_cols < meta["samples"])
        raw_cols, cols = raw_cols[valid], raw_cols[valid]
    rows = np.arange(row0, row1 + 1)
    if rows.size < 3 or cols.size < 3:
        raise ValueError("El DEM no cubre la ventana con al menos 3 x 3 píxeles")
    # Leer solo las filas necesarias. Un memmap del LDEM_64 completo puede
    # reservar >500 MB de espacio virtual en Windows aunque el recorte sea pequeño.
    row_indices = np.arange(row0, row1 + 1)
    row_bytes = meta["samples"] * meta["dtype"].itemsize
    z = np.empty((row_indices.size, cols.size), dtype=np.float64)
    with meta["image_path"].open("rb") as raster:
        for output_row, source_row in enumerate(row_indices):
            raster.seek(meta["data_offset"] + int(source_row) * row_bytes)
            full_row = np.fromfile(raster, dtype=meta["dtype"], count=meta["samples"])
            if full_row.size != meta["samples"]:
                raise ValueError(f"Fila incompleta al leer el DEM: {source_row}")
            z[output_row] = full_row[cols]
    z = z.astype(np.float64, copy=False)
    for null in meta["nulls"]:
        z[z == null] = np.nan
    z[np.abs(z) > 1e30] = np.nan
    z = z * meta["scale"] + (meta["offset"] if abs(meta["offset"]) < 500 else 0.0)
    # Algunos LDEM lunares almacenan kilómetros; MOLA MEGDR normalmente metros.
    finite = z[np.isfinite(z)]
    if finite.size and np.nanmax(np.abs(finite)) < 50:
        z *= 1000.0
    if np.isfinite(z).mean() < 0.8:
        raise ValueError(f"Cobertura válida insuficiente en ventana ({np.isfinite(z).mean():.0%})")
    # Píxel centro y resolución métrica local (proyección cilíndrica planetocéntrica).
    lats = (meta["line_offset"] - rows) / res
    dy = radius_km * 1000 * math.radians(1 / res)
    dx = radius_km * 1000 * np.cos(np.radians(lats)) * math.radians(1 / fac)
    # Corta al rectángulo geográfico de 25 km; omite el margen añadido al leer.
    keep_r = np.abs(lats - lat) <= dlat / 2 + 1e-10
    lons = (center_lon + (raw_cols - meta["sample_offset"]) / fac) % 360
    keep_c = np.abs(_wrap180(lons - lon)) <= dlon / 2 + 1e-10
    z = z[np.ix_(keep_r, keep_c)]
    if z.size < 9 or min(z.shape) < 3:
        raise ValueError("Ventana resultante demasiado pequeña para derivadas")
    return z, float(np.nanmedian(dx)), float(dy)


def _metrics(z: np.ndarray, dx: float, dy: float) -> dict[str, float]:
    valid = np.isfinite(z)
    if not valid.all():
        # Gradientes/vecindarios se calculan donde la ventana está completa para evitar
        # que valores faltantes generen pendientes espurias.
        z = z.copy()
        z[~valid] = np.nanmedian(z)
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    relief = zmax - zmin
    dz_south, dz_east = np.gradient(z, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(dz_east, dz_south)))
    # Riley TRI: raíz de la suma de diferencias cuadráticas con los ocho vecinos.
    padded = np.pad(z, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    squared = np.zeros_like(z)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                squared += (padded[1 + di:1 + di + z.shape[0], 1 + dj:1 + dj + z.shape[1]] - center) ** 2
    tri = np.sqrt(squared)
    windows = sliding_window_view(np.pad(z, 1, mode="edge"), (3, 3))
    rough = np.sqrt(np.mean((windows - np.mean(windows, axis=(-1, -2), keepdims=True)) ** 2, axis=(-1, -2)))
    return {
        "relief_m": relief,
        "slope_mean_deg": float(np.mean(slope)),
        "slope_p90_deg": float(np.percentile(slope, 90)),
        "ruggedness_tri_m": float(np.mean(tri)),
        "roughness_rms_m": float(np.mean(rough)),
        "hypsometric_integral": float((np.mean(z) - zmin) / relief) if relief else 0.0,
    }


def extract(
    sites_path: Path,
    output_path: Path,
    side_km: float,
    per_body: int,
    dem_labels: dict[str, Path] | None = None,
    download: bool = True,
) -> int:
    with sites_path.open(encoding="utf-8-sig", newline="") as stream:
        sites = list(csv.DictReader(stream))
    selected: list[dict[str, str]] = []
    for body in ("moon", "mars"):
        selected.extend([row for row in sites if row["cuerpo"].strip().lower() == body][:per_body])
    if len(selected) != 2 * per_body:
        raise ValueError(f"Se esperaban {per_body} sitios por cuerpo; encontrados {len(selected)} en {sites_path}")
    metas: dict[str, dict[str, Any]] = {}
    dem_labels = dem_labels or DEM_LABELS
    if download:
        if dem_labels != DEM_LABELS:
            raise ValueError("Las rutas DEM personalizadas no se pueden combinar con descarga reproducible")
        for body in ("moon", "mars"):
            _download_dem(body, dem_labels[body])
    for body in ("moon", "mars"):
        label = dem_labels[body]
        if not label.exists():
            raise FileNotFoundError(f"Falta el DEM PDS3 para {body}: {label}")
        metas[body], _ = _label(label)
    rows_out = []
    for site in selected:
        body = site["cuerpo"].strip().lower()
        z, dx, dy = _read_window(metas[body], float(site["lat"]), float(site["lon"]), body, side_km)
        rows_out.append({"nombre": site["nombre"], "cuerpo": body, **_metrics(z, dx, dy)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["nombre", "cuerpo", *METRICS]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)
    return len(rows_out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=Path, default=SITES_CSV, help="CSV con nombre,cuerpo,lat,lon")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "processed/topografia_sitios_25km.csv",
    )
    parser.add_argument("--side-km", type=float, default=25.0)
    parser.add_argument("--per-body", type=int, default=5, help="Primeras N ubicaciones de cada cuerpo según el CSV")
    parser.add_argument("--moon-dem", type=Path, default=DEM_LABELS["moon"], help="Etiqueta .LBL local si se usa --no-download")
    parser.add_argument("--mars-dem", type=Path, default=DEM_LABELS["mars"], help="Etiqueta .LBL local si se usa --no-download")
    parser.add_argument("--no-download", action="store_true", help="Procesar DEM locales sin descargarlos ni reemplazarlos")
    args = parser.parse_args()
    try:
        count = extract(
            args.sites,
            args.output,
            args.side_km,
            args.per_body,
            {"moon": args.moon_dem, "mars": args.mars_dem},
            download=not args.no_download,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error de extracción: {exc}", file=sys.stderr)
        return 2
    print(f"Listo: {count} sitios escritos en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

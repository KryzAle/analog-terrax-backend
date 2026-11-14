#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extraer_variables_similitud.py
==============================
Extrae, para una lista de sitios (Luna, Marte, Tierra), las variables que usa
similarity_engine.py:

  A. Topografia (6)  : relief_m, slope_mean_deg, slope_p90_deg, ruggedness_tri_m,
                       roughness_rms_m, hypsometric_integral      (ventana 25x25 km)
  B. Geologia  (7)   : fraccion de area de cada clase litologica armonizada
  C. Mineralogia (9) : indices 0-1 (o NaN si el cuerpo no los mide)
  D. Banderas        : ndvi, is_impact, has_ice, is_volcanic

Uso rapido
----------
  pip install numpy pandas requests rasterio geopandas shapely pyproj

  python scripts/extraer_variables_similitud.py --autotest            # prueba con datos sinteticos
  python scripts/extraer_variables_similitud.py --descargar           # baja mapas geologicos y crateres
  python scripts/extraer_variables_similitud.py --sitios app/data/input/sitios.csv \
        --salida salida/

sitios.csv necesita las columnas:  nombre,cuerpo,lat,lon     (cuerpo = moon | mars | earth)
Longitudes en grados Este (0-360 o -180..180, ambas valen).

Lo que NO se puede automatizar con certeza (ver los avisos [REVISAR] en el codigo):
  * La reclasificacion de unidades geologicas a las 7 clases: el script escribe un CSV
    borrador (unidades_<cuerpo>_revisar.csv). Revisalo y guardalo como
    harmonizacion_<cuerpo>.csv; el script usara ese archivo.
  * Mineralogia de Marte (CRISM): hay que indicar los rasteres de parametros espectrales.
  * Mineralogia de la Luna (Kaguya MI): hay que pegar las URLs de los mapas en MI_RASTERS.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# CONFIGURACION
# =============================================================================
RADIO_KM = {"moon": 1737.4, "mars": 3396.0, "earth": 6371.0}
VENTANA_KM = 25.0                      # lado de la ventana de analisis
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "app" / "data"
DIR_EXT = DATA_DIR / "reference"  # fuentes y mapas externos

URLS = {
    # Verificadas en las paginas de USGS Astrogeology / Astropedia:
    "geo_moon_zip": "https://asc-astropedia.s3.us-west-2.amazonaws.com/Moon/Geology/"
                    "Unified_Geologic_Map_of_the_Moon_GIS_v2.zip",             # ~224 MB
    "dem_mars": "https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif",  # ~2 GB
    "craters_moon": "https://astrogeology.usgs.gov/ckan/dataset/"
                    "f89f5478-b69a-486c-b9b5-30d7b0c5ad2b/resource/"
                    "c4f25cc2-4f8a-4207-a845-5e176da3ac5a/download/"
                    "lunar_crater_database_robbins_2018",                       # ~93 MB
    "craters_mars_api": "https://astrogeology.usgs.gov/pygeoapi/collections/mars/robbinsv1/items",
    # [REVISAR] El ZIP del mapa geologico de Marte (SIM 3292, ~790 MB) esta en
    # https://pubs.usgs.gov/publication/sim3292  ("Database ZIP"). Pega aqui su URL
    # o descargalo a mano y guardalo como app/data/reference/mars_geologia/SIM3292.zip
    "geo_mars_zip": None,
}

# [REVISAR] Mapas minerales Kaguya MI (Lemelin et al.). Pega la URL o ruta local de cada
# GeoTIFF desde https://astrogeology.usgs.gov/search/map/lunar-kaguya-multiband-imager-mosaics
MI_RASTERS = {
    "plagioclase": "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Lunar_MI_mineral_maps/Lunar_Kaguya_MIMap_MineralDeconv_PlagioclasePercent_50N50S.tif",
    "olivine":     "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Lunar_MI_mineral_maps/Lunar_Kaguya_MIMap_MineralDeconv_OlivinePercent_50N50S.tif",
    "lcp":         "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Lunar_MI_mineral_maps/Lunar_Kaguya_MIMap_MineralDeconv_OrthopyroxenePercent_50N50S.tif",
    "hcp":         "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Lunar_MI_mineral_maps/Lunar_Kaguya_MIMap_MineralDeconv_ClinopyroxenePercent_50N50S.tif",
    "feo":         "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Lunar_MI_mineral_maps/Lunar_Kaguya_MIMap_MineralDeconv_FeOWeightPercent_50N50S.tif",
}
# [REVISAR] Parametros espectrales CRISM (Viviano-Beck et al. 2014) como rasteres
# (mosaico/MRDR convertido a GeoTIFF, un parametro por archivo).
CRISM_RASTERS = {
    "olivine": None,            # OLINDEX3
    "pyroxene_lcp": None,       # LCPINDEX2
    "pyroxene_hcp": None,       # HCPINDEX2
    "phyllosilicates_clays": None,  # D2300 o BD1900_2
    "sulfates": None,           # SINDEX2
    "fe_oxides": None,          # BD530_2
}
# Valor del parametro que se mapea a indice 1.0 (clip a [0,1]). None -> se guarda el
# valor crudo en <col>_raw y el indice queda NaN hasta que definas la escala.
CRISM_VMAX = {k: None for k in CRISM_RASTERS}

CLASES = ["basaltic_volcanic", "feldspathic_plutonic", "pyroclastic",
          "impact_breccia_regolith", "clastic_sedimentary", "evaporite_chemical", "ice_polar"]
MINERALES = ["plagioclase_feldspar", "pyroxene", "olivine", "fe_oxides",
             "phyllosilicates_clays", "sulfates", "carbonates", "silica_glass", "h_water_ice"]

# Banderas
CRATER_MIN_KM = 5.0        # solo craters >= este diametro cuentan para is_impact
UMBRAL_IMPACT = 0.25       # fraccion de la ventana cubierta por craters
UMBRAL_VOLC = 0.50         # fraccion basaltic_volcanic + pyroclastic
UMBRAL_HIELO = 0.10        # fraccion ice_polar

# Reglas BORRADOR para sugerir clase a partir del nombre/simbolo de la unidad geologica.
# Son heuristicas: hay que revisarlas a mano (sobre todo en Marte).
REGLAS = [
    ("ice_polar", r"\bice\b|polar|glaci|permafrost|\bcap\b|vastitas|dorsa argentea|volatile"),
    ("evaporite_chemical", r"evaporit|sulf|chlorid|\bsalt|salar|carbonat"),
    ("pyroclastic", r"pyroclast|dark mantle|tephra|explosive|\bash\b"),
    ("basaltic_volcanic", r"\bmare\b|mare |basalt|volcan|\blava|flow|dome|shield|igneous|ridged plains"),
    ("clastic_sedimentary", r"sediment|fluvial|lacustr|channel|delta|alluvi|valley|\bfan\b|apron|"
                            r"layered|clastic|aeolian|eolian|dune|chaotic"),
    ("feldspathic_plutonic", r"\bterra\b|highland|anorthos|plutonic|massif|basement|crustal"),
    ("impact_breccia_regolith", r"crater|basin|ejecta|impact|breccia|regolith|\brim\b|peak|plains|"
                                r"cayley|apollonian|undivided"),
]


# =============================================================================
# 0. UTILIDADES
# =============================================================================
def bbox_km(lat, lon, lado_km, radio_km):
    dlat = math.degrees(lado_km / radio_km)
    dlon = dlat / max(math.cos(math.radians(lat)), 1e-6)
    return dict(center_lat=lat, center_lon=lon, minlat=lat - dlat / 2, maxlat=lat + dlat / 2,
                delta_lon_deg=dlon, delta_lat_deg=dlat)


def _wrap180(x):
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def descargar(url: str, destino: Path, forzar=False) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and not forzar:
        print(f"  = ya existe {destino}")
        return destino
    print(f"  v {url}\n    -> {destino}")
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "similitud-extractor/1.0"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        hecho = 0
        with open(destino, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                hecho += len(chunk)
                if total:
                    print(f"\r    {hecho/1e6:8.1f} / {total/1e6:.1f} MB", end="")
        print()
    return destino


def descargar_y_extraer(url: str, carpeta: Path) -> Path:
    """Descarga; si es un ZIP lo descomprime en `carpeta`."""
    carpeta.mkdir(parents=True, exist_ok=True)
    arch = descargar(url, carpeta / "descarga.bin")
    if not zipfile.is_zipfile(arch):
        raise ValueError(f"El archivo descargado no es un ZIP válido: {arch}")
    with zipfile.ZipFile(arch) as z:
        z.extractall(carpeta)
    print(f"  zip extraido en {carpeta}")
    return carpeta


def descargar_crateres_luna() -> Path:
    carpeta = DIR_EXT / "craters_moon"
    csv_esperado = carpeta / "lunar_crater_database_robbins_2018.csv"
    if csv_esperado.is_file():
        print(f"  = ya existe {csv_esperado}")
        return csv_esperado

    descargar_y_extraer(URLS["craters_moon"], carpeta)
    encontrados = list(carpeta.rglob(csv_esperado.name))
    if len(encontrados) != 1:
        raise FileNotFoundError(
            f"Se esperaba encontrar un CSV llamado {csv_esperado.name} dentro de {carpeta}; "
            f"se encontraron {len(encontrados)}."
        )
    if encontrados[0] != csv_esperado:
        shutil.move(str(encontrados[0]), str(csv_esperado))
    return csv_esperado


# =============================================================================
# 1. TOPOGRAFIA
# =============================================================================
def topo_features(z, dx_m, dy_m, min_validos=0.9):
    """
    Las 6 variables topograficas a partir de una matriz de elevacion (metros).
    z      : 2D (filas crecen hacia el SUR).
    dx_m   : tamano de pixel en x (m): escalar o vector por fila (varia con la latitud).
    dy_m   : tamano de pixel en y (m).
    USAR LA MISMA FUNCION PARA LOS SITIOS DE LA TIERRA, a la misma resolucion (~500 m).
    """
    z = np.asarray(z, dtype=float)
    nan_out = dict(relief_m=np.nan, slope_mean_deg=np.nan, slope_p90_deg=np.nan,
                   ruggedness_tri_m=np.nan, roughness_rms_m=np.nan, hypsometric_integral=np.nan)
    ok = np.isfinite(z)
    if z.ndim != 2 or min(z.shape) < 5 or ok.mean() < min_validos:
        return nan_out
    dx_row = np.full(z.shape[0], float(dx_m)) if np.ndim(dx_m) == 0 else np.asarray(dx_m, float)

    # pendiente (diferencias centrales en metros reales)
    gN = -np.gradient(z, axis=0) / dy_m
    gE = np.gradient(z, axis=1) / dx_row[:, None]
    pend = np.degrees(np.arctan(np.hypot(gE, gN)))

    # TRI: media de |z_centro - z_vecino| en las 8 celdas vecinas
    w = sliding_window_view(np.pad(z, 1, constant_values=np.nan), (3, 3))
    d = np.abs(w - z[..., None, None])
    d[..., 1, 1] = np.nan
    tri = np.nanmean(d, axis=(2, 3))

    # RMS sin tendencia planar (plano ajustado por minimos cuadrados, coordenadas en metros)
    ny, nx = z.shape
    yy, xx = np.meshgrid(np.arange(ny) * dy_m, np.arange(nx) * float(np.nanmean(dx_row)), indexing="ij")
    A = np.c_[xx[ok], yy[ok], np.ones(ok.sum())]
    coef, *_ = np.linalg.lstsq(A, z[ok], rcond=None)
    rms = float(np.sqrt(np.mean((z[ok] - A @ coef) ** 2)))

    zmin, zmax = np.nanmin(z), np.nanmax(z)
    hyp = float((np.nanmean(z) - zmin) / (zmax - zmin)) if zmax > zmin else np.nan

    return dict(relief_m=float(np.nanpercentile(z, 95) - np.nanpercentile(z, 5)),
                slope_mean_deg=float(np.nanmean(pend)),
                slope_p90_deg=float(np.nanpercentile(pend, 90)),
                ruggedness_tri_m=float(np.nanmean(tri)),
                roughness_rms_m=rms,
                hypsometric_integral=hyp)


def _recorte_centro(z, lats, dx_rows, dy_m, lado_km):
    """Recorta la ventana al tamano exacto lado_km x lado_km (en pixeles) alrededor del centro."""
    ny = int(round(lado_km * 1000.0 / dy_m))
    nx = int(round(lado_km * 1000.0 / float(np.nanmean(dx_rows))))
    ny, nx = min(ny, z.shape[0]), min(nx, z.shape[1])
    i0, j0 = (z.shape[0] - ny) // 2, (z.shape[1] - nx) // 2
    return z[i0:i0 + ny, j0:j0 + nx], dx_rows[i0:i0 + ny]


# ---- Luna (y cualquier mapa PDS3 cilindrico): lector tomado de tu notebook -----------------------
def leer_etiqueta_pds3(ruta_lbl) -> dict:
    objetos, pila = {"": {}}, [""]
    for linea in Path(ruta_lbl).read_text(errors="ignore").splitlines():
        l = linea.strip()
        if not l or l.startswith("/*"):
            continue
        if l.upper() == "END":
            break
        m = re.match(r"^OBJECT\s*=\s*(\S+)", l, re.I)
        if m:
            pila.append(m.group(1).upper()); objetos.setdefault(pila[-1], {}); continue
        if re.match(r"^END_OBJECT", l, re.I):
            if len(pila) > 1: pila.pop()
            continue
        if "=" in l:
            k, v = l.split("=", 1)
            objetos[pila[-1]].setdefault(k.strip().upper(), v.strip())
    return objetos


def _get(et, clave, objs=("IMAGE", "IMAGE_MAP_PROJECTION", "")):
    for o in objs:
        if o in et and clave in et[o]:
            return et[o][clave]
    for o in et:
        if clave in et[o]:
            return et[o][clave]
    return None


def _num(v, default=None):
    if v is None: return default
    m = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", str(v))
    return float(m.group()) if m else default


def _txt(v):
    return None if v is None else str(v).split("<")[0].replace('"', "").strip()


def _buscar_ci(directorio: Path, nombre: str):
    for p in directorio.iterdir():
        if p.name.lower() == nombre.lower():
            return p
    return None


def _puntero_imagen(et, ruta_lbl: Path):
    ptr = et[""].get("^IMAGE") or et.get("IMAGE", {}).get("^IMAGE")
    rec = int(_num(et[""].get("RECORD_BYTES"), 1))
    if ptr is None:
        for ext in (".img", ".IMG"):
            if ruta_lbl.with_suffix(ext).exists(): return ruta_lbl.with_suffix(ext), 0
        raise FileNotFoundError("Etiqueta sin ^IMAGE y sin .IMG junto a ella")
    nombre = re.findall(r'"([^"]+)"', ptr)
    n = _num(re.sub(r'"[^"]+"', "", ptr), None)
    off = 0 if n is None else int(n - 1) * (1 if "<BYTES>" in ptr.upper() else rec)
    if nombre:
        arch = _buscar_ci(ruta_lbl.parent, nombre[0])
        if arch is None: raise FileNotFoundError(f"No esta {nombre[0]} junto a {ruta_lbl.name}")
        return arch, off
    return ruta_lbl, off


def _dtype_pds(et):
    tipo = (_txt(_get(et, "SAMPLE_TYPE")) or "").upper()
    bits = int(_num(_get(et, "SAMPLE_BITS"), 16))
    if bits == 8:
        return np.dtype("u1" if "UNSIGNED" in tipo or not tipo else "i1")
    e = "<" if (tipo.startswith(("PC", "LSB", "VAX")) or "LSB" in tipo) else ">"
    if "REAL" in tipo or "FLOAT" in tipo:
        return np.dtype(f"{e}f{bits // 8}")
    return np.dtype(f"{e}{'u' if 'UNSIGNED' in tipo else 'i'}{bits // 8}")


def leer_ventana_pds(ruta_lbl, bbox, radio_km_defecto, margen_px=2, banda=0):
    ruta_lbl = Path(ruta_lbl)
    et = leer_etiqueta_pds3(ruta_lbl)
    proy = (_txt(_get(et, "MAP_PROJECTION_TYPE")) or "").upper()
    if not any(s in proy for s in ("CYLINDRICAL", "EQUIRECTANGULAR")):
        raise NotImplementedError(f"Proyeccion '{proy or 'desconocida'}' no soportada")
    lineas, muestras = int(_num(_get(et, "LINES"))), int(_num(_get(et, "LINE_SAMPLES")))
    nb = int(_num(_get(et, "BANDS"), 1))
    res = _num(_get(et, "MAP_RESOLUTION"))
    lpo, spo = _num(_get(et, "LINE_PROJECTION_OFFSET")), _num(_get(et, "SAMPLE_PROJECTION_OFFSET"))
    clon, clat = _num(_get(et, "CENTER_LONGITUDE"), 0.0), _num(_get(et, "CENTER_LATITUDE"), 0.0)
    radio_km = _num(_get(et, "A_AXIS_RADIUS"), radio_km_defecto)
    if radio_km < 100: radio_km = radio_km_defecto
    fac = res * math.cos(math.radians(clat))

    i0 = max(int(math.floor(lpo - bbox["maxlat"] * res)) - margen_px, 0)
    i1 = min(int(math.ceil(lpo - bbox["minlat"] * res)) + margen_px, lineas - 1)
    if i1 < i0: raise ValueError("AOI fuera de la cobertura en latitud")
    dc, d = float(_wrap180(bbox["center_lon"] - clon)), bbox["delta_lon_deg"]
    j0 = int(math.floor(spo + (dc - d / 2) * fac)) - margen_px
    j1 = int(math.ceil(spo + (dc + d / 2) * fac)) + margen_px
    js = np.arange(j0, j1 + 1)
    if muestras / fac >= 359.0:
        cols = js % muestras
    else:
        cols = js[(js >= 0) & (js < muestras)]; js = cols
        if cols.size == 0: raise ValueError("AOI fuera de la cobertura en longitud")

    arch, off = _puntero_imagen(et, ruta_lbl)
    dt = _dtype_pds(et)
    shape = (nb, lineas, muestras) if nb > 1 else (lineas, muestras)
    mm = np.memmap(arch, dtype=dt, mode="r", offset=off, shape=shape)
    mm = mm[banda] if nb > 1 else mm
    bloque = np.asarray(mm[i0:i1 + 1])[:, cols].astype(np.float64)
    for k in ("MISSING_CONSTANT", "CORE_NULL", "NULL_CONSTANT", "INVALID_CONSTANT"):
        raw = _get(et, k)
        if raw is not None and re.fullmatch(r"\s*[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?\s*(<.*>)?\s*", str(raw)):
            bloque[bloque == float(_num(raw))] = np.nan
    bloque[np.abs(bloque) > 1e30] = np.nan
    escala, off_dn = _num(_get(et, "SCALING_FACTOR"), 1.0), _num(_get(et, "OFFSET"), 0.0)
    valores = bloque * escala + (off_dn if abs(off_dn) < 500 else 0.0)   # OFFSET ~ radio -> alturas relativas
    if 0 < np.nanmax(np.abs(valores)) < 50:          # LDEM *_float viene en km
        valores = valores * 1000.0
    lats = (lpo - np.arange(i0, i1 + 1)) / res
    R = radio_km * 1000.0
    dy = R * math.radians(1.0 / res)
    dx = R * np.cos(np.radians(lats)) * math.radians(1.0 / fac)
    return dict(valores=valores, lats=lats, dx_m=dx, dy_m=dy)


def topo_pds(ruta_lbl, lat, lon, cuerpo, lado_km=VENTANA_KM):
    R = RADIO_KM[cuerpo]
    w = leer_ventana_pds(ruta_lbl, bbox_km(lat, lon, lado_km, R), R)
    z, dx = _recorte_centro(w["valores"], w["lats"], w["dx_m"], w["dy_m"], lado_km)
    return topo_features(z, dx, w["dy_m"])


# ---- Marte: MOLA 463 m (GeoTIFF cilindrico simple, 128 ppd) via rasterio, lectura por ventana ----
def topo_mola_tif(ruta_o_url, lat, lon, lado_km=VENTANA_KM):
    import rasterio
    from rasterio.windows import Window
    R = RADIO_KM["mars"]
    src = ruta_o_url if not str(ruta_o_url).startswith("http") else "/vsicurl/" + ruta_o_url
    with rasterio.open(src) as ds:
        ppd = ds.width / 360.0                       # 128 ppd
        lat_top = ds.height / ppd / 2.0              # +-88 grados
        dlat = math.degrees(lado_km / R)
        dlon = dlat / max(math.cos(math.radians(lat)), 1e-6)
        lon0 = lon % 360.0                           # supone columna 0 = longitud 0 E (como el mosaico USGS)
        r0 = int(math.floor((lat_top - (lat + dlat / 2)) * ppd)) - 1
        r1 = int(math.ceil((lat_top - (lat - dlat / 2)) * ppd)) + 1
        c0 = int(math.floor((lon0 - dlon / 2) * ppd)) - 1
        c1 = int(math.ceil((lon0 + dlon / 2) * ppd)) + 1
        if c0 < 0 or c1 > ds.width or r0 < 0 or r1 > ds.height:
            raise ValueError("Ventana fuera del mosaico o cruzando el meridiano 0/360 (no soportado)")
        z = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(float)
        if ds.nodata is not None:
            z[z == ds.nodata] = np.nan
    lats = lat_top - (np.arange(r0, r1) + 0.5) / ppd
    dy = R * 1000.0 * math.radians(1.0 / ppd)
    dx = R * 1000.0 * np.cos(np.radians(lats)) * math.radians(1.0 / ppd)
    z, dx = _recorte_centro(z, lats, dx, dy, lado_km)
    return topo_features(z, dx, dy)


# ---- Cualquier DEM geografico (EPSG:4326 u otro CRS geografico): sirve para la TIERRA ----------
def topo_raster_geografico(ruta, lat, lon, cuerpo="earth", lado_km=VENTANA_KM):
    import rasterio
    R = RADIO_KM[cuerpo]
    with rasterio.open(ruta) as ds:
        if not ds.crs or not ds.crs.is_geographic:
            raise ValueError("El DEM debe estar en coordenadas geograficas (grados); reproyectalo antes")
        b = bbox_km(lat, lon, lado_km, R)
        l0, l1 = b["center_lon"] - b["delta_lon_deg"] / 2, b["center_lon"] + b["delta_lon_deg"] / 2
        if ds.bounds.left >= 0 and l0 < 0:           # DEM en 0..360
            l0, l1 = l0 % 360, l1 % 360
        win = ds.window(l0, b["minlat"], l1, b["maxlat"]).round_offsets().round_lengths()
        z = ds.read(1, window=win, boundless=True, fill_value=np.nan).astype(float)
        if ds.nodata is not None:
            z[z == ds.nodata] = np.nan
        tr = ds.window_transform(win)
        lats = tr.f + (np.arange(z.shape[0]) + 0.5) * tr.e
        dy = R * 1000.0 * math.radians(abs(tr.e))
        dx = R * 1000.0 * np.cos(np.radians(lats)) * math.radians(abs(tr.a))
    z, dx = _recorte_centro(z, lats, dx, dy, lado_km)
    return topo_features(z, dx, dy)


# =============================================================================
# 2. GEOLOGIA
# =============================================================================
def _sugerir_clase(texto: str):
    t = (texto or "").lower()
    for clase, patron in REGLAS:
        if re.search(patron, t):
            return clase
    return None


def cargar_geologia(cuerpo: str, carpeta: Path):
    """Devuelve un GeoDataFrame de poligonos de unidades + (col_simbolo, col_nombre)."""
    import geopandas as gpd
    shps = sorted(carpeta.rglob("*.shp"))
    if not shps:
        raise FileNotFoundError(f"No hay .shp en {carpeta}. Descarga/extrae el mapa geologico primero.")
    mejor = None
    for shp in shps:
        try:
            g = gpd.read_file(shp)
        except Exception:
            continue
        if g.empty or not g.geom_type.astype(str).str.contains("Polygon").all():
            continue
        cols = [c for c in g.columns if re.search(r"unit|symbol|map_?unit|geo", c, re.I)]
        if cols and (mejor is None or len(g) > len(mejor[0])):
            mejor = (g, shp, cols)
    if mejor is None:
        raise RuntimeError("No encontre una capa de poligonos con columna de unidad. "
                           f"Capas vistas: {[s.name for s in shps]}")
    g, shp, cols = mejor
    col_sim = next((c for c in cols if re.search(r"symbol|^unit$|first_unit|map_?unit", c, re.I)), cols[0])
    col_nom = next((c for c in g.columns if re.search(r"name|descr|label|unitname", c, re.I)), col_sim)
    print(f"  capa geologica: {shp.name} | {len(g)} poligonos | simbolo='{col_sim}' nombre='{col_nom}'")
    return g, col_sim, col_nom


def tabla_armonizacion(g, col_sim, col_nom, cuerpo: str, dir_salida: Path) -> dict:
    """Lee harmonizacion_<cuerpo>.csv (unidad,clase) o genera un borrador para revisar."""
    ruta = dir_salida / f"harmonizacion_{cuerpo}.csv"
    if ruta.exists():
        t = pd.read_csv(ruta)
        malas = set(t["clase"].dropna()) - set(CLASES)
        if malas:
            raise ValueError(f"Clases invalidas en {ruta}: {malas}. Validas: {CLASES}")
        print(f"  usando {ruta}")
        return dict(zip(t["unidad"].astype(str), t["clase"]))
    u = g[[col_sim, col_nom]].astype(str).drop_duplicates(col_sim).copy()
    u.columns = ["unidad", "nombre"]
    u["clase"] = [_sugerir_clase(f"{n} {s}") for s, n in zip(u["unidad"], u["nombre"])]
    borrador = dir_salida / f"unidades_{cuerpo}_revisar.csv"
    dir_salida.mkdir(parents=True, exist_ok=True)
    u.to_csv(borrador, index=False)
    print(f"  [REVISAR] clasificacion BORRADOR -> {borrador}\n"
          f"            corrigela y guardala como {ruta.name}; mientras tanto uso el borrador.")
    return dict(zip(u["unidad"], u["clase"]))


def _a_esfera(g, R_m):
    """Lleva el GeoDataFrame a lon/lat sobre una esfera de radio R (independiente del .prj original)."""
    from pyproj import CRS
    if g.crs is not None and not g.crs.is_geographic:
        g = g.to_crs(g.crs.geodetic_crs)
    return g.set_crs(CRS.from_proj4(f"+proj=longlat +R={R_m} +no_defs"), allow_override=True)


def fracciones_litologia(g, col_sim, mapa, lat, lon, cuerpo, lado_km=VENTANA_KM):
    from pyproj import CRS
    from shapely.geometry import box
    R = RADIO_KM[cuerpo] * 1000.0
    gs = _a_esfera(g, R)
    b = bbox_km(lat, lon, lado_km, RADIO_KM[cuerpo])
    dominio180 = gs.total_bounds[0] < -1.0
    lon_c = float(_wrap180(lon)) if dominio180 else lon % 360.0
    m = 0.05
    sub = gs.cx[lon_c - b["delta_lon_deg"] / 2 - m: lon_c + b["delta_lon_deg"] / 2 + m,
                lat - b["delta_lat_deg"] / 2 - m: lat + b["delta_lat_deg"] / 2 + m]
    out = {c: np.nan for c in CLASES}
    out["litologia_cobertura"] = 0.0
    if sub.empty:
        return out
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon_c} +R={R} +units=m +no_defs")
    sub = sub.to_crs(aeqd)
    h = lado_km * 500.0
    sitio = box(-h, -h, h, h)
    areas = {c: 0.0 for c in CLASES}
    for geom, uni in zip(sub.geometry, sub[col_sim].astype(str)):
        clase = mapa.get(uni)
        if clase is None or not geom.is_valid and not geom.buffer(0).is_valid:
            continue
        a = geom.buffer(0).intersection(sitio).area
        if clase in areas:
            areas[clase] += a
    tot = sum(areas.values())
    out["litologia_cobertura"] = min(tot / sitio.area, 1.0)
    if tot > 0:
        out.update({c: areas[c] / tot for c in CLASES})    # normalizado: suma 1 sobre lo clasificado
    return out


# =============================================================================
# 3. CRATERES -> is_impact
# =============================================================================
def craters_mars_api(lat, lon, lado_km=VENTANA_KM, min_km=CRATER_MIN_KM):
    b = bbox_km(lat, lon, lado_km * 3, RADIO_KM["mars"])      # margen: craters grandes que entran de lado
    lon180 = float(_wrap180(lon))
    bb = f"{lon180 - b['delta_lon_deg']/2},{b['minlat']},{lon180 + b['delta_lon_deg']/2},{b['maxlat']}"
    url, params, filas = URLS["craters_mars_api"], {"f": "json", "bbox": bb, "limit": 5000}, []
    while url:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        for ft in js.get("features", []):
            p = ft.get("properties", {})
            lo = p.get("lon_e", ft["geometry"]["coordinates"][0] if ft.get("geometry") else np.nan)
            la = p.get("lat", ft["geometry"]["coordinates"][1] if ft.get("geometry") else np.nan)
            filas.append((la, lo, p.get("diamkm", np.nan)))
        url = next((l["href"] for l in js.get("links", []) if l.get("rel") == "next"), None)
        params = None
    df = pd.DataFrame(filas, columns=["lat", "lon", "diam_km"]).dropna()
    return df[df.diam_km >= min_km]


def craters_csv(ruta, col_lat="LAT_CIRC_IMG", col_lon="LON_CIRC_IMG", col_diam="DIAM_CIRC_IMG"):
    df = pd.read_csv(ruta, usecols=[col_lat, col_lon, col_diam])
    df.columns = ["lat", "lon", "diam_km"]
    return df.dropna()


def fraccion_craterizada(df, lat, lon, cuerpo, lado_km=VENTANA_KM, min_km=CRATER_MIN_KM):
    from pyproj import CRS, Transformer
    from shapely.geometry import Point, box
    from shapely.ops import unary_union
    d = df[df.diam_km >= min_km]
    if d.empty:
        return 0.0
    R = RADIO_KM[cuerpo] * 1000.0
    geo = CRS.from_proj4(f"+proj=longlat +R={R} +no_defs")
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={float(_wrap180(lon))} +R={R} +units=m +no_defs")
    tr = Transformer.from_crs(geo, aeqd, always_xy=True)
    x, y = tr.transform(_wrap180(d.lon.to_numpy()), d.lat.to_numpy())
    h = lado_km * 500.0
    sitio = box(-h, -h, h, h)
    circulos = [Point(xi, yi).buffer(di * 500.0) for xi, yi, di in zip(x, y, d.diam_km.to_numpy())
                if abs(xi) < h + di * 500 and abs(yi) < h + di * 500]
    if not circulos:
        return 0.0
    return float(unary_union(circulos).intersection(sitio).area / sitio.area)


# =============================================================================
# 4. MINERALOGIA
# =============================================================================
def muestrear_raster(ruta_o_url, lat, lon, cuerpo, lado_km=VENTANA_KM, estadistico=np.nanmean):
    """Estadistico del raster dentro de la ventana (cualquier CRS; usa la esfera del cuerpo)."""
    import rasterio
    from pyproj import CRS, Transformer
    from rasterio.windows import from_bounds
    R = RADIO_KM[cuerpo] * 1000.0
    src = ruta_o_url if not str(ruta_o_url).startswith("http") else "/vsicurl/" + str(ruta_o_url)
    with rasterio.open(src) as ds:
        b = bbox_km(lat, lon, lado_km, RADIO_KM[cuerpo])
        lons = np.array([lon - b["delta_lon_deg"] / 2, lon + b["delta_lon_deg"] / 2])
        lats = np.array([b["minlat"], b["maxlat"]])
        if ds.crs and ds.crs.is_geographic:
            xs, ys = (lons % 360.0 if ds.bounds.left >= 0 else _wrap180(lons)), lats
        else:
            geo = CRS.from_proj4(f"+proj=longlat +R={R} +no_defs")
            xs, ys = Transformer.from_crs(geo, ds.crs, always_xy=True).transform(
                np.repeat(lons, 2), np.tile(lats, 2))
        win = from_bounds(min(xs), min(ys), max(xs), max(ys), ds.transform)
        z = ds.read(1, window=win, boundless=True, fill_value=np.nan).astype(float)
        if ds.nodata is not None:
            z[z == ds.nodata] = np.nan
    return float(estadistico(z)) if np.isfinite(z).any() else np.nan


def mineralogia_luna(lat, lon):
    """Kaguya MI (wt%) -> indices 0-1. Los minerales sin fuente quedan en NaN (null)."""
    out = {m: np.nan for m in MINERALES}
    v = {k: (muestrear_raster(p, lat, lon, "moon") if p else np.nan) for k, p in MI_RASTERS.items()}
    if np.isfinite(v["plagioclase"]): out["plagioclase_feldspar"] = np.clip(v["plagioclase"] / 100.0, 0, 1)
    if np.isfinite(v["olivine"]): out["olivine"] = np.clip(v["olivine"] / 100.0, 0, 1)
    if np.isfinite(v["lcp"]) and np.isfinite(v["hcp"]): out["pyroxene"] = np.clip((v["lcp"] + v["hcp"]) / 100.0, 0, 1)
    # [REVISAR] FeO wt% es solo un proxy de fe_oxides; se normaliza con 25 wt% como maximo tipico mare.
    if np.isfinite(v["feo"]): out["fe_oxides"] = np.clip(v["feo"] / 25.0, 0, 1)
    return out


def mineralogia_marte(lat, lon):
    """CRISM (parametros espectrales) -> indices. Sin escala definida (CRISM_VMAX) solo se guarda el valor crudo."""
    out = {m: np.nan for m in MINERALES}
    for k, p in CRISM_RASTERS.items():
        if not p:
            continue
        val = muestrear_raster(p, lat, lon, "mars")
        col = {"pyroxene_lcp": "pyroxene", "pyroxene_hcp": "pyroxene"}.get(k, k)
        out[f"{k}_raw"] = val
        vmax = CRISM_VMAX.get(k)
        if vmax and np.isfinite(val):
            idx = float(np.clip(val / vmax, 0, 1))
            out[col] = idx if not np.isfinite(out[col]) else max(out[col], idx)   # pyroxene = max(LCP, HCP)
    return out


# =============================================================================
# 5. ORQUESTACION
# =============================================================================
def procesar(sitios: pd.DataFrame, args) -> pd.DataFrame:
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    geo, crat = {}, {}
    filas = []
    for _, s in sitios.iterrows():
        cuerpo, lat, lon, nombre = s["cuerpo"].lower(), float(s["lat"]), float(s["lon"]), s["nombre"]
        print(f"\n== {nombre} ({cuerpo}) lat={lat} lon={lon}")
        fila = dict(nombre=nombre, cuerpo=cuerpo, lat=lat, lon=lon)

        # --- A. topografia
        try:
            if cuerpo == "moon":
                fila.update(topo_pds(args.ldem, lat, lon, "moon"))
            elif cuerpo == "mars":
                fila.update(topo_mola_tif(args.mars_dem or URLS["dem_mars"], lat, lon))
            else:
                fila.update(topo_raster_geografico(args.tierra_dem, lat, lon, "earth"))
        except Exception as e:
            print(f"  [topografia] {type(e).__name__}: {e}")

        # --- B. geologia (solo Luna/Marte; la Tierra ya la tienes aparte)
        if cuerpo in ("moon", "mars"):
            try:
                if cuerpo not in geo:
                    g, cs, cn = cargar_geologia(cuerpo, DIR_EXT / f"{cuerpo}_geologia")
                    geo[cuerpo] = (g, cs, tabla_armonizacion(g, cs, cn, cuerpo, salida))
                g, cs, mapa = geo[cuerpo]
                fila.update(fracciones_litologia(g, cs, mapa, lat, lon, cuerpo))
            except Exception as e:
                print(f"  [geologia] {type(e).__name__}: {e}")

        # --- C. mineralogia
        try:
            if cuerpo == "moon":
                fila.update(mineralogia_luna(lat, lon))
            elif cuerpo == "mars":
                fila.update(mineralogia_marte(lat, lon))
        except Exception as e:
            print(f"  [mineralogia] {type(e).__name__}: {e}")

        # --- D. banderas
        try:
            if cuerpo == "mars":
                df_c = craters_mars_api(lat, lon)
            elif cuerpo == "moon":
                if "moon" not in crat:
                    crat["moon"] = craters_csv(args.craters_moon)
                df_c = crat["moon"]
            else:
                df_c = None
            if df_c is not None:
                fc = fraccion_craterizada(df_c, lat, lon, cuerpo)
                fila["fraccion_craterizada"], fila["is_impact"] = fc, bool(fc >= UMBRAL_IMPACT)
        except Exception as e:
            print(f"  [craters] {type(e).__name__}: {e}")
        if cuerpo != "earth":
            fila["ndvi"] = 0.0
            volc = np.nansum([fila.get("basaltic_volcanic", np.nan), fila.get("pyroclastic", np.nan)])
            fila["is_volcanic"] = bool(volc >= UMBRAL_VOLC) if "basaltic_volcanic" in fila else None
            ice = fila.get("ice_polar", np.nan)
            fila["has_ice"] = bool(ice >= UMBRAL_HIELO) if np.isfinite(ice) else None
        filas.append(fila)

    df = pd.DataFrame(filas)
    ruta = salida / "sitios_variables.csv"
    df.to_csv(ruta, index=False)
    print(f"\nListo -> {ruta}")
    faltan = [c for c in MINERALES if c not in df.columns or df[c].isna().all()]
    if faltan:
        print("Minerales sin datos (quedan null en el motor):", ", ".join(faltan))
    return df


def descargas(args):
    print("Descargando fuentes externas...")
    crater_csv = descargar_crateres_luna()
    print(f"  catalogo de crateres lunares listo: {crater_csv}")

    descargar_y_extraer(URLS["geo_moon_zip"], DIR_EXT / "moon_geologia")
    if URLS["geo_mars_zip"]:
        descargar_y_extraer(URLS["geo_mars_zip"], DIR_EXT / "mars_geologia")
    else:
        print("  [REVISAR] geo_mars_zip no configurado: descarga 'Database ZIP' de "
              "https://pubs.usgs.gov/publication/sim3292 y extraelo en app/data/reference/mars_geologia/")
    print("DEM de Marte: se lee por ventanas desde la URL (no hace falta bajar 2 GB). "
          "Si falla /vsicurl/, bajalo y usa --mars-dem ruta.tif")


# =============================================================================
# 6. AUTOTEST (datos sinteticos, sin red)
# =============================================================================
def autotest():
    import tempfile
    print("1) topografia sobre un plano inclinado + ruido")
    n, px = 53, 474.0
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))
    z = 0.02 * px * jj + np.random.default_rng(0).normal(0, 1.0, (n, n))          # pendiente 2 % hacia el este
    r = topo_features(z, px, px)
    esperada = math.degrees(math.atan(0.02))
    print("  ", {k: round(v, 3) for k, v in r.items()}, f"(pendiente esperada ~{esperada:.2f} deg)")
    assert abs(r["slope_mean_deg"] - esperada) < 0.3
    assert 0.5 < r["roughness_rms_m"] < 1.5          # el plano se elimina, queda el ruido (~1 m)

    print("2) fracciones de litologia con 2 poligonos")
    import geopandas as gpd
    from shapely.geometry import box
    lat, lon = 5.0, 30.0
    g = gpd.GeoDataFrame({"UnitSymbol": ["Im", "Nt"]},
                         geometry=[box(lon - 1, lat - 1, lon, lat + 1), box(lon, lat - 1, lon + 1, lat + 1)],
                         crs=None)
    mapa = {"Im": "basaltic_volcanic", "Nt": "feldspathic_plutonic"}
    f = fracciones_litologia(g, "UnitSymbol", mapa, lat, lon, "moon")
    print("  ", {k: round(v, 3) for k, v in f.items() if v == v and v > 0})
    assert abs(f["basaltic_volcanic"] - 0.5) < 0.02 and abs(f["feldspathic_plutonic"] - 0.5) < 0.02

    print("3) cobertura de craters (crater de 10 km en el centro de 25x25 km)")
    fc = fraccion_craterizada(pd.DataFrame(dict(lat=[lat], lon=[lon], diam_km=[10.0])), lat, lon, "moon")
    print(f"   fraccion = {fc:.4f}  (esperado {math.pi*25/625:.4f})")
    assert abs(fc - math.pi * 25 / 625) < 0.005

    print("4) lectura de DEM geografico (GeoTIFF sintetico)")
    import rasterio
    from rasterio.transform import from_origin
    with tempfile.TemporaryDirectory() as td:
        ruta = Path(td) / "dem.tif"
        ny, nx, res = 600, 600, 1 / 220.0
        zz = np.random.default_rng(1).normal(2500, 30, (ny, nx)).astype("float32")
        with rasterio.open(ruta, "w", driver="GTiff", height=ny, width=nx, count=1, dtype="float32",
                           crs="EPSG:4326", transform=from_origin(-79.0, -1.0, res, res)) as ds:
            ds.write(zz, 1)
        t = topo_raster_geografico(ruta, -2.4, -77.6, "earth")
        print("  ", {k: round(v, 2) for k, v in t.items()})
        assert np.isfinite(t["relief_m"])
        m = muestrear_raster(ruta, -2.4, -77.6, "earth")
        print(f"   media del raster en la ventana: {m:.1f} (esperado ~2500)")
        assert abs(m - 2500) < 20
    print("\nAUTOTEST OK")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sitios", help="CSV con nombre,cuerpo,lat,lon")
    ap.add_argument("--salida", default="salida")
    ap.add_argument("--ldem", default=str(
        DATA_DIR / "raw" / "moon" / "topography" / "LUN-TOP-01" / "ldem_64.lbl"
    ))
    ap.add_argument("--mars-dem", default=None, help="MOLA 463 m GeoTIFF local (si no, se lee por URL)")
    ap.add_argument("--tierra-dem", default=None, help="DEM geografico de la Tierra (GeoTIFF)")
    ap.add_argument("--craters-moon", default=str(DIR_EXT / "craters_moon" / "lunar_crater_database_robbins_2018.csv"))
    ap.add_argument("--descargar", action="store_true")
    ap.add_argument("--autotest", action="store_true")
    a = ap.parse_args()
    if a.autotest:
        autotest(); return
    if a.descargar:
        descargas(a)
    if a.sitios:
        procesar(pd.read_csv(a.sitios), a)
    elif not a.descargar:
        ap.print_help()


if __name__ == "__main__":
    main()

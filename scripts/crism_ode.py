#!/usr/bin/env python3
"""
crism_ode.py — Mars mineralogy from CRISM through the ODE REST API
==================================================================
For each Mars site:
1. Query ODE (`oderest.rsl.wustl.edu/live2`) for CRISM products covering the site.
2. Download only the small PDS3 labels first and inspect their band names.
3. Download images only when a label contains spectral parameters (OLINDEX3,
   LCPINDEX2, HCPINDEX2, D2300/BD1900_2, SINDEX2, BD530_2, etc.), then average
   each parameter over the 25 × 25 km site window.
4. Write `output/crism_sites.csv`; with `--merge`, merge values into
   `output/site_variables.csv`.

Requires `extract_similarity_features.py` in the same scripts directory.

Usage
-----
  python scripts/crism_ode.py --autotest
  python scripts/crism_ode.py --list-product-types
  python scripts/crism_ode.py --sites app/data/input/sites.csv --output output \
      --merge output/site_variables.csv

Known limitations
-----------------
* If product labels do not list band names, the script cannot identify the band
  for each parameter. It reports this and saves the label for inspection.
* Only cylindrical maps are supported. Other projections are skipped.
* Values are normalized to 0–1 only when `CRISM_VMAX` is configured; otherwise
  the raw values are saved.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

import extract_similarity_features as ev
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =============================================================================
# CONFIGURACION
# =============================================================================
ODE_REST_URL = "https://oderest.rsl.wustl.edu/live2"
PT_TOKENS = ["MRDR", "MTRDR", "TRDR", "MSP"]  # tipos de producto a probar, en orden de preferencia
DIR_CRISM = Path(__file__).resolve().parents[1] / "app" / "data" / "reference" / "crism"

# Parameter key -> accepted band names (the first match wins).
PARAMS = {
    "olivine": ["OLINDEX3", "OLINDEX2", "OLINDEX"],
    "pyroxene_lcp": ["LCPINDEX2", "LCPINDEX"],
    "pyroxene_hcp": ["HCPINDEX2", "HCPINDEX"],
    "phyllosilicates_clays": ["D2300", "BD1900_2", "BD1900R2"],
    "sulfates": ["SINDEX2", "SINDEX"],
    "fe_oxides": ["BD530_2", "BD530"],
}
# Valor que se mapea a indice 1.0 (clip 0-1). None = solo valor crudo (<param>_raw).
# [REVISAR] No invento escalas: definelas tu (p. ej. percentil 99 de los parametros en Marte).
CRISM_VMAX = {k: None for k in PARAMS}

MIN_COBERTURA = 0.30  # fraccion minima de pixeles validos en la ventana para aceptar un producto
R_MARTE = ev.RADIO_KM["mars"]


# =============================================================================
# CLIENTE ODE (mismo patron que tu notebook)
# =============================================================================
def crear_sesion():
    s = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "crism-ode-extractor/1.0"
    return s


SESION = crear_sesion()


def ode_get(params: dict, timeout: int = 90) -> dict:
    params = {k: v for k, v in params.items() if v is not None}
    params.setdefault("output", "JSON")
    r = SESION.get(ODE_REST_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    res = data.get("ODEResults", data)
    estado = str(res.get("Status", "")).lower()
    if estado and estado not in ("success", "ok"):
        raise RuntimeError(f"ODE devolvio error: {res.get('Error', res)}")
    return res


def _buscar_dicts(obj, claves):
    out = []
    if isinstance(obj, dict):
        if all(k in obj for k in claves):
            out.append(obj)
        for v in obj.values():
            out += _buscar_dicts(v, claves)
    elif isinstance(obj, list):
        for v in obj:
            out += _buscar_dicts(v, claves)
    return out


def _f(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def listar_pts_crism() -> pd.DataFrame:
    """Todos los (IHID, IID, PT) de CRISM en ODE, con el numero de productos."""
    res = ode_get({"query": "iipt", "target": "mars"})
    filas = _buscar_dicts(res, ["IHID", "IID", "PT"])
    if not filas:
        raise RuntimeError(
            "La respuesta 'iipt' no trae IHID/IID/PT; revisa la documentacion de ODE REST."
        )
    df = pd.DataFrame(filas).drop_duplicates(subset=["IHID", "IID", "PT"])
    mask = df["IID"].astype(str).str.contains("CRISM", case=False) | df.get(
        "IIName", pd.Series("", index=df.index)
    ).astype(str).str.contains("CRISM", case=False)
    df = df[mask].copy()
    col_n = next(
        (
            c
            for c in df.columns
            if c.lower() in ("numberproducts", "number_products", "productcount", "count")
        ),
        None,
    )
    df["n_productos"] = pd.to_numeric(df[col_n], errors="coerce") if col_n else np.nan
    return df.reset_index(drop=True)


def elegir_pts(df_pts: pd.DataFrame):
    """Orden de preferencia segun PT_TOKENS; devuelve [(ihid, iid, pt), ...]."""
    out = []
    for tok in PT_TOKENS:
        for r in df_pts.itertuples():
            if tok.lower() in str(r.PT).lower() and (r.IHID, r.IID, r.PT) not in out:
                out.append((r.IHID, r.IID, r.PT))
    return out


def bbox_ode(lat, lon, lado_km=ev.VENTANA_KM, radio_km=R_MARTE):
    dlat = lado_km / (radio_km * math.pi / 180.0)
    dlon = min(
        lado_km / (radio_km * max(math.cos(math.radians(lat)), 0.001) * math.pi / 180.0), 359.0
    )
    return dict(
        minlat=round(lat - dlat / 2, 5),
        maxlat=round(lat + dlat / 2, 5),
        westernlon=round((lon - dlon / 2) % 360.0, 5),
        easternlon=round((lon + dlon / 2) % 360.0, 5),
    )


def productos_en_bbox(ihid, iid, pt, bb, limite=60, pagina=50):
    """Lista de productos ODE (dict con pdsid, escala y archivos) que intersectan el bbox."""
    base = {
        "target": "mars",
        "query": "product",
        "results": "pmf",
        "ihid": ihid,
        "iid": iid,
        "pt": pt,
        **bb,
    }
    prods, offset = [], 0
    while len(prods) < limite:
        lim = min(pagina, limite - len(prods))
        res = ode_get(dict(base, limit=lim, offset=offset))
        bloque = res.get("Products", {})
        bloque = bloque.get("Product", []) if isinstance(bloque, dict) else []
        bloque = [bloque] if isinstance(bloque, dict) else bloque
        prods += bloque
        if len(bloque) < lim:
            break
        offset += len(bloque)
    out = []
    for p in prods:
        fs = (p.get("Product_files") or {}).get("Product_file", [])
        fs = [fs] if isinstance(fs, dict) else fs
        out.append(
            dict(
                pdsid=p.get("pdsid") or p.get("Product_name") or p.get("ode_id"),
                escala=_f(p.get("Map_scale")),
                archivos=[
                    dict(nombre=f.get("FileName"), kb=_f(f.get("KBytes"), 0.0), url=f.get("URL"))
                    for f in fs
                    if f.get("FileName")
                ],
            )
        )
    return out


def descargar_archivo(url: str, destino: Path) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and destino.stat().st_size > 0:
        return destino
    tmp = destino.with_suffix(destino.suffix + ".part")
    with SESION.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                if chunk:
                    f.write(chunk)
    tmp.replace(destino)
    return destino


# =============================================================================
# ETIQUETAS Y PARAMETROS
# =============================================================================
def nombres_bandas(texto: str):
    """Nombres de banda declarados en una etiqueta PDS3 (lista en MAYUSCULAS) o []."""
    for m in re.finditer(
        r"(?:BAND_NAME|BAND_BIN_BAND_NAME|BAND_ID)\s*=\s*\(([^)]*)\)", texto, re.I | re.S
    ):
        nombres = [a or b for a, b in re.findall(r'"([^"]+)"|([A-Za-z0-9_]+)', m.group(1))]
        if nombres:
            return [n.strip().upper() for n in nombres]
    m = re.search(r'BAND_NAME\s*=\s*"([^"]+)"', texto, re.I)
    return [m.group(1).strip().upper()] if m else []


def parametros_en_etiqueta(nombres):
    """{clave_motor: (indice_banda, nombre)} para los parametros que esta etiqueta contiene."""
    out = {}
    for clave, candidatos in PARAMS.items():
        for cand in candidatos:  # respeta el orden de preferencia
            if cand in nombres:
                out[clave] = (nombres.index(cand), cand)
                break
    return out


def _imagen_de(prod, lbl_nombre):
    stem = Path(lbl_nombre).stem.lower()
    imgs = [f for f in prod["archivos"] if f["name"].lower().endswith((".img", ".tif"))]
    mismos = [f for f in imgs if Path(f["name"]).stem.lower() == stem]
    return (mismos or imgs or [None])[0]


# =============================================================================
# PROCESO POR SITIO
# =============================================================================
def procesar_sitio(nombre, lat, lon, pts, max_productos=40, max_mb=800.0):
    """Devuelve dict con <clave>_raw y <clave>_fuente para cada parametro hallado."""
    mejor = {}  # clave -> dict(valor, cobertura, pdsid, banda)
    bb = bbox_ode(lat, lon)
    bbox_lec = ev.bbox_km(lat, lon, ev.VENTANA_KM, R_MARTE)
    vistos = set()
    for ihid, iid, pt in pts:
        try:
            prods = productos_en_bbox(ihid, iid, pt, bb, limite=max_productos)
        except Exception as e:
            print(f"  [ODE] {ihid}/{iid}/{pt}: {type(e).__name__}: {e}")
            continue
        print(f"  {pt}: {len(prods)} productos intersectan el sitio")
        for prod in prods:
            if prod["pdsid"] in vistos:
                continue
            vistos.add(prod["pdsid"])
            lbls = [f for f in prod["archivos"] if f["name"].lower().endswith(".lbl")]
            if not lbls:
                continue
            lbl = lbls[0]
            carpeta = DIR_CRISM / str(prod["pdsid"])
            try:
                ruta_lbl = descargar_archivo(lbl["url"], carpeta / lbl["name"])
                nombres = nombres_bandas(ruta_lbl.read_text(errors="ignore"))
            except Exception as e:
                print(f"    {prod['pdsid']}: no pude leer la etiqueta ({e})")
                continue
            hallados = parametros_en_etiqueta(nombres)
            if not hallados:
                continue  # este producto no trae parametros utiles
            img = _imagen_de(prod, lbl["name"])
            if img is None or img["kb"] / 1024.0 > max_mb:
                print(f"    {prod['pdsid']}: imagen ausente o > {max_mb:.0f} MB, la salto")
                continue
            try:
                descargar_archivo(img["url"], carpeta / img["name"])
            except Exception as e:
                print(f"    {prod['pdsid']}: fallo la descarga de la imagen ({e})")
                continue
            for clave, (idx, nom) in hallados.items():
                try:
                    w = ev.leer_ventana_pds(ruta_lbl, bbox_lec, R_MARTE, banda=idx, a_metros=False)
                except Exception as e:
                    print(f"    {prod['pdsid']} [{nom}]: {type(e).__name__}: {e}")
                    continue
                z = np.where(np.abs(w["valores"]) > 1e4, np.nan, w["valores"])
                cob = float(np.isfinite(z).mean())
                if cob < MIN_COBERTURA:
                    continue
                if clave not in mejor or cob > mejor[clave]["cobertura"]:
                    mejor[clave] = dict(
                        valor=float(np.nanmean(z)), cobertura=cob, pdsid=prod["pdsid"], banda=nom
                    )
    out = {}
    for clave, d in mejor.items():
        out[f"{clave}_raw"] = d["valor"]
        out[f"{clave}_fuente"] = f"{d['pdsid']}:{d['banda']}"
        out[f"{clave}_cobertura"] = round(d["cobertura"], 3)
    if not mejor:
        print(
            "  [!] Ningun producto con parametros legibles. Revisa las etiquetas descargadas en "
            f"{DIR_CRISM} (busca los nombres de banda) o ejecuta --list-product-types."
        )
    return out


def a_indices(fila: dict) -> dict:
    """Pasa los valores crudos a indices 0-1 (solo si CRISM_VMAX esta definido)."""
    idx = {}

    def i(clave):
        v, vmax = fila.get(f"{clave}_raw"), CRISM_VMAX.get(clave)
        return (
            float(np.clip(v / vmax, 0, 1))
            if (v is not None and vmax and np.isfinite(v))
            else np.nan
        )

    idx["olivine"] = i("olivine")
    px = [x for x in (i("pyroxene_lcp"), i("pyroxene_hcp")) if np.isfinite(x)]
    idx["pyroxene"] = max(px) if px else np.nan
    idx["phyllosilicates_clays"] = i("phyllosilicates_clays")
    idx["sulfates"] = i("sulfates")
    idx["fe_oxides"] = i("fe_oxides")
    return idx


def ejecutar(sitios: pd.DataFrame, salida: Path, merge: Path | None, max_productos, max_mb):
    salida.mkdir(parents=True, exist_ok=True)
    df_pts = listar_pts_crism()
    pts = elegir_pts(df_pts)
    if not pts:
        print("No CRISM product types matched these tokens:", PT_TOKENS)
        print(df_pts.to_string())
        return None
    print("Product types to be queried:", [p[2] for p in pts])
    filas = []
    for _, site in sitios[sitios["body"].str.lower() == "mars"].iterrows():
        print(f"\n== {site['name']} lat={site['lat']} lon={site['lon']}")
        row = {"name": site["name"]}
        row.update(
            procesar_sitio(
                site["name"], float(site["lat"]), float(site["lon"]), pts, max_productos, max_mb
            )
        )
        row.update(a_indices(row))
        filas.append(row)
    df = pd.DataFrame(filas)
    ruta = salida / "crism_sites.csv"
    df.to_csv(ruta, index=False)
    print(f"\nDone: wrote {ruta}")
    if merge and merge.exists():
        base = pd.read_csv(merge)
        shutil.copy(merge, merge.with_suffix(".bak.csv"))
        for col in ("olivine", "pyroxene", "phyllosilicates_clays", "sulfates", "fe_oxides"):
            if col not in df.columns:
                continue
            m = dict(zip(df["name"], df[col], strict=True))
            nuevo = base["name"].map(m)
            base[col] = nuevo.where(nuevo.notna(), base[col] if col in base.columns else np.nan)
        base.to_csv(merge, index=False)
        print(f"Merged indices into {merge} (backup: {merge.with_suffix('.bak.csv').name})")
    if all(v is None for v in CRISM_VMAX.values()):
        print(
            "[TODO] CRISM_VMAX is not configured: only raw values (*_raw) are available; 0–1 indices are empty."
        )
    return df


# =============================================================================
# AUTOTEST OFFLINE (ODE simulado + imagen PDS3 sintetica)
# =============================================================================
def autotest():
    import tempfile

    global DIR_CRISM, ode_get, descargar_archivo
    print("1) Band names")
    nb = nombres_bandas('BAND_NAME = ("OLINDEX3","HCPINDEX2","SINDEX2")')
    assert nb == ["OLINDEX3", "HCPINDEX2", "SINDEX2"], nb
    assert parametros_en_etiqueta(nb) == {
        "olivine": (0, "OLINDEX3"),
        "pyroxene_hcp": (1, "HCPINDEX2"),
        "sulfates": (2, "SINDEX2"),
    }
    print("   OK")

    print("2) End-to-end pipeline with mocked ODE")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ny = nx = 640
        cubo = np.stack(
            [np.full((ny, nx), 0.020), np.full((ny, nx), 0.050), np.full((ny, nx), 0.010)]
        ).astype("<f4")
        cubo[0, :, :100] = 65535  # franja sin datos (MISSING_CONSTANT)
        (td / "T0001_FAKE.IMG").write_bytes(cubo.tobytes())
        (td / "T0001_FAKE.LBL").write_text(f"""PDS_VERSION_ID = PDS3
RECORD_TYPE = FIXED_LENGTH
RECORD_BYTES = {nx * 4}
^IMAGE = "T0001_FAKE.IMG"
OBJECT = IMAGE
  LINES = {ny}
  LINE_SAMPLES = {nx}
  BANDS = 3
  BAND_STORAGE_TYPE = BAND_SEQUENTIAL
  SAMPLE_TYPE = PC_REAL
  SAMPLE_BITS = 32
  BAND_NAME = ("OLINDEX3","HCPINDEX2","SINDEX2")
  MISSING_CONSTANT = 65535
END_OBJECT = IMAGE
OBJECT = IMAGE_MAP_PROJECTION
  MAP_PROJECTION_TYPE = "SIMPLE CYLINDRICAL"
  A_AXIS_RADIUS = 3396.0 <KM>
  MAP_RESOLUTION = 64 <PIXEL/DEGREE>
  CENTER_LONGITUDE = 0.0
  CENTER_LATITUDE = 0.0
  LINE_PROJECTION_OFFSET = 1280.0
  SAMPLE_PROJECTION_OFFSET = -4480.0
END_OBJECT = IMAGE_MAP_PROJECTION
END
""")

        def fake_ode(params, timeout=90):
            if params.get("query") == "iipt":
                return {
                    "IIPTSets": {
                        "IIPT": [
                            {
                                "IHID": "MRO",
                                "IID": "CRISM",
                                "IIName": "CRISM",
                                "PT": "MRDR",
                                "NumberProducts": "5",
                            }
                        ]
                    }
                }
            return {
                "Products": {
                    "Product": [
                        {
                            "pdsid": "T0001_FAKE",
                            "Map_scale": "460",
                            "Product_files": {
                                "Product_file": [
                                    {
                                        "FileName": "T0001_FAKE.LBL",
                                        "KBytes": "2",
                                        "URL": "fake://lbl",
                                    },
                                    {
                                        "FileName": "T0001_FAKE.IMG",
                                        "KBytes": "3000",
                                        "URL": "fake://img",
                                    },
                                ]
                            },
                        },
                        {
                            "pdsid": "T0002_SIN_PARAMS",
                            "Map_scale": "460",
                            "Product_files": {
                                "Product_file": [
                                    {
                                        "FileName": "T0002_SIN_PARAMS.LBL",
                                        "KBytes": "2",
                                        "URL": "fake://lbl2",
                                    }
                                ]
                            },
                        },
                    ]
                }
            }

        def fake_dl(url, destino):
            destino.parent.mkdir(parents=True, exist_ok=True)
            if url == "fake://lbl2":
                destino.write_text('BAND_NAME = ("R770","R440")')
            else:
                shutil.copy(
                    td / ("T0001_FAKE.LBL" if url == "fake://lbl" else "T0001_FAKE.IMG"), destino
                )
            return destino

        DIR_CRISM = td / "crism"
        ode_get, descargar_archivo = fake_ode, fake_dl
        CRISM_VMAX.update(olivine=0.1, pyroxene_hcp=0.1, sulfates=0.05)
        sit = pd.DataFrame([dict(name="jezero_crater_sample", body="mars", lat=15.0, lon=75.0)])
        base = pd.DataFrame(
            [dict(name="jezero_crater_sample", body="mars", lat=15.0, lon=75.0, relief_m=123.0)]
        )
        base.to_csv(td / "site_variables.csv", index=False)
        df = ejecutar(sit, td / "salida", td / "site_variables.csv", 10, 800)
        r = df.iloc[0]
        print(
            "  ",
            {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in r.items()
                if "source" not in k
            },
        )
        assert abs(r["olivine_raw"] - 0.020) < 1e-4 and abs(r["pyroxene_hcp_raw"] - 0.050) < 1e-4
        assert (
            abs(r["olivine"] - 0.2) < 1e-3
            and abs(r["pyroxene"] - 0.5) < 1e-3
            and abs(r["sulfates"] - 0.2) < 1e-3
        )
        fusion = pd.read_csv(td / "site_variables.csv")
        assert abs(fusion.loc[0, "pyroxene"] - 0.5) < 1e-3 and fusion.loc[0, "relief_m"] == 123.0
    print("\nAUTOTEST OK")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sites", dest="sites", help="CSV with name,body,lat,lon columns; Mars rows are used"
    )
    ap.add_argument("--output", default="output")
    ap.add_argument("--merge", help="Path to site_variables.csv to merge the indices into")
    ap.add_argument("--max-products", dest="max_products", type=int, default=40)
    ap.add_argument("--max-mb", type=float, default=800.0)
    ap.add_argument("--list-product-types", dest="list_product_types", action="store_true")
    ap.add_argument("--autotest", action="store_true")
    a = ap.parse_args()
    if a.autotest:
        autotest()
    elif a.list_product_types:
        print(listar_pts_crism().to_string())
    elif a.sites:
        ejecutar(
            pd.read_csv(a.sites),
            Path(a.output),
            Path(a.merge) if a.merge else None,
            a.max_products,
            a.max_mb,
        )
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

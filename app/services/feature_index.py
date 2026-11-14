from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.schemas.analog import SingleLaserProfile
from app.schemas.planetary import (
    CelestialBody,
    DimensionSpec,
    EarthdataAtl03Record,
    EarthdataGranuleSummary,
    EnvironmentalFlags,
    LithologyClassSpec,
    PlanetarySite,
    TaxonomyResponse,
    TopographyFingerprint,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class BodyMatrixIndex:
    """In-memory vectorized NumPy matrices for a celestial body."""

    body: CelestialBody
    cell_ids: NDArray[np.str_]
    site_ids: NDArray[np.str_]
    lats: NDArray[np.float32]
    lons: NDArray[np.float32]
    elevations: NDArray[np.float32]
    is_curated: NDArray[np.bool_]
    is_known_analog: NDArray[np.bool_]
    topo: NDArray[np.float32]
    geo: NDArray[np.float32]
    min_vec: NDArray[np.float32]
    min_mask: NDArray[np.bool_]
    ndvi: NDArray[np.float32]
    is_impact: NDArray[np.bool_]
    has_ice: NDArray[np.bool_]
    is_volcanic: NDArray[np.bool_]


class FeatureIndexStore:
    """Singleton-cached store holding precomputed planetary feature indices and metadata."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir
        self.taxonomy_raw: dict[str, Any] = json.loads(
            (data_dir / "taxonomy.json").read_text(encoding="utf-8")
        )
        self.feature_stats_raw: dict[str, Any] = json.loads(
            (data_dir / "feature_stats.json").read_text(encoding="utf-8")
        )
        self.sites_raw: dict[str, list[dict[str, Any]]] = json.loads(
            (data_dir / "sites.json").read_text(encoding="utf-8")
        )
        self.profiles_raw: dict[str, dict[str, Any]] = json.loads(
            (data_dir / "profiles" / "laser_profiles.json").read_text(encoding="utf-8")
        )
        atl03_path = data_dir / "earth_icesat2_atl03.json"
        self.earth_atl03_raw: dict[str, dict[str, Any]] = (
            json.loads(atl03_path.read_text(encoding="utf-8")) if atl03_path.exists() else {}
        )

        self.lithology_ids: list[str] = [c["id"] for c in self.taxonomy_raw["lithology_classes"]]
        self.mineral_ids: list[str] = [m["id"] for m in self.taxonomy_raw["mineral_dimensions"]]
        self.topo_ids: list[str] = [t["id"] for t in self.feature_stats_raw["topography_features"]]

        self.topo_medians = np.asarray(
            [t["median"] for t in self.feature_stats_raw["topography_features"]],
            dtype=np.float32,
        )
        self.topo_iqrs = np.asarray(
            [t["iqr"] for t in self.feature_stats_raw["topography_features"]],
            dtype=np.float32,
        )
        self.sigma_topo = float(self.feature_stats_raw.get("sigma_topo", 1.65))
        self.lithology_matrix = np.asarray(
            self.taxonomy_raw["lithology_similarity_matrix"], dtype=np.float32
        )

        self.indices: dict[CelestialBody, BodyMatrixIndex] = {
            "moon": self._load_npz("moon", data_dir / "moon_index.npz"),
            "mars": self._load_npz("mars", data_dir / "mars_index.npz"),
            "earth": self._load_npz("earth", data_dir / "earth_index.npz"),
        }

        self._sites_by_id: dict[str, dict[str, Any]] = {}
        self._sites_by_cell: dict[str, dict[str, Any]] = {}
        for key in ("moon", "mars", "earth_all"):
            for s in self.sites_raw.get(key, []):
                self._sites_by_id[s["site_id"]] = s
                self._sites_by_cell[s["cell_id"]] = s

    def _load_npz(self, body: CelestialBody, path: Path) -> BodyMatrixIndex:
        with np.load(path, allow_pickle=False) as data:
            return BodyMatrixIndex(
                body=body,
                cell_ids=data["cell_ids"],
                site_ids=data["site_ids"],
                lats=data["lats"],
                lons=data["lons"],
                elevations=data["elevations"],
                is_curated=data["is_curated"],
                is_known_analog=data["is_known_analog"],
                topo=data["topo"],
                geo=data["geo"],
                min_vec=data["min_vec"],
                min_mask=data["min_mask"],
                ndvi=data["ndvi"],
                is_impact=data["is_impact"],
                has_ice=data["has_ice"],
                is_volcanic=data["is_volcanic"],
            )

    def to_planetary_site(self, raw: dict[str, Any]) -> PlanetarySite:
        topo_vals = raw["topo"]
        topo = TopographyFingerprint(
            relief_m=float(topo_vals[0]),
            slope_mean_deg=float(topo_vals[1]),
            slope_p90_deg=float(topo_vals[2]),
            ruggedness_tri_m=float(topo_vals[3]),
            roughness_rms_m=float(topo_vals[4]),
            hypsometric_integral=float(topo_vals[5]),
        )
        geo_dict = {
            self.lithology_ids[i]: round(float(raw["geo"][i]), 3)
            for i in range(len(self.lithology_ids))
        }
        min_dict: dict[str, float | None] = {}
        for i, m_id in enumerate(self.mineral_ids):
            val = float(raw["min"][i])
            min_dict[m_id] = round(val, 3) if val >= 0.0 else None

        fl = raw.get("flags", {})
        eg_raw = raw.get("earthdata_granule")
        earthdata_granule = EarthdataGranuleSummary(**eg_raw) if eg_raw else None
        return PlanetarySite(
            site_id=raw["site_id"],
            cell_id=raw["cell_id"],
            body=raw["body"],
            name=raw["name"],
            region=raw["region"],
            lat=float(raw["lat"]),
            lon=float(raw["lon"]),
            elevation_m=float(raw["elevation_m"]),
            is_curated_site=bool(raw.get("is_curated_site", True)),
            is_known_analog=bool(raw.get("is_known_analog", False)),
            analog_category=raw.get("analog_category"),
            mission_relevance=raw["mission_relevance"],
            topography=topo,
            geology_fractions=geo_dict,
            mineralogy=min_dict,
            flags=EnvironmentalFlags(
                ndvi=float(fl.get("ndvi", 0.0)),
                is_impact=bool(fl.get("is_impact", False)),
                has_ice=bool(fl.get("has_ice", False)),
                is_volcanic=bool(fl.get("is_volcanic", False)),
            ),
            earthdata_granule=earthdata_granule,
        )

    def list_curated_sites(self, body: CelestialBody) -> list[PlanetarySite]:
        raw_list = self.sites_raw.get(body, [])
        return [self.to_planetary_site(item) for item in raw_list]

    def get_site_by_id_or_cell(self, identifier: str) -> PlanetarySite | None:
        raw = self._sites_by_id.get(identifier) or self._sites_by_cell.get(identifier)
        if raw is None:
            return None
        return self.to_planetary_site(raw)

    def snap_coordinates_to_site(
        self, body: CelestialBody, lat: float, lon: float
    ) -> PlanetarySite:
        """Find the nearest indexed cell on the specified body using great-circle angular distance."""
        idx = self.indices[body]
        lat1 = np.radians(lat)
        lon1 = np.radians(lon)
        lat2 = np.radians(idx.lats.astype(np.float64))
        lon2 = np.radians(idx.lons.astype(np.float64))

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        best_pos = int(np.argmin(a))
        cell_id = str(idx.cell_ids[best_pos])
        raw = self._sites_by_cell[cell_id]
        return self.to_planetary_site(raw)

    def get_taxonomy_response(self) -> TaxonomyResponse:
        return TaxonomyResponse(
            window_size_km=float(self.feature_stats_raw["window_size_km"]),
            resampled_gsd_m=float(self.feature_stats_raw["resampled_gsd_m"]),
            topography_features=[
                DimensionSpec(**item) for item in self.feature_stats_raw["topography_features"]
            ],
            lithology_classes=[
                LithologyClassSpec(**item) for item in self.taxonomy_raw["lithology_classes"]
            ],
            lithology_similarity_matrix=self.taxonomy_raw["lithology_similarity_matrix"],
            mineral_dimensions=[
                DimensionSpec(**item) for item in self.taxonomy_raw["mineral_dimensions"]
            ],
        )

    def get_laser_profile(self, identifier: str) -> SingleLaserProfile | None:
        site = self.get_site_by_id_or_cell(identifier)
        cell_id = site.cell_id if site else identifier
        raw = self.profiles_raw.get(cell_id)
        if raw is None:
            return None
        return SingleLaserProfile(**raw)

    def list_earthdata_atl03(self) -> list[EarthdataAtl03Record]:
        return [EarthdataAtl03Record(**rec) for rec in self.earth_atl03_raw.values()]

    def get_earthdata_atl03(self, identifier: str) -> EarthdataAtl03Record | None:
        rec = self.earth_atl03_raw.get(identifier)
        if rec is None:
            site = self.get_site_by_id_or_cell(identifier)
            if site is not None:
                rec = self.earth_atl03_raw.get(site.site_id)
        if rec is None:
            return None
        return EarthdataAtl03Record(**rec)


@lru_cache
def get_feature_index_store() -> FeatureIndexStore:
    """Return cached singleton FeatureIndexStore for the warm instance."""
    return FeatureIndexStore()

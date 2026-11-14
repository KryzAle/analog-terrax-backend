"""
Convert extracted ATLAS/ICESat-2 L2A (ATL03) geolocated photon & segment data
into normalized laser altimeter profiles for cross-planetary comparison with
LRO LOLA (Moon) and MGS MOLA (Mars).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers on Earth."""
    r_earth = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r_earth * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_profile_from_atl03_sample(sample_json_path: Path) -> dict[str, Any]:
    """Read real extracted ATL03 JSON and format a standardized along-track elevation transect."""
    raw = json.loads(sample_json_path.read_text(encoding="utf-8"))
    segments = raw.get("geolocation_segments", [])
    photons = raw.get("photons", [])

    points: list[dict[str, float]] = []
    if segments:
        lat0 = float(segments[0]["reference_lat"])
        lon0 = float(segments[0]["reference_lon"])
        base_elev = float(segments[0]["dem_height_m"] or 0.0)
        for seg in segments:
            lat = float(seg["reference_lat"])
            lon = float(seg["reference_lon"])
            elev = float(seg["dem_height_m"] or base_elev)
            dist_km = haversine_km(lat0, lon0, lat, lon)
            points.append(
                {
                    "distance_km": round(dist_km, 4),
                    "elevation_m": round(elev, 3),
                    "relative_elevation_m": round(elev - base_elev, 3),
                    "lat": lat,
                    "lon": lon,
                }
            )

    return {
        "instrument": "ATLAS / ICESat-2 (ATL03 v007)",
        "granule_id": raw.get("granule", {}).get(
            "granule_id", "ATL03_20240323120922_00742306_007_01.h5"
        ),
        "beam": raw.get("sampled_beam", {}).get("beam_id", "gt1r"),
        "points": points,
        "photon_sample_count": len(photons),
    }

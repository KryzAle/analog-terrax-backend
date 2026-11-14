"""Fixed-scale topographic feature extraction for 25 km planetary windows (~500 m GSD)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_topographic_fingerprint_from_dem(
    dem_m: NDArray[np.floating],
    pixel_size_m: float = 500.0,
) -> dict[str, float]:
    """
    Compute the 6-dimensional topographic fingerprint from a 2D elevation array (in meters)
    resampled to a uniform Ground Sampling Distance (default 500 m).
    """
    valid = dem_m[np.isfinite(dem_m)]
    if valid.size < 4:
        return {
            "relief_m": 0.0,
            "slope_mean_deg": 0.0,
            "slope_p90_deg": 0.0,
            "ruggedness_tri_m": 0.0,
            "roughness_rms_m": 0.0,
            "hypsometric_integral": 0.5,
        }

    p5, p95 = np.percentile(valid, [5.0, 95.0])
    relief_m = float(max(0.0, p95 - p5))

    # Gradient -> slope in degrees
    gy, gx = np.gradient(dem_m, pixel_size_m)
    slope_rad = np.arctan(np.hypot(gx, gy))
    slope_deg = np.degrees(slope_rad)
    slope_mean_deg = float(np.nanmean(slope_deg))
    slope_p90_deg = float(np.nanpercentile(slope_deg, 90.0))

    # Terrain Ruggedness Index (mean absolute difference with neighbors)
    dy = np.abs(np.diff(dem_m, axis=0))
    dx = np.abs(np.diff(dem_m, axis=1))
    ruggedness_tri_m = float(0.5 * (np.nanmean(dy) + np.nanmean(dx)))

    # Detrended RMS roughness (residuals after planar detrend)
    ny, nx = dem_m.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    a_mat = np.column_stack([xx.ravel(), yy.ravel(), np.ones(dem_m.size)])
    coeffs, _, _, _ = np.linalg.lstsq(a_mat, dem_m.ravel(), rcond=None)
    plane = (a_mat @ coeffs).reshape(dem_m.shape)
    residuals = dem_m - plane
    roughness_rms_m = float(np.sqrt(np.nanmean(residuals**2)))

    # Hypsometric integral: (mean - min) / (max - min)
    elev_range = float(np.max(valid) - np.min(valid))
    hypsometric_integral = (
        float((np.mean(valid) - np.min(valid)) / elev_range) if elev_range > 1e-3 else 0.5
    )

    return {
        "relief_m": round(relief_m, 2),
        "slope_mean_deg": round(slope_mean_deg, 2),
        "slope_p90_deg": round(slope_p90_deg, 2),
        "ruggedness_tri_m": round(ruggedness_tri_m, 2),
        "roughness_rms_m": round(roughness_rms_m, 2),
        "hypsometric_integral": round(hypsometric_integral, 3),
    }


def summarize_laser_profile(
    distances_km: list[float],
    elevations_m: list[float],
) -> dict[str, float]:
    """Compute laser altimeter transect statistics (relief, mean slope, detrended RMS roughness)."""
    if len(elevations_m) < 2:
        return {"relief_m": 0.0, "mean_slope_deg": 0.0, "rms_roughness_m": 0.0}

    h = np.asarray(elevations_m, dtype=np.float64)
    x_m = np.asarray(distances_km, dtype=np.float64) * 1000.0
    dx = np.diff(x_m)
    dh = np.diff(h)
    valid_steps = np.abs(dx) > 1e-3
    slopes_deg = np.degrees(np.arctan(np.abs(dh[valid_steps] / dx[valid_steps])))

    # Linear detrend for 1D laser transect RMS roughness
    slope_fit, intercept_fit = np.polyfit(x_m, h, 1)
    residuals = h - (slope_fit * x_m + intercept_fit)

    return {
        "relief_m": round(float(np.max(h) - np.min(h)), 3),
        "mean_slope_deg": round(float(np.mean(slopes_deg)) if slopes_deg.size else 0.0, 3),
        "rms_roughness_m": round(float(np.sqrt(np.mean(residuals**2))), 3),
    }

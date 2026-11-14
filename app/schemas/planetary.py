from typing import Literal

from pydantic import BaseModel, Field

CelestialBody = Literal["moon", "mars", "earth"]


class TopographyFingerprint(BaseModel):
    """Fixed-scale (500m GSD, 25km window) topographic metrics."""

    relief_m: float = Field(..., description="Local relief (p95 - p5 elevation) in meters")
    slope_mean_deg: float = Field(..., description="Mean window slope in degrees")
    slope_p90_deg: float = Field(..., description="90th percentile slope in degrees")
    ruggedness_tri_m: float = Field(..., description="Terrain Ruggedness Index (TRI) in meters")
    roughness_rms_m: float = Field(
        ..., description="Planar-detrended RMS surface roughness in meters"
    )
    hypsometric_integral: float = Field(
        ..., description="Hypsometric integral (mean - min) / (max - min)"
    )


class EnvironmentalFlags(BaseModel):
    """Toggleable hard-filter metadata flags for a location."""

    ndvi: float = Field(default=0.0, description="MODIS vegetation index (0 = barren)")
    is_impact: bool = Field(default=False, description="Contains impact crater or ejecta structure")
    has_ice: bool = Field(
        default=False, description="Contains polar ice, permafrost, or PSR volatiles"
    )
    is_volcanic: bool = Field(
        default=False, description="Contains volcanic basalt flows, tephra, or lava tubes"
    )


class PlanetarySite(BaseModel):
    """Complete multi-planetary fingerprint for a Moon, Mars, or Earth location."""

    site_id: str = Field(..., description="Unique site identifier")
    cell_id: str = Field(..., description="Grid cell identifier")
    body: CelestialBody = Field(..., description="Celestial body ('moon', 'mars', or 'earth')")
    name: str = Field(..., description="Human-readable site or region name")
    region: str = Field(..., description="Physiographic or planetary region")
    lat: float = Field(..., description="Latitude in decimal degrees")
    lon: float = Field(..., description="Longitude in decimal degrees")
    elevation_m: float = Field(..., description="Mean datum elevation in meters")
    is_curated_site: bool = Field(
        default=True,
        description="True if curated target/analog site, False if background grid cell",
    )
    is_known_analog: bool = Field(
        default=False,
        description="True if established benchmark analog, False if newly surfaced candidate",
    )
    analog_category: str | None = Field(
        default=None, description="Analog classification or discovery tier"
    )
    mission_relevance: str = Field(
        ..., description="Scientific and operational rationale for permanent base / analog testing"
    )
    topography: TopographyFingerprint
    geology_fractions: dict[str, float] = Field(
        ...,
        description="Fractional coverage across the 7 harmonized cross-planet lithology classes",
    )
    mineralogy: dict[str, float | None] = Field(
        ...,
        description="Relative abundance across the 9 shared mineral dimensions (None if unobserved)",
    )
    flags: EnvironmentalFlags


class LithologyClassSpec(BaseModel):
    id: str
    label: str
    earth_source: str
    moon_source: str
    mars_source: str


class DimensionSpec(BaseModel):
    id: str
    label: str
    unit: str | None = None
    median: float | None = None
    iqr: float | None = None


class TaxonomyResponse(BaseModel):
    """Harmonized cross-planetary taxonomy, feature normalization stats, and similarity matrix."""

    window_size_km: float
    resampled_gsd_m: float
    topography_features: list[DimensionSpec]
    lithology_classes: list[LithologyClassSpec]
    lithology_similarity_matrix: list[list[float]]
    mineral_dimensions: list[DimensionSpec]

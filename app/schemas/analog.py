from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.planetary import CelestialBody, PlanetarySite


class SimilarityWeights(BaseModel):
    """User-adjustable weights across the three planetary comparison families."""

    topography: float = Field(default=0.45, ge=0.0, le=1.0, description="Weight for topography")
    geology: float = Field(default=0.35, ge=0.0, le=1.0, description="Weight for geology")
    mineralogy: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for mineralogy (semi-quantitative indicator)",
    )


class AnalogSearchFilters(BaseModel):
    """Optional toggleable filters for narrowing Earth analog candidates."""

    curated_only: bool = Field(
        default=True,
        description="Return curated benchmark + under-studied sites (False includes raw grid cells)",
    )
    exclude_vegetated: bool = Field(
        default=True, description="Exclude Earth cells with MODIS NDVI > max_ndvi"
    )
    max_ndvi: float = Field(default=0.30, ge=0.0, le=1.0, description="Maximum allowed NDVI")
    require_impact: bool = Field(
        default=False, description="Only match Earth sites with impact structures"
    )
    require_ice: bool = Field(
        default=False, description="Only match Earth sites with polar ice or permafrost"
    )
    require_volcanic: bool = Field(default=False, description="Only match volcanic Earth sites")


class AnalogSearchRequest(BaseModel):
    """Request payload for finding the most similar Earth locations to a Moon or Mars site."""

    target_body: Literal["moon", "mars"] = Field(
        default="moon", description="Target planetary body ('moon' or 'mars')"
    )
    target_site_id: str | None = Field(
        default="moon_shackleton_rim",
        max_length=128,
        description="Curated target site ID (if omitted, snaps target_lat/target_lon to nearest cell)",
    )
    target_lat: float | None = Field(
        default=None, ge=-90.0, le=90.0, description="Custom target latitude on Moon or Mars"
    )
    target_lon: float | None = Field(
        default=None, ge=-180.0, le=180.0, description="Custom target longitude on Moon or Mars"
    )
    weights: SimilarityWeights = Field(default_factory=SimilarityWeights)
    filters: AnalogSearchFilters = Field(default_factory=AnalogSearchFilters)
    k: int = Field(default=8, ge=1, le=50, description="Number of top Earth matches to return")


class ScoreBreakdown(BaseModel):
    """Normalized similarity scores in [0, 1] and percentage [0, 100]."""

    overall_score: float = Field(..., description="Weighted composite similarity score (0 to 1)")
    overall_pct: float = Field(..., description="Weighted composite match percentage (0 to 100)")
    topography_score: float = Field(..., description="Gaussian kernel score on robust z-scores")
    geology_score: float = Field(..., description="Bilinear lithology similarity matrix score")
    mineralogy_score: float = Field(..., description="Masked cosine similarity on shared minerals")
    compared_mineral_dimensions: list[str] = Field(
        ..., description="Mineral dimensions present in both target and Earth site"
    )


class FeatureContribution(BaseModel):
    """Explainability item describing a key matching feature between target and Earth analog."""

    family: Literal["topography", "geology", "mineralogy"]
    feature_id: str
    label: str
    target_value: float
    earth_value: float
    unit: str
    closeness_pct: float = Field(..., description="Closeness score (0-100%) for this feature")


class AnalogMatchResult(BaseModel):
    """Ranked Earth analog site with full score breakdown and feature explanations."""

    rank: int
    site: PlanetarySite
    scores: ScoreBreakdown
    top_contributing_features: list[FeatureContribution]


class AnalogSearchResponse(BaseModel):
    """Response returned by POST /api/v1/analogs/search."""

    target: PlanetarySite
    normalized_weights: SimilarityWeights
    total_candidates_evaluated: int
    matches: list[AnalogMatchResult]


class HeatmapCellScore(BaseModel):
    """Compact cell similarity score for rendering a global Earth heatmap layer."""

    cell_id: str
    site_id: str
    name: str
    lat: float
    lon: float
    is_curated_site: bool
    score_uint8: int = Field(..., ge=0, le=100, description="Quantized similarity score (0-100)")
    topography_uint8: int = Field(..., ge=0, le=100)
    geology_uint8: int = Field(..., ge=0, le=100)
    mineralogy_uint8: int = Field(..., ge=0, le=100)


class AnalogHeatmapResponse(BaseModel):
    """Global Earth similarity heatmap response."""

    target_site_id: str
    target_name: str
    target_body: Literal["moon", "mars"]
    cells: list[HeatmapCellScore]


class LaserProfilePoint(BaseModel):
    distance_km: float
    elevation_m: float
    relative_elevation_m: float
    lat: float
    lon: float


class LaserProfileStats(BaseModel):
    relief_m: float
    mean_slope_deg: float
    rms_roughness_m: float


class SingleLaserProfile(BaseModel):
    cell_id: str
    site_id: str
    body: CelestialBody
    instrument: str
    granule_id: str
    concept_id: str | None = None
    beam: str | None = None
    time_start: str | None = None
    https_url: str | None = None
    photon_sample_count: int | None = None
    points: list[LaserProfilePoint]
    stats: LaserProfileStats


class LaserProfileComparisonResponse(BaseModel):
    """Side-by-side laser altimeter transect comparison (ICESat-2 vs LOLA / MOLA)."""

    target_profile: SingleLaserProfile
    earth_analog_profile: SingleLaserProfile
    roughness_ratio: float = Field(
        ..., description="Ratio of Earth RMS roughness to target planetary RMS roughness"
    )
    slope_delta_deg: float = Field(
        ..., description="Absolute difference in laser transect mean slope (degrees)"
    )

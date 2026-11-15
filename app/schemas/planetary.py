from typing import Any, Literal

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


class EarthdataGranuleSummary(BaseModel):
    """Summary of the real NASA Earthdata ATLAS/ICESat-2 L2A (ATL03 v007) granule extracted for an Earth site."""

    dataset: str = Field(
        default="ATLAS/ICESat-2 L2A Global Geolocated Photon Data (ATL03)",
        description="NASA Earthdata dataset title",
    )
    granule_id: str = Field(..., description="Producer granule HDF5 filename")
    concept_id: str = Field(..., description="NASA CMR concept ID")
    time_start: str = Field(..., description="Granule acquisition start UTC timestamp")
    size_mb: float = Field(..., description="Source HDF5 granule size in megabytes")
    https_url: str = Field(..., description="NSIDC Earthdata Cloud HTTPS URL")
    beam_id: str = Field(..., description="Sampled ATLAS laser beam identifier (e.g. gt1l, gt2r)")
    photon_sample_count: int = Field(..., description="Number of geolocated photons sampled")
    segment_count: int = Field(..., description="Number of 20m geolocation segments sampled")
    along_track_relief_m: float = Field(
        ..., description="Along-track local relief (p95 - p5) from ICESat-2 segments in meters"
    )
    along_track_mean_elevation_m: float = Field(
        ..., description="Along-track mean elevation from ICESat-2 segments in meters"
    )


class Dem2dTopography(BaseModel):
    source_tile_url: str
    window_size_km: float
    resampled_grid_shape: list[int]
    resampled_gsd_m: float
    mean_elevation_m: float
    min_elevation_m: float
    max_elevation_m: float
    relief_m: float
    slope_mean_deg: float
    slope_p90_deg: float
    ruggedness_tri_m: float
    roughness_rms_m: float
    hypsometric_integral: float


class MacrostratUnit(BaseModel):
    map_id: int | None = None
    source_id: int | None = None
    unit_name: str
    age: str
    lithology: str
    description: str = ""


class MacrostratGeology(BaseModel):
    api_endpoint: str
    units_count: int
    mapped_units: list[MacrostratUnit]


class CmrGranuleRef(BaseModel):
    short_name: str
    concept_id: str
    granule_id: str
    time_start: str
    https_url: str


class EmitDetectedMineral(BaseModel):
    mineral_id: int
    mineral_name: str
    pixel_count: int
    mean_band_depth: float


class EmitStreamedMineralogy(BaseModel):
    granule_id: str
    concept_id: str
    https_url: str
    window_shape: list[int]
    group_1_mean_band_depth: float
    group_2_mean_band_depth: float
    top_group_1_minerals: list[EmitDetectedMineral]
    top_group_2_minerals: list[EmitDetectedMineral]


class EarthdataSatellitesBundle(BaseModel):
    dem_granule: CmrGranuleRef | None = None
    modis_ndvi_granule: CmrGranuleRef | None = None
    modis_cryosphere_granule: CmrGranuleRef | None = None
    emit_mineralogy_granule: CmrGranuleRef | None = None
    emit_streamed_mineralogy: EmitStreamedMineralogy | None = None


class EarthMultiSourceProvenance(BaseModel):
    """Real multi-source Earth extraction provenance (2D DEM, Macrostrat Geology API, NASA EMIT/MODIS/NASADEM)."""

    topography_2d_dem: Dem2dTopography
    geology_macrostrat: MacrostratGeology
    earthdata_satellites: EarthdataSatellitesBundle


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
    earthdata_granule: EarthdataGranuleSummary | None = Field(
        default=None,
        description="Real NASA Earthdata ICESat-2 ATL03 granule metadata (for curated Earth sites)",
    )
    earth_provenance: EarthMultiSourceProvenance | None = Field(
        default=None,
        description="Real multi-source provenance (2D DEM, Macrostrat Geology, NASA EMIT L2B & MODIS)",
    )
    planetary_provenance: dict[str, Any] | None = Field(
        default=None,
        description="Real NASA PDS & USGS Astrogeology multi-source provenance for Moon and Mars sites",
    )


class Atl03GranuleMetadata(BaseModel):
    granule_id: str
    concept_id: str
    version: str
    time_start: str
    time_end: str
    size_mb: float
    https_url: str


class Atl03OrbitInfo(BaseModel):
    orbit_number: list[int]
    rgt: list[int]
    cycle_number: list[int]
    sc_orient: list[int]


class Atl03SampledBeam(BaseModel):
    beam_id: str
    matched_query_bbox: bool
    window_segments_evaluated: int
    total_segments_in_beam: int


class Atl03ComputedTopography(BaseModel):
    relief_m: float
    slope_mean_deg: float
    slope_p90_deg: float
    ruggedness_tri_m: float
    roughness_rms_m: float
    hypsometric_integral: float
    mean_elevation_m: float


class Atl03GeolocationSegment(BaseModel):
    segment_offset: int
    reference_lat: float
    reference_lon: float
    dem_height_m: float | None = None
    solar_elevation_deg: float | None = None
    photon_count_in_segment: int


class Atl03PhotonSample(BaseModel):
    photon_index: int
    lat: float
    lon: float
    height_wgs84_m: float
    land_signal_confidence: int
    quality_flag: int
    delta_time_s: float


class EarthdataAtl03Record(BaseModel):
    """Complete extracted NASA Earthdata ATLAS/ICESat-2 L2A (ATL03 v007) record for an Earth analog site."""

    site_id: str
    cell_id: str
    site_name: str
    query_bbox: list[float]
    dataset: str
    granule: Atl03GranuleMetadata
    orbit_info: Atl03OrbitInfo
    sampled_beam: Atl03SampledBeam
    computed_topography: Atl03ComputedTopography
    geolocation_segments: list[Atl03GeolocationSegment]
    photons: list[Atl03PhotonSample]
    earth_provenance: EarthMultiSourceProvenance | None = None


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

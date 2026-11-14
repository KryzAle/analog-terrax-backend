from typing import Literal

from fastapi import APIRouter, Depends, Query

from app.api.deps import SimilarityEngineDep
from app.core.cache import cache_control
from app.schemas.analog import (
    AnalogHeatmapResponse,
    AnalogSearchRequest,
    AnalogSearchResponse,
    LaserProfileComparisonResponse,
    SimilarityWeights,
)

router = APIRouter(prefix="/analogs", tags=["analogs"])


@router.post(
    "/search",
    response_model=AnalogSearchResponse,
    summary="Find top-K similar Earth analog locations for a Moon or Mars site",
    dependencies=[Depends(cache_control(s_maxage=600, stale_while_revalidate=3600))],
)
async def search_earth_analogs(
    payload: AnalogSearchRequest,
    engine: SimilarityEngineDep,
) -> AnalogSearchResponse:
    """
    Execute weighted multi-criteria similarity search across topography (robust z-score kernel),
    geology (harmonized lithology similarity matrix), and mineralogy (masked cosine similarity).
    """
    return engine.search_analogs(payload)


@router.get(
    "/heatmap",
    response_model=AnalogHeatmapResponse,
    summary="Get quantized similarity scores across all Earth cells for global heatmap rendering",
    dependencies=[Depends(cache_control(s_maxage=3600, stale_while_revalidate=86400))],
)
async def get_earth_analog_heatmap(
    engine: SimilarityEngineDep,
    target_body: Literal["moon", "mars"] = Query(default="moon"),
    target_site_id: str = Query(default="moon_shackleton_rim"),
    w_topo: float = Query(default=0.45, ge=0.0, le=1.0),
    w_geo: float = Query(default=0.35, ge=0.0, le=1.0),
    w_min: float = Query(default=0.20, ge=0.0, le=1.0),
) -> AnalogHeatmapResponse:
    """Return quantized uint8 (0-100) similarity scores across all indexed Earth cells."""
    weights = SimilarityWeights(topography=w_topo, geology=w_geo, mineralogy=w_min)
    return engine.build_heatmap(
        target_body=target_body,
        target_site_id=target_site_id,
        weights=weights,
    )


@router.get(
    "/{earth_cell}/profile",
    response_model=LaserProfileComparisonResponse,
    summary="Compare laser altimeter elevation profiles (ICESat-2 vs LOLA / MOLA)",
    dependencies=[Depends(cache_control(s_maxage=3600, stale_while_revalidate=86400))],
)
async def get_laser_profile_comparison(
    earth_cell: str,
    engine: SimilarityEngineDep,
    target: str = Query(
        default="moon_shackleton_rim",
        description="Target Moon/Mars site_id or cell_id for laser profile comparison",
    ),
) -> LaserProfileComparisonResponse:
    """Return side-by-side laser altimeter transects (ICESat-2 ATLAS vs LRO LOLA or MGS MOLA)."""
    return engine.compare_laser_profiles(
        earth_identifier=earth_cell,
        target_identifier=target,
    )

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from app.api.deps import FeatureStoreDep
from app.core.cache import cache_control
from app.schemas.planetary import (
    CelestialBody,
    EarthdataAtl03Record,
    PlanetarySite,
    TaxonomyResponse,
)

router = APIRouter(tags=["planetary"])


@router.get(
    "/bodies/earth/earthdata",
    response_model=list[EarthdataAtl03Record],
    summary="List real extracted NASA Earthdata ATLAS/ICESat-2 L2A (ATL03 v007) records for all Earth sites",
    dependencies=[Depends(cache_control(s_maxage=3600, stale_while_revalidate=86400))],
)
async def list_earthdata_atl03_records(
    store: FeatureStoreDep,
) -> list[EarthdataAtl03Record]:
    """Return full extracted NASA Earthdata ATL03 v007 records (granules, orbit info, geolocation segments, and photons) for all curated Earth sites."""
    return store.list_earthdata_atl03()


@router.get(
    "/bodies/earth/sites/{site_id}/earthdata",
    response_model=EarthdataAtl03Record,
    summary="Get real extracted NASA Earthdata ATLAS/ICESat-2 L2A (ATL03 v007) record for a specific Earth site",
    dependencies=[Depends(cache_control(s_maxage=3600, stale_while_revalidate=86400))],
)
async def get_earthdata_atl03_record(
    site_id: Annotated[str, Path(max_length=128)],
    store: FeatureStoreDep,
) -> EarthdataAtl03Record:
    """Retrieve the extracted NASA Earthdata ATL03 v007 granule, segments, and photon samples for an Earth site_id or cell_id."""
    record = store.get_earthdata_atl03(site_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Earthdata ATL03 record not found for Earth site '{site_id}'.",
        )
    return record


@router.get(
    "/taxonomy",
    response_model=TaxonomyResponse,
    summary="Get harmonized cross-planet taxonomy and feature normalization stats",
    dependencies=[Depends(cache_control(s_maxage=3600, stale_while_revalidate=86400))],
)
async def get_taxonomy(store: FeatureStoreDep) -> TaxonomyResponse:
    """Return shared lithology classes, similarity matrix, mineral dimensions, and topo stats."""
    return store.get_taxonomy_response()


@router.get(
    "/bodies/{body}/sites",
    response_model=list[PlanetarySite],
    summary="List curated reference sites for Moon, Mars, or Earth",
    dependencies=[Depends(cache_control(s_maxage=3600, stale_while_revalidate=86400))],
)
async def list_body_sites(
    body: CelestialBody,
    store: FeatureStoreDep,
) -> list[PlanetarySite]:
    """Return curated candidate base locations (Moon/Mars) or terrestrial analogs (Earth)."""
    return store.list_curated_sites(body)


@router.get(
    "/bodies/{body}/sites/{site_id}",
    response_model=PlanetarySite,
    summary="Get fingerprint for a specific site or cell ID",
    dependencies=[Depends(cache_control(s_maxage=3600, stale_while_revalidate=86400))],
)
async def get_body_site(
    body: CelestialBody,
    site_id: Annotated[str, Path(max_length=128)],
    store: FeatureStoreDep,
) -> PlanetarySite:
    """Retrieve a single planetary site fingerprint by site_id or cell_id."""
    site = store.get_site_by_id_or_cell(site_id)
    if site is None or site.body != body:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Site '{site_id}' not found on body '{body}'.",
        )
    return site


@router.get(
    "/bodies/{body}/cells",
    response_model=PlanetarySite,
    summary="Snap latitude/longitude coordinates to the nearest indexed cell",
    dependencies=[Depends(cache_control(s_maxage=3600, stale_while_revalidate=86400))],
)
async def snap_body_cell(
    body: CelestialBody,
    store: FeatureStoreDep,
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees"),
) -> PlanetarySite:
    """Snap a click at (lat, lon) on Moon, Mars, or Earth to its nearest indexed cell."""
    return store.snap_coordinates_to_site(body=body, lat=lat, lon=lon)

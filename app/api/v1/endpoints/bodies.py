from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import FeatureStoreDep
from app.core.cache import cache_control
from app.schemas.planetary import CelestialBody, PlanetarySite, TaxonomyResponse

router = APIRouter(tags=["planetary"])


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
    site_id: str,
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

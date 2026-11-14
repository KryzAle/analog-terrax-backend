from httpx import AsyncClient


async def test_get_taxonomy(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/taxonomy")
    assert response.status_code == 200
    data = response.json()
    assert data["window_size_km"] == 25.0
    assert data["resampled_gsd_m"] == 500.0
    assert len(data["topography_features"]) == 6
    assert len(data["lithology_classes"]) == 7
    assert len(data["mineral_dimensions"]) == 9


async def test_list_moon_and_mars_sites(async_client: AsyncClient) -> None:
    moon_resp = await async_client.get("/api/v1/bodies/moon/sites")
    assert moon_resp.status_code == 200
    moon_sites = moon_resp.json()
    assert len(moon_sites) >= 6
    assert any(s["site_id"] == "moon_shackleton_rim" for s in moon_sites)

    mars_resp = await async_client.get("/api/v1/bodies/mars/sites")
    assert mars_resp.status_code == 200
    mars_sites = mars_resp.json()
    assert len(mars_sites) >= 6
    assert any(s["site_id"] == "mars_jezero_crater" for s in mars_sites)


async def test_snap_coordinates_to_cell(async_client: AsyncClient) -> None:
    resp = await async_client.get("/api/v1/bodies/mars/cells?lat=18.4&lon=77.6")
    assert resp.status_code == 200
    site = resp.json()
    assert site["site_id"] == "mars_jezero_crater"


async def test_search_analogs_rediscovers_expected_matches(async_client: AsyncClient) -> None:
    # 1. Marius Hills Lava Tube Pit (Moon) -> Lanzarote Monte Corona Lava Tube should rank in top 2
    marius_resp = await async_client.post(
        "/api/v1/analogs/search",
        json={
            "target_body": "moon",
            "target_site_id": "moon_marius_hills_pit",
            "weights": {"topography": 0.45, "geology": 0.35, "mineralogy": 0.20},
            "k": 5,
        },
    )
    assert marius_resp.status_code == 200
    marius_data = marius_resp.json()
    top_ids = [m["site"]["site_id"] for m in marius_data["matches"][:2]]
    assert "earth_lanzarote_corona_tube" in top_ids

    # 2. Shackleton Crater Rim (Moon South Pole) -> Queen Maud Anorthosite & McMurdo Dry Valleys
    shackleton_resp = await async_client.post(
        "/api/v1/analogs/search",
        json={
            "target_body": "moon",
            "target_site_id": "moon_shackleton_rim",
            "weights": {"topography": 0.45, "geology": 0.35, "mineralogy": 0.20},
            "k": 5,
        },
    )
    assert shackleton_resp.status_code == 200
    shackleton_top = [m["site"]["site_id"] for m in shackleton_resp.json()["matches"][:3]]
    assert "earth_queen_maud_anorthosite" in shackleton_top
    assert "earth_mcmurdo_dry_valleys" in shackleton_top

    # 3. Jezero Crater (Mars) -> Lake Salda Fan-Delta
    jezero_resp = await async_client.post(
        "/api/v1/analogs/search",
        json={
            "target_body": "mars",
            "target_site_id": "mars_jezero_crater",
            "weights": {"topography": 0.40, "geology": 0.35, "mineralogy": 0.25},
            "k": 5,
        },
    )
    assert jezero_resp.status_code == 200
    jezero_top = [m["site"]["site_id"] for m in jezero_resp.json()["matches"][:2]]
    assert "earth_lake_salda_delta" in jezero_top


async def test_heatmap_and_laser_profile_endpoints(async_client: AsyncClient) -> None:
    hm_resp = await async_client.get(
        "/api/v1/analogs/heatmap?target_body=moon&target_site_id=moon_shackleton_rim"
    )
    assert hm_resp.status_code == 200
    hm_data = hm_resp.json()
    assert len(hm_data["cells"]) >= 100
    assert all(0 <= c["score_uint8"] <= 100 for c in hm_data["cells"])

    # Compare real extracted ICESat-2 ATL03 profile (earth_c013 and earth_c001) against Moon Shackleton LOLA
    for cell_id in ("earth_c001", "earth_c004", "earth_c009", "earth_c013"):
        prof_resp = await async_client.get(
            f"/api/v1/analogs/{cell_id}/profile?target=moon_shackleton_rim"
        )
        assert prof_resp.status_code == 200
        prof_data = prof_resp.json()
        assert "ICESat-2" in prof_data["earth_analog_profile"]["instrument"]
        assert prof_data["earth_analog_profile"]["granule_id"].startswith("ATL03_")
        assert prof_data["earth_analog_profile"]["granule_id"].endswith(".h5")
        assert "LOLA" in prof_data["target_profile"]["instrument"]
        assert len(prof_data["earth_analog_profile"]["points"]) == 25


async def test_earthdata_atl03_endpoints_for_all_14_earth_sites(
    async_client: AsyncClient,
) -> None:
    earth_sites_resp = await async_client.get("/api/v1/bodies/earth/sites")
    assert earth_sites_resp.status_code == 200
    earth_sites = earth_sites_resp.json()
    assert len(earth_sites) == 14
    for site in earth_sites:
        gran = site["earthdata_granule"]
        assert gran is not None
        assert gran["granule_id"].startswith("ATL03_")
        assert gran["concept_id"].endswith("-NSIDC_CPRD")
        assert gran["segment_count"] == 25
        assert gran["photon_sample_count"] == 25

    all_ed_resp = await async_client.get("/api/v1/bodies/earth/earthdata")
    assert all_ed_resp.status_code == 200
    all_ed = all_ed_resp.json()
    assert len(all_ed) == 14

    single_ed_resp = await async_client.get(
        "/api/v1/bodies/earth/sites/earth_haughton_crater/earthdata"
    )
    assert single_ed_resp.status_code == 200
    single_ed = single_ed_resp.json()
    assert single_ed["site_id"] == "earth_haughton_crater"
    assert len(single_ed["geolocation_segments"]) == 25
    assert len(single_ed["photons"]) == 25

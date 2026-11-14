from __future__ import annotations

import numpy as np
from fastapi import HTTPException, status
from numpy.typing import NDArray

from app.schemas.analog import (
    AnalogHeatmapResponse,
    AnalogMatchResult,
    AnalogSearchFilters,
    AnalogSearchRequest,
    AnalogSearchResponse,
    FeatureContribution,
    HeatmapCellScore,
    LaserProfileComparisonResponse,
    ScoreBreakdown,
    SimilarityWeights,
)
from app.schemas.planetary import PlanetarySite
from app.services.feature_index import BodyMatrixIndex, FeatureIndexStore


class SimilarityEngine:
    """Vectorized multi-criteria planetary analog similarity scoring engine."""

    def __init__(self, store: FeatureIndexStore) -> None:
        self.store = store

    def _resolve_target_site(self, req: AnalogSearchRequest) -> PlanetarySite:
        if req.target_site_id:
            site = self.store.get_site_by_id_or_cell(req.target_site_id)
            if site is not None and site.body == req.target_body:
                return site
        if req.target_lat is not None and req.target_lon is not None:
            return self.store.snap_coordinates_to_site(
                req.target_body, req.target_lat, req.target_lon
            )
        if req.target_site_id:
            site = self.store.get_site_by_id_or_cell(req.target_site_id)
            if site is not None:
                return site
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target site '{req.target_site_id}' not found on {req.target_body}.",
        )

    @staticmethod
    def _normalize_weights(weights: SimilarityWeights) -> SimilarityWeights:
        total = weights.topography + weights.geology + weights.mineralogy
        if total <= 1e-6:
            return SimilarityWeights(topography=0.45, geology=0.35, mineralogy=0.20)
        return SimilarityWeights(
            topography=round(weights.topography / total, 4),
            geology=round(weights.geology / total, 4),
            mineralogy=round(weights.mineralogy / total, 4),
        )

    def _compute_vectorized_scores(
        self, target: PlanetarySite
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """
        Compute (S_topo, S_geo, S_min) across all Earth cells simultaneously.
        - S_topo: exp(-mean((z_target - z_cell)^2) / sigma^2)
        - S_geo : normalized bilinear form (f_t^T M f_c) / sqrt((f_t^T M f_t)(f_c^T M f_c))
        - S_min : masked cosine similarity over shared valid mineral dimensions
        """
        earth_idx = self.store.indices["earth"]
        raw_target = self.store.get_site_by_id_or_cell(target.cell_id)
        if raw_target is None:
            raise HTTPException(status_code=404, detail="Target cell vector missing.")

        t_topo = np.array(
            [
                target.topography.relief_m,
                target.topography.slope_mean_deg,
                target.topography.slope_p90_deg,
                target.topography.ruggedness_tri_m,
                target.topography.roughness_rms_m,
                target.topography.hypsometric_integral,
            ],
            dtype=np.float32,
        )
        z_target = (t_topo - self.store.topo_medians) / self.store.topo_iqrs
        z_earth = (earth_idx.topo - self.store.topo_medians) / self.store.topo_iqrs
        mean_sq_dist = np.mean((z_earth - z_target) ** 2, axis=1)
        s_topo = np.exp(-mean_sq_dist / (self.store.sigma_topo**2)).astype(np.float32)

        # Geology bilinear similarity with self-similarity normalization
        t_geo = np.array(
            [target.geology_fractions[c_id] for c_id in self.store.lithology_ids],
            dtype=np.float32,
        )
        m_mat = self.store.lithology_matrix
        mt_t = m_mat @ t_geo  # shape (7,)
        cross_sim = earth_idx.geo @ mt_t  # shape (N,)
        self_t = float(t_geo @ mt_t)
        self_e = np.einsum("ij,jk,ik->i", earth_idx.geo, m_mat, earth_idx.geo)
        denom = np.sqrt(np.maximum(self_t * self_e, 1e-8))
        s_geo = np.clip(cross_sim / denom, 0.0, 1.0).astype(np.float32)

        # Mineralogy masked cosine similarity
        t_min_list = [target.mineralogy.get(m_id) for m_id in self.store.mineral_ids]
        t_min_mask = np.array([v is not None and v >= 0.0 for v in t_min_list], dtype=np.bool_)
        t_min_vec = np.array(
            [float(v) if (v is not None and v >= 0.0) else 0.0 for v in t_min_list],
            dtype=np.float32,
        )

        joint_mask = earth_idx.min_mask & t_min_mask[None, :]  # (N, 9)
        t_masked = np.where(joint_mask, t_min_vec[None, :], 0.0)
        e_masked = np.where(joint_mask, earth_idx.min_vec, 0.0)

        dot_prod = np.sum(t_masked * e_masked, axis=1)
        norm_t = np.linalg.norm(t_masked, axis=1)
        norm_e = np.linalg.norm(e_masked, axis=1)
        s_min = np.where(
            (norm_t > 1e-6) & (norm_e > 1e-6),
            np.clip(dot_prod / (norm_t * norm_e), 0.0, 1.0),
            0.5,
        ).astype(np.float32)

        return s_topo, s_geo, s_min

    def _explain_top_features(
        self, target: PlanetarySite, earth_site: PlanetarySite
    ) -> list[FeatureContribution]:
        contributions: list[FeatureContribution] = []

        # Topographic feature closeness
        topo_specs = self.store.feature_stats_raw["topography_features"]
        t_topo_dict = target.topography.model_dump()
        e_topo_dict = earth_site.topography.model_dump()

        for spec in topo_specs:
            f_id = spec["id"]
            t_val = float(t_topo_dict[f_id])
            e_val = float(e_topo_dict[f_id])
            iqr = float(spec["iqr"])
            z_diff = abs(t_val - e_val) / max(iqr, 1e-3)
            closeness = float(np.exp(-(z_diff**2) / (self.store.sigma_topo**2)) * 100.0)
            contributions.append(
                FeatureContribution(
                    family="topography",
                    feature_id=f_id,
                    label=spec["label"],
                    target_value=round(t_val, 2),
                    earth_value=round(e_val, 2),
                    unit=spec["unit"],
                    closeness_pct=round(closeness, 1),
                )
            )

        # Dominant shared lithology class
        for l_spec in self.store.taxonomy_raw["lithology_classes"]:
            l_id = l_spec["id"]
            t_frac = float(target.geology_fractions.get(l_id, 0.0))
            e_frac = float(earth_site.geology_fractions.get(l_id, 0.0))
            if t_frac >= 0.15 and e_frac >= 0.15:
                closeness = (1.0 - abs(t_frac - e_frac)) * 100.0
                contributions.append(
                    FeatureContribution(
                        family="geology",
                        feature_id=l_id,
                        label=l_spec["label"],
                        target_value=round(t_frac * 100.0, 1),
                        earth_value=round(e_frac * 100.0, 1),
                        unit="% area",
                        closeness_pct=round(closeness, 1),
                    )
                )

        # Shared prominent mineral dimensions
        for m_spec in self.store.taxonomy_raw["mineral_dimensions"]:
            m_id = m_spec["id"]
            t_m = target.mineralogy.get(m_id)
            e_m = earth_site.mineralogy.get(m_id)
            if t_m is not None and e_m is not None and t_m >= 0.20 and e_m >= 0.20:
                closeness = (1.0 - min(abs(t_m - e_m), 1.0)) * 100.0
                contributions.append(
                    FeatureContribution(
                        family="mineralogy",
                        feature_id=m_id,
                        label=m_spec["label"],
                        target_value=round(t_m * 100.0, 1),
                        earth_value=round(e_m * 100.0, 1),
                        unit="index %",
                        closeness_pct=round(closeness, 1),
                    )
                )

        contributions.sort(key=lambda c: c.closeness_pct, reverse=True)
        return contributions[:4]

    @staticmethod
    def _apply_filters(
        earth_idx: BodyMatrixIndex, filters: AnalogSearchFilters
    ) -> NDArray[np.bool_]:
        mask = np.ones(len(earth_idx.cell_ids), dtype=np.bool_)
        if filters.curated_only:
            mask &= earth_idx.is_curated
        if filters.exclude_vegetated:
            mask &= earth_idx.ndvi <= filters.max_ndvi
        if filters.require_impact:
            mask &= earth_idx.is_impact
        if filters.require_ice:
            mask &= earth_idx.has_ice
        if filters.require_volcanic:
            mask &= earth_idx.is_volcanic
        return mask

    def search_analogs(self, req: AnalogSearchRequest) -> AnalogSearchResponse:
        target = self._resolve_target_site(req)
        norm_w = self._normalize_weights(req.weights)
        s_topo, s_geo, s_min = self._compute_vectorized_scores(target)

        overall = norm_w.topography * s_topo + norm_w.geology * s_geo + norm_w.mineralogy * s_min

        earth_idx = self.store.indices["earth"]
        valid_mask = self._apply_filters(earth_idx, req.filters)
        candidate_indices = np.where(valid_mask)[0]

        if candidate_indices.size == 0:
            return AnalogSearchResponse(
                target=target,
                normalized_weights=norm_w,
                total_candidates_evaluated=0,
                matches=[],
            )

        ranked_sub = np.argsort(overall[candidate_indices])[::-1][: req.k]
        top_indices = candidate_indices[ranked_sub]

        matches: list[AnalogMatchResult] = []
        for rank_idx, idx_pos in enumerate(top_indices, start=1):
            cell_id = str(earth_idx.cell_ids[idx_pos])
            earth_site = self.store.get_site_by_id_or_cell(cell_id)
            if earth_site is None:
                continue

            compared_minerals = [
                m_id
                for m_id in self.store.mineral_ids
                if target.mineralogy.get(m_id) is not None
                and earth_site.mineralogy.get(m_id) is not None
            ]

            ov_val = float(overall[idx_pos])
            scores = ScoreBreakdown(
                overall_score=round(ov_val, 4),
                overall_pct=round(ov_val * 100.0, 1),
                topography_score=round(float(s_topo[idx_pos]), 4),
                geology_score=round(float(s_geo[idx_pos]), 4),
                mineralogy_score=round(float(s_min[idx_pos]), 4),
                compared_mineral_dimensions=compared_minerals,
            )

            matches.append(
                AnalogMatchResult(
                    rank=rank_idx,
                    site=earth_site,
                    scores=scores,
                    top_contributing_features=self._explain_top_features(target, earth_site),
                )
            )

        return AnalogSearchResponse(
            target=target,
            normalized_weights=norm_w,
            total_candidates_evaluated=int(candidate_indices.size),
            matches=matches,
        )

    def build_heatmap(
        self,
        target_body: str,
        target_site_id: str,
        weights: SimilarityWeights,
    ) -> AnalogHeatmapResponse:
        req = AnalogSearchRequest(
            target_body="moon" if target_body == "moon" else "mars",
            target_site_id=target_site_id,
            weights=weights,
        )
        target = self._resolve_target_site(req)
        norm_w = self._normalize_weights(weights)
        s_topo, s_geo, s_min = self._compute_vectorized_scores(target)
        overall = norm_w.topography * s_topo + norm_w.geology * s_geo + norm_w.mineralogy * s_min

        earth_idx = self.store.indices["earth"]
        cells: list[HeatmapCellScore] = []
        for i in range(len(earth_idx.cell_ids)):
            c_id = str(earth_idx.cell_ids[i])
            site = self.store.get_site_by_id_or_cell(c_id)
            cells.append(
                HeatmapCellScore(
                    cell_id=c_id,
                    site_id=str(earth_idx.site_ids[i]),
                    name=site.name if site else c_id,
                    lat=round(float(earth_idx.lats[i]), 4),
                    lon=round(float(earth_idx.lons[i]), 4),
                    is_curated_site=bool(earth_idx.is_curated[i]),
                    score_uint8=int(np.clip(np.round(overall[i] * 100.0), 0, 100)),
                    topography_uint8=int(np.clip(np.round(s_topo[i] * 100.0), 0, 100)),
                    geology_uint8=int(np.clip(np.round(s_geo[i] * 100.0), 0, 100)),
                    mineralogy_uint8=int(np.clip(np.round(s_min[i] * 100.0), 0, 100)),
                )
            )

        return AnalogHeatmapResponse(
            target_site_id=target.site_id,
            target_name=target.name,
            target_body="moon" if target.body == "moon" else "mars",
            cells=cells,
        )

    def compare_laser_profiles(
        self, earth_identifier: str, target_identifier: str
    ) -> LaserProfileComparisonResponse:
        earth_prof = self.store.get_laser_profile(earth_identifier)
        if earth_prof is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Laser altimetry profile not found for Earth site '{earth_identifier}'.",
            )
        target_prof = self.store.get_laser_profile(target_identifier)
        if target_prof is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Laser altimetry profile not found for target site '{target_identifier}'.",
            )

        t_rms = max(target_prof.stats.rms_roughness_m, 1e-3)
        e_rms = earth_prof.stats.rms_roughness_m
        slope_delta = abs(earth_prof.stats.mean_slope_deg - target_prof.stats.mean_slope_deg)

        return LaserProfileComparisonResponse(
            target_profile=target_prof,
            earth_analog_profile=earth_prof,
            roughness_ratio=round(e_rms / t_rms, 3),
            slope_delta_deg=round(slope_delta, 3),
        )

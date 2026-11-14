from typing import Annotated

import httpx
from fastapi import Depends

from app.core.config import Settings, get_settings
from app.core.http_client import get_http_client
from app.services.feature_index import FeatureIndexStore, get_feature_index_store
from app.services.similarity_engine import SimilarityEngine


def get_similarity_engine(
    store: Annotated[FeatureIndexStore, Depends(get_feature_index_store)],
) -> SimilarityEngine:
    return SimilarityEngine(store=store)


SettingsDep = Annotated[Settings, Depends(get_settings)]
HttpClientDep = Annotated[httpx.AsyncClient, Depends(get_http_client)]
FeatureStoreDep = Annotated[FeatureIndexStore, Depends(get_feature_index_store)]
SimilarityEngineDep = Annotated[SimilarityEngine, Depends(get_similarity_engine)]

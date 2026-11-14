from typing import Annotated

import httpx
from fastapi import Depends

from app.core.config import Settings, get_settings
from app.core.http_client import get_http_client

SettingsDep = Annotated[Settings, Depends(get_settings)]
HttpClientDep = Annotated[httpx.AsyncClient, Depends(get_http_client)]

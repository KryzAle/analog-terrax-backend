from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class UpstreamServiceError(Exception):
    """Raised when an external upstream service (e.g. NASA API) fails or times out."""

    def __init__(self, service: str, message: str, status_code: int = 502) -> None:
        self.service = service
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def _error_payload(
    request: Request,
    code: str,
    message: str,
    details: Any = None,
) -> dict[str, Any]:
    request_id = getattr(request.state, "request_id", None)
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details,
            "request_id": request_id,
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    """Register standardized JSON exception handlers on the FastAPI app."""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                request=request,
                code=f"HTTP_{exc.status_code}",
                message=str(exc.detail),
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_payload(
                request=request,
                code="VALIDATION_ERROR",
                message="Request payload or parameters failed validation.",
                details=exc.errors(),
            ),
        )

    @app.exception_handler(UpstreamServiceError)
    async def upstream_exception_handler(
        request: Request, exc: UpstreamServiceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                request=request,
                code="UPSTREAM_SERVICE_ERROR",
                message=exc.message,
                details={"service": exc.service},
            ),
        )

"""FastAPI application (M10).

Transactions must arrive in time order. There is no authentication: put the service behind an
authenticating gateway before exposing it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError

from fraud.api.schemas import Health, ScoreResponse, Transaction
from fraud.api.scorer import InvalidField, Scorer
from fraud.config import Config, load_config


def create_app(config: Config | None = None, scorer: Scorer | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.scorer = scorer or Scorer(config or load_config())
        yield

    app = FastAPI(title="Fraud scoring API", lifespan=lifespan)

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        current: Scorer = request.app.state.scorer
        return Health(
            status="ok",
            model_version=current.config.api.model_version,
            store_rows=current.store.row_count(),
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(transaction: Transaction, request: Request) -> dict:
        current: Scorer = request.app.state.scorer
        try:
            return current.score(transaction.model_dump())
        except InvalidField as error:
            raise RequestValidationError(
                [{"loc": ("body", error.field), "msg": str(error), "type": "type_error"}]
            ) from error

    @app.post("/store/reset")
    def reset(request: Request) -> dict[str, str]:
        current: Scorer = request.app.state.scorer
        if not current.config.api.allow_reset:
            raise HTTPException(status_code=403, detail="store reset is disabled")
        current.store.reset()
        return {"status": "reset"}

    return app


app = create_app()

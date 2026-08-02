"""Run the HTTP service with values from config.yaml."""

from __future__ import annotations

import uvicorn

from .main import app


def main() -> None:
    loaded = app.state.loaded_config
    uvicorn.run(
        app,
        host=loaded.settings.server.host,
        port=loaded.settings.server.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

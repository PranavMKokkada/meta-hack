"""Server entry point for openenv multi-mode deployment."""

import os
import sys

# Add parent directory to path so imports work from both root and server/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app  # noqa: E402


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    port = int(os.environ.get("PORT", port))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

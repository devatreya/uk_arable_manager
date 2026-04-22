"""
ORS server entry point for uk_arable_manager.

Usage:
  python app.py                          # runs on port 8080
  python app.py --port 9000
  uvicorn app:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import argparse

from openreward.environments import Server

from env import UKArableManager

server = Server(environments=[UKArableManager])
app = server.app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="uk_arable_manager ORS server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    print(f"Starting uk_arable_manager ORS server on {args.host}:{args.port}")
    print(f"Environment: UKArableManager  |  Splits: train / validation / test")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)

"""
OpenReward server entry point for uk_arable_manager.

Usage:
  python server.py                  # runs on 0.0.0.0:8080
  python server.py --port 9000
"""
from __future__ import annotations

import argparse

from openreward.environments import Server

from env import UKArableManager

server = Server(environments=[UKArableManager])
app = server.app

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="uk_arable_manager OpenReward server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    print(f"Starting uk_arable_manager on {args.host}:{args.port}")
    print("Environment: UKArableManager  |  Splits: train / validation / test")
    server.run(host=args.host, port=args.port)

"""LOCAL DEV MOCK for testing the landing page demo without a real backend.

Serves the landing page at / and a fake /segment that returns the same JSON
shape as the real Cloud Run API (deploy/cloudrun/main.py):
    {n_neurons, cells:[{cell_id,area_px,centroid_x,centroid_y}], overlay_png_b64, model}

This is NOT part of the deployment — it exists only so the front-end wiring can be
verified end-to-end (upload -> fetch -> overlay + count + CSV). Same-origin, so no
CORS involved. Run:  python landing/mock_server.py   then open http://localhost:8765
"""
import base64
import io
import json
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
PORT = 8765


def fake_overlay(n=42, size=384):
    """A dark microscopy-ish image with n red neuron outlines."""
    rng = random.Random(0)
    arr = (np.random.default_rng(0).normal(28, 8, (size, size, 3)).clip(0, 60)).astype("uint8")
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    cells = []
    for i in range(1, n + 1):
        cx, cy = rng.randint(20, size - 20), rng.randint(20, size - 20)
        r = rng.randint(7, 14)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 60, 60), width=2)
        cells.append({"cell_id": i, "area_px": int(3.14 * r * r),
                      "centroid_x": round(cx + rng.uniform(-1, 1), 1),
                      "centroid_y": round(cy + rng.uniform(-1, 1), 1)})
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), cells


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        html = (HERE / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html)

    def do_POST(self):
        if self.path.rstrip("/") != "/segment":
            self.send_response(404); self.end_headers(); return
        # drain the uploaded body (we don't actually process it)
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        n = random.Random().randint(180, 340)
        b64, cells = fake_overlay(n)
        payload = json.dumps({
            "n_neurons": len(cells), "cells": cells,
            "overlay_png_b64": b64, "filename": "uploaded",
            "model": "MOCK (local dev server — not a real model)",
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    print(f"Mock BlueSpotter server on http://localhost:{PORT}  (Ctrl-C to stop)")
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()

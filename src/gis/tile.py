import math
import os
import requests
from pathlib import Path

from gis.config import Config

config = Config()


# Convert latitude/longitude to tile numbers at a given zoom level
def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

# Download one tile
def download_tile(z, x, y, out_dir: Path = config.mnt_path / 'image'):
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    out_path = out_dir / str(z) /  f"{z}_{x}_{y}.jpg"
    if not out_path.exists():
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            out_path.write_bytes(r.content)
            print(f"Saved {out_path}")
            return True
        else:
            print(f"Failed: {url} ({r.status_code})")

    else:
        print(f"Exists: {out_path}")
        return False

# Download tiles for a bounding box
def download_bbox(lat_min, lon_min, lat_max, lon_max, zoom, out_dir="tiles"):
    x_min, y_max = latlon_to_tile(lat_min, lon_min, zoom)
    x_max, y_min = latlon_to_tile(lat_max, lon_max, zoom)

    os.makedirs(out_dir, exist_ok=True)

    for x in range(min(x_min, x_max), max(x_min, x_max) + 1):
        for y in range(min(y_min, y_max), max(y_min, y_max) + 1):
            download_tile(zoom, x, y, out_dir)


# lat_min, lon_min = -31.731489757606823, 137.23765792723077
# lat_max, lon_max = -31.73430481216368, 137.24318327780156


# download_bbox(lat_min, lon_min, lat_max, lon_max, 18, out_dir="test18")

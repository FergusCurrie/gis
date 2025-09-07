import math
import os
import requests
from pathlib import Path

from gis.config import Config

config = Config()

import pyproj
import mercantile

def extend_line_and_get_tiles(lat1, lon1, lat2, lon2, extension_distance_m, zoom_level, 
                             local_crs='EPSG:3857'):
    """
    Extends a line between two lat/lon points and returns mercantile tiles along the extended line.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates  
        extension_distance_m: How far to extend the line in meters
        zoom_level: Zoom level for tile calculation
        local_crs: Projected CRS for accurate distance calculations
        
    Returns:
        list: Mercantile tiles covering the extended line
    """
    
    transformer_to_proj = pyproj.Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    transformer_to_geo = pyproj.Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
    
    x1, y1 = transformer_to_proj.transform(lon1, lat1)
    x2, y2 = transformer_to_proj.transform(lon2, lat2)
    
    dx = x2 - x1
    dy = y2 - y1

    line_length = math.sqrt(dx*dx + dy*dy)


    unit_dx = dx / line_length
    unit_dy = dy / line_length
    
    back_x = x1 - unit_dx * extension_distance_m
    back_y = y1 - unit_dy * extension_distance_m
 
    forward_x = x2 + unit_dx * extension_distance_m
    forward_y = y2 + unit_dy * extension_distance_m
    
    num_samples = max(10, int((line_length + 2*extension_distance_m) / 100))  # Sample every ~100m
    
    sample_points = []
    for i in range(num_samples + 1):
        t = i / num_samples
        sample_x = back_x + t * (forward_x - back_x)
        sample_y = back_y + t * (forward_y - back_y)
        sample_lon, sample_lat = transformer_to_geo.transform(sample_x, sample_y)
        sample_points.append((sample_lat, sample_lon))

    tiles = set()
    for lat, lon in sample_points:
        tile = mercantile.tile(lon, lat, zoom_level)
        tiles.add(tile)
    return list(tiles)

def lat_lon_to_tile_pixel(latitude, longitude, zoom, width=256):
    """
    Converts latitude/longitude coordinates to tile position and pixel coordinates within that tile.
    
    Args:
        latitude (float): Latitude in degrees
        longitude (float): Longitude in degrees  
        zoom (int): Zoom level
        width (int): Tile width in pixels (default 256)
        
    Returns:
        tuple: (tile_x, tile_y, pixel_x, pixel_y)
    """
    # Convert lat/lon to Web Mercator normalized coordinates
    mercator_x = longitude / 360.0 + 0.5
    mercator_y = 0.5 - (math.log(math.tan(math.radians(latitude)) + 1/math.cos(math.radians(latitude))) / (2 * math.pi))
    
    # Convert to global pixel coordinates
    map_width = width * (2 ** zoom)
    global_pixel_x = mercator_x * map_width
    global_pixel_y = mercator_y * map_width
    
    # Calculate tile coordinates
    tile_x = int(global_pixel_x // width)
    tile_y = int(global_pixel_y // width)
    
    # Calculate pixel coordinates within the tile
    pixel_x = global_pixel_x - (tile_x * width)
    pixel_y = global_pixel_y - (tile_y * width)
    
    return tile_x, tile_y, pixel_x, pixel_y


def tile_to_latlon(z, x, y):
    """
    Converts web Mercator tile coordinates (x, y) and zoom level to 
    latitude and longitude.
    """
    n = 2.0 ** z
    lon_rad = x / n * 2 * math.pi - math.pi
    lat_rad = math.atan(math.sinh(math.pi - (2 * math.pi * y) / n))

    lon_deg = math.degrees(lon_rad)
    lat_deg = math.degrees(lat_rad)

    return lat_deg, lon_deg

def tile_pixel_to_lat_lon(zoom, tile_x, tile_y, pixel_x, pixel_y, width=256):
    """
    Converts a pixel position on tile + tile location to lat, lon of pixel 
    """
    global_pixel_x = tile_x * width + pixel_x
    global_pixel_y = tile_y * width + pixel_y
    map_width = width * (2 ** zoom)
    mercator_x = (global_pixel_x / map_width) - 0.5
    mercator_y = 0.5 - (global_pixel_y / map_width)
    longitude = mercator_x * 360.0
    latitude = math.degrees(math.atan(math.sinh(mercator_y * 2 * math.pi)))
    return latitude, longitude

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

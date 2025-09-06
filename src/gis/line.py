import numpy as np
import math
from scipy import ndimage

def tile_pixel_to_latlon(tile_x, tile_y, tile_z, pixel_x, pixel_y, tile_size=256):
    """
    Convert pixel coordinates within a tile to latitude/longitude.
    
    Args:
        tile_x, tile_y: Tile coordinates (x=longitude, y=latitude direction)
        tile_z: Zoom level
        pixel_x, pixel_y: Pixel coordinates within the tile (0 to tile_size-1)
        tile_size: Size of tile in pixels (default 256)
    
    Returns:
        (longitude, latitude) tuple
    """
    # Convert tile + pixel to global pixel coordinates at this zoom level
    global_pixel_x = tile_x * tile_size + pixel_x
    global_pixel_y = tile_y * tile_size + pixel_y
    
    # Convert to lat/lon using Web Mercator math
    n = 2.0 ** tile_z
    
    # Longitude is straightforward
    lon = (global_pixel_x / (tile_size * n)) * 360.0 - 180.0
    
    # Latitude requires inverse Mercator projection
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * global_pixel_y / (tile_size * n))))
    lat = math.degrees(lat_rad)
    
    return lon, lat

def extract_skeleton_points(skeleton_mask):
    """
    Extract ordered points from a skeleton mask.
    
    Args:
        skeleton_mask: Binary skeleton from morphology.skeletonize()
    
    Returns:
        List of (y, x) pixel coordinates along the skeleton
    """
    # Find all skeleton pixels
    skeleton_pixels = np.where(skeleton_mask)
    skeleton_coords = list(zip(skeleton_pixels[0], skeleton_pixels[1]))
    
    if len(skeleton_coords) < 2:
        return skeleton_coords
    
    # Order the points by following the skeleton
    return order_skeleton_points(skeleton_coords, skeleton_mask)

def order_skeleton_points(skeleton_coords, skeleton_mask):
    """
    Order skeleton points to form a continuous line.
    """
    if len(skeleton_coords) < 2:
        return skeleton_coords
    
    # Build adjacency graph
    coord_set = set(skeleton_coords)
    adjacency = {}
    
    for y, x in skeleton_coords:
        neighbors = []
        # Check 8-connected neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) in coord_set:
                    neighbors.append((ny, nx))
        adjacency[(y, x)] = neighbors
    
    # Find endpoints (nodes with only 1 neighbor)
    endpoints = [coord for coord, neighbors in adjacency.items() if len(neighbors) == 1]
    
    if not endpoints:
        # No clear endpoints, start from any point
        start_point = skeleton_coords[0]
    else:
        start_point = endpoints[0]
    
    # Traverse the skeleton
    ordered_points = []
    visited = set()
    current = start_point
    
    while current and current not in visited:
        ordered_points.append(current)
        visited.add(current)
        
        # Find next unvisited neighbor
        next_point = None
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                next_point = neighbor
                break
        current = next_point
    
    return ordered_points

def skeleton_to_wkt_linestring(skeleton_mask, tile_x, tile_y, tile_z, tile_size=256, simplify_tolerance=0.0001):
    """
    Convert skeleton mask to WKT LINESTRING with geographic coordinates.
    
    Args:
        skeleton_mask: Binary skeleton from morphology.skeletonize()
        tile_x, tile_y, tile_z: Tile coordinates
        tile_size: Tile size in pixels
        simplify_tolerance: Tolerance for coordinate simplification (degrees)
    
    Returns:
        WKT LINESTRING string
    """
    # Extract ordered skeleton points
    skeleton_points = extract_skeleton_points(skeleton_mask)
    
    if len(skeleton_points) < 2:
        return None
    
    # Convert to lat/lon coordinates
    coordinates = []
    for pixel_y, pixel_x in skeleton_points:
        lon, lat = tile_pixel_to_latlon(tile_x, tile_y, tile_z, pixel_x, pixel_y, tile_size)
        coordinates.append((lon, lat))
    
    # Optional: Simplify coordinates to reduce file size
    if simplify_tolerance > 0:
        coordinates = simplify_coordinates(coordinates, simplify_tolerance)
    
    # Format as WKT
    coord_strings = [f"{lon} {lat}" for lon, lat in coordinates]
    wkt = f"LINESTRING({', '.join(coord_strings)})"
    
    return wkt

def simplify_coordinates(coordinates, tolerance):
    """
    Simple coordinate simplification using distance threshold.
    For more advanced simplification, consider using the Ramer-Douglas-Peucker algorithm.
    """
    if len(coordinates) <= 2:
        return coordinates
    
    simplified = [coordinates[0]]  # Always keep first point
    
    for i in range(1, len(coordinates) - 1):
        # Calculate distance to last kept point
        dx = coordinates[i][0] - simplified[-1][0]
        dy = coordinates[i][1] - simplified[-1][1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > tolerance:
            simplified.append(coordinates[i])
    
    simplified.append(coordinates[-1])  # Always keep last point
    return simplified

def skeleton_to_wkt_multilinestring(skeleton_mask, tile_x, tile_y, tile_z, tile_size=256):
    """
    Convert skeleton mask to WKT MULTILINESTRING for cases with multiple disconnected lines.
    """
    # Label connected components
    labeled_skeleton = ndimage.label(skeleton_mask)[0]
    
    linestrings = []
    
    for component_id in range(1, labeled_skeleton.max() + 1):
        component_mask = labeled_skeleton == component_id
        
        # Extract skeleton for this component
        wkt_line = skeleton_to_wkt_linestring(component_mask, tile_x, tile_y, tile_z, tile_size)
        
        if wkt_line:
            # Extract just the coordinate part
            coord_part = wkt_line.replace('LINESTRING(', '').replace(')', '')
            linestrings.append(f"({coord_part})")
    # print(len(linestrings))
    if not linestrings:
        return None
    elif len(linestrings) == 1:
        return f"LINESTRING{linestrings[0]}"
    else:
        return f"MULTILINESTRING({', '.join(linestrings)})"

def process_rail_skeleton_from_tiles(skeleton_mask, tile_info_list):
    """
    Process skeleton that spans multiple tiles.
    
    Args:
        skeleton_mask: Combined skeleton mask from stitched tiles
        tile_info_list: List of dictionaries with 'x', 'y', 'z', 'offset_x', 'offset_y' for each tile
    
    Returns:
        WKT string
    """
    # This is more complex - you'd need to track which pixels belong to which tile
    # For now, assuming single tile or that you can determine tile boundaries
    
    # Find skeleton points
    skeleton_points = extract_skeleton_points(skeleton_mask)
    
    coordinates = []
    for pixel_y, pixel_x in skeleton_points:
        # Determine which tile this pixel belongs to
        tile_info = find_tile_for_pixel(pixel_x, pixel_y, tile_info_list)
        
        if tile_info:
            # Convert to tile-local coordinates
            local_x = pixel_x - tile_info['offset_x']
            local_y = pixel_y - tile_info['offset_y']
            
            # Convert to lat/lon
            lon, lat = tile_pixel_to_latlon(
                tile_info['x'], tile_info['y'], tile_info['z'], 
                local_x, local_y
            )
            coordinates.append((lon, lat))
    
    if len(coordinates) < 2:
        return None
    
    coord_strings = [f"{lon} {lat}" for lon, lat in coordinates]
    return f"LINESTRING({', '.join(coord_strings)})"

def find_tile_for_pixel(pixel_x, pixel_y, tile_info_list, tile_size=256):
    """Helper function to find which tile a pixel belongs to."""
    for tile_info in tile_info_list:
        min_x = tile_info['offset_x']
        max_x = min_x + tile_size
        min_y = tile_info['offset_y']
        max_y = min_y + tile_size
        
        if min_x <= pixel_x < max_x and min_y <= pixel_y < max_y:
            return tile_info
    
    return None

# Example usage
def example_usage():
    """Example of how to use the skeleton to WKT conversion."""
    
    # Create a simple skeleton for demonstration
    test_skeleton = np.zeros((50, 50), dtype=bool)
    # Draw a diagonal line
    for i in range(45):
        test_skeleton[i, i] = True
    
    # Example tile coordinates (Melbourne area, zoom 18)
    tile_x = 236870
    tile_y = 156616
    tile_z = 18
    
    # Convert to WKT
    wkt_result = skeleton_to_wkt_linestring(test_skeleton, tile_x, tile_y, tile_z)
    
    print("Example WKT output:")
    print(wkt_result)
    
    # For multiple disconnected lines
    wkt_multi = skeleton_to_wkt_multilinestring(test_skeleton, tile_x, tile_y, tile_z)
    print("\nMultilinestring version:")
    print(wkt_multi)


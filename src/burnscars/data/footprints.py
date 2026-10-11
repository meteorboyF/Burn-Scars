"""Chip footprints and pairwise geometry in one equal-area CRS.

Chips come in several UTM zones. Footprints (raster bounds in the native CRS)
are densified and reprojected to DIST_CRS (EPSG:5070, CONUS Albers equal-area;
choice in docs/DECISIONS.md D-009). Distances are minimum edge-to-edge between
polygons (0 when touching/overlapping). A geodesic cross-check re-measures the
same nearest-point pair on the WGS84 ellipsoid.
"""
import numpy as np
import shapely
from pyproj import CRS, Geod, Transformer
from shapely.geometry import Polygon

DIST_CRS = "EPSG:5070"
_GEOD = Geod(ellps="WGS84")


def densified_ring(bounds, n=16):
    """Boundary points of an axis-aligned box, n points per edge."""
    left, bottom, right, top = bounds
    xs = np.linspace(left, right, n)
    ys = np.linspace(bottom, top, n)
    return np.concatenate([
        np.c_[xs, np.full(n, bottom)], np.c_[np.full(n, right), ys],
        np.c_[xs[::-1], np.full(n, top)], np.c_[np.full(n, left), ys[::-1]],
    ])


def reproject_footprint(bounds, src_crs, dst_crs=DIST_CRS, n=16) -> Polygon:
    tr = Transformer.from_crs(CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs),
                              always_xy=True)
    ring = densified_ring(bounds, n)
    x, y = tr.transform(ring[:, 0], ring[:, 1])
    return Polygon(np.c_[x, y])


def pairwise_distance(polys_a, polys_b) -> np.ndarray:
    """(len(a), len(b)) matrix of min edge-to-edge distances in CRS units (m)."""
    a = np.asarray(polys_a, dtype=object)
    b = np.asarray(polys_b, dtype=object)
    return shapely.distance(a[:, None], b[None, :])


def pairwise_overlap(polys_a, polys_b, dist=None):
    """Intersection area and footprint IoU, computed only where distance == 0."""
    a = np.asarray(polys_a, dtype=object)
    b = np.asarray(polys_b, dtype=object)
    if dist is None:
        dist = pairwise_distance(a, b)
    inter = np.zeros(dist.shape)
    iou = np.zeros(dist.shape)
    ii, jj = np.nonzero(dist == 0)
    if len(ii):
        ia = shapely.area(shapely.intersection(a[ii], b[jj]))
        ua = shapely.area(a[ii]) + shapely.area(b[jj]) - ia
        inter[ii, jj] = ia
        iou[ii, jj] = np.where(ua > 0, ia / ua, 0)
    return inter, iou


def geodesic_nearest(poly_a: Polygon, poly_b: Polygon, crs=DIST_CRS) -> float:
    """Geodesic length (m) of the planar nearest-point segment between two footprints."""
    if poly_a.intersects(poly_b):
        return 0.0
    line = shapely.shortest_line(poly_a, poly_b)
    (x0, y0), (x1, y1) = line.coords
    tr = Transformer.from_crs(CRS.from_user_input(crs), "EPSG:4326", always_xy=True)
    lon, lat = tr.transform([x0, x1], [y0, y1])
    return float(_GEOD.inv(lon[0], lat[0], lon[1], lat[1])[2])


def single_linkage_groups(dist: np.ndarray, threshold_m: float) -> np.ndarray:
    """Connected components of the graph {i~j : dist[i,j] <= threshold}. Labels are
    renumbered by first appearance so they are deterministic given the chip order."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    adj = csr_matrix(dist <= threshold_m)
    _, labels = connected_components(adj, directed=False)
    remap, out = {}, np.empty_like(labels)
    for i, lab in enumerate(labels):
        out[i] = remap.setdefault(lab, len(remap))
    return out

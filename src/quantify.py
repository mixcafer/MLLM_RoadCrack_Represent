from __future__ import annotations

from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology


NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


def quantify_cracks(
    mask_path: str | Path,
    image_path: str | Path | None = None,
    scale_pixels_per_cm: float | None = None,
    config: dict | None = None,
    min_area: int = 10,
    min_path_length: int = 20,
    sample_step: int = 30,
    max_gap: int = 300,
    max_width: float = 120,
) -> dict:
    """Module 3: quantify crack length and width from a binary mask."""
    if config:
        min_area = int(config.get("min_crack_area", min_area))
        min_path_length = int(config.get("min_path_length", min_path_length))
        sample_step = int(config.get("dist_between_points", sample_step))
        max_gap = int(config.get("max_gap", max_gap))
        max_width = float(config.get("max_valid_width", max_width))

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    mask = mask > 0

    labels = measure.label(mask, connectivity=2)
    cracks = []
    for label_id in range(1, labels.max() + 1):
        component = labels == label_id
        if int(component.sum()) < min_area:
            continue

        crack = _measure_one_crack(
            component=component,
            crack_id=len(cracks) + 1,
            scale_pixels_per_cm=scale_pixels_per_cm,
            sample_step=sample_step,
            max_gap=max_gap,
            max_width=max_width,
        )
        if crack and crack["length_px"] >= min_path_length:
            cracks.append(crack)

    return {
        "image_path": str(image_path) if image_path else None,
        "mask_path": str(mask_path),
        "scale_pixels_per_cm": scale_pixels_per_cm,
        "cracks": cracks,
    }


def _measure_one_crack(
    component: np.ndarray,
    crack_id: int,
    scale_pixels_per_cm: float | None,
    sample_step: int,
    max_gap: int,
    max_width: float,
) -> dict | None:
    skeleton = morphology.skeletonize(component)
    points = np.column_stack(np.nonzero(skeleton))
    if len(points) < 2:
        return None

    graph = _skeleton_graph(skeleton)
    path = _longest_shortest_path(graph)
    if len(path) < 2:
        return None

    length_px = sum(
        float(np.hypot(y2 - y1, x2 - x1))
        for (y1, x1), (y2, x2) in zip(path[:-1], path[1:], strict=False)
    )
    distance = ndi.distance_transform_edt(component)
    widths = _sample_widths(component, distance, path, sample_step, max_gap, max_width)
    if not widths:
        return None

    avg_width_px = float(np.mean(widths))
    max_width_px = float(np.max(widths))

    return {
        "crack_id": crack_id,
        "length_px": float(length_px),
        "avg_width_px": avg_width_px,
        "max_width_px": max_width_px,
        "width_samples_px": [float(w) for w in widths],
        "length_cm": _to_cm(length_px, scale_pixels_per_cm),
        "avg_width_cm": _to_cm(avg_width_px, scale_pixels_per_cm),
        "max_width_cm": _to_cm(max_width_px, scale_pixels_per_cm),
    }


def _skeleton_graph(skeleton: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    rows, cols = skeleton.shape
    ys, xs = np.nonzero(skeleton)
    for y, x in zip(ys, xs, strict=False):
        node = (int(y), int(x))
        graph.add_node(node)
        for dy, dx in NEIGHBORS_8:
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx_ < cols and skeleton[ny, nx_]:
                graph.add_edge(node, (int(ny), int(nx_)), weight=float(np.hypot(dy, dx)))
    return graph


def _longest_shortest_path(graph: nx.Graph) -> list[tuple[int, int]]:
    if graph.number_of_nodes() < 2:
        return []
    endpoints = [node for node, degree in graph.degree() if degree == 1]
    starts = endpoints if len(endpoints) >= 2 else list(graph.nodes)

    best_start = starts[0]
    best_end = starts[0]
    best_length = -1.0
    for start in starts:
        lengths = nx.single_source_dijkstra_path_length(graph, start, weight="weight")
        end = max(lengths, key=lengths.get)
        if lengths[end] > best_length:
            best_start, best_end, best_length = start, end, lengths[end]
    return nx.shortest_path(graph, best_start, best_end, weight="weight")


def _sample_widths(
    component: np.ndarray,
    distance: np.ndarray,
    path: list[tuple[int, int]],
    sample_step: int,
    max_gap: int,
    max_width: float,
) -> list[float]:
    widths = []
    for idx in range(0, len(path), max(1, sample_step)):
        tangent = _local_tangent(path, idx)
        if tangent is None:
            continue
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        width = _cast_width(component, path[idx], normal, max_gap)
        if width is None:
            width = float(distance[path[idx]] * 2.0)
        if 0 < width <= max_width:
            widths.append(float(width))
    return widths


def _local_tangent(path: list[tuple[int, int]], idx: int) -> np.ndarray | None:
    points = np.array(path[max(0, idx - 5): min(len(path), idx + 6)], dtype=float)
    if len(points) < 2:
        return None
    _, _, vh = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    return vh[0]


def _cast_width(component: np.ndarray, point: tuple[int, int], normal: np.ndarray, max_gap: int) -> float | None:
    rows, cols = component.shape
    origin = np.array(point, dtype=float)
    edges = []
    for direction in (-1, 1):
        last_inside = origin.copy()
        for step in range(1, max_gap + 1):
            candidate = origin + direction * step * normal
            y, x = np.rint(candidate).astype(int)
            if y < 0 or y >= rows or x < 0 or x >= cols:
                break
            if not component[y, x]:
                edges.append(last_inside)
                break
            last_inside = candidate
    if len(edges) != 2:
        return None
    return float(np.linalg.norm(edges[0] - edges[1]))


def _to_cm(value_px: float, scale_pixels_per_cm: float | None) -> float | None:
    if not scale_pixels_per_cm:
        return None
    return float(value_px / scale_pixels_per_cm)

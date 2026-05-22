"""Bambu two-layer color card OBJ generator.

Dependencies:
    pip install numpy pillow opencv-python

Run:
    python bambu_color_voxelizer.py

The app loads an image, detects/edit colors, previews the quantized palette,
and exports a welded triangle-only OBJ plus MTL for two-height color printing.
"""

from __future__ import annotations

import queue
import struct
import threading
import traceback
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image, ImageTk, ImageDraw

try:
    import cv2
except ImportError:  # pragma: no cover - handled in main()
    cv2 = None

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk


ALPHA_VISIBLE_THRESHOLD = 10
DEFAULT_PREVIEW_BG_LIGHT = np.array([238, 238, 238], dtype=np.uint8)
DEFAULT_PREVIEW_BG_DARK = np.array([205, 205, 205], dtype=np.uint8)
__version__ = "0.2"

try:
    LANCZOS = Image.Resampling.LANCZOS
    NEAREST = Image.Resampling.NEAREST
except AttributeError:  # Pillow < 9
    LANCZOS = Image.LANCZOS
    NEAREST = Image.NEAREST


@dataclass(frozen=True)
class PaletteSnapshot:
    rgb: tuple[int, int, int]


@dataclass(frozen=True)
class ExportSettings:
    max_x_mm: float
    max_y_mm: float
    corner_radius_mm: float
    base_thickness_mm: float
    grid_resolution: int
    color_thickness_mm: float = 0.2
    bridge_diagonal_contacts: bool = True
    base_rgb: tuple[int, int, int] = (0, 0, 0)
    frame_enabled: bool = False
    frame_width_mm: float = 0.0
    frame_rgb: tuple[int, int, int] = (0, 0, 0)


@dataclass
class PaletteRow:
    frame: ttk.Frame
    swatch: tk.Button
    hex_label: ttk.Label
    rgb: tuple[int, int, int]


def clamp_u8(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0, 255).astype(np.uint8)


def rgb_to_hex(rgb: Iterable[int]) -> str:
    r, g, b = [int(v) for v in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.strip().lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def apply_image_adjustments(
    rgba: np.ndarray,
    denoise_kernel: int,
    brightness: float,
    contrast: float,
) -> np.ndarray:
    """Apply blur, brightness, and contrast while respecting transparency."""
    out = rgba.copy()
    rgb = out[:, :, :3].astype(np.float32)
    alpha = out[:, :, 3].astype(np.float32) / 255.0

    denoise_kernel = int(denoise_kernel)
    if denoise_kernel > 1:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for cv2 blur denoise.")
        if denoise_kernel % 2 == 0:
            denoise_kernel += 1

        premultiplied = rgb * alpha[:, :, None]
        blurred_rgb = cv2.blur(premultiplied, (denoise_kernel, denoise_kernel))
        blurred_alpha = cv2.blur(alpha, (denoise_kernel, denoise_kernel))
        safe_alpha = np.maximum(blurred_alpha, 1.0 / 255.0)
        rgb = blurred_rgb / safe_alpha[:, :, None]
        rgb[alpha <= 0] = 0

    rgb = (rgb - 127.5) * float(contrast) + 127.5
    rgb = rgb * float(brightness)
    out[:, :, :3] = clamp_u8(rgb)
    return out


def nearest_palette_indices(
    rgb: np.ndarray,
    palette: np.ndarray,
    chunk_size: int = 150_000,
) -> np.ndarray:
    """Return the closest palette index for each RGB pixel."""
    if palette.size == 0:
        raise ValueError("Palette is empty.")

    h, w = rgb.shape[:2]
    flat = rgb.reshape(-1, 3).astype(np.float32)
    pal = palette.astype(np.float32)
    out = np.empty(flat.shape[0], dtype=np.int32)

    for start in range(0, flat.shape[0], chunk_size):
        end = min(start + chunk_size, flat.shape[0])
        diff = flat[start:end, None, :] - pal[None, :, :]
        dist = np.einsum("ijk,ijk->ij", diff, diff)
        out[start:end] = np.argmin(dist, axis=1)

    return out.reshape(h, w)


def make_checkerboard(width: int, height: int, tile: int = 12) -> np.ndarray:
    y, x = np.indices((height, width))
    mask = ((x // tile) + (y // tile)) % 2
    bg = np.empty((height, width, 3), dtype=np.uint8)
    bg[mask == 0] = DEFAULT_PREVIEW_BG_LIGHT
    bg[mask == 1] = DEFAULT_PREVIEW_BG_DARK
    return bg


def composite_rgba_for_preview(rgba: np.ndarray) -> np.ndarray:
    h, w = rgba.shape[:2]
    bg = make_checkerboard(w, h)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    return clamp_u8(rgb * alpha + bg.astype(np.float32) * (1.0 - alpha))


def detect_palette_kmeans(
    adjusted_rgba: np.ndarray,
    color_count: int,
    sample_limit: int = 120_000,
) -> list[tuple[int, int, int]]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for KMeans color detection.")

    cv2.setRNGSeed(42)  # Make K-Means deterministic

    visible = adjusted_rgba[:, :, 3] > ALPHA_VISIBLE_THRESHOLD
    pixels = adjusted_rgba[:, :, :3][visible]
    if pixels.size == 0:
        raise ValueError("The image has no visible pixels to analyze.")

    rng = np.random.default_rng(42)
    if pixels.shape[0] > sample_limit:
        indices = rng.choice(pixels.shape[0], size=sample_limit, replace=False)
        pixels = pixels[indices]

    unique = np.unique(pixels.reshape(-1, 3), axis=0)
    k = max(1, min(int(color_count), unique.shape[0], pixels.shape[0]))
    if k == unique.shape[0]:
        centers = unique.astype(np.uint8)
        counts = np.ones(unique.shape[0], dtype=np.int64)
    else:
        data = pixels.astype(np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            50,
            0.35,
        )
        _, labels, centers = cv2.kmeans(
            data,
            k,
            None,
            criteria,
            4,
            cv2.KMEANS_PP_CENTERS,
        )
        centers = clamp_u8(centers)
        counts = np.bincount(labels.flatten(), minlength=k)

    # Stable, useful default: darker colors lower, ties by larger cluster first.
    luminance = (
        centers[:, 0].astype(np.float32) * 0.2126
        + centers[:, 1].astype(np.float32) * 0.7152
        + centers[:, 2].astype(np.float32) * 0.0722
    )
    order = np.lexsort((-counts, luminance))
    return [tuple(int(v) for v in centers[i]) for i in order]


def compute_fit_geometry(
    img_width: int,
    img_height: int,
    settings: ExportSettings,
) -> tuple[int, int, float, float, float, float]:
    width_mm, height_mm = compute_physical_fit_size(
        img_width,
        img_height,
        settings.max_x_mm,
        settings.max_y_mm,
    )

    grid = max(1, min(600, int(settings.grid_resolution)))
    if width_mm >= height_mm:
        out_w = grid
        out_h = max(1, int(round(grid * height_mm / width_mm)))
    else:
        out_h = grid
        out_w = max(1, int(round(grid * width_mm / height_mm)))

    dx = width_mm / out_w
    dy = height_mm / out_h
    return out_w, out_h, width_mm, height_mm, dx, dy


def compute_physical_fit_size(
    img_width: int,
    img_height: int,
    max_x_mm: float,
    max_y_mm: float,
) -> tuple[float, float]:
    if img_width <= 0 or img_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    aspect = img_width / img_height
    width_mm = float(max_x_mm)
    height_mm = width_mm / aspect
    if height_mm > float(max_y_mm):
        height_mm = float(max_y_mm)
        width_mm = height_mm * aspect

    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("Max Size X/Y must be greater than zero.")

    return width_mm, height_mm


def rounded_rectangle_mask(
    width: int,
    height: int,
    width_mm: float,
    height_mm: float,
    dx: float,
    dy: float,
    corner_radius_mm: float,
) -> np.ndarray:
    radius = max(0.0, min(float(corner_radius_mm), width_mm / 2.0, height_mm / 2.0))
    if radius <= 0:
        return np.ones((height, width), dtype=bool)

    xs = (np.arange(width, dtype=np.float64) + 0.5) * dx - width_mm / 2.0
    ys = height_mm / 2.0 - (np.arange(height, dtype=np.float64) + 0.5) * dy
    x_grid, y_grid = np.meshgrid(xs, ys)

    qx = np.abs(x_grid) - (width_mm / 2.0 - radius)
    qy = np.abs(y_grid) - (height_mm / 2.0 - radius)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) - radius
    return outside <= 0.0


def rounded_rectangle_mask_for_rect(
    width: int,
    height: int,
    total_width_mm: float,
    total_height_mm: float,
    dx: float,
    dy: float,
    rect_width_mm: float,
    rect_height_mm: float,
    corner_radius_mm: float,
) -> np.ndarray:
    rect_width_mm = max(0.0, float(rect_width_mm))
    rect_height_mm = max(0.0, float(rect_height_mm))
    if rect_width_mm <= 0.0 or rect_height_mm <= 0.0:
        return np.zeros((height, width), dtype=bool)

    radius = max(
        0.0,
        min(float(corner_radius_mm), rect_width_mm / 2.0, rect_height_mm / 2.0),
    )
    xs = (np.arange(width, dtype=np.float64) + 0.5) * dx - total_width_mm / 2.0
    ys = total_height_mm / 2.0 - (np.arange(height, dtype=np.float64) + 0.5) * dy
    x_grid, y_grid = np.meshgrid(xs, ys)

    qx = np.abs(x_grid) - (rect_width_mm / 2.0 - radius)
    qy = np.abs(y_grid) - (rect_height_mm / 2.0 - radius)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) - radius
    return outside <= 0.0


def rounded_rectangle_frame_mask(
    width: int,
    height: int,
    width_mm: float,
    height_mm: float,
    dx: float,
    dy: float,
    corner_radius_mm: float,
    frame_width_mm: float,
) -> np.ndarray:
    frame_width_mm = max(0.0, float(frame_width_mm))
    outer = rounded_rectangle_mask(width, height, width_mm, height_mm, dx, dy, corner_radius_mm)
    if frame_width_mm <= 0.0:
        return np.zeros_like(outer)

    inner_width = width_mm - 2.0 * frame_width_mm
    inner_height = height_mm - 2.0 * frame_width_mm
    if inner_width <= 0.0 or inner_height <= 0.0:
        return outer

    inner = rounded_rectangle_mask_for_rect(
        width,
        height,
        width_mm,
        height_mm,
        dx,
        dy,
        inner_width,
        inner_height,
        max(0.0, float(corner_radius_mm) - frame_width_mm),
    )
    return outer & ~inner


def bridge_diagonal_contacts(heights: np.ndarray, materials: np.ndarray) -> int:
    h, w = heights.shape
    bridged = 0
    epsilon = 1e-9

    for _pass_index in range(32):
        pass_bridged = 0
        for y in range(h - 1):
            for x in range(w - 1):
                a = float(heights[y, x])
                b = float(heights[y, x + 1])
                c = float(heights[y + 1, x])
                d = float(heights[y + 1, x + 1])

                target_height = min(a, d)
                if (
                    target_height > 0
                    and b + epsilon < target_height
                    and c + epsilon < target_height
                ):
                    if b >= c:
                        ty, tx = y, x + 1
                    else:
                        ty, tx = y + 1, x
                    source_material = materials[y, x] if a <= d else materials[y + 1, x + 1]
                    if heights[ty, tx] + epsilon < target_height:
                        heights[ty, tx] = target_height
                        materials[ty, tx] = source_material
                        pass_bridged += 1

                target_height = min(b, c)
                if (
                    target_height > 0
                    and a + epsilon < target_height
                    and d + epsilon < target_height
                ):
                    if a >= d:
                        ty, tx = y, x
                    else:
                        ty, tx = y + 1, x + 1
                    source_material = materials[y, x + 1] if b <= c else materials[y + 1, x]
                    if heights[ty, tx] + epsilon < target_height:
                        heights[ty, tx] = target_height
                        materials[ty, tx] = source_material
                        pass_bridged += 1

        bridged += pass_bridged
        if pass_bridged == 0:
            break

    return bridged


def build_height_and_material_maps(
    adjusted_rgba: np.ndarray,
    palette: list[PaletteSnapshot],
    settings: ExportSettings,
    progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]], int, float, float, float, float]:
    if not palette:
        raise ValueError("No colors are defined. Use Auto-Detect or add a palette first.")

    if progress:
        progress("Scaling image and calculating voxel size...")

    src_h, src_w = adjusted_rgba.shape[:2]
    out_w, out_h, width_mm, height_mm, dx, dy = compute_fit_geometry(
        src_w,
        src_h,
        settings,
    )

    image = Image.fromarray(adjusted_rgba, mode="RGBA")
    resized = image.resize((out_w, out_h), NEAREST)
    scaled = np.array(resized, dtype=np.uint8)

    if progress:
        progress("Building rounded mask and assigning palette colors...")

    mask = rounded_rectangle_mask(
        out_w,
        out_h,
        width_mm,
        height_mm,
        dx,
        dy,
        settings.corner_radius_mm,
    )

    color_palette_rgb = [entry.rgb for entry in palette]
    color_palette = np.array(color_palette_rgb, dtype=np.uint8)
    nearest = nearest_palette_indices(scaled[:, :, :3], color_palette)

    base_height = float(settings.base_thickness_mm)
    color_height = base_height + float(settings.color_thickness_mm)
    palette_rgb_list = [tuple(int(v) for v in settings.base_rgb), *color_palette_rgb]
    base_material = 0
    visible = scaled[:, :, 3] > ALPHA_VISIBLE_THRESHOLD

    heights = np.zeros((out_h, out_w), dtype=np.float64)
    materials = np.full((out_h, out_w), base_material, dtype=np.int32)

    base_cells = mask
    heights[base_cells] = base_height
    materials[base_cells] = base_material

    color_cells = mask & visible
    heights[color_cells] = color_height
    materials[color_cells] = nearest[color_cells] + 1

    if settings.frame_enabled and settings.frame_width_mm > 0.0:
        frame_rgb = tuple(int(v) for v in settings.frame_rgb)
        try:
            frame_material = palette_rgb_list.index(frame_rgb)
        except ValueError:
            frame_material = len(palette_rgb_list)
            palette_rgb_list.append(frame_rgb)

        frame_cells = rounded_rectangle_frame_mask(
            out_w,
            out_h,
            width_mm,
            height_mm,
            dx,
            dy,
            settings.corner_radius_mm,
            settings.frame_width_mm,
        )
        heights[frame_cells] = color_height
        materials[frame_cells] = frame_material

    if settings.bridge_diagonal_contacts:
        bridged = bridge_diagonal_contacts(heights, materials)
        if progress and bridged:
            progress(f"Bridged {bridged:,} diagonal contacts for manifold export...")

    return heights, materials, palette_rgb_list, base_material, width_mm, height_mm, dx, dy


def write_mtl(mtl_path: Path, palette_rgb: list[tuple[int, int, int]]) -> None:
    with mtl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for idx, rgb in enumerate(palette_rgb, start=1):
            r, g, b = [v / 255.0 for v in rgb]
            handle.write(f"newmtl Material_Color_{idx}\n")
            handle.write("Ka 0.000000 0.000000 0.000000\n")
            handle.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
            handle.write("Ks 0.000000 0.000000 0.000000\n")
            handle.write("d 1.000000\n")
            handle.write("illum 1\n\n")


def validate_closed_triangle_mesh(
    faces_by_material: list[list[tuple[int, int, int]]],
) -> None:
    edge_counts: Counter[tuple[int, int]] = Counter()
    face_counts: Counter[tuple[int, int, int]] = Counter()
    degenerate_faces = 0

    for faces in faces_by_material:
        for face in faces:
            if len(set(face)) != 3:
                degenerate_faces += 1
                continue
            face_counts[tuple(sorted(face))] += 1
            a, b, c = face
            edge_counts[tuple(sorted((a, b)))] += 1
            edge_counts[tuple(sorted((b, c)))] += 1
            edge_counts[tuple(sorted((c, a)))] += 1

    bad_edges = sum(1 for count in edge_counts.values() if count != 2)
    duplicate_faces = sum(1 for count in face_counts.values() if count > 1)
    if degenerate_faces or bad_edges or duplicate_faces:
        raise ValueError(
            "Generated mesh is not manifold "
            f"({bad_edges} bad edges, {duplicate_faces} duplicate triangles, "
            f"{degenerate_faces} degenerate triangles). "
            "Try a slightly lower Grid Resolution or keep diagonal bridging enabled."
        )


def mesh_heightmap_to_obj(
    obj_path: Path,
    mtl_path: Path,
    heights: np.ndarray,
    materials: np.ndarray,
    palette_rgb: list[tuple[int, int, int]],
    base_material: int,
    base_surface_height: float,
    height_mm: float,
    dx: float,
    dy: float,
    progress: Callable[[str], None] | None = None,
) -> None:
    h, w = heights.shape
    vertices: list[tuple[float, float, float]] = []
    vertex_map: dict[tuple[str, int, int, float], int] = {}
    object_faces: dict[tuple[str, int], list[tuple[int, int, int]]] = {}
    faces_by_material: list[list[tuple[int, int, int]]] = [
        [] for _ in range(len(palette_rgb))
    ]

    def vertex_id(scope: str, ix: int, iy: int, z: float) -> int:
        z_key = round(float(z), 6)
        key = (scope, int(ix), int(iy), z_key)
        found = vertex_map.get(key)
        if found is not None:
            return found

        x_coord = round(ix * dx, 6)
        y_coord = round(height_mm - iy * dy, 6)
        z_coord = z_key
        vertices.append((x_coord, y_coord, z_coord))
        new_id = len(vertices)
        vertex_map[key] = new_id
        return new_id

    def add_quad(
        scope: str,
        object_name: str,
        material_index: int,
        corners: tuple[tuple[int, int, float], ...],
    ) -> None:
        v1, v2, v3, v4 = [vertex_id(scope, ix, iy, z) for ix, iy, z in corners]
        triangles = ((v1, v2, v3), (v1, v3, v4))
        material_index = int(material_index)
        faces_by_material[material_index].extend(triangles)
        object_faces.setdefault((object_name, material_index), []).extend(triangles)

    epsilon = 1e-9
    base_surface_height = round(float(base_surface_height), 6)
    base_mask = heights > 0
    color_mask = heights > base_surface_height + epsilon
    active_cells = int(np.count_nonzero(base_mask))
    color_cells = int(np.count_nonzero(color_mask))

    if progress:
        progress(f"Meshing {active_cells:,} base voxels and {color_cells:,} color voxels...")

    def add_slab_from_mask(
        scope: str,
        mask: np.ndarray,
        material_index: int,
        z0: float,
        z1: float,
        isolate_cells: bool = False,
    ) -> None:
        z0 = round(float(z0), 6)
        z1 = round(float(z1), 6)
        if z1 <= z0 + epsilon or not np.any(mask):
            return

        for y, x in np.argwhere(mask):
            y = int(y)
            x = int(x)
            cell_scope = f"{scope}_cell_{y}_{x}" if isolate_cells else scope
            object_name = "Base" if material_index == base_material else f"Color_{material_index}"

            add_quad(cell_scope, object_name, material_index, ((x, y, z0), (x + 1, y, z0), (x + 1, y + 1, z0), (x, y + 1, z0)))
            add_quad(cell_scope, object_name, material_index, ((x, y, z1), (x, y + 1, z1), (x + 1, y + 1, z1), (x + 1, y, z1)))

            if isolate_cells or x == 0 or not mask[y, x - 1]:
                add_quad(cell_scope, object_name, material_index, ((x, y, z1), (x, y, z0), (x, y + 1, z0), (x, y + 1, z1)))

            if isolate_cells or x == w - 1 or not mask[y, x + 1]:
                add_quad(cell_scope, object_name, material_index, ((x + 1, y, z1), (x + 1, y + 1, z1), (x + 1, y + 1, z0), (x + 1, y, z0)))

            if isolate_cells or y == 0 or not mask[y - 1, x]:
                add_quad(cell_scope, object_name, material_index, ((x, y, z1), (x + 1, y, z1), (x + 1, y, z0), (x, y, z0)))

            if isolate_cells or y == h - 1 or not mask[y + 1, x]:
                add_quad(cell_scope, object_name, material_index, ((x, y + 1, z1), (x, y + 1, z0), (x + 1, y + 1, z0), (x + 1, y + 1, z1)))

    def iter_connected_components(mask: np.ndarray) -> Iterable[tuple[int, np.ndarray]]:
        visited = np.zeros(mask.shape, dtype=bool)
        component_index = 0

        for start_y, start_x in np.argwhere(mask):
            start_y = int(start_y)
            start_x = int(start_x)
            if visited[start_y, start_x]:
                continue

            component = np.zeros(mask.shape, dtype=bool)
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True

            while stack:
                y, x = stack.pop()
                component[y, x] = True

                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            component_index += 1
            yield component_index, component

    add_slab_from_mask("base", base_mask, base_material, 0.0, base_surface_height)

    if progress and color_cells:
        progress("Meshing separate color layer solids...")

    for material_index in sorted(int(v) for v in np.unique(materials[color_mask])):
        material_mask = color_mask & (materials == material_index)
        top_levels = sorted(round(float(v), 6) for v in np.unique(heights[material_mask]))
        for top_level in top_levels:
            level_mask = material_mask & np.isclose(heights, top_level, atol=epsilon)
            for component_index, component_mask in iter_connected_components(level_mask):
                scope = f"mat_{material_index}_z_{top_level:.6f}_c_{component_index}"
                add_slab_from_mask(
                    scope,
                    component_mask,
                    material_index,
                    base_surface_height,
                    top_level,
                    isolate_cells=True,
                )

    if progress:
        progress("Validating manifold edges...")
    validate_closed_triangle_mesh(faces_by_material)

    if progress:
        progress("Writing OBJ and MTL files...")

    write_mtl(mtl_path, palette_rgb)

    with obj_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"mtllib {mtl_path.name}\n")
        handle.write("o Bambu_Two_Layer_Color_Card\n")
        for x_coord, y_coord, z_coord in vertices:
            handle.write(f"v {x_coord:.6f} {y_coord:.6f} {z_coord:.6f}\n")

        for (object_name, material_index), faces in object_faces.items():
            if not faces:
                continue
            obj_material_index = material_index + 1
            handle.write(f"\no {object_name}\n")
            handle.write(f"g {object_name}\n")
            handle.write(f"usemtl Material_Color_{obj_material_index}\n")
            for a, b, c in faces:
                handle.write(f"f {a} {b} {c}\n")


def export_obj_mtl(
    obj_path: Path,
    adjusted_rgba: np.ndarray,
    palette: list[PaletteSnapshot],
    settings: ExportSettings,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, Path]:
    if obj_path.suffix.lower() != ".obj":
        obj_path = obj_path.with_suffix(".obj")
    mtl_path = obj_path.with_suffix(".mtl")

    (
        heights, materials, palette_rgb, base_material, _width_mm, height_mm, dx, dy
    ) = build_height_and_material_maps(adjusted_rgba, palette, settings, progress)

    mesh_heightmap_to_obj(
        obj_path, mtl_path, heights, materials, palette_rgb, base_material,
        settings.base_thickness_mm, height_mm, dx, dy, progress
    )
    return obj_path, mtl_path


def mask_to_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    rectangles: list[tuple[int, int, int, int]] = []

    for y in range(h):
        for x in range(w):
            if visited[y, x] or not mask[y, x]:
                continue

            x1 = x
            while x1 < w and mask[y, x1] and not visited[y, x1]:
                x1 += 1

            y1 = y + 1
            while y1 < h and np.all(mask[y1, x:x1] & ~visited[y1, x:x1]):
                y1 += 1

            visited[y:y1, x:x1] = True
            rectangles.append((x, y, x1, y1))

    return rectangles


def rectangles_to_mesh(
    rectangles: list[tuple[int, int, int, int]],
    z0: float,
    z1: float,
    height_mm: float,
    dx: float,
    dy: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []

    def coord(ix: int, iy: int, z: float) -> tuple[float, float, float]:
        return (round(ix * dx, 6), round(height_mm - iy * dy, 6), round(float(z), 6))

    def add_quad(a: int, b: int, c: int, d: int) -> None:
        triangles.append((a, b, c))
        triangles.append((a, c, d))

    for x0, y0, x1, y1 in rectangles:
        start = len(vertices)
        vertices.extend([
            coord(x0, y0, z0), coord(x1, y0, z0), coord(x1, y1, z0), coord(x0, y1, z0),
            coord(x0, y0, z1), coord(x0, y1, z1), coord(x1, y1, z1), coord(x1, y0, z1)
        ])
        b0, b1, b2, b3, t0, t1, t2, t3 = range(start, start + 8)
        add_quad(b0, b1, b2, b3)  # bottom
        add_quad(t0, t1, t2, t3)  # top
        add_quad(t0, b0, b3, t1)  # left
        add_quad(t3, t2, b2, b1)  # right
        add_quad(t0, t3, b1, b0)  # front
        add_quad(t1, b3, b2, t2)  # back

    return vertices, triangles


def write_binary_stl(
    path: Path,
    name: str,
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
) -> None:
    header = f"Bambu Color Voxelizer {name}".encode("ascii", errors="replace")[:80]
    header = header.ljust(80, b" ")

    with path.open("wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<I", len(triangles)))

        for a, b, c in triangles:
            p1 = np.array(vertices[a], dtype=np.float64)
            p2 = np.array(vertices[b], dtype=np.float64)
            p3 = np.array(vertices[c], dtype=np.float64)
            normal = np.cross(p2 - p1, p3 - p1)
            norm = float(np.linalg.norm(normal))
            if norm > 0.0:
                normal /= norm
            else:
                normal[:] = 0.0

            handle.write(
                struct.pack(
                    "<12fH",
                    float(normal[0]), float(normal[1]), float(normal[2]),
                    float(p1[0]), float(p1[1]), float(p1[2]),
                    float(p2[0]), float(p2[1]), float(p2[2]),
                    float(p3[0]), float(p3[1]), float(p3[2]), 0,
                )
            )


def export_stl_parts(
    path: Path,
    adjusted_rgba: np.ndarray,
    palette: list[PaletteSnapshot],
    settings: ExportSettings,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, ...]:
    if path.suffix.lower() != ".stl":
        path = path.with_suffix(".stl")

    (
        heights, materials, _palette_rgb, _base_material, _width_mm, height_mm, dx, dy
    ) = build_height_and_material_maps(adjusted_rgba, palette, settings, progress)

    base_height = round(float(settings.base_thickness_mm), 6)
    color_top = round(base_height + float(settings.color_thickness_mm), 6)
    base_mask = heights > 0
    color_mask = heights > base_height + 1e-9

    outputs: list[Path] = []
    stem = path.with_suffix("")

    if progress:
        progress("Writing separate STL parts...")

    base_rectangles = mask_to_rectangles(base_mask)
    base_vertices, base_triangles = rectangles_to_mesh(base_rectangles, 0.0, base_height, height_mm, dx, dy)
    base_path = stem.with_name(f"{stem.name}_base.stl")
    write_binary_stl(base_path, "base", base_vertices, base_triangles)
    outputs.append(base_path)

    for material_index in sorted(int(v) for v in np.unique(materials[color_mask])):
        material_mask = color_mask & (materials == material_index)
        rectangles = mask_to_rectangles(material_mask)
        vertices, triangles = rectangles_to_mesh(rectangles, base_height, color_top, height_mm, dx, dy)
        color_path = stem.with_name(f"{stem.name}_color_{material_index}.stl")
        write_binary_stl(color_path, f"color_{material_index}", vertices, triangles)
        outputs.append(color_path)

    return tuple(outputs)


def write_3mf(
    path: Path,
    parts: list[tuple[str, int, list[tuple[float, float, float]], list[tuple[int, int, int]]]],
    palette_rgb: list[tuple[int, int, int]],
) -> None:
    object_entries = []
    component_entries = []
    part_settings = []
    next_object_id = 2

    for name, material_index, vertices, triangles in parts:
        if not vertices or not triangles:
            continue

        object_id = next_object_id
        next_object_id += 1
        safe_name = escape(name)

        vertex_xml = "\n".join(f'<vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>' for x, y, z in vertices)
        triangle_xml = "\n".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in triangles)
        
        object_entries.append(
            f'<object id="{object_id}" type="model" name="{safe_name}">'
            f"<mesh><vertices>{vertex_xml}</vertices><triangles>{triangle_xml}</triangles></mesh>"
            "</object>"
        )
        component_entries.append(f'<component objectid="{object_id}"/>')
        part_settings.append(
            f'<part id="{object_id}" subtype="normal_part">'
            f'<metadata key="name" value="{safe_name}"/>'
            f'<metadata key="extruder" value="{material_index + 1}"/>'
            "</part>"
        )

    assembly_object_id = 1
    object_entries.append(
        f'<object id="{assembly_object_id}" type="model" name="Bambu_Two_Layer_Color_Card">'
        f'<components>{"".join(component_entries)}</components>'
        "</object>"
    )

    model_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">\n'
        '<metadata name="Application">BambuStudio-02.06.00.00</metadata>\n'
        '<metadata name="BambuStudio:3mfVersion">1</metadata>\n'
        "<resources>\n"
        f'{"".join(object_entries)}\n'
        "</resources>\n"
        f'<build><item objectid="{assembly_object_id}"/></build>\n'
        "</model>\n"
    )

    model_settings_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<config>\n"
        "  <plate>\n"
        '    <metadata key="plater_id" value="1"/>\n'
        '    <metadata key="plater_name" value=""/>\n'
        '    <metadata key="thumbnail_file" value="Metadata/plate_1.png"/>\n'
        '    <metadata key="thumbnail_no_light_file" value="Metadata/plate_no_light_1.png"/>\n'
        '    <metadata key="top_file" value="Metadata/top_1.png"/>\n'
        '    <metadata key="pick_file" value="Metadata/pick_1.png"/>\n'
        f'    <object_on_plate>\n'
        f'      <metadata key="object_id" value="{assembly_object_id}"/>\n'
        f'    </object_on_plate>\n'
        "  </plate>\n"
        f'  <object id="{assembly_object_id}">\n'
        '    <metadata key="name" value="Bambu_Two_Layer_Color_Card"/>\n'
        '    <metadata key="extruder" value="1"/>\n'
        f'    {"".join(part_settings)}\n'
        "  </object>\n"
        "</config>\n"
    )

    model_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/Metadata/model_settings.config" Id="rel-1" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/model-settings"/>'
        "</Relationships>\n"
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        '<Default Extension="config" ContentType="application/octet-stream"/>'
        "</Types>\n"
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        "</Relationships>\n"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("3D/3dmodel.model", model_xml)
        archive.writestr("3D/_rels/3dmodel.model.rels", model_rels)
        archive.writestr("Metadata/model_settings.config", model_settings_xml)


def export_3mf(
    path: Path,
    adjusted_rgba: np.ndarray,
    palette: list[PaletteSnapshot],
    settings: ExportSettings,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path]:
    if path.suffix.lower() != ".3mf":
        path = path.with_suffix(".3mf")

    (
        heights, materials, palette_rgb, base_material, _width_mm, height_mm, dx, dy
    ) = build_height_and_material_maps(adjusted_rgba, palette, settings, progress)

    base_height = round(float(settings.base_thickness_mm), 6)
    color_top = round(base_height + float(settings.color_thickness_mm), 6)
    base_mask = heights > 0
    color_mask = heights > base_height + 1e-9

    if progress:
        progress("Building compact 3MF rectangles...")

    parts: list[tuple[str, int, list[tuple[float, float, float]], list[tuple[int, int, int]]]] = []

    base_rectangles = mask_to_rectangles(base_mask)
    base_vertices, base_triangles = rectangles_to_mesh(base_rectangles, 0.0, base_height, height_mm, dx, dy)
    parts.append(("Base", base_material, base_vertices, base_triangles))

    for material_index in sorted(int(v) for v in np.unique(materials[color_mask])):
        material_mask = color_mask & (materials == material_index)
        rectangles = mask_to_rectangles(material_mask)
        vertices, triangles = rectangles_to_mesh(rectangles, base_height, color_top, height_mm, dx, dy)
        parts.append((f"Color_{material_index}", material_index, vertices, triangles))

    if progress:
        progress("Writing grouped Bambu 3MF file...")
    write_3mf(path, parts, palette_rgb)
    return (path,)


def export_model(
    path: Path,
    adjusted_rgba: np.ndarray,
    palette: list[PaletteSnapshot],
    settings: ExportSettings,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, ...]:
    if path.suffix.lower() == ".stl":
        return export_stl_parts(path, adjusted_rgba, palette, settings, progress)
    if path.suffix.lower() == ".obj":
        return export_obj_mtl(path, adjusted_rgba, palette, settings, progress)
    return export_3mf(path, adjusted_rgba, palette, settings, progress)


def build_3d_preview(
    heights: np.ndarray,
    materials: np.ndarray,
    palette_rgb: list[tuple[int, int, int]],
    canvas_w: int,
    canvas_h: int,
) -> Image.Image:
    """Render a fast software axonometric 3D projection of the card heightmap."""
    h_orig, w_orig = heights.shape
    max_dim_3d = 300
    
    # Render at exactly 1:1 voxel detail up to 300x300, or gracefully subsample above that.
    if max(h_orig, w_orig) <= max_dim_3d:
        h_3d, w_3d = h_orig, w_orig
    else:
        if w_orig > h_orig:
            w_3d = max_dim_3d
            h_3d = max(1, int(round(max_dim_3d * h_orig / w_orig)))
        else:
            h_3d = max_dim_3d
            w_3d = max(1, int(round(max_dim_3d * w_orig / h_orig)))

    y_indices = np.linspace(0, h_orig - 1, h_3d, dtype=int)
    x_indices = np.linspace(0, w_orig - 1, w_3d, dtype=int)

    h_small = heights[y_indices][:, x_indices]
    m_small = materials[y_indices][:, x_indices]

    img_3d = Image.new("RGB", (canvas_w, canvas_h), (44, 44, 44))
    draw = ImageDraw.Draw(img_3d)

    w_half = w_3d / 2.0
    h_half = h_3d / 2.0

    max_z = float(np.max(h_small)) if h_small.size > 0 else 1.0
    if max_z <= 0:
        max_z = 1.0

    # Ensure the Z height remains visually chunky and perfectly proportioned
    # regardless of whether the grid is 50x50 or 300x300.
    z_factor = 5.0 * (max(w_3d, h_3d) / 80.0)

    def raw_project(cx: float, cy: float, cz: float) -> tuple[float, float]:
        iso_x = (cx - w_half) * 1.0 - (cy - h_half) * 1.0
        iso_y = (cx - w_half) * 0.5 + (cy - h_half) * 0.5
        return iso_x, iso_y - cz * z_factor

    corners = [
        (0, 0, 0), (w_3d, 0, 0), (0, h_3d, 0), (w_3d, h_3d, 0),
        (0, 0, max_z), (w_3d, 0, max_z), (0, h_3d, max_z), (w_3d, h_3d, max_z)
    ]
    proj_corners = [raw_project(cx, cy, cz) for cx, cy, cz in corners]
    xs = [p[0] for p in proj_corners]
    ys = [p[1] for p in proj_corners]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max_x - min_x if max_x > min_x else 1.0
    span_y = max_y - min_y if max_y > min_y else 1.0

    scale_x = (canvas_w * 0.75) / span_x
    scale_y = (canvas_h * 0.75) / span_y
    scale = min(scale_x, scale_y)
    z_scale = scale * z_factor

    center_x = canvas_w / 2.0
    center_y = canvas_h / 2.0

    offset_x = center_x - ((min_x + max_x) / 2.0) * scale
    offset_y = center_y - ((min_y + max_y) / 2.0) * scale

    def project(cx: float, cy: float, cz: float) -> tuple[float, float]:
        iso_x = (cx - w_half) * 1.0 - (cy - h_half) * 1.0
        iso_y = (cx - w_half) * 0.5 + (cy - h_half) * 0.5
        u = offset_x + iso_x * scale
        v = offset_y + iso_y * scale - cz * z_scale
        return u, v

    def shade(rgb_color: tuple[int, int, int], shade_factor: float) -> tuple[int, int, int]:
        return tuple(int(v) for v in clamp_u8(np.array(rgb_color) * shade_factor))

    for y in range(h_3d):
        for x in range(w_3d):
            z_curr = h_small[y, x]
            if z_curr <= 0:
                continue

            mat_idx = m_small[y, x]
            if mat_idx < 0 or mat_idx >= len(palette_rgb):
                base_rgb = palette_rgb[0] if palette_rgb else (128, 128, 128)
            else:
                base_rgb = palette_rgb[mat_idx]

            v0 = project(x, y, z_curr)
            v1 = project(x + 1, y, z_curr)
            v2 = project(x + 1, y + 1, z_curr)
            v3 = project(x, y + 1, z_curr)

            z_front = h_small[y + 1, x] if y + 1 < h_3d else 0.0
            if z_curr > z_front:
                vf0 = v3
                vf1 = v2
                vf2 = project(x + 1, y + 1, z_front)
                vf3 = project(x, y + 1, z_front)
                draw.polygon([vf0, vf1, vf2, vf3], fill=shade(base_rgb, 0.75))

            z_right = h_small[y, x + 1] if x + 1 < w_3d else 0.0
            if z_curr > z_right:
                vr1 = v1
                vr2 = v2
                vr3 = project(x + 1, y + 1, z_right)
                vr4 = project(x + 1, y, z_right)
                draw.polygon([vr1, vr2, vr3, vr4], fill=shade(base_rgb, 0.60))

            draw.polygon([v0, v1, v2, v3], fill=base_rgb, outline=shade(base_rgb, 0.95))

    return img_3d



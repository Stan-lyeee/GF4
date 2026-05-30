"""Utility functions for GF4 Week 3 sparse reconstruction.

Plotting, CSV, and PLY helpers are mostly provided.
The core calibrated geometry, reprojection checks,
and third-view registration steps are marked with TODO and should be completed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Iterable

import cv2
import numpy as np


@dataclass
class TwoViewResult:
    """Container for Week 3 two-view reconstruction metrics."""

    image_i: str
    image_j: str
    filtered_matches: int
    essential_inliers: int
    pose_inliers: int
    triangulated_points: int
    positive_depth_points: int
    kept_points: int
    median_reprojection_error_px: float | None
    mean_reprojection_error_px: float | None
    focal_length_px: float

    def as_dict(self) -> dict:
        return {
            "image_i": self.image_i,
            "image_j": self.image_j,
            "filtered_matches": self.filtered_matches,
            "essential_inliers": self.essential_inliers,
            "pose_inliers": self.pose_inliers,
            "triangulated_points": self.triangulated_points,
            "positive_depth_points": self.positive_depth_points,
            "kept_points": self.kept_points,
            "median_reprojection_error_px": self.median_reprojection_error_px,
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
            "focal_length_px": self.focal_length_px,
        }


@dataclass
class ThirdViewResult:
    """Container for Week 3 third-view registration metrics."""

    image_k: str
    anchor_matches: int
    pnp_correspondences: int
    pnp_inliers: int
    median_reprojection_error_px: float | None
    mean_reprojection_error_px: float | None
    focal_length_px: float

    def as_dict(self) -> dict:
        return {
            "image_k": self.image_k,
            "anchor_matches": self.anchor_matches,
            "pnp_correspondences": self.pnp_correspondences,
            "pnp_inliers": self.pnp_inliers,
            "median_reprojection_error_px": self.median_reprojection_error_px,
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
            "focal_length_px": self.focal_length_px,
        }


def ensure_dir(path: Path) -> Path:
    """Create an output directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(path: Path, rows: Iterable[dict]) -> None:
    """Save a list of dictionaries as CSV."""
    rows = list(rows)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_camera_matrix(
    image_shape: tuple[int, ...],
    focal_length_px: float | None = None,
    principal_point: tuple[float, float] | None = None,
) -> np.ndarray:
    """Build a simple camera intrinsic matrix for a resized image."""
    height, width = image_shape[:2]
    if focal_length_px is None:
        focal_length_px = 1.2 * max(width, height)
    if principal_point is None:
        principal_point = ((width - 1) / 2.0, (height - 1) / 2.0)

    cx, cy = principal_point
    return np.array(
        [
            [focal_length_px, 0.0, cx],
            [0.0, focal_length_px, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_essential_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    threshold: float = 1.0,
    confidence: float = 0.999,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the essential matrix with OpenCV RANSAC.

    TODO: Complete this function.

    """
    raise NotImplementedError("TODO: estimate essential matrix")


def recover_relative_pose(
    E: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    inlier_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover relative camera pose from an essential matrix.

    TODO: Complete this function.

    """
    raise NotImplementedError("TODO: recover relative pose")


def make_projection_matrices(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create projection matrices P1 = K[I|0] and P2 = K[R|t].

    TODO: Complete this function.
    """
    raise NotImplementedError("TODO: create projection matrices")


def triangulate_points(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Triangulate 3D points from corresponding image points.

    TODO: Complete this function.

    """
    raise NotImplementedError("TODO: triangulate 3D points")


def project_points(
    points3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Project 3D points into an image using camera matrix K[R|t].

    TODO: Complete this function.
    """
    raise NotImplementedError("TODO: project 3D points")


def compute_reprojection_errors(
    points3d: np.ndarray,
    observed_pts: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean reprojection error in pixels.

    TODO: Complete this function by projecting points3d and comparing with
    observed_pts.
    """
    raise NotImplementedError("TODO: compute reprojection errors")


def compute_depths(
    points3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute point depths in camera 1 and camera 2 coordinates.

    TODO: Complete this function.

    Camera 1 has extrinsics [I|0]. Camera 2 has extrinsics [R|t].
    """
    raise NotImplementedError("TODO: compute point depths")


def filter_reconstructed_points(
    points3d: np.ndarray,
    errors1: np.ndarray,
    errors2: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    max_reprojection_error: float = 4.0,
) -> np.ndarray:
    """Return a boolean mask for valid triangulated points.

    TODO: Complete this function.

    Keep points that:
    - have finite 3D coordinates,
    - have positive depth in both cameras,
    - have reprojection error at most max_reprojection_error in both images.
    """
    raise NotImplementedError("TODO: filter reconstructed points")


def build_2d3d_correspondences(
    reconstructed_anchor_indices: np.ndarray,
    reconstructed_points: np.ndarray,
    anchor_to_new_matches: list[cv2.DMatch],
    new_keypoints: list[cv2.KeyPoint],
) -> tuple[np.ndarray, np.ndarray]:
    """Build 2D-3D correspondences for registering a third image.

    The two-view reconstruction gives one 3D point for each kept feature in an
    anchor image. This function matches that anchor image to a new image and
    keeps only those matches whose anchor feature already has a reconstructed
    3D point.

    TODO: Complete this function.

    Inputs:
    - reconstructed_anchor_indices[i] is the anchor-image keypoint index for
      reconstructed_points[i].
    - anchor_to_new_matches are OpenCV matches from the anchor image to the new
      image, so match.queryIdx is an anchor-image keypoint index and
      match.trainIdx is a new-image keypoint index.

    Return:
    - points3d: Nx3 reconstructed 3D points
    - pts_new: Nx2 feature coordinates in the new image
    """
    raise NotImplementedError("TODO: build 2D-3D correspondences")


def estimate_camera_pose_pnp(
    points3d: np.ndarray,
    pts2d: np.ndarray,
    K: np.ndarray,
    threshold: float = 6.0,
    confidence: float = 0.999,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a new camera pose from 2D-3D correspondences using PnP.

    TODO: Complete this function with cv2.solvePnPRansac and cv2.Rodrigues.

    Return:
    - R: 3x3 world-to-camera rotation for the new image
    - t: 3x1 world-to-camera translation for the new image
    - inlier_mask: boolean array of shape (N,)
    """
    raise NotImplementedError("TODO: estimate camera pose with PnP")


def sample_point_colours(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.uint8)

    height, width = image.shape[:2]
    xy = np.rint(pts).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
    bgr = image[xy[:, 1], xy[:, 0]]
    return bgr[:, ::-1].astype(np.uint8)


def draw_reprojection_overlay(
    image1: np.ndarray,
    image2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    points3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    output_path: Path,
    max_draw: int = 120,
) -> None:
    """Save a visual check comparing observed and reprojected image points.

    TODO: Complete this function.

    Required behaviour:
    - project points3d into image 1 using camera pose [I|0],
    - project points3d into image 2 using camera pose [R|t],
    - show the two input images side-by-side,
    - draw observed points as one marker style,
    - draw reprojected points as a different marker style,
    - draw a short line from each observed point to its reprojection,
    - save the figure to output_path.

    This is one of the main ways to check whether your reconstruction is
    geometrically meaningful.
    """
    raise NotImplementedError("TODO: draw two-view reprojection overlay")


def draw_single_image_reprojection_overlay(
    image: np.ndarray,
    observed_pts: np.ndarray,
    points3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    output_path: Path,
    max_draw: int = 120,
) -> None:
    """Save a reprojection check for one registered camera.

    TODO: Complete this function.

    Required behaviour:
    - project points3d into the image using camera pose [R|t],
    - draw observed_pts and projected points on top of the image,
    - draw a short line from each observed point to its reprojection,
    - save the figure to output_path.

    This is the corresponding correctness check for the image registered by
    PnP.
    """
    raise NotImplementedError("TODO: draw single-image reprojection overlay")


def _camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (-R.T @ t.reshape(3, 1)).ravel()


def _camera_frustum(
    R_wc: np.ndarray,
    center: np.ndarray,
    scale: float,
) -> np.ndarray:
    corners = np.array(
        [
            [-0.5, -0.35, 1.0],
            [0.5, -0.35, 1.0],
            [0.5, 0.35, 1.0],
            [-0.5, 0.35, 1.0],
        ],
        dtype=np.float64,
    )
    return center[None, :] + scale * (R_wc @ corners.T).T


def _set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    radius = 0.5 * max(x_range, y_range, z_range, 1e-9)

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim3d([x_mid - radius, x_mid + radius])
    ax.set_ylim3d([y_mid - radius, y_mid + radius])
    ax.set_zlim3d([z_mid - radius, z_mid + radius])


def _robust_bounds(points: np.ndarray, extra_points: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    point_cloud = np.empty((0, 3), dtype=np.float64)
    if len(points):
        point_cloud = points[np.isfinite(points).all(axis=1)]

    extra_cloud = np.empty((0, 3), dtype=np.float64)
    if extra_points is not None and len(extra_points):
        extra_cloud = extra_points[np.isfinite(extra_points).all(axis=1)]

    if len(point_cloud) == 0 and len(extra_cloud) == 0:
        return np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])

    if len(point_cloud) > 10:
        lower = np.percentile(point_cloud, 2, axis=0)
        upper = np.percentile(point_cloud, 98, axis=0)
    elif len(point_cloud):
        lower = np.min(point_cloud, axis=0)
        upper = np.max(point_cloud, axis=0)
    else:
        lower = np.min(extra_cloud, axis=0)
        upper = np.max(extra_cloud, axis=0)

    if len(extra_cloud):
        lower = np.minimum(lower, np.min(extra_cloud, axis=0))
        upper = np.maximum(upper, np.max(extra_cloud, axis=0))

    span = np.maximum(upper - lower, 1e-6)
    padding = 0.18 * span
    return lower - padding, upper + padding


def _apply_equal_limits(ax, lower: np.ndarray, upper: np.ndarray) -> None:
    center = 0.5 * (lower + upper)
    radius = 0.5 * max(np.max(upper - lower), 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_two_view_reconstruction(
    points3d: np.ndarray,
    colours: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    output_path: Path,
    max_points: int = 5000,
) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)

    points = points3d
    point_colours = colours
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
        point_colours = point_colours[indices]

    center1 = np.zeros(3)
    center2 = _camera_center(R, t)
    baseline = np.linalg.norm(center2 - center1)
    frustum_scale = max(0.65, 0.55 * baseline)

    camera_points = np.vstack([center1, center2])
    frustum1 = _camera_frustum(np.eye(3), center1, frustum_scale)
    frustum2 = _camera_frustum(R.T, center2, frustum_scale)
    plot_extra = np.vstack([camera_points, frustum1, frustum2])
    lower, upper = _robust_bounds(points, extra_points=plot_extra)
    point_size = 24 if len(points) < 200 else 9 if len(points) < 1000 else 3
    rgb = point_colours / 255.0 if len(point_colours) else "#456990"

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.28, 1.0], height_ratios=[1.0, 0.82])
    ax = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xz = fig.add_subplot(grid[0, 1])
    ax_cam = fig.add_subplot(grid[1, 1])

    def draw_camera_3d(axis, center, frustum, colour, label):
        axis.scatter(
            [center[0]],
            [center[1]],
            [center[2]],
            c=colour,
            s=170,
            edgecolors="white",
            linewidths=1.2,
            depthshade=False,
            label=label,
        )
        closed = np.vstack([frustum, frustum[0]])
        axis.plot(closed[:, 0], closed[:, 1], closed[:, 2], c=colour, linewidth=2.0)
        for corner in frustum:
            axis.plot(
                [center[0], corner[0]],
                [center[1], corner[1]],
                [center[2], corner[2]],
                c=colour,
                linewidth=1.2,
            )

    def draw_camera_xz(
        axis,
        center,
        frustum,
        colour,
        label,
        marker_size=150,
        label_offset=(0.0, 0.0),
        label_va="center",
    ):
        axis.scatter(
            [center[0]],
            [center[2]],
            c=colour,
            s=marker_size,
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
            label=label,
        )
        closed = np.vstack([frustum, frustum[0]])
        axis.plot(closed[:, 0], closed[:, 2], c=colour, linewidth=2.0, zorder=4)
        for corner in frustum:
            axis.plot(
                [center[0], corner[0]],
                [center[2], corner[2]],
                c=colour,
                linewidth=1.2,
                zorder=4,
            )
        axis.text(
            center[0] + label_offset[0],
            center[2] + label_offset[1],
            f"  {label}",
            color=colour,
            weight="bold",
            va=label_va,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            zorder=6,
        )

    if len(points):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=point_size,
            c=rgb,
            alpha=0.9,
            depthshade=False,
        )
    else:
        ax.text(0, 0, 0, "No points after filtering", ha="center")

    draw_camera_3d(ax, center1, frustum1, "#2458a6", "Camera 1")
    draw_camera_3d(ax, center2, frustum2, "#a33b3b", "Camera 2")
    ax.plot([center1[0], center2[0]], [center1[1], center2[1]], [center1[2], center2[2]], c="#5b6675", linewidth=2.0)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z / depth")
    ax.set_title(f"Scene view ({len(points)} points)")
    ax.legend(loc="upper right")
    _apply_equal_limits(ax, lower, upper)
    ax.view_init(elev=18, azim=-68)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass

    if len(points):
        ax_xz.scatter(points[:, 0], points[:, 2], s=point_size, c=rgb, alpha=0.9)
    ax_xz.plot([center1[0], center2[0]], [center1[2], center2[2]], c="#5b6675", linewidth=1.8)
    draw_camera_xz(ax_xz, center1, frustum1, "#2458a6", "C1")
    draw_camera_xz(ax_xz, center2, frustum2, "#a33b3b", "C2")

    x_padding = 0.05 * max(upper[0] - lower[0], 1e-6)
    z_padding = 0.05 * max(upper[2] - lower[2], 1e-6)
    ax_xz.set_xlim(lower[0] - x_padding, upper[0] + x_padding)
    ax_xz.set_ylim(lower[2] - z_padding, upper[2] + z_padding)
    ax_xz.grid(True, alpha=0.25)
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z / depth")
    ax_xz.set_title("Full X-Z depth view")
    ax_xz.legend(loc="upper right")
    ax_xz.text(
        0.02,
        0.02,
        "Translation scale is arbitrary",
        transform=ax_xz.transAxes,
        fontsize=9,
        color="#5b6675",
    )

    detail_points = np.vstack([camera_points, frustum1, frustum2])
    detail_x = detail_points[:, 0]
    detail_z = detail_points[:, 2]
    detail_center = np.array([np.mean(detail_x), np.mean(detail_z)])
    detail_radius = 0.85 * max(np.ptp(detail_x), np.ptp(detail_z), baseline, 1.0)
    ax_cam.plot([center1[0], center2[0]], [center1[2], center2[2]], c="#5b6675", linewidth=2.2)
    label_offset = (0.02 * detail_radius, 0.05 * detail_radius)
    draw_camera_xz(
        ax_cam,
        center1,
        frustum1,
        "#2458a6",
        "Camera 1",
        marker_size=190,
        label_offset=label_offset,
        label_va="bottom",
    )
    draw_camera_xz(
        ax_cam,
        center2,
        frustum2,
        "#a33b3b",
        "Camera 2",
        marker_size=190,
        label_offset=label_offset,
        label_va="bottom",
    )
    ax_cam.set_xlim(detail_center[0] - detail_radius, detail_center[0] + detail_radius)
    ax_cam.set_ylim(detail_center[1] - detail_radius, detail_center[1] + detail_radius)
    ax_cam.set_aspect("equal", adjustable="box")
    ax_cam.grid(True, alpha=0.25)
    ax_cam.set_xlabel("X")
    ax_cam.set_ylabel("Z / depth")
    ax_cam.set_title("Camera baseline zoom")
    ax_cam.text(
        0.02,
        0.98,
        "Frustums show viewing direction; scale is arbitrary",
        transform=ax_cam.transAxes,
        fontsize=9,
        color="#5b6675",
        va="top",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_multi_view_reconstruction(
    points3d: np.ndarray,
    colours: np.ndarray,
    camera_poses: list[tuple[str, np.ndarray, np.ndarray]],
    output_path: Path,
    max_points: int = 5000,
) -> None:
    """Save a sparse reconstruction plot consistent with the two-view figure."""
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)

    points = points3d
    point_colours = colours
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
        point_colours = point_colours[indices]

    if camera_poses:
        camera_centers = np.array([_camera_center(R, t) for _, R, t in camera_poses])
    else:
        camera_centers = np.empty((0, 3), dtype=np.float64)

    if len(camera_centers) >= 2:
        baseline = np.median(
            [
                np.linalg.norm(camera_centers[i] - camera_centers[j])
                for i in range(len(camera_centers))
                for j in range(i + 1, len(camera_centers))
            ]
        )
    else:
        baseline = 1.0
    frustum_scale = max(0.65, 0.55 * baseline)

    frustums = [
        _camera_frustum(R.T, center, frustum_scale)
        for center, (_, R, _) in zip(camera_centers, camera_poses)
    ]
    plot_extra = camera_centers
    if frustums:
        plot_extra = np.vstack([camera_centers, *frustums])

    lower, upper = _robust_bounds(points, extra_points=plot_extra)
    point_size = 24 if len(points) < 200 else 9 if len(points) < 1000 else 3
    rgb = point_colours / 255.0 if len(point_colours) else "#456990"

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.28, 1.0], height_ratios=[1.0, 0.82])
    ax = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xz = fig.add_subplot(grid[0, 1])
    ax_cam = fig.add_subplot(grid[1, 1])

    colours_by_camera = ["#2458a6", "#a33b3b", "#2f7d32", "#7a4ea3", "#b46b00"]

    def draw_camera_3d(axis, center, frustum, colour, label):
        axis.scatter(
            [center[0]],
            [center[1]],
            [center[2]],
            c=colour,
            s=170,
            edgecolors="white",
            linewidths=1.2,
            depthshade=False,
            label=label,
        )
        closed = np.vstack([frustum, frustum[0]])
        axis.plot(closed[:, 0], closed[:, 1], closed[:, 2], c=colour, linewidth=2.0)
        for corner in frustum:
            axis.plot(
                [center[0], corner[0]],
                [center[1], corner[1]],
                [center[2], corner[2]],
                c=colour,
                linewidth=1.2,
            )

    def draw_camera_xz(
        axis,
        center,
        frustum,
        colour,
        label,
        marker_size=150,
        label_offset=(0.0, 0.0),
        label_va="center",
    ):
        axis.scatter(
            [center[0]],
            [center[2]],
            c=colour,
            s=marker_size,
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
            label=label,
        )
        closed = np.vstack([frustum, frustum[0]])
        axis.plot(closed[:, 0], closed[:, 2], c=colour, linewidth=2.0, zorder=4)
        for corner in frustum:
            axis.plot(
                [center[0], corner[0]],
                [center[2], corner[2]],
                c=colour,
                linewidth=1.2,
                zorder=4,
            )
        axis.text(
            center[0] + label_offset[0],
            center[2] + label_offset[1],
            f"  {label}",
            color=colour,
            weight="bold",
            va=label_va,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            zorder=6,
        )

    if len(points):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=point_size,
            c=rgb,
            alpha=0.9,
            depthshade=False,
        )
    else:
        ax.text(0, 0, 0, "No points after filtering", ha="center")

    for idx, ((label, _, _), center, frustum) in enumerate(zip(camera_poses, camera_centers, frustums)):
        colour = colours_by_camera[idx % len(colours_by_camera)]
        draw_camera_3d(ax, center, frustum, colour, label)

    if len(camera_centers) >= 2:
        ax.plot(
            camera_centers[:, 0],
            camera_centers[:, 1],
            camera_centers[:, 2],
            c="#5b6675",
            linewidth=2.0,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z / depth")
    ax.set_title(f"Scene view ({len(points)} points, {len(camera_poses)} cameras)")
    ax.legend(loc="upper right")
    _apply_equal_limits(ax, lower, upper)
    ax.view_init(elev=18, azim=-68)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass

    if len(points):
        ax_xz.scatter(points[:, 0], points[:, 2], s=point_size, c=rgb, alpha=0.9)
    if len(camera_centers) >= 2:
        ax_xz.plot(camera_centers[:, 0], camera_centers[:, 2], c="#5b6675", linewidth=1.8)
    for idx, ((_, _, _), center, frustum) in enumerate(zip(camera_poses, camera_centers, frustums)):
        colour = colours_by_camera[idx % len(colours_by_camera)]
        draw_camera_xz(ax_xz, center, frustum, colour, f"C{idx + 1}")

    x_padding = 0.05 * max(upper[0] - lower[0], 1e-6)
    z_padding = 0.05 * max(upper[2] - lower[2], 1e-6)
    ax_xz.set_xlim(lower[0] - x_padding, upper[0] + x_padding)
    ax_xz.set_ylim(lower[2] - z_padding, upper[2] + z_padding)
    ax_xz.grid(True, alpha=0.25)
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z / depth")
    ax_xz.set_title("Full X-Z depth view")
    ax_xz.legend(loc="upper right")
    ax_xz.text(
        0.02,
        0.02,
        "Translation scale is arbitrary",
        transform=ax_xz.transAxes,
        fontsize=9,
        color="#5b6675",
    )

    if len(plot_extra):
        detail_x = plot_extra[:, 0]
        detail_z = plot_extra[:, 2]
        detail_center = np.array([np.mean(detail_x), np.mean(detail_z)])
        detail_radius = 0.85 * max(np.ptp(detail_x), np.ptp(detail_z), baseline, 1.0)
    else:
        detail_center = np.zeros(2)
        detail_radius = 1.0

    if len(camera_centers) >= 2:
        ax_cam.plot(camera_centers[:, 0], camera_centers[:, 2], c="#5b6675", linewidth=2.2)
    label_offset = (0.02 * detail_radius, 0.05 * detail_radius)
    for idx, ((label, _, _), center, frustum) in enumerate(zip(camera_poses, camera_centers, frustums)):
        colour = colours_by_camera[idx % len(colours_by_camera)]
        draw_camera_xz(
            ax_cam,
            center,
            frustum,
            colour,
            label,
            marker_size=190,
            label_offset=label_offset,
            label_va="bottom",
        )

    ax_cam.set_xlim(detail_center[0] - detail_radius, detail_center[0] + detail_radius)
    ax_cam.set_ylim(detail_center[1] - detail_radius, detail_center[1] + detail_radius)
    ax_cam.set_aspect("equal", adjustable="box")
    ax_cam.grid(True, alpha=0.25)
    ax_cam.set_xlabel("X")
    ax_cam.set_ylabel("Z / depth")
    ax_cam.set_title("Camera trajectory zoom")
    ax_cam.text(
        0.02,
        0.98,
        "Frustums show viewing direction; scale is arbitrary",
        transform=ax_cam.transAxes,
        fontsize=9,
        color="#5b6675",
        va="top",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def _extract_rgb_patch(image: np.ndarray, point: np.ndarray, patch_radius: int) -> np.ndarray:
    """Extract a small RGB image patch centred on a floating-point image coordinate."""
    height, width = image.shape[:2]
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    x0 = max(0, x - patch_radius)
    x1 = min(width, x + patch_radius + 1)
    y0 = max(0, y - patch_radius)
    y1 = min(height, y + patch_radius + 1)

    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        size = 2 * patch_radius + 1
        return np.full((size, size, 3), 220, dtype=np.uint8)
    return cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)


def plot_patch_cloud_reconstruction(
    points3d: np.ndarray,
    source_image: np.ndarray,
    source_points: np.ndarray,
    camera_poses: list[tuple[str, np.ndarray, np.ndarray]],
    output_path: Path,
    max_patches: int = 70,
    patch_radius: int = 14,
    patch_zoom: float = 0.55,
) -> None:
    """Save an X-Z reconstruction view where points are shown as image patches."""
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    ensure_dir(output_path.parent)

    n = min(len(points3d), len(source_points))
    points = points3d[:n]
    image_points = source_points[:n]

    if camera_poses:
        camera_centers = np.array([_camera_center(R, t) for _, R, t in camera_poses])
    else:
        camera_centers = np.empty((0, 3), dtype=np.float64)

    if len(camera_centers) >= 2:
        baseline = np.median(
            [
                np.linalg.norm(camera_centers[i] - camera_centers[j])
                for i in range(len(camera_centers))
                for j in range(i + 1, len(camera_centers))
            ]
        )
    else:
        baseline = 1.0
    frustum_scale = max(0.65, 0.55 * baseline)

    frustums = [
        _camera_frustum(R.T, center, frustum_scale)
        for center, (_, R, _) in zip(camera_centers, camera_poses)
    ]
    plot_extra = camera_centers
    if frustums:
        plot_extra = np.vstack([camera_centers, *frustums])
    lower, upper = _robust_bounds(points, extra_points=plot_extra)

    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    if len(points):
        ax.scatter(points[:, 0], points[:, 2], s=12, c="#9aa5b1", alpha=0.32, label="all points")
        if len(points) > max_patches:
            indices = np.linspace(0, len(points) - 1, max_patches, dtype=int)
        else:
            indices = np.arange(len(points))

        for idx in indices:
            patch_rgb = _extract_rgb_patch(source_image, image_points[idx], patch_radius)
            image_box = OffsetImage(patch_rgb, zoom=patch_zoom)
            ab = AnnotationBbox(
                image_box,
                (points[idx, 0], points[idx, 2]),
                frameon=True,
                pad=0.08,
                bboxprops={
                    "edgecolor": "#263238",
                    "linewidth": 0.6,
                    "alpha": 0.9,
                },
                zorder=4,
            )
            ax.add_artist(ab)
    else:
        ax.text(0.5, 0.5, "No points after filtering", transform=ax.transAxes, ha="center")

    colours_by_camera = ["#2458a6", "#a33b3b", "#2f7d32", "#7a4ea3", "#b46b00"]
    if len(camera_centers) >= 2:
        ax.plot(camera_centers[:, 0], camera_centers[:, 2], c="#5b6675", linewidth=2.0, zorder=2)

    for idx, ((label, _, _), center, frustum) in enumerate(zip(camera_poses, camera_centers, frustums)):
        colour = colours_by_camera[idx % len(colours_by_camera)]
        ax.scatter(
            [center[0]],
            [center[2]],
            c=colour,
            s=170,
            edgecolors="white",
            linewidths=1.2,
            zorder=6,
            label=label,
        )
        closed = np.vstack([frustum, frustum[0]])
        ax.plot(closed[:, 0], closed[:, 2], c=colour, linewidth=2.0, zorder=5)
        for corner in frustum:
            ax.plot([center[0], corner[0]], [center[2], corner[2]], c=colour, linewidth=1.2, zorder=5)
        ax.text(
            center[0],
            center[2],
            f"  {label}",
            color=colour,
            weight="bold",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            zorder=7,
        )

    x_padding = 0.05 * max(upper[0] - lower[0], 1e-6)
    z_padding = 0.05 * max(upper[2] - lower[2], 1e-6)
    ax.set_xlim(lower[0] - x_padding, upper[0] + x_padding)
    ax.set_ylim(lower[2] - z_padding, upper[2] + z_padding)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("X")
    ax.set_ylabel("Z / depth")
    ax.set_title("Patch cloud X-Z view")
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.02,
        "Each thumbnail is cropped around the source feature in image 1",
        transform=ax.transAxes,
        fontsize=9,
        color="#5b6675",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def write_ply(path: Path, points3d: np.ndarray, colours: np.ndarray) -> None:
    """Write a simple ASCII PLY point cloud with RGB colours."""
    ensure_dir(path.parent)
    if len(colours) != len(points3d):
        colours = np.full((len(points3d), 3), 200, dtype=np.uint8)

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, colour in zip(points3d, colours):
            f.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(colour[0])} {int(colour[1])} {int(colour[2])}\n"
            )

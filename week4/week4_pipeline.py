"""Week 4 incremental sparse SfM pipeline.

This script extends the Week 3 two-view + third-view pipeline to a small
multi-view reconstruction. It registers new images with PnP, then triangulates
new points after each accepted camera.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEEK2_DIR = ROOT / "week2"
DEFAULT_WEEK3_DIR = ROOT / "week3"
cv2 = None


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GF4 Week 4 incremental SfM")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--week2-dir", type=Path, default=DEFAULT_WEEK2_DIR)
    parser.add_argument("--week3-dir", type=Path, default=DEFAULT_WEEK3_DIR)
    parser.add_argument("--max-images", type=int, default=12, help="Use 0 for all images.")
    parser.add_argument("--target-registered", type=int, default=10)
    parser.add_argument("--initial-pair", nargs=2, type=int, metavar=("I", "J"))
    parser.add_argument("--max-image-size", type=int, default=1600, help="Use 0 to disable resizing.")
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--focal-length-px", type=float, default=None)
    parser.add_argument("--ransac-threshold", type=float, default=1.0)
    parser.add_argument("--confidence", type=float, default=0.999)
    parser.add_argument("--max-reprojection-error", type=float, default=4.0)
    parser.add_argument("--pnp-ransac-threshold", type=float, default=6.0)
    parser.add_argument("--min-initial-points", type=int, default=30)
    parser.add_argument("--min-pnp-correspondences", type=int, default=20)
    parser.add_argument("--min-pnp-inliers", type=int, default=12)
    parser.add_argument("--min-pnp-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--max-registration-error", type=float, default=8.0)
    parser.add_argument("--max-observation-error", type=float, default=6.0)
    parser.add_argument("--max-new-points-per-pair", type=int, default=800)
    args = parser.parse_args()

    if args.max_images == 0:
        args.max_images = None
    if args.max_image_size == 0:
        args.max_image_size = None
    if args.target_registered < 2:
        parser.error("--target-registered must be at least 2")
    return args


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_stem(path: Path) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in path.stem)


def median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if len(values) else None


def mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def none_large(value: float | None) -> float:
    return 1e9 if value is None else value


def point_array(points: list[np.ndarray]) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 3) if points else np.empty((0, 3))


def colour_array(colours: list[np.ndarray]) -> np.ndarray:
    return np.asarray(colours, dtype=np.uint8).reshape(-1, 3) if colours else np.empty((0, 3), dtype=np.uint8)


def projection(cam: dict) -> np.ndarray:
    return cam["K"] @ np.hstack([cam["R"], cam["t"].reshape(3, 1)])


def depth(points3d: np.ndarray, cam: dict) -> np.ndarray:
    return (cam["R"] @ points3d.T + cam["t"].reshape(3, 1)).T[:, 2]


def triangulate_between(pts_a: np.ndarray, pts_b: np.ndarray, cam_a: dict, cam_b: dict) -> np.ndarray:
    if len(pts_a) == 0:
        return np.empty((0, 3), dtype=np.float64)
    points4d = cv2.triangulatePoints(projection(cam_a), projection(cam_b), pts_a.T, pts_b.T)
    points3d = np.full((points4d.shape[1], 3), np.nan, dtype=np.float64)
    ok = np.abs(points4d[3]) > 1e-12
    points3d[ok] = (points4d[:3, ok] / points4d[3, ok]).T
    return points3d


def get_matches(week2, features, i: int, j: int, ratio: float, cache: dict) -> list:
    key = (i, j)
    if key not in cache:
        cache[key] = week2.match_descriptors(
            features[i].descriptors,
            features[j].descriptors,
            ratio=ratio,
        )
    return cache[key]


def pair_reconstruction(week2, w3, features, Ks, i: int, j: int, args, cache: dict) -> dict | None:
    matches = get_matches(week2, features, i, j, args.ratio, cache)
    if len(matches) < 8:
        return None

    pts_i, pts_j = week2.matched_keypoint_coords(features[i].keypoints, features[j].keypoints, matches)
    try:
        E, essential_mask = w3.estimate_essential_matrix(
            pts_i,
            pts_j,
            Ks[i],
            threshold=args.ransac_threshold,
            confidence=args.confidence,
        )
        R, t, pose_mask = w3.recover_relative_pose(E, pts_i, pts_j, Ks[i], inlier_mask=essential_mask)
    except Exception:
        return None

    pose_matches = [m for m, keep in zip(matches, pose_mask) if keep]
    pts_i_pose = pts_i[pose_mask]
    pts_j_pose = pts_j[pose_mask]
    cam_i = {"idx": i, "K": Ks[i], "R": np.eye(3), "t": np.zeros((3, 1))}
    cam_j = {"idx": j, "K": Ks[j], "R": R, "t": t}

    points3d = triangulate_between(pts_i_pose, pts_j_pose, cam_i, cam_j)
    err_i = w3.compute_reprojection_errors(points3d, pts_i_pose, Ks[i], cam_i["R"], cam_i["t"])
    err_j = w3.compute_reprojection_errors(points3d, pts_j_pose, Ks[j], R, t)
    keep = (
        np.isfinite(points3d).all(axis=1)
        & (depth(points3d, cam_i) > 0)
        & (depth(points3d, cam_j) > 0)
        & (err_i <= args.max_reprojection_error)
        & (err_j <= args.max_reprojection_error)
    )

    kept_errors = 0.5 * (err_i[keep] + err_j[keep])
    kept_matches = [m for m, ok in zip(pose_matches, keep) if ok]
    return {
        "i": i,
        "j": j,
        "R": R,
        "t": t,
        "E": E,
        "points": points3d[keep],
        "colours": w3.sample_point_colours(features[i].image, pts_i_pose[keep]),
        "pts_i": pts_i_pose[keep],
        "pts_j": pts_j_pose[keep],
        "matches": kept_matches,
        "metrics": {
            "image_i": features[i].path.name,
            "image_j": features[j].path.name,
            "filtered_matches": len(matches),
            "essential_inliers": int(np.sum(essential_mask)),
            "pose_inliers": int(np.sum(pose_mask)),
            "kept_points": int(np.sum(keep)),
            "median_reprojection_error_px": median(kept_errors),
            "mean_reprojection_error_px": mean(kept_errors),
            "focal_length_px": float(Ks[i][0, 0]),
        },
    }


def choose_initial_pair(week2, w3, features, Ks, args, cache: dict) -> dict:
    if args.initial_pair is not None:
        i, j = args.initial_pair
        if i == j or i < 0 or j < 0 or i >= len(features) or j >= len(features):
            raise ValueError("--initial-pair must contain two different zero-based image indices")
        result = pair_reconstruction(week2, w3, features, Ks, i, j, args, cache)
        if result is None:
            raise ValueError("The requested initial pair could not be reconstructed")
        return result

    best = None
    best_score = None
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            result = pair_reconstruction(week2, w3, features, Ks, i, j, args, cache)
            if result is None or result["metrics"]["kept_points"] < args.min_initial_points:
                continue
            med = none_large(result["metrics"]["median_reprojection_error_px"])
            score = (
                result["metrics"]["kept_points"],
                result["metrics"]["pose_inliers"],
                result["metrics"]["filtered_matches"],
                -med,
            )
            if best_score is None or score > best_score:
                best = result
                best_score = score
    if best is None:
        raise ValueError("No usable initial pair found. Try --initial-pair or lower --min-initial-points.")
    return best


def build_pnp_candidate(week2, w3, features, Ks, registered, candidate_idx, obs, points, args, cache):
    pids, pts2d, kps = [], [], []
    total_matches = 0
    used_points, used_kps = set(), set()

    for cam in registered:
        matches = get_matches(week2, features, cam["idx"], candidate_idx, args.ratio, cache)
        total_matches += len(matches)
        for m in sorted(matches, key=lambda x: x.distance):
            pid = obs[cam["idx"]].get(m.queryIdx)
            if pid is None or pid in used_points or m.trainIdx in used_kps:
                continue
            used_points.add(pid)
            used_kps.add(m.trainIdx)
            pids.append(pid)
            kps.append(m.trainIdx)
            pts2d.append(features[candidate_idx].keypoints[m.trainIdx].pt)

    row_base = {
        "image": features[candidate_idx].path.name,
        "anchor_matches": total_matches,
        "pnp_correspondences": len(pids),
    }
    if len(pids) < args.min_pnp_correspondences:
        return {**row_base, "ok": False, "reason": "too_few_2d3d_correspondences"}

    X = point_array([points[pid] for pid in pids])
    pts2d = np.asarray(pts2d, dtype=np.float64).reshape(-1, 2)
    R, t, mask = w3.estimate_camera_pose_pnp(
        X,
        pts2d,
        Ks[candidate_idx],
        threshold=args.pnp_ransac_threshold,
        confidence=args.confidence,
    )
    X_in = X[mask]
    pts_in = pts2d[mask]
    errors = w3.compute_reprojection_errors(X_in, pts_in, Ks[candidate_idx], R, t) if len(X_in) else np.array([])
    inliers = int(np.sum(mask))
    inlier_ratio = inliers / len(pids) if pids else 0.0
    med_error = median(errors)

    ok = (
        inliers >= args.min_pnp_inliers
        and inlier_ratio >= args.min_pnp_inlier_ratio
        and med_error is not None
        and med_error <= args.max_registration_error
    )
    reason = "accepted" if ok else "weak_pnp_or_high_reprojection_error"
    return {
        **row_base,
        "ok": ok,
        "reason": reason,
        "R": R,
        "t": t,
        "mask": mask,
        "pids": pids,
        "kps": kps,
        "X": X,
        "pts2d": pts2d,
        "pnp_inliers": inliers,
        "pnp_inlier_ratio": inlier_ratio,
        "median_reprojection_error_px": med_error,
        "mean_reprojection_error_px": mean(errors),
    }


def candidate_csv_row(round_idx: int, result: dict, selected: bool) -> dict:
    return {
        "round": round_idx,
        "image": result["image"],
        "selected": int(selected),
        "accepted": int(result["ok"]),
        "reason": result["reason"],
        "anchor_matches": result["anchor_matches"],
        "pnp_correspondences": result["pnp_correspondences"],
        "pnp_inliers": result.get("pnp_inliers", 0),
        "pnp_inlier_ratio": result.get("pnp_inlier_ratio", 0.0),
        "median_reprojection_error_px": result.get("median_reprojection_error_px"),
        "mean_reprojection_error_px": result.get("mean_reprojection_error_px"),
    }


def add_observation_if_consistent(w3, features, cam, obs, points, image_idx, kp_idx, pid, max_error) -> int:
    if kp_idx in obs[image_idx]:
        return 0
    pt = np.asarray([features[image_idx].keypoints[kp_idx].pt], dtype=np.float64)
    X = np.asarray([points[pid]], dtype=np.float64)
    err = w3.compute_reprojection_errors(X, pt, cam["K"], cam["R"], cam["t"])
    if len(err) and np.isfinite(err[0]) and err[0] <= max_error:
        obs[image_idx][kp_idx] = pid
        return 1
    return 0


def grow_points(week2, w3, features, new_cam, anchors, obs, points, colours, args, cache):
    new_count = 0
    added_obs = 0
    rows = []
    new_idx = new_cam["idx"]

    for anchor in anchors:
        a_idx = anchor["idx"]
        matches = sorted(get_matches(week2, features, a_idx, new_idx, args.ratio, cache), key=lambda x: x.distance)
        pts_a, pts_n, kp_a, kp_n = [], [], [], []
        used_a, used_n = set(), set()

        for m in matches:
            pid_a = obs[a_idx].get(m.queryIdx)
            pid_n = obs[new_idx].get(m.trainIdx)
            if pid_a is not None and pid_n is None:
                added_obs += add_observation_if_consistent(
                    w3, features, new_cam, obs, points, new_idx, m.trainIdx, pid_a, args.max_observation_error
                )
                continue
            if pid_n is not None and pid_a is None:
                added_obs += add_observation_if_consistent(
                    w3, features, anchor, obs, points, a_idx, m.queryIdx, pid_n, args.max_observation_error
                )
                continue
            if pid_a is not None or pid_n is not None:
                continue
            if m.queryIdx in used_a or m.trainIdx in used_n:
                continue
            used_a.add(m.queryIdx)
            used_n.add(m.trainIdx)
            kp_a.append(m.queryIdx)
            kp_n.append(m.trainIdx)
            pts_a.append(features[a_idx].keypoints[m.queryIdx].pt)
            pts_n.append(features[new_idx].keypoints[m.trainIdx].pt)
            if len(pts_a) >= args.max_new_points_per_pair:
                break

        if not pts_a:
            rows.append({
                "new_image": features[new_idx].path.name,
                "anchor_image": features[a_idx].path.name,
                "lowe_matches": len(matches),
                "triangulation_candidates": 0,
                "new_points": 0,
            })
            continue

        pts_a = np.asarray(pts_a, dtype=np.float64)
        pts_n = np.asarray(pts_n, dtype=np.float64)
        X = triangulate_between(pts_a, pts_n, anchor, new_cam)
        err_a = w3.compute_reprojection_errors(X, pts_a, anchor["K"], anchor["R"], anchor["t"])
        err_n = w3.compute_reprojection_errors(X, pts_n, new_cam["K"], new_cam["R"], new_cam["t"])
        keep = (
            np.isfinite(X).all(axis=1)
            & (depth(X, anchor) > 0)
            & (depth(X, new_cam) > 0)
            & (err_a <= args.max_reprojection_error)
            & (err_n <= args.max_reprojection_error)
        )
        new_colours = w3.sample_point_colours(features[a_idx].image, pts_a[keep])
        added_this_anchor = 0

        for point, colour, a_kp, n_kp in zip(X[keep], new_colours, np.asarray(kp_a)[keep], np.asarray(kp_n)[keep]):
            if a_kp in obs[a_idx] or n_kp in obs[new_idx]:
                continue
            pid = len(points)
            points.append(point)
            colours.append(colour)
            obs[a_idx][int(a_kp)] = pid
            obs[new_idx][int(n_kp)] = pid
            new_count += 1
            added_this_anchor += 1

        rows.append({
            "new_image": features[new_idx].path.name,
            "anchor_image": features[a_idx].path.name,
            "lowe_matches": len(matches),
            "triangulation_candidates": len(pts_a),
            "new_points": added_this_anchor,
        })
    return new_count, added_obs, rows


def save_camera_files(features, registered, output_dir: Path) -> list[dict]:
    camera_dir = ensure(output_dir / "cameras")
    rows = []
    for order, cam in enumerate(registered):
        stem = f"{order:02d}_{safe_stem(features[cam['idx']].path)}"
        np.savetxt(camera_dir / f"{stem}_K.txt", cam["K"])
        np.savetxt(camera_dir / f"{stem}_R.txt", cam["R"])
        np.savetxt(camera_dir / f"{stem}_t.txt", cam["t"])
        centre = (-cam["R"].T @ cam["t"]).ravel()
        rows.append({
            "order": order,
            "image": features[cam["idx"]].path.name,
            "camera_center_x": float(centre[0]),
            "camera_center_y": float(centre[1]),
            "camera_center_z": float(centre[2]),
        })
    return rows


def run(args: argparse.Namespace) -> None:
    global cv2
    import cv2 as cv2_module

    cv2 = cv2_module
    week2 = load_module("week2_sfm_utils", args.week2_dir / "sfm_utils.py")
    w3 = load_module("week3_two_view_utils", args.week3_dir / "two_view_utils.py")
    output_dir = ensure(args.output_dir)
    overlay_dir = ensure(output_dir / "reprojection_overlays")

    image_paths = week2.list_image_paths(args.image_dir, max_images=args.max_images)
    if len(image_paths) < 2:
        raise ValueError("Need at least two images")
    features = week2.precompute_image_features(
        image_paths,
        max_features=args.max_features,
        max_image_size=args.max_image_size,
    )
    Ks = [
        w3.make_camera_matrix(f.image.shape, focal_length_px=args.focal_length_px)
        for f in features
    ]

    match_cache = {}
    init = choose_initial_pair(week2, w3, features, Ks, args, match_cache)
    i, j = init["i"], init["j"]
    obs = [dict() for _ in features]
    points, colours = [], []

    for point, colour, match in zip(init["points"], init["colours"], init["matches"]):
        if match.queryIdx in obs[i] or match.trainIdx in obs[j]:
            continue
        pid = len(points)
        points.append(point)
        colours.append(colour)
        obs[i][match.queryIdx] = pid
        obs[j][match.trainIdx] = pid

    registered = [
        {"idx": i, "K": Ks[i], "R": np.eye(3), "t": np.zeros((3, 1))},
        {"idx": j, "K": Ks[j], "R": init["R"], "t": init["t"]},
    ]
    remaining = set(range(len(features))) - {i, j}

    if hasattr(week2, "draw_matches"):
        week2.draw_matches(
            features[i].image,
            features[i].keypoints,
            features[j].image,
            features[j].keypoints,
            init["matches"],
            output_dir / "initial_pose_inlier_matches.png",
        )
    w3.draw_reprojection_overlay(
        features[i].image,
        features[j].image,
        init["pts_i"],
        init["pts_j"],
        init["points"],
        Ks[i],
        init["R"],
        init["t"],
        output_dir / "initial_reprojection_overlay.png",
    )
    w3.plot_multi_view_reconstruction(
        init["points"],
        init["colours"],
        [(features[i].path.name, np.eye(3), np.zeros((3, 1))), (features[j].path.name, init["R"], init["t"])],
        output_dir / "initial_reconstruction.png",
    )

    candidate_rows = []
    registration_rows = []
    triangulation_rows = []
    round_idx = 1

    while remaining and len(registered) < min(args.target_registered, len(features)):
        attempts = [
            build_pnp_candidate(week2, w3, features, Ks, registered, idx, obs, points, args, match_cache)
            for idx in sorted(remaining)
        ]
        accepted = [r for r in attempts if r["ok"]]
        best = None
        if accepted:
            best = max(
                accepted,
                key=lambda r: (
                    r["pnp_inliers"],
                    r["pnp_inlier_ratio"],
                    -none_large(r["median_reprojection_error_px"]),
                ),
            )
        for result in attempts:
            candidate_rows.append(candidate_csv_row(round_idx, result, best is result))
        if best is None:
            break

        new_idx = next(idx for idx in remaining if features[idx].path.name == best["image"])
        new_cam = {"idx": new_idx, "K": Ks[new_idx], "R": best["R"], "t": best["t"]}
        for pid, kp, keep in zip(best["pids"], best["kps"], best["mask"]):
            if keep and kp not in obs[new_idx]:
                obs[new_idx][kp] = pid

        anchors = list(registered)
        registered.append(new_cam)
        remaining.remove(new_idx)

        X_in = best["X"][best["mask"]]
        pts_in = best["pts2d"][best["mask"]]
        w3.draw_single_image_reprojection_overlay(
            features[new_idx].image,
            pts_in,
            X_in,
            Ks[new_idx],
            best["R"],
            best["t"],
            overlay_dir / f"{safe_stem(features[new_idx].path)}.png",
        )
        new_points, added_obs, tri_rows = grow_points(
            week2, w3, features, new_cam, anchors, obs, points, colours, args, match_cache
        )
        triangulation_rows.extend(tri_rows)
        registration_rows.append({
            "order": len(registered) - 1,
            "image": features[new_idx].path.name,
            "anchor_matches": best["anchor_matches"],
            "pnp_correspondences": best["pnp_correspondences"],
            "pnp_inliers": best["pnp_inliers"],
            "pnp_inlier_ratio": best["pnp_inlier_ratio"],
            "median_reprojection_error_px": best["median_reprojection_error_px"],
            "mean_reprojection_error_px": best["mean_reprojection_error_px"],
            "new_triangulated_points": new_points,
            "added_existing_observations": added_obs,
            "total_sparse_points": len(points),
        })
        print(
            f"registered {features[new_idx].path.name}: "
            f"{best['pnp_inliers']} PnP inliers, +{new_points} points"
        )
        round_idx += 1

    final_points = point_array(points)
    final_colours = colour_array(colours)
    poses = [(features[c["idx"]].path.name, c["R"], c["t"]) for c in registered]
    w3.plot_multi_view_reconstruction(final_points, final_colours, poses, output_dir / "final_reconstruction.png")
    w3.write_ply(output_dir / "points3d.ply", final_points, final_colours)

    first_idx = registered[0]["idx"]
    inv_obs = {pid: kp for kp, pid in obs[first_idx].items()}
    patch_ids = [pid for pid in range(len(points)) if pid in inv_obs]
    if patch_ids:
        patch_pts = np.asarray([features[first_idx].keypoints[inv_obs[pid]].pt for pid in patch_ids])
        w3.plot_patch_cloud_reconstruction(
            final_points[patch_ids],
            features[first_idx].image,
            patch_pts,
            poses,
            output_dir / "final_patch_cloud.png",
        )

    w3.save_csv(output_dir / "initial_pair_metrics.csv", [init["metrics"]])
    w3.save_csv(output_dir / "candidate_metrics.csv", candidate_rows)
    w3.save_csv(output_dir / "registration_metrics.csv", registration_rows)
    w3.save_csv(output_dir / "triangulation_metrics.csv", triangulation_rows)
    w3.save_csv(output_dir / "cameras.csv", save_camera_files(features, registered, output_dir))
    w3.save_csv(output_dir / "unregistered_images.csv", [
        {"image": features[idx].path.name, "reason": "target_reached" if len(registered) >= args.target_registered else "no_candidate_met_thresholds"}
        for idx in sorted(remaining)
    ])
    w3.save_csv(output_dir / "summary.csv", [{
        "input_images": len(features),
        "registered_images": len(registered),
        "unregistered_images": len(remaining),
        "sparse_points": len(points),
        "feature_observations": sum(len(x) for x in obs),
        "initial_image_i": features[i].path.name,
        "initial_image_j": features[j].path.name,
    }])
    print(f"wrote outputs to {output_dir}")


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

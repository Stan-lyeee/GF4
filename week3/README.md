# GF4 Structure from Motion - Week 3 Guide
## Sparse Reconstruction: Two Views plus One Registered View

**Goal:** Turn Week 2's pairwise feature-matching code into a small sparse reconstruction with 3D points and three registered cameras.

**Main deliverable for the week:** A Python pipeline that recovers relative camera pose from a good image pair, triangulates sparse 3D points, registers one additional image, and visualises the sparse points with three camera poses.

There is no separate Week 3 submission. Treat this week as the week where you start generating the core evidence for your final report and presentation.

---

## Week 3 learning objectives

By the end of Week 3, you should be able to:

1. Estimate the essential matrix from matched image points.
2. Recover relative camera rotation and translation direction.
3. Triangulate sparse 3D points from two calibrated views.
4. Filter triangulated points using positive depth and reprojection error.
5. Use reprojection overlays to check whether triangulated points are consistent with the input images.
6. Build simple 2D-3D correspondences between reconstructed points and a third image.
7. Register a third image using PnP with RANSAC.
8. Visualise a small sparse reconstruction with multiple camera poses.
9. Explain why many correct-looking matches may still produce poor 3D structure.

---

## Required reading

Read the following:

1. OpenCV: Camera Calibration and 3D Reconstruction

   https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

   On this page, look up:

   - `findEssentialMat`
   - `recoverPose`
   - `triangulatePoints`
   - `solvePnPRansac`

2. OpenCV: Epipolar Geometry

   https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html

3. OpenCV: Camera Calibration

   https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html


---

## Relationship to Week 2

Week 3 builds directly on your Week 2 code.

```text
Week 2:
images -> SIFT -> descriptor matching -> filtering -> geometric verification

Week 3:
Lowe-filtered matches -> essential matrix RANSAC -> relative pose -> triangulation -> 3D points
3D points + new image -> 2D-3D correspondences -> PnP -> third camera pose
```

The Week 3 starter code expects to use your completed Week 2 `sfm_utils.py` for:

- SIFT feature detection,
- descriptor matching,
- converting OpenCV matches into aligned point-coordinate arrays,
- optional match visualisation.

---

## Starter code

Starter code is provided:

```text
week3/week3_pipeline.py
week3/two_view_utils.py
```

The starter code handles:

- command-line argument parsing,
- loading your Week 2 `sfm_utils.py`,
- running Week 2 feature matching,
- saving metrics and figures,
- camera/frustum plotting,
- patch-cloud visualisation,
- writing a simple coloured `.ply` point cloud.

You have to complete the Week 3 TODO functions in `two_view_utils.py`.

Continue using your Week 1/2 Python environment. The Week 3 code uses OpenCV,
NumPy, and Matplotlib.

---

## Suggested command-line interface

Run from the repository root.

For a good image pair plus a third image:

```bash
python week3/week3_pipeline.py \
  --image1 path/to/good_01.jpg \
  --image2 path/to/good_08.jpg \
  --image3 path/to/good_12.jpg \
  --output-dir week3/output/three_view \
  --week2-dir week2 \
  --max-features 4000 \
  --ratio 0.75
```

For pair-only debugging, omit `--image3`:

```bash
python week3/week3_pipeline.py \
  --image1 path/to/good_01.jpg \
  --image2 path/to/good_08.jpg \
  --output-dir week3/output/pair_only \
  --week2-dir week2 \
  --max-features 4000 \
  --ratio 0.75
```

The default camera intrinsics are an approximation:

```text
focal length = 1.2 * max(image width, image height)
principal point = image centre
```

If you know a better focal length in pixels, pass it explicitly:

```bash
--focal-length-px 1800
```

We will not refine the camera intrinsics in this simplified pipeline.

---

## Useful OpenCV functions

You may use OpenCV for the heavy geometry:

- `cv2.findEssentialMat`
- `cv2.recoverPose`
- `cv2.triangulatePoints`
- `cv2.solvePnPRansac`
- `cv2.Rodrigues`

You should implement the pipeline logic yourself:

- create projection matrices,
- convert homogeneous triangulation output into 3D coordinates,
- project 3D points back into image coordinates,
- compute reprojection errors,
- filter invalid 3D points,
- draw reprojection overlays for the two-view reconstruction,
- connect reconstructed points to third-image features,
- draw the third-view reprojection overlay after PnP,
- interpret the PnP inliers and reprojection errors,
- interpret the result.

The standard camera/frustum plots and patch-cloud plots are provided. The reprojection overlays are deliberately not provided because they are part of the core SfM evaluation: you need to understand how a 3D point projects back into each image.

---

## Required implementation checklist

Your group should implement the following Week 3 functions:

- `estimate_essential_matrix`
- `recover_relative_pose`
- `make_projection_matrices`
- `triangulate_points`
- `project_points`
- `compute_reprojection_errors`
- `compute_depths`
- `filter_reconstructed_points`
- `draw_reprojection_overlay`
- `build_2d3d_correspondences`
- `estimate_camera_pose_pnp`
- `draw_single_image_reprojection_overlay`

In particular, the two reprojection overlay functions should:

- project the reconstructed 3D points into the relevant image or images,
- draw the observed 2D feature locations,
- draw the reprojected 3D point locations,
- draw line segments showing the reprojection error,
- save the resulting figure.

---

## AI use policy

You may use AI tools to help you learn the background material, understand the concepts, and interpret documentation. For example, you may use AI to ask:

- what SIFT keypoints and descriptors represent,
- how Lowe's ratio test works,
- what OpenCV functions such as `cv2.SIFT_create`, `BFMatcher.knnMatch`, or `cv2.findFundamentalMat` return.

You may **not** use AI tools to write code for this assignment, edit code, fix code, debug code, generate tests, or diagnose implementation errors. The implementation you submit must be written and debugged by your group.

Any use of AI tools must be clearly mentioned in your final report.

---

## Phase A - Camera intrinsics

Introduce the camera intrinsic matrix:

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

For this simplified pipeline, you may use:

```text
fx = fy = approximate focal length in pixels
cx = (image width - 1) / 2
cy = (image height - 1) / 2
```

---

## Phase B - Essential matrix and relative pose

The Week 2 fundamental matrix works in pixel coordinates. With camera intrinsics, use the essential matrix:

```text
E = K.T F K
```

or estimate it directly:

```python
E, mask = cv2.findEssentialMat(
    pts1,
    pts2,
    K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0,
)
```

Recover relative pose:

```python
_, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
```

Important interpretation:

- `R` is the relative rotation from camera 1 to camera 2,
- `t` is the relative translation direction,
- the scale of `t` is unknown.

SfM can recover structure only up to an arbitrary global scale unless extra metric information is provided.

---

## Phase C - Triangulation

Construct projection matrices:

```text
P1 = K [I | 0]
P2 = K [R | t]
```

Use pose inlier correspondences and triangulate:

```python
points4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
points3d = points4d[:3] / points4d[3]
```

Then filter points:

- positive depth in camera 1,
- positive depth in camera 2,
- finite coordinates,
- reprojection error below a chosen threshold.

Start with:

```text
maximum reprojection error: 4 px
```

The pipeline should also save:

```text
reprojection_overlay.png
two_view_patch_cloud.png
```

The reprojection overlay is the most direct Week 3 correctness check, and you must implement it. It projects the kept 3D points back into both images. The observed matched keypoints and reprojected points should lie close together; long line segments indicate large reprojection error or an implementation problem.

The patch-cloud visualisation is provided as an additional interpretability aid. It shows an X-Z reconstruction view where selected 3D points are drawn using small image patches cropped around their source features in image 1. Use this to connect the sparse 3D points back to actual image content.

---

## Phase D - Register a third image

Use your reconstructed points from images 1 and 2. Then match image 1 against image 3 and image 2 against image 3. A match to image 3 becomes useful for PnP only if the matched feature in image 1 or image 2 already has a reconstructed 3D point.

Conceptually:

```text
image 1 feature <-> image 2 feature -> triangulated 3D point
image 1 or image 2 feature <-> image 3 feature -> 2D observation in image 3

therefore:
triangulated 3D point <-> image 3 feature
```

Estimate the third camera pose with:

```python
success, rvec, tvec, inliers = cv2.solvePnPRansac(
    points3d,
    pts3,
    K3,
    None,
    reprojectionError=6.0,
    confidence=0.999,
)
R3, _ = cv2.Rodrigues(rvec)
```

Here `K3` is the intrinsic matrix for image 3. If all images have the same camera and resize, it will usually be the same approximation as `K`.

The pipeline should save:

```text
third_view_reprojection_overlay.png
three_view_reconstruction.png
three_view_patch_cloud.png
third_view_metrics.csv
```

You must implement `third_view_reprojection_overlay.png` yourself using your `project_points` function. The other visualisations are provided.


Questions to answer:

- Do the observed and reprojected points overlap in `third_view_reprojection_overlay.png`?
- Does the third camera pose look plausible relative to the first two cameras?
- Did PnP keep most of the available 2D-3D correspondences, or reject many of them?

---

## Phase E - Evidence pack for final deliverables

There is no Week 3 submission, but by the end of the week your group should have:

- a working two-view reconstruction for at least one good pair,
- a working third-view registration for one additional overlapping image,
- saved figures of matches, reprojection overlays, camera poses, triangulated points, and patch-cloud views,
- small metrics tables for the two-view reconstruction and third-view PnP,
- a short note on what succeeded, what failed, and why,

---

## Looking ahead to Week 4

Week 4 will focus on:

- extending the sparse reconstruction to a small sequence of roughly 5-10 images,
- comparing your pipeline with COLMAP,
- preparing the final report and presentation,
- making sure every group member understands the code and results.

"""
可视化推理结果并在2D图像上投影3D框。
检查是否需要旋转BEV以匹配相机视角。
"""
import argparse
import os
import sys
import numpy as np
import cv2
from pathlib import Path

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from config import kitti_config as cnf
from data_process.lidar_mapping import read_lidar_file_with_fallback
from data_process.kitti_data_utils import Calibration
from utils.visualization_utils import compute_box_3d, project_to_image


def draw_3d_boxes_on_image(img, detections, calib):
    """Draw 3D boxes projected onto 2D image."""
    for line in detections:
        parts = line.strip().split()
        if len(parts) < 15:
            continue

        cls_name = parts[0]
        score = float(parts[-1])

        # Parse 3D box in camera coordinates (KITTI format)
        h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
        x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])
        ry = float(parts[14])

        if z_cam <= 0:
            continue

        # Compute 3D box corners in camera coordinates
        location = np.array([x_cam, y_cam, z_cam])
        dims = np.array([h, w, l])

        corners_3d = compute_box_3d(dims, location, ry)
        corners_2d = project_to_image(corners_3d, calib.P2)

        # Check if box is within image bounds
        x_min, y_min = corners_2d[:, 0].min(), corners_2d[:, 1].min()
        x_max, y_max = corners_2d[:, 0].max(), corners_2d[:, 1].max()

        img_h, img_w = img.shape[:2]
        if x_max < 0 or y_max < 0 or x_min >= img_w or y_min >= img_h:
            continue

        # Clip to image bounds
        x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
        x_max, y_max = min(img_w-1, int(x_max)), min(img_h-1, int(y_max))

        # Color based on class
        if cls_name == 'Car':
            color = (0, 255, 0)  # Green
        elif cls_name == 'Pedestrian':
            color = (255, 0, 0)  # Blue
        elif cls_name == 'Cyclist':
            color = (0, 0, 255)  # Red
        else:
            color = (255, 255, 0)  # Cyan

        # Draw 2D bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        # Draw class label and score
        label = f'{cls_name}: {score:.2f}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img, (x_min, y_min - 25), (x_min + label_size[0], y_min), color, -1)
        cv2.putText(img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw 3D box projection
        corners_2d = corners_2d.astype(np.int32)

        # Draw the 8 corners
        for corner in corners_2d:
            if 0 <= corner[0] < img_w and 0 <= corner[1] < img_h:
                cv2.circle(img, tuple(corner), 2, color, -1)

        # Draw edges of the 3D box
        # Bottom face
        for i in range(4):
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[(i+1)%4])
            if all(0 <= p < img_w for p in [pt1[0], pt2[0]]) and all(0 <= p < img_h for p in [pt1[1], pt2[1]]):
                cv2.line(img, pt1, pt2, color, 2)

        # Top face
        for i in range(4, 8):
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[4 + (i-4+1)%4])
            if all(0 <= p < img_w for p in [pt1[0], pt2[0]]) and all(0 <= p < img_h for p in [pt1[1], pt2[1]]):
                cv2.line(img, pt1, pt2, color, 2)

        # Vertical edges
        for i in range(4):
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[i+4])
            if all(0 <= p < img_w for p in [pt1[0], pt2[0]]) and all(0 <= p < img_h for p in [pt1[1], pt2[1]]):
                cv2.line(img, pt1, pt2, color, 2)

    return img


def create_bev_from_points(points, cnf, rotate_180=False):
    """Create BEV image from point cloud with optional rotation."""
    xyz_i = points[:, :4].copy()

    # Filter points within boundaries
    x_lim = (cnf.boundary['minX'], cnf.boundary['maxX'])
    y_lim = (cnf.boundary['minY'], cnf.boundary['maxY'])
    z_lim = (cnf.boundary['minZ'], cnf.boundary['maxZ'])

    mask = (xyz_i[:, 0] >= x_lim[0]) & (xyz_i[:, 0] < x_lim[1]) & \
           (xyz_i[:, 1] >= y_lim[0]) & (xyz_i[:, 1] < y_lim[1]) & \
           (xyz_i[:, 2] >= z_lim[0]) & (xyz_i[:, 2] < z_lim[1])

    xyz_i = xyz_i[mask]

    if len(xyz_i) == 0:
        return np.zeros((cnf.BEV_HEIGHT, cnf.BEV_WIDTH), dtype=np.uint8)

    bev_height = cnf.BEV_HEIGHT
    bev_width = cnf.BEV_WIDTH

    x = xyz_i[:, 0]
    y = xyz_i[:, 1]
    intensity = xyz_i[:, 3]

    # Apply 180-degree rotation if requested
    if rotate_180:
        x_rot = -x
        y_rot = -y
        x, y = x_rot, y_rot

    # Normalize to BEV grid
    x_img = np.floor((y - y_lim[0]) / (y_lim[1] - y_lim[0]) * bev_width).astype(np.int32)
    y_img = np.floor((x - x_lim[0]) / (x_lim[1] - x_lim[0]) * bev_height).astype(np.int32)

    x_img = np.clip(x_img, 0, bev_width - 1)
    y_img = np.clip(y_img, 0, bev_height - 1)

    # Create BEV intensity map
    bev_map = np.zeros((bev_height, bev_width), dtype=np.float32)

    for i in range(len(x_img)):
        bev_map[y_img[i], x_img[i]] = max(bev_map[y_img[i], x_img[i]], intensity[i])

    # Normalize for visualization
    bev_map = np.clip(bev_map * 255, 0, 255).astype(np.uint8)

    # If rotated, also rotate the image back for visualization
    if rotate_180:
        bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

    return bev_map


def draw_boxes_on_bev(bev_img, detections, cnf, rotate_180=False):
    """Draw detection boxes on BEV image."""
    if bev_img is None:
        return None

    # Convert grayscale to RGB for colored boxes
    if len(bev_img.shape) == 2:
        bev_img = cv2.cvtColor(bev_img, cv2.COLOR_GRAY2BGR)

    for line in detections:
        parts = line.strip().split()
        if len(parts) < 15:
            continue

        cls_name = parts[0]
        score = float(parts[-1])

        # Parse 3D box in camera coordinates
        h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
        x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])

        # Convert camera to lidar coordinates
        x_lidar = z_cam
        y_lidar = -x_cam

        # Apply rotation if requested
        if rotate_180:
            x_lidar = -x_lidar
            y_lidar = -y_lidar

        # Convert to BEV pixel coordinates
        x_bev = int((y_lidar - cnf.boundary['minY']) / cnf.bound_size_y * cnf.BEV_WIDTH)
        y_bev = int((x_lidar - cnf.boundary['minX']) / cnf.bound_size_x * cnf.BEV_HEIGHT)

        # Box dimensions in BEV
        l_bev = int(l / cnf.bound_size_x * cnf.BEV_HEIGHT)
        w_bev = int(w / cnf.bound_size_y * cnf.BEV_WIDTH)

        # Draw box
        if 0 <= x_bev < cnf.BEV_WIDTH and 0 <= y_bev < cnf.BEV_HEIGHT:
            # Color based on class
            if cls_name == 'Car':
                color = (0, 255, 0)  # Green
            elif cls_name == 'Pedestrian':
                color = (255, 0, 0)  # Blue
            elif cls_name == 'Cyclist':
                color = (0, 0, 255)  # Red
            else:
                color = (255, 255, 0)  # Cyan

            # Draw center point
            cv2.circle(bev_img, (x_bev, y_bev), 3, color, -1)

            # Draw box
            pt1 = (x_bev - w_bev // 2, y_bev - l_bev // 2)
            pt2 = (x_bev + w_bev // 2, y_bev + l_bev // 2)
            cv2.rectangle(bev_img, pt1, pt2, color, 2)

            # Draw score
            cv2.putText(bev_img, f'{cls_name[:3]}:{score:.2f}', (x_bev + 5, y_bev - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return bev_img


def visualize_sample(sample_id, dataset_dir, result_dir, output_dir=None, rotate_bev=False):
    """Visualize a single sample with 3D boxes and BEV rotation check."""
    # Paths
    bin_path = os.path.join(dataset_dir, 'testing', 'velodyne', f'{sample_id:06d}.bin')
    img_path = os.path.join(dataset_dir, 'testing', 'image_2', f'{sample_id:06d}.png')
    result_path = os.path.join(result_dir, f'{sample_id:06d}.txt')
    calib_path = os.path.join(dataset_dir, 'testing', 'calib', f'{sample_id:06d}.txt')

    print(f"\n{'='*80}")
    print(f"Sample {sample_id:06d} - BEV Rotation: {'ON' if rotate_bev else 'OFF'}")
    print(f"{'='*80}")

    # Load point cloud
    points = read_lidar_file_with_fallback(bin_path)
    print(f"Point cloud shape: {points.shape}")

    # Load detections
    detections = []
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            detections = f.readlines()
        print(f"Detections: {len(detections)}")

    # Load camera calibration
    calib = None
    if os.path.exists(calib_path):
        calib = Calibration(calib_path)

    # Load camera image and draw 3D boxes
    img_rgb = None
    img_with_boxes = None
    if os.path.exists(img_path):
        img_rgb = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        if calib is not None:
            img_with_boxes = draw_3d_boxes_on_image(img_rgb.copy(), detections, calib)

    # Create BEV map with optional rotation
    bev_map = create_bev_from_points(points, cnf, rotate_bev)
    bev_with_boxes = draw_boxes_on_bev(bev_map, detections, cnf, rotate_bev)

    # Create combined visualization
    if img_with_boxes is not None and bev_with_boxes is not None:
        # Resize images to same height
        h_img = img_with_boxes.shape[0]
        h_bev = bev_with_boxes.shape[0]

        if h_img != h_bev:
            scale = h_img / h_bev
            new_w = int(bev_with_boxes.shape[1] * scale)
            bev_with_boxes = cv2.resize(bev_with_boxes, (new_w, h_img))

        # Combine horizontally
        combined = np.hstack([img_with_boxes, bev_with_boxes])

        # Add title
        combined = cv2.copyMakeBorder(combined, 60, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Count detections by class
        class_counts = {}
        for det in detections:
            parts = det.strip().split()
            if len(parts) >= 1:
                cls_name = parts[0]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        det_str = ', '.join([f'{cls}:{cnt}' for cls, cnt in sorted(class_counts.items())])
        rotation_text = "BEV ROTATED 180°" if rotate_bev else "BEV NORMAL"
        title = f'Sample {sample_id:06d} - {rotation_text} - Total: {len(detections)} ({det_str})'

        cv2.putText(combined, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Add coordinate system labels for BEV
        if rotate_bev:
            cv2.putText(combined, "BEV: ↑ Forward, ← Left", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        else:
            cv2.putText(combined, "BEV: ↓ Forward, → Left", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Save or display
        suffix = "_rotated" if rotate_bev else ""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{sample_id:06d}_3d_boxes{suffix}.png')
            cv2.imwrite(output_path, combined)
            print(f"Saved to: {output_path}")
        else:
            cv2.imshow(f'Sample {sample_id:06d} {suffix}', combined)
            print(f"Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return len(detections)


def main():
    parser = argparse.ArgumentParser(description='Visualize with 3D boxes and BEV rotation check')
    parser.add_argument('--dataset-dir', type=str,
                       default=r'D:\bev\SFA3D-modified\DRadDataset',
                       help='Dataset root directory')
    parser.add_argument('--result-dir', type=str,
                       default=r'D:\bev\SFA3D-modified\ultra_aggressive_results',
                       help='Inference results directory')
    parser.add_argument('--output-dir', type=str,
                       default=r'D:\bev\SFA3D-modified\visualizations_3d_boxes',
                       help='Output directory for visualizations')
    parser.add_argument('--sample-ids', type=int, nargs='+',
                       default=[0, 1, 2, 10, 625],
                       help='Sample IDs to visualize')
    parser.add_argument('--rotate-bev', action='store_true',
                       help='Rotate BEV by 180 degrees')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("3D BOX VISUALIZATION WITH BEV ROTATION CHECK")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Results: {args.result_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Rotate BEV: {args.rotate_bev}")
    print(f"{'='*80}\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Visualize samples (both normal and rotated for comparison)
    for sample_id in args.sample_ids:
        print(f"\n--- Processing Sample {sample_id:06d} ---")

        # Visualize normal BEV
        visualize_sample(sample_id, args.dataset_dir, args.result_dir, args.output_dir, rotate_bev=False)

        # Visualize rotated BEV
        visualize_sample(sample_id, args.dataset_dir, args.result_dir, args.output_dir, rotate_bev=True)


if __name__ == '__main__':
    main()

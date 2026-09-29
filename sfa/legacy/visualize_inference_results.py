"""
可视化推理结果并验证雷达设备（0、1、2、4）是否正确加载。
"""
import argparse
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from config import kitti_config as cnf
from data_process.kitti_data_utils import Calibration


def load_point_cloud(bin_path):
    """Load point cloud from .bin file and check its structure."""
    points = np.fromfile(bin_path, dtype=np.float32)

    # Check different possible formats
    print(f"\nLoading: {bin_path}")
    print(f"Total values: {len(points)}")

    # Try to determine the format
    if len(points) % 8 == 0:
        # 8D radar format (x, y, z, intensity, vx, vy, device_id, timestamp)
        points = points.reshape(-1, 8)
        print(f"Format: 8D radar (x, y, z, intensity, vx, vy, device_id, timestamp)")
        print(f"Number of points: {points.shape[0]}")

        if points.shape[0] > 0:
            # Check device IDs
            device_ids = points[:, 6].astype(int)
            unique_devices = np.unique(device_ids)
            print(f"Unique device IDs: {unique_devices}")

            # Count points per device
            for dev_id in unique_devices:
                count = np.sum(device_ids == dev_id)
                print(f"  Device {dev_id}: {count} points")

            # Check if devices 0, 1, 2, 4 are present
            expected_devices = [0, 1, 2, 4]
            present_devices = [d for d in expected_devices if d in unique_devices]
            missing_devices = [d for d in expected_devices if d not in unique_devices]

            if missing_devices:
                print(f"\nWARNING: Missing devices: {missing_devices}")
            else:
                print(f"\nGOOD: All expected devices {expected_devices} are present!")

    elif len(points) % 4 == 0:
        points = points.reshape(-1, 4)
        print(f"Format: 4D LiDAR/Radar (x, y, z, intensity)")
        print(f"Number of points: {points.shape[0]}")
    else:
        print(f"Unknown format - cannot reshape evenly")
        return None

    return points


def create_bev_from_points(points, cnf):
    """Create BEV image from point cloud."""
    # Use only x, y, z, intensity (first 4 dimensions)
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
        return None

    # Create BEV image
    bev_height = cnf.BEV_HEIGHT
    bev_width = cnf.BEV_WIDTH

    # Convert to BEV coordinates
    x = xyz_i[:, 0]
    y = xyz_i[:, 1]
    z = xyz_i[:, 2]
    intensity = xyz_i[:, 3]

    # Normalize to BEV grid
    x_img = np.floor((y - y_lim[0]) / (y_lim[1] - y_lim[0]) * bev_width).astype(np.int32)
    y_img = np.floor((x - x_lim[0]) / (x_lim[1] - x_lim[0]) * bev_height).astype(np.int32)

    x_img = np.clip(x_img, 0, bev_width - 1)
    y_img = np.clip(y_img, 0, bev_height - 1)

    # Create BEV intensity map
    bev_map = np.zeros((bev_height, bev_width), dtype=np.float32)

    for i in range(len(x_img)):
        bev_map[y_img[i], x_img[i]] = max(bev_map[y_img[i], x_img[i]], intensity[i])

    return bev_map


def draw_boxes_on_bev(bev_img, detections, cnf):
    """Draw detection boxes on BEV image."""
    if bev_img is None:
        return None

    # Convert grayscale to RGB for colored boxes
    if len(bev_img.shape) == 2:
        bev_img = cv2.cvtColor((bev_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for line in detections:
        parts = line.strip().split()
        if len(parts) < 15:
            continue

        cls_name = parts[0]
        score = float(parts[-1])

        # Parse 3D box in camera coordinates
        h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
        x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])
        ry = float(parts[14])

        # Convert camera to lidar coordinates (simplified)
        # This is approximate - you may need calibration for exact conversion
        x_lidar = z_cam
        y_lidar = -x_cam

        # Convert to BEV pixel coordinates
        x_bev = int((y_lidar - cnf.boundary['minY']) / cnf.bound_size_y * cnf.BEV_WIDTH)
        y_bev = int((x_lidar - cnf.boundary['minX']) / cnf.bound_size_x * cnf.BEV_HEIGHT)

        # Box dimensions in BEV
        l_bev = int(l / cnf.bound_size_x * cnf.BEV_HEIGHT)
        w_bev = int(w / cnf.bound_size_y * cnf.BEV_WIDTH)

        # Draw box center
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

            # Draw box (simplified as rectangle)
            pt1 = (x_bev - w_bev // 2, y_bev - l_bev // 2)
            pt2 = (x_bev + w_bev // 2, y_bev + l_bev // 2)
            cv2.rectangle(bev_img, pt1, pt2, color, 1)

            # Draw score
            cv2.putText(bev_img, f'{score:.2f}', (x_bev + 5, y_bev - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return bev_img


def visualize_sample(sample_id, dataset_dir, result_dir):
    """Visualize a single sample with detections."""
    # Paths
    bin_path = os.path.join(dataset_dir, 'testing', 'velodyne', f'{sample_id:06d}.bin')
    img_path = os.path.join(dataset_dir, 'testing', 'image_2', f'{sample_id:06d}.png')
    result_path = os.path.join(result_dir, f'{sample_id:06d}.txt')

    print(f"\n{'='*80}")
    print(f"Visualizing Sample {sample_id:06d}")
    print(f"{'='*80}")

    # Load point cloud and check devices
    points = load_point_cloud(bin_path)
    if points is None:
        print("Failed to load point cloud")
        return

    # Load detections
    detections = []
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            detections = f.readlines()
        print(f"\nDetections: {len(detections)}")
        for det in detections:
            parts = det.strip().split()
            if len(parts) >= 15:
                print(f"  {parts[0]} - score: {parts[-1]}")

    # Create BEV map
    bev_map = create_bev_from_points(points, cnf)

    # Draw detections on BEV
    bev_with_boxes = draw_boxes_on_bev(bev_map, detections, cnf)

    # Load image
    img_rgb = None
    if os.path.exists(img_path):
        img_rgb = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    # Plot
    fig = plt.figure(figsize=(16, 8))

    if img_rgb is not None:
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title(f'Camera Image - Sample {sample_id:06d}')
        plt.axis('off')

    if bev_with_boxes is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(bev_with_boxes)
        plt.title(f'BEV with Detections - {len(detections)} objects')
        plt.xlabel('Y (lateral)')
        plt.ylabel('X (forward)')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Car'),
            Patch(facecolor='blue', label='Pedestrian'),
            Patch(facecolor='red', label='Cyclist')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize inference results')
    parser.add_argument('--dataset-dir', type=str,
                       default=r'D:\bev\SFA3D-modified\DRadDataset',
                       help='Dataset root directory')
    parser.add_argument('--result-dir', type=str,
                       default=r'D:\bev\SFA3D-modified\超激进P',
                       help='Inference results directory')
    parser.add_argument('--sample-ids', type=int, nargs='+',
                       default=[0, 1, 10, 100, 625],
                       help='Sample IDs to visualize')
    parser.add_argument('--check-all', action='store_true',
                       help='Check all samples for device IDs')

    args = parser.parse_args()

    if args.check_all:
        print("\nChecking device IDs in all samples...")
        velodyne_dir = os.path.join(args.dataset_dir, 'testing', 'velodyne')
        bin_files = sorted(Path(velodyne_dir).glob('*.bin'))

        all_devices = set()
        for i, bin_file in enumerate(bin_files[:10]):  # Check first 10
            points = load_point_cloud(str(bin_file))
            if points is not None and points.shape[1] >= 7:
                device_ids = points[:, 6].astype(int)
                all_devices.update(np.unique(device_ids))

        print(f"\nDevices found across {min(10, len(bin_files))} samples: {sorted(all_devices)}")
        expected = [0, 1, 2, 4]
        if set(expected).issubset(all_devices):
            print(f"SUCCESS: All expected devices {expected} are present!")
        else:
            missing = set(expected) - all_devices
            print(f"WARNING: Missing devices: {sorted(missing)}")

    # Visualize specific samples
    for sample_id in args.sample_ids:
        visualize_sample(sample_id, args.dataset_dir, args.result_dir)


if __name__ == '__main__':
    main()

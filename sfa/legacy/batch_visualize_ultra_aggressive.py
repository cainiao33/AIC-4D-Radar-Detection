"""
批量可视化超激进推理结果。
处理所有测试样本并生成可视化图像。
"""
import argparse
import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from config import kitti_config as cnf
from data_process.lidar_mapping import read_lidar_file_with_fallback


def create_bev_from_points(points, cnf):
    """Create BEV image from point cloud."""
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
    return bev_map


def draw_boxes_on_bev(bev_img, detections, cnf):
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

        # Convert camera to lidar coordinates (simplified)
        x_lidar = z_cam
        y_lidar = -x_cam

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


def process_sample(sample_id, dataset_dir, result_dir, output_dir):
    """Process a single sample and save visualization."""
    # Paths
    bin_path = os.path.join(dataset_dir, 'testing', 'velodyne', f'{sample_id:06d}.bin')
    img_path = os.path.join(dataset_dir, 'testing', 'image_2', f'{sample_id:06d}.png')
    result_path = os.path.join(result_dir, f'{sample_id:06d}.txt')

    # Check if files exist
    if not os.path.exists(bin_path):
        return None

    # Load point cloud
    try:
        points = read_lidar_file_with_fallback(bin_path)
    except Exception as e:
        print(f"Error loading {bin_path}: {e}")
        return None

    # Load detections
    detections = []
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            detections = f.readlines()

    # Create BEV map
    bev_map = create_bev_from_points(points, cnf)

    # Draw detections on BEV
    bev_with_boxes = draw_boxes_on_bev(bev_map, detections, cnf)

    # Load camera image
    img_rgb = None
    if os.path.exists(img_path):
        img_rgb = cv2.imread(img_path)

    # Create combined visualization
    if img_rgb is not None and bev_with_boxes is not None:
        # Resize images to same height
        h_img = img_rgb.shape[0]
        h_bev = bev_with_boxes.shape[0]

        if h_img != h_bev:
            scale = h_img / h_bev
            new_w = int(bev_with_boxes.shape[1] * scale)
            bev_with_boxes = cv2.resize(bev_with_boxes, (new_w, h_img))

        # Combine horizontally
        combined = np.hstack([img_rgb, bev_with_boxes])

        # Add title
        combined = cv2.copyMakeBorder(combined, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Count detections by class
        class_counts = {}
        for det in detections:
            parts = det.strip().split()
            if len(parts) >= 1:
                cls_name = parts[0]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        det_str = ', '.join([f'{cls}:{cnt}' for cls, cnt in sorted(class_counts.items())])
        title = f'Sample {sample_id:06d} - Total: {len(detections)} ({det_str})'

        cv2.putText(combined, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Save
        output_path = os.path.join(output_dir, f'{sample_id:06d}_vis.png')
        cv2.imwrite(output_path, combined)

        return len(detections)

    return None


def main():
    parser = argparse.ArgumentParser(description='Batch visualize ultra-aggressive inference results')
    parser.add_argument('--dataset-dir', type=str,
                       default=r'D:\bev\SFA3D-modified\DRadDataset',
                       help='Dataset root directory')
    parser.add_argument('--result-dir', type=str,
                       default=r'D:\bev\SFA3D-modified\ultra_aggressive_results',
                       help='Inference results directory')
    parser.add_argument('--output-dir', type=str,
                       default=r'D:\bev\SFA3D-modified\ultra_aggressive_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Process only first N samples (default: all)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("BATCH VISUALIZATION - ULTRA-AGGRESSIVE DETECTIONS")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Results: {args.result_dir}")
    print(f"Output:  {args.output_dir}")
    print(f"{'='*80}\n")

    # Get all velodyne files
    velodyne_dir = os.path.join(args.dataset_dir, 'testing', 'velodyne')
    bin_files = sorted(Path(velodyne_dir).glob('*.bin'))

    if args.num_samples:
        bin_files = bin_files[:args.num_samples]

    print(f"Processing {len(bin_files)} samples...\n")

    # Process all samples
    t_start = time.time()
    total_detections = 0
    processed = 0

    for bin_file in tqdm(bin_files, desc="Generating visualizations"):
        sample_id = int(bin_file.stem)
        num_det = process_sample(sample_id, args.dataset_dir, args.result_dir, args.output_dir)

        if num_det is not None:
            total_detections += num_det
            processed += 1

    elapsed = time.time() - t_start

    print(f"\n{'='*80}")
    print(f"COMPLETED")
    print(f"{'='*80}")
    print(f"Processed samples: {processed}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per sample: {total_detections / max(processed, 1):.2f}")
    print(f"Time elapsed: {elapsed:.2f}s ({processed / max(elapsed, 1e-9):.2f} samples/s)")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Write OUTPUT_DIR/trajectory_path.ply (camera path coloured by video time: blue = start, red = end) and
trajectory_points.ply (sparse points, grey). Open both in CloudCompare or MeshLab to see whether the walk is one
consistent path.

Usage: python3 export_trajectory.py OUTPUT_DIR
"""
import json
import math
from pathlib import Path
import struct
import sys

from diagnose_frames import clock, frame_face


def camera_centers(images_bin):
    """{name: (x, y, z)} camera centres from COLMAP images.bin."""
    centers = {}
    with images_bin.open('rb') as src:
        (n,) = struct.unpack('<Q', src.read(8))
        for _ in range(n):
            _, qw, qx, qy, qz, tx, ty, tz, _ = struct.unpack('<i7di', src.read(64))
            name = bytearray()
            while (c := src.read(1)) != b'\0':
                name += c
            (points,) = struct.unpack('<Q', src.read(8))
            src.seek(points * 24, 1)
            # centre = -R^T t, R from the (w, x, y, z) quaternion
            r = ((1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
                 (2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)),
                 (2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)))
            centers[name.decode()] = tuple(-sum(r[i][k] * t for i, t in enumerate((tx, ty, tz))) for k in range(3))
    return centers


def sparse_points(points_bin):
    with points_bin.open('rb') as src:
        (n,) = struct.unpack('<Q', src.read(8))
        for _ in range(n):
            _, x, y, z, *_ = struct.unpack('<Q3d3Bd', src.read(43))
            (track,) = struct.unpack('<Q', src.read(8))
            src.seek(track * 8, 1)
            yield x, y, z


def time_colour(f):
    return int(255 * f), int(255 * (1 - abs(2 * f - 1))), int(255 * (1 - f))


def main(root):
    model = root / json.loads((root / 'selected_model.json').read_text(encoding='utf-8'))['path']
    path = {}
    for name, c in camera_centers(model / 'images.bin').items():
        if (ff := frame_face(name)) and ff[0] not in path:
            path[ff[0]] = c  # rig faces share one centre; keep one per time sample
    samples = sorted(path)
    span = max(samples[-1] - samples[0], 1)
    # Fill each step between consecutive samples so the path reads as a line, not scattered dots.
    line = [(*(a + (b - a) * k / STEPS for a, b in zip(path[s], path[t])),
             *time_colour((s - samples[0] + k / STEPS * (t - s)) / span))
            for s, t in zip(samples, samples[1:]) for k in range(STEPS)]
    points = [(*p, 128, 128, 128) for p in sparse_points(model / 'points3D.bin')]
    for name, vertices in (('trajectory_path.ply', line), ('trajectory_points.ply', points)):
        write_ply(root / name, vertices)
    print(f'{root / "trajectory_path.ply"}: カメラ経路 時間サンプル {len(samples)} 個 (青=開始 → 緑 → 赤=終了)')
    print(f'{root / "trajectory_points.ply"}: 疎な点 {len(points)} 個 (灰色)')
    # A walk moves a similar distance every sample; a step far above the median is a jump in the reconstruction.
    config = json.loads((root / 'state.json').read_text(encoding='utf-8'))['config']
    fps, start = float(config['fps']), float(config['start'])
    steps = sorted(((math.dist(path[s], path[t]), s) for s, t in zip(samples, samples[1:])), reverse=True)
    median = steps[len(steps) // 2][0] or 1e-9
    print(f'\n1サンプルあたりの移動量 (中央値 {median:.3f}) が大きい順:')
    for d, s in steps[:10]:
        print(f'  {clock(start + (s - 1) / fps)} → {clock(start + s / fps)}: {d:9.3f}  (中央値の {d / median:6.1f} 倍)')


STEPS = 20


def write_ply(out, vertices):
    with out.open('wb') as dst:
        dst.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\n'
                   'property float x\nproperty float y\nproperty float z\n'
                   'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
        for v in vertices:
            dst.write(struct.pack('<3f3B', *v))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))

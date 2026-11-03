#!/usr/bin/env python3
"""Show how many frames were extracted and which video time ranges each SfM component covers.

Usage: python3 diagnose_frames.py OUTPUT_DIR
"""
import json
from pathlib import Path
import re
import sqlite3
import struct
import sys


def registered_names(images_bin):
    names = []
    with images_bin.open('rb') as src:
        (n,) = struct.unpack('<Q', src.read(8))
        for _ in range(n):
            src.read(4 + 8 * 4 + 8 * 3 + 4)  # image_id, qvec, tvec, camera_id
            name = bytearray()
            while (c := src.read(1)) != b'\0':
                name += c
            names.append(name.decode())
            (points,) = struct.unpack('<Q', src.read(8))
            src.seek(points * 24, 1)
    return names


def frame_face(name):
    """Return (time sample, direction) for v3 'cam3/00000007.jpg' or v2 '00000007_3.jpg' names."""
    if m := re.fullmatch(r'cam(\d)/(\d+)\.jpg', name):
        return int(m.group(2)), int(m.group(1))
    if m := re.fullmatch(r'(\d+)_(\d)\.jpg', name):
        return int(m.group(1)), int(m.group(2))
    return None


def ranges(samples):
    out, start, prev = [], None, None
    for s in sorted(samples):
        if start is None:
            start = prev = s
        elif s == prev + 1:
            prev = s
        else:
            out.append((start, prev))
            start = prev = s
    if start is not None:
        out.append((start, prev))
    return out


MAX_IMAGES = 2147483647  # COLMAP pair_id = image_id1 * MAX_IMAGES + image_id2
MIN_MATCHES = 15  # Mapper.min_num_matches default


def cut_links(db):
    """Per cut t|t+1: (verified pairs spanning it, inlier total) from database.db."""
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    try:
        sample = {i: ff[0] for i, n in con.execute('SELECT image_id, name FROM images') if (ff := frame_face(n))}
        links = {}
        for pair_id, rows in con.execute('SELECT pair_id, rows FROM two_view_geometries WHERE config >= 2 AND rows >= ?',
                                         (MIN_MATCHES,)):
            a, b = sorted((sample.get(pair_id // MAX_IMAGES), sample.get(pair_id % MAX_IMAGES)), key=lambda x: x or 0)
            if a is None or a == b:
                continue
            for t in range(a, b):
                pairs, inliers = links.get(t, (0, 0))
                links[t] = (pairs + 1, inliers + rows)
        return links
    finally:
        con.close()


def clock(seconds):
    return f'{int(seconds // 60):02d}:{seconds % 60:04.1f}'


def main(root):
    config = json.loads((root / 'state.json').read_text(encoding='utf-8'))['config']
    fps, start = float(config['fps']), float(config['start'])
    samples = {}
    for p in (root / 'images').rglob('*.jpg'):
        if ff := frame_face(p.relative_to(root / 'images').as_posix()):
            samples.setdefault(ff[0], set()).add(ff[1])
    if not samples:
        print('images/ に画像がありません。')
        return
    print(f'抽出: 時間サンプル {len(samples)} 個 (#{min(samples)}..#{max(samples)}), '
          f'画像 {sum(len(v) for v in samples.values())} 枚')
    print(f'      動画上の範囲 {clock(start)} .. {clock(start + (max(samples) - 1) / fps)}')
    selected = json.loads((root / 'selected_model.json').read_text(encoding='utf-8'))
    chosen = (root / selected['path']).resolve()
    ends = set()
    for comp in sorted(chosen.parent.iterdir()):
        if not (comp / 'images.bin').is_file():
            continue
        names = registered_names(comp / 'images.bin')
        idx = {ff[0] for n in names if (ff := frame_face(n))}
        mark = ' <= 採用' if comp.resolve() == chosen else ''
        print(f'\n成分 {comp.name}: 登録画像 {len(names)} 枚 / 時間サンプル {len(idx)} 個{mark}')
        for a, b in ranges(idx):
            if b - a >= 4:
                print(f'  {clock(start + (a - 1) / fps)} .. {clock(start + (b - 1) / fps)}  ({b - a + 1} サンプル)')
                ends.add(b)
    if not (root / 'database.db').is_file():
        return
    links = cut_links(root / 'database.db')
    last = max(samples)
    # Cuts near either end of the video are spanned by fewer pairs by construction; leave them out of the ranking.
    inner = range(min(samples) + 8, last - 8) or range(min(samples), last)
    weakest = sorted(inner, key=lambda t: links.get(t, (0, 0)))[:10]
    typical = sorted(links.get(t, (0, 0))[0] for t in inner)[len(inner) // 2]
    print(f'\n時刻の切れ目をまたぐ照合 ({MIN_MATCHES} 点以上一致した画像ペア数 / 一致点の合計, 中央値 {typical} ペア)')
    for t in sorted(set(weakest) | (ends - {last})):
        pairs, inliers = links.get(t, (0, 0))
        tags = [x for x, on in (('成分の境目', t in ends), ('弱い順上位10', t in weakest)) if on]
        print(f'  {clock(start + (t - 1) / fps)} | {clock(start + t / fps)}: {pairs:5d} ペア / {inliers:7d} 点  '
              f'[{", ".join(tags)}]')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))

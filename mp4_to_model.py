#!/usr/bin/env python3
"""Stitched equirectangular MP4 -> textured OBJ (Linux, FFmpeg/COLMAP/OpenMVS)."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
import zlib

FACES = ((0, 0), (90, 0), (180, 0), (-90, 0), (0, 90), (0, -90))


def save(path, data):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def require(paths):
    paths = list(paths)
    if not paths:
        raise RuntimeError('No output files produced.')
    for p in paths:
        if not p.is_file() or not p.stat().st_size:
            raise RuntimeError(f'Missing or empty output: {p}')
    return paths


def identity(path):
    h = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b''):
            h.update(block)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def count(path):
    with path.open('rb') as src:
        data = src.read(8)
    if len(data) != 8:
        raise RuntimeError(f'Invalid COLMAP model: {path}')
    return struct.unpack('<Q', data)[0]


def schema(config, version):
    """`config` as written by an older version: drop keys added later, set that version."""
    added = {'sfm': 4, 'loop_images': 5}
    return {**{k: v for k, v in config.items() if version >= added.get(k, 0)}, 'version': version}


def migrate_v1_state(root, state, config):
    """Reuse v1 outputs only when input and all processing settings match."""
    old = state['config']
    if (old.get('version') != 1 or 'decode' in old or
            config.get('version') != 2 or config.get('decode') != 'auto' or
            {**old, 'version': 2, 'decode': 'auto'} != config or
            '01_frames' not in state['completed']):
        return False
    require(root / x for x in state['completed']['01_frames'])
    info = root / 'decode.json'
    if info.exists():
        json.loads(info.read_text(encoding='utf-8'))
    backup = root / f'state.before-v2-{time.time_ns()}.json'
    shutil.copy2(root / 'state.json', backup)
    if not info.exists():
        save(info, {'mode': 'legacy-cache', 'gpu': None, 'single_decode': None,
                    'fallback_reason': 'Reusing v1 frames; decoder was not recorded.'})
    state['config'] = dict(config)
    if 'decode.json' not in state['completed']['01_frames']:
        state['completed']['01_frames'].append('decode.json')
    save(root / 'state.json', state)
    print(f'Migrated compatible v1 checkpoint; backup: {backup.name}', flush=True)
    return True


V2_FRAME = re.compile(r'images/(\d+)_(\d)\.jpg')
V3_KEEP = {'images', 'logs', 'decode.json', 'video.json', 'state.json', '.lock', 'v2-backup'}
MATCH_STAGES = ('01_frames', '02_features', '02_rig', '03_matches')


def migrate_v2_state(root, state, config):
    """Reuse v2 frames in the v3 per-camera layout; set aside all later v2 outputs."""
    old, target = state['config'], schema(config, 3)
    if (old.get('version') != 2 or config.get('version', 0) < 3 or
            {**old, 'version': 3} != target or '01_frames' not in state['completed']):
        return False
    frames = []
    for name in state['completed']['01_frames']:
        m = V2_FRAME.fullmatch(name)
        if not m:
            frames.append(name)
            continue
        new = f'images/cam{m.group(2)}/{m.group(1)}.jpg'
        if (root / name).exists():
            (root / new).parent.mkdir(parents=True, exist_ok=True)
            (root / name).replace(root / new)
        frames.append(new)
    require(root / x for x in frames)
    backup = root / 'v2-backup'
    backup.mkdir(exist_ok=True)
    shutil.copy2(root / 'state.json', backup / 'state.json')
    for p in root.iterdir():
        if p.name not in V3_KEEP and not p.name.startswith('state.before-'):
            p.replace(backup / p.name)
    state['config'] = target
    state['completed'] = {'01_frames': frames}
    save(root / 'state.json', state)
    print('Migrated v2 frames to the per-camera rig layout; earlier SfM/dense/mesh outputs moved to '
          'v2-backup/ (safe to delete to free disk space).', flush=True)
    return True


def restart_after_matches(root, state, config, version):
    """Upgrade a `version` job by one: keep frames, features and sequential matches, redo the rest."""
    if (state['config'].get('version') != version or config.get('version', 0) <= version or
            schema(config, version) != state['config']):
        return False
    kept = {k: v for k, v in state['completed'].items() if k in MATCH_STAGES}
    for outputs in kept.values():
        require(root / x for x in outputs)
    keep = {Path(x).parts[0] for outputs in kept.values() for x in outputs}
    keep |= {'logs', 'video.json', 'state.json', '.lock'}
    moved = [p for p in root.iterdir() if p.name not in keep and not p.name.startswith('state.before-')
             and not re.fullmatch(r'v\d+-backup', p.name)]
    if moved:
        backup = root / f'v{version}-backup'
        backup.mkdir(exist_ok=True)
        shutil.copy2(root / 'state.json', backup / 'state.json')
        for p in moved:
            p.replace(backup / p.name)
        print(f'Kept v{version} frames, features and matches; earlier SfM/dense/mesh outputs moved to '
              f'{backup.name}/ (safe to delete to free disk space).', flush=True)
    state['config'] = schema(config, version + 1)
    state['completed'] = kept
    save(root / 'state.json', state)
    return True


# Settings used only from a late stage on, with the first stage each affects. Changing one on
# --resume redoes that stage and everything after it instead of refusing the job.
RESTAGE = {'target_faces': '09_', 'free_space_support': '09_', 'close_holes': '09_', 'floor': '09b_',
           'texture_size': '10_'}


def restage(state, config):
    changed = [k for k in RESTAGE if k in config and state['config'].get(k) != config[k]]
    if changed:
        first = min(RESTAGE[k] for k in changed)
        state['completed'] = {k: v for k, v in state['completed'].items() if k < first}
        state['config'].update({k: config[k] for k in changed})
    return changed


class Job:
    def __init__(self, root, config, resume=False):
        self.root, self.state_path = root, root / 'state.json'
        if self.state_path.exists():
            if not resume:
                raise RuntimeError('Job exists. Use --resume or another output directory.')
            self.state = json.loads(self.state_path.read_text(encoding='utf-8'))
            # Thread count only changes speed (e.g. after adding CPUs); older jobs recorded it.
            self.state['config'].pop('threads', None)
            restaged = restage(self.state, config)
            if self.state['config'] != config:
                # Each step upgrades one version, so older checkpoints chain through to the current one.
                migrate_v1_state(root, self.state, config) or migrate_v2_state(root, self.state, config)
                for version in (3, 4):
                    restart_after_matches(root, self.state, config, version)
                if self.state['config'] != config:
                    old = self.state['config']
                    diff = ', '.join(f'{k}: {old.get(k)!r} -> {config.get(k)!r}'
                                     for k in sorted(old.keys() | config.keys()) if old.get(k) != config.get(k))
                    raise RuntimeError(f'Input/settings changed ({diff}). Choose a new output directory.')
            if restaged:
                print(f'Settings changed ({", ".join(restaged)}); redoing the stages they affect.', flush=True)
                save(self.state_path, self.state)
        else:
            if any(p.name != '.lock' for p in root.iterdir()):
                raise RuntimeError('New job requires an empty output directory.')
            self.state = {'config': config, 'completed': {}}
            save(self.state_path, self.state)
        (root / 'logs').mkdir(exist_ok=True)

    def run(self, name, command):
        command = [str(x) for x in command]
        print(f'[{name}] {shlex.join(command)}', flush=True)
        log = self.root / 'logs' / f'{name}.log'
        with log.open('a', encoding='utf-8') as dst:
            dst.write(f'\n{time.ctime()} {shlex.join(command)}\n')
            dst.flush()
            result = subprocess.run(command, cwd=self.root, stdout=dst, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, QT_QPA_PLATFORM='offscreen'))
        if result.returncode:
            raise RuntimeError(f'{name}: exit {result.returncode}. See {log}')

    def stage(self, name, action):
        if name in self.state['completed']:
            require(self.root / x for x in self.state['completed'][name])
            print(f'[{name}] completed; skipping', flush=True)
            return
        outputs = require(action())
        self.state['completed'][name] = [str(p.relative_to(self.root)) for p in outputs]
        save(self.state_path, self.state)

    def command(self, name, command, outputs):
        def action():
            self.run(name, command)
            return outputs() if callable(outputs) else outputs
        self.stage(name, action)


@contextlib.contextmanager
def job_lock(root):
    import fcntl
    with (root / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another process is using this output directory.') from None
        yield


def preflight(gpu_spec):
    for name in ('ffmpeg', 'ffprobe', 'colmap', 'InterfaceCOLMAP', 'ReconstructMesh', 'TextureMesh', 'nvidia-smi'):
        if not shutil.which(name):
            raise RuntimeError(f'{name} not found. Use the supplied Docker image.')
    result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                            text=True, capture_output=True, check=True)
    available = {int(x.strip()) for x in result.stdout.splitlines() if x.strip()}
    if not available:
        raise RuntimeError('No NVIDIA GPU detected.')
    gpus = sorted(available) if gpu_spec == 'all' else [int(x) for x in gpu_spec.split(',')]
    if not set(gpus) <= available:
        raise RuntimeError(f'GPUs {gpus} unavailable; detected: {sorted(available)}')
    for command, option in (('feature_extractor', 'FeatureExtraction.gpu_index'),
                            ('sequential_matcher', 'FeatureMatching.gpu_index'),
                            ('patch_match_stereo', 'PatchMatchStereo.gpu_index'),
                            ('mapper', 'Mapper.ba_use_gpu'),
                            ('global_mapper', 'GlobalMapper.gp_use_gpu'),
                            ('vocab_tree_matcher', 'VocabTreeMatching.num_images')):
        result = subprocess.run(['colmap', command, '-h'], text=True, capture_output=True)
        if option not in result.stdout + result.stderr:
            raise RuntimeError(f'Unsupported COLMAP CLI: {option}. Supplied image uses 4.2.1.')
    result = subprocess.run(['ffmpeg', '-hide_banner', '-filters'], text=True, capture_output=True, check=True)
    if 'v360' not in result.stdout:
        raise RuntimeError('FFmpeg v360 filter is required.')
    return gpus


def probe(video):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                             '-show_entries', 'stream=width,height,codec_name,pix_fmt:format=duration',
                             '-of', 'json', str(video)], text=True, capture_output=True, check=True)
    data = json.loads(result.stdout)
    if not data.get('streams'):
        raise RuntimeError('No video stream found.')
    width, height = data['streams'][0]['width'], data['streams'][0]['height']
    if abs(width / height - 2) > .02:
        raise RuntimeError(f'Expected stitched 2:1 equirectangular video; got {width}x{height}.')
    duration = float(data['format']['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError('Invalid video duration.')
    return {'width': width, 'height': height, 'duration': duration,
            'codec': data['streams'][0].get('codec_name', ''),
            'pix_fmt': data['streams'][0].get('pix_fmt', '')}


def texture_outputs(root):
    obj, mtl = root / 'model.obj', root / 'model.mtl'
    require([obj, mtl])
    vertices = faces = 0
    with obj.open(encoding='utf-8') as src:
        for line in src:
            vertices += line.startswith('v ')
            faces += line.startswith('f ')
    if not vertices or not faces:
        raise RuntimeError('OBJ contains no mesh geometry.')
    textures = []
    for line in mtl.read_text(encoding='utf-8').splitlines():
        if line.startswith('map_Kd '):
            p = (root / line[7:].strip()).resolve()
            if not p.is_relative_to(root.resolve()):
                raise RuntimeError('Texture reference is outside the output directory.')
            textures.append(p)
    return [obj, mtl, *require(textures)], vertices, faces


def opaque_materials(root):
    # OpenMVS writes "Tr 1.000000" (fully transparent). Unity/Blender then import the
    # material as transparent with alpha 0, so the texture renders black or invisible.
    mtl = root / 'model.mtl'
    lines = mtl.read_text(encoding='utf-8').splitlines()
    fixed = []
    for line in lines:
        if line.split()[:1] in (['Tr'], ['d']):
            continue
        fixed.append(line)
        if line.startswith('newmtl '):
            fixed.append('d 1.000000')
    if fixed != lines:
        tmp = mtl.with_suffix('.mtl.tmp')
        tmp.write_text('\n'.join(fixed) + '\n', encoding='utf-8')
        tmp.replace(mtl)


def quat_mul(a, b):
    return (a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[1] * b[0] + a[0] * b[1] + a[2] * b[3] - a[3] * b[2],
            a[2] * b[0] + a[0] * b[2] + a[3] * b[1] - a[1] * b[3],
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1])


def rig_config(faces):
    # FFmpeg v360 maps a camera ray v to the panorama direction q v q*, q = qy(yaw) qx(pitch),
    # in x-right/y-down/z-forward axes (the COLMAP camera convention). With cam0 (yaw=pitch=0)
    # as the rig frame, cam_i_from_rig is the inverse rotation conj(q_i); all share one center.
    cameras = [{'image_prefix': 'cam0/', 'ref_sensor': True}]
    for i, (yaw, pitch) in enumerate(FACES[1:faces], 1):
        y, p = math.radians(yaw) / 2, math.radians(pitch) / 2
        w, x, yy, z = quat_mul((math.cos(y), 0, math.sin(y), 0), (math.cos(p), math.sin(p), 0, 0))
        cameras.append({'image_prefix': f'cam{i}/', 'cam_from_rig_rotation': [w, -x, -yy, -z],
                        'cam_from_rig_translation': [0, 0, 0]})
    return [{'cameras': cameras}]


def png(width, height, value):
    """Uniform 8-bit greyscale PNG."""
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress((b'\0' + bytes([value]) * width) * height, 9)) + chunk(b'IEND', b''))


def fusion_masks(dense, size):
    # The downward face mostly shows the photographer, who moves with the camera; at indoor
    # distances that body is fused as floating, smeared geometry. Mask that face out of fusion
    # (0 = ignored); the floor is still seen obliquely by the horizontal faces.
    down = f'cam{FACES.index((0, -90))}/'
    data = {v: png(size, size, v) for v in (0, 255)}
    masks = []
    for image in sorted((dense / 'images').rglob('*.jpg')):
        name = image.relative_to(dense / 'images').as_posix()
        mask = dense / 'masks' / f'{name}.png'
        mask.parent.mkdir(parents=True, exist_ok=True)
        mask.write_bytes(data[0 if name.startswith(down) else 255])
        masks.append(mask)
    return masks


def camera_poses(images_bin):
    """{name: (rotation rows, centre)} from COLMAP images.bin."""
    poses = {}
    with images_bin.open('rb') as src:
        (n,) = struct.unpack('<Q', src.read(8))
        for _ in range(n):
            _, qw, qx, qy, qz, tx, ty, tz, _ = struct.unpack('<i7di', src.read(64))
            name = bytearray()
            while (c := src.read(1)) != b'\0':
                if not c:
                    raise RuntimeError(f'Invalid COLMAP model: {images_bin}')
                name += c
            (points,) = struct.unpack('<Q', src.read(8))
            src.seek(points * 24, 1)
            r = ((1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
                 (2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)),
                 (2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)))
            # centre = -R^T t
            poses[name.decode()] = r, tuple(-sum(r[i][k] * t for i, t in enumerate((tx, ty, tz))) for k in range(3))
    return poses


PLY_TYPES = {'char': 'b', 'int8': 'b', 'uchar': 'B', 'uint8': 'B', 'short': 'h', 'int16': 'h',
             'ushort': 'H', 'uint16': 'H', 'int': 'i', 'int32': 'i', 'uint': 'I', 'uint32': 'I',
             'float': 'f', 'float32': 'f', 'double': 'd', 'float64': 'd'}


def ply_vertices(path):
    """(header, vertex struct, vertex rows, xyz columns, bytes after the vertex block) of a binary PLY."""
    data = path.read_bytes()
    end = data.find(b'end_header\n')
    if not data.startswith(b'ply\n') or end < 0:
        raise RuntimeError(f'Not a PLY file: {path}')
    end += len(b'end_header\n')
    lines = data[:end].decode('ascii').splitlines()
    if 'format binary_little_endian 1.0' not in lines:
        raise RuntimeError(f'Expected a binary little-endian PLY: {path}')
    elements = [line.split() for line in lines if line.startswith(('element ', 'property '))]
    if not elements or elements[0][:2] != ['element', 'vertex']:
        raise RuntimeError(f'PLY must start with the vertex element: {path}')
    n, props = int(elements[0][2]), []
    for words in elements[1:]:
        if words[0] == 'element':
            break
        if words[1] == 'list' or words[1] not in PLY_TYPES:
            raise RuntimeError(f'Unsupported vertex property in {path}: {" ".join(words)}')
        props.append(words)
    names = [w[2] for w in props]
    row = struct.Struct('<' + ''.join(PLY_TYPES[w[1]] for w in props))
    block = data[end:end + n * row.size]
    if len(block) != n * row.size:
        raise RuntimeError(f'Truncated PLY: {path}')
    return data[:end], row, [list(v) for v in row.iter_unpack(block)], [names.index(c) for c in 'xyz'], data[end + len(block):]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def unit(v):
    n = math.sqrt(dot(v, v))
    if not n:
        raise RuntimeError('Degenerate direction while estimating the floor.')
    return tuple(x / n for x in v)


def flatten_floor(images_bin, mesh, out):
    """Snap mesh vertices that sank below the floor (mirror images in glossy floors, noisy
    depth at grazing angles) up onto it. The floor is found from the walk itself: up is the
    level front camera's -y axis, and the floor lies a constant eye height below the cameras."""
    poses = camera_poses(images_bin)
    front = [poses[k] for k in sorted(poses) if k.startswith('cam0/')]
    if len(front) < 2:
        raise RuntimeError('Floor estimate needs at least two registered front (cam0) views.')
    up = unit([-sum(r[1][k] for r, _ in front) for k in range(3)])
    side = unit(next(c for c in ((up[1] * a[2] - up[2] * a[1], up[2] * a[0] - up[0] * a[2], up[0] * a[1] - up[1] * a[0])
                                 for a in ((1, 0, 0), (0, 1, 0))) if dot(c, c) > .1))
    ahead = (up[1] * side[2] - up[2] * side[1], up[2] * side[0] - up[0] * side[2], up[0] * side[1] - up[1] * side[0])
    def plan(p):
        return dot(p, side), dot(p, ahead)
    walk = [(*plan(c), dot(c, up)) for _, c in front]
    steps = sorted(math.dist(a[:2], b[:2]) for a, b in zip(walk, walk[1:]))
    # About 10 samples of walking per cell: fine enough for stairs, coarse enough to average out bobbing.
    cell = max(10 * steps[len(steps) // 2], 1e-6 * max(1, *(abs(x) for w in walk for x in w)))
    heights = {}
    for x, y, h in walk:
        heights.setdefault((math.floor(x / cell), math.floor(y / cell)), []).append(h)
    heights = {k: sorted(v)[len(v) // 2] for k, v in heights.items()}
    nearest = {}
    def camera_height(key):
        """(median camera height of the nearest walked cell, distance in cells)."""
        if key not in nearest:
            for ring in range(64):
                found = [(math.dist(key, k), heights[k]) for k in
                         ((key[0] + dx, key[1] + dy) for dx in range(-ring, ring + 1) for dy in range(-ring, ring + 1)
                          if max(abs(dx), abs(dy)) == ring) if k in heights]
                if found:
                    d, h = min(found)
                    nearest[key] = h, d
                    break
            else:
                nearest[key] = None
        return nearest[key]

    header, row, vertices, xyz, rest = ply_vertices(mesh)
    below = []  # (camera height above vertex, vertex index, cells to the walk)
    for i, v in enumerate(vertices):
        x, y = plan([v[k] for k in xyz])
        found = camera_height((math.floor(x / cell), math.floor(y / cell)))
        if found:
            below.append((found[0] - dot([v[k] for k in xyz], up), i, found[1]))
    # Eye height: the floor under the walk is the most common height below the cameras.
    near = sorted(o for o, _, d in below if d <= 1 and o > 0)
    if len(near) < 100:
        raise RuntimeError('Too few mesh vertices under the camera path to find the floor.')
    width = near[int(.99 * (len(near) - 1))] / 200
    bins = {}
    for o in near:
        bins[int(o / width)] = bins.get(int(o / width), 0) + 1
    peak = max(bins, key=lambda b: (sum(bins.get(b + k, 0) for k in (-1, 0, 1)), -b))
    near = [o for o in near if abs(o / width - peak - .5) <= 2]
    eye = near[len(near) // 2]
    # A reflected ceiling can sink up to a room height (about two eye heights) below the floor;
    # keep anything deeper (a lower storey seen through a stairwell) and anything clearly above.
    reach, depth, tolerance = 5 * eye / cell, 2 * eye, .03 * eye
    moved = 0
    for offset, i, d in below:
        lift = offset - eye  # > 0: vertex lies below the local floor
        if d <= reach and -tolerance <= lift <= depth:
            v = vertices[i]
            for k, u in zip(xyz, up):
                v[k] += lift * u
            moved += 1
    tmp = out.with_suffix('.ply.tmp')
    tmp.write_bytes(header + b''.join(row.pack(*v) for v in vertices) + rest)
    tmp.replace(out)
    report = {'up': up, 'eye_height': eye, 'cell': cell, 'vertices': len(vertices), 'flattened': moved}
    print(f'Floor: eye height {eye:.3f}, flattened {moved}/{len(vertices)} vertices.', flush=True)
    return report


def frame_command(a, video, images, cuda, surface):
    # Decode only once, then split sampled frames into all virtual directions.
    cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'warning', '-y',
           '-ss', a.start]
    if a.duration is not None:
        cmd += ['-t', a.duration]
    if cuda:
        cmd += ['-hwaccel', 'cuda', '-hwaccel_device', a.gpus.split(',')[0],
                '-hwaccel_output_format', 'cuda']
    cmd += ['-noautorotate', '-i', video, '-filter_complex_threads', a.threads]
    download = f'hwdownload,format={surface},' if cuda else ''
    graph = (f'[0:v:0]{download}fps={a.fps},format=yuv420p,split={a.faces}' +
             ''.join(f'[s{i}]' for i in range(a.faces)))
    for i, (yaw, pitch) in enumerate(FACES[:a.faces]):
        graph += (f';[s{i}]v360=input=equirect:output=flat:h_fov=100:v_fov=100:'
                  f'yaw={yaw}:pitch={pitch}:w={a.face_size}:h={a.face_size},'
                  f'format=yuvj420p[out{i}]')
    cmd += ['-filter_complex', graph]
    for i in range(a.faces):
        cmd += ['-map', f'[out{i}]', '-an', '-q:v', '2', '-threads:v', '1',
                '-frames:v', a.max_frames + 1, images / f'cam{i}' / '%08d.jpg']
    return cmd


def decode_surface(info):
    # Common stitched Insta360 outputs: 8/10-bit 4:2:0. Other formats use CPU.
    if info.get('codec') not in ('h264', 'hevc', 'vp9', 'vp8', 'mpeg2video', 'mpeg4', 'vc1'):
        return None
    return {'yuv420p': 'nv12', 'yuvj420p': 'nv12',
            'yuv420p10le': 'p010le', 'nv12': 'nv12',
            'p010le': 'p010le'}.get(info.get('pix_fmt'))


def reconstruct(j, a, video, info):
    root, size = j.root, a.face_size
    images, db, dense = root / 'images', root / 'database.db', root / 'dense'

    def extract():
        for i in range(a.faces):
            (images / f'cam{i}').mkdir(parents=True, exist_ok=True)
        surface = decode_surface(info)
        use_cuda = a.decode != 'cpu' and surface is not None
        if a.decode == 'cuda' and surface is None:
            raise RuntimeError('GPU decode requested but input codec/pixel format is not configured for NVDEC.')
        fallback = None
        if use_cuda:
            try:
                j.run('01_frames_cuda', frame_command(a, video, images, True, surface))
                mode = 'nvdec'
            except RuntimeError as error:
                if a.decode == 'cuda':
                    raise
                fallback = str(error)
                print(f'NVDEC unavailable; retrying with CPU decode. {error}', flush=True)
                j.run('01_frames_cpu', frame_command(a, video, images, False, None))
                mode = 'cpu'
        else:
            if a.decode != 'cpu':
                fallback = f"Codec/pixel format {info.get('codec')}/{info.get('pix_fmt')} is not configured for NVDEC."
                print(f'{fallback} Using CPU decode.', flush=True)
            j.run('01_frames_cpu', frame_command(a, video, images, False, None))
            mode = 'cpu'
        expected = None
        for i in range(a.faces):
            n = len(list((images / f'cam{i}').glob('*.jpg')))
            if n < 10 or n > a.max_frames:
                raise RuntimeError(f'Got {n} time samples; allowed range: 10..{a.max_frames}.')
            if expected is not None and n != expected:
                raise RuntimeError('Frame counts differ between virtual views.')
            expected = n
        save(root / 'decode.json', {'mode': mode, 'gpu': a.gpus.split(',')[0] if mode == 'nvdec' else None,
                                    'single_decode': True, 'fallback_reason': fallback})
        return [*sorted(images.glob('cam*/*.jpg')), root / 'decode.json']
    j.stage('01_frames', extract)

    focal = size / (2 * math.tan(math.radians(50)))
    # One camera per direction folder, so the six directions can form one rig below.
    j.command('02_features', ['colmap', 'feature_extractor', '--database_path', db,
        '--image_path', images, '--ImageReader.camera_model', 'PINHOLE', '--ImageReader.single_camera_per_folder', 1,
        '--ImageReader.camera_params', f'{focal},{focal},{size/2},{size/2}',
        '--FeatureExtraction.use_gpu', 1, '--FeatureExtraction.gpu_index', a.gpus,
        '--FeatureExtraction.num_threads', a.threads], [db])

    # Without a rig, the mapper treats the six directions of each time sample as unrelated
    # cameras and the sequence splits into many components (only the largest was kept).
    # The rig fixes them to one center with known relative rotations.
    rig = root / 'rig.json'
    def configure_rig():
        rig.write_text(json.dumps(rig_config(a.faces), indent=2), encoding='utf-8')
        j.run('02_rig', ['colmap', 'rig_configurator', '--database_path', db, '--rig_config_path', rig])
        return [rig, db]
    j.stage('02_rig', configure_rig)
    # Overlap now counts rig frames; each image is matched with all directions of its neighbors.
    j.command('03_matches', ['colmap', 'sequential_matcher', '--database_path', db,
        '--SequentialMatching.overlap', a.overlap, '--SequentialMatching.quadratic_overlap', 0,
        '--SequentialMatching.expand_rig_images', 1, '--FeatureMatching.rig_verification', 1,
        '--FeatureMatching.skip_image_pairs_in_same_frame', 1,
        '--FeatureMatching.use_gpu', 1, '--FeatureMatching.gpu_index', a.gpus,
        '--FeatureMatching.num_threads', a.threads], [db])
    if a.loop_images:
        # The walk revisits places after cuts and loops; sequential matching only links nearby times.
        # Retrieve visually similar images anywhere in the video (vocabulary tree; COLMAP downloads it).
        j.command('03_loops', ['colmap', 'vocab_tree_matcher', '--database_path', db,
            '--VocabTreeMatching.num_images', a.loop_images, '--VocabTreeMatching.num_threads', a.threads,
            '--FeatureMatching.skip_image_pairs_in_same_frame', 1,
            '--FeatureMatching.use_gpu', 1, '--FeatureMatching.gpu_index', a.gpus,
            '--FeatureMatching.num_threads', a.threads], [db])

    def sparse():
        # Keep failed attempts, but never pick their incomplete models.
        attempt = root / 'sparse' / f'attempt_{time.time_ns()}'
        attempt.mkdir(parents=True)
        gpu = a.gpus.split(',')[0]
        if a.sfm == 'global':
            # Solves all rig frames at once, so a weak stretch of video does not split the walk into
            # separate models the way incremental registration does.
            j.run('04_sparse', ['colmap', 'global_mapper', '--database_path', db, '--image_path', images,
                '--output_path', attempt, '--GlobalMapper.num_threads', a.threads,
                '--GlobalMapper.gp_use_gpu', 1, '--GlobalMapper.gp_gpu_index', gpu,
                '--GlobalMapper.ba_ceres_use_gpu', 1, '--GlobalMapper.ba_gpu_index', gpu,
                '--GlobalMapper.ba_refine_focal_length', 0, '--GlobalMapper.ba_refine_principal_point', 0,
                '--GlobalMapper.ba_refine_extra_params', 0, '--GlobalMapper.refine_sensor_from_rig', 0])
        else:
            j.run('04_sparse', ['colmap', 'mapper', '--database_path', db, '--image_path', images,
                '--output_path', attempt, '--Mapper.num_threads', a.threads,
                '--Mapper.ba_use_gpu', 1, '--Mapper.ba_gpu_index', gpu,
                '--Mapper.ba_refine_focal_length', 0, '--Mapper.ba_refine_principal_point', 0,
                '--Mapper.ba_refine_extra_params', 0, '--Mapper.ba_refine_sensor_from_rig', 0])
        candidates = [x.parent for x in attempt.glob('*/images.bin')]
        if not candidates:
            raise RuntimeError('No reconstruction. Check camera movement, overlap, blur and white walls.')
        best = max(candidates, key=lambda x: count(x / 'images.bin'))
        n, total, points = count(best / 'images.bin'), len(list(images.glob('cam*/*.jpg'))), count(best / 'points3D.bin')
        if n < 10 or points < 100 or n / total < a.min_registered_ratio:
            raise RuntimeError(f'Insufficient reconstruction: {n}/{total} views, {points} points. Dense stage stopped.')
        selection = root / 'selected_model.json'
        save(selection, {'path': str(best.relative_to(root)), 'registered_views': n,
            'total_views': total, 'registered_ratio': n / total, 'points': points, 'components': len(candidates)})
        print(f'Using largest of {len(candidates)} components: {n}/{total} registered views.', flush=True)
        return [selection, *(best / f for f in ('cameras.bin', 'images.bin', 'points3D.bin'))]
    j.stage('04_sparse', sparse)
    selection = json.loads((root / 'selected_model.json').read_text(encoding='utf-8'))
    dense.mkdir(exist_ok=True)
    j.command('05_undistort', ['colmap', 'image_undistorter', '--image_path', images,
        '--input_path', root / selection['path'], '--output_path', dense, '--output_type', 'COLMAP',
        '--max_image_size', size], lambda: [*(dense / 'sparse' / f for f in ('cameras.bin', 'images.bin', 'points3D.bin')),
                                           *require((dense / 'images').rglob('*.jpg'))])
    j.command('06_depth', ['colmap', 'patch_match_stereo', '--workspace_path', dense,
        '--workspace_format', 'COLMAP', '--PatchMatchStereo.gpu_index', a.gpus,
        '--PatchMatchStereo.cache_size', 8, '--PatchMatchStereo.num_threads', 1,
        '--PatchMatchStereo.geom_consistency', 1, '--PatchMatchStereo.max_image_size', size],
        lambda: list((dense / 'stereo' / 'depth_maps').rglob('*.geometric.bin')))
    masks = []
    if a.faces > FACES.index((0, -90)):
        j.stage('07_masks', lambda: fusion_masks(dense, size))
        masks = ['--StereoFusion.mask_path', dense / 'masks']
    # Stages 07-10 were renamed (v2/v4) when the down-face mask was added, so --resume refuses
    # nothing and redoes fusion onward for jobs finished without it. The 4GB cache keeps
    # ~13k-view jobs inside 31GB RAM next to the per-view fused-pixel masks.
    j.command('07_fusion_v2', ['colmap', 'stereo_fusion', '--workspace_path', dense,
        '--workspace_format', 'COLMAP', '--input_type', 'geometric',
        '--StereoFusion.use_cache', 1, '--StereoFusion.cache_size', 4,
        '--StereoFusion.num_threads', a.threads, *masks,
        '--output_path', dense / 'fused.ply'], [dense / 'fused.ply', dense / 'fused.ply.vis'])
    # Retain dense visibility: InterfaceCOLMAP requires fused.ply AND fused.ply.vis.
    j.command('08_import_v2', ['InterfaceCOLMAP', '-i', dense, '-o', root / 'scene.mvs',
        '--image-folder', dense / 'images'], [root / 'scene.mvs', root / 'scene.ply'])
    # White walls yield few depth points, so rays to points behind them carve the wall away;
    # free-space support keeps such weakly-supported surfaces, and close-holes patches the rest.
    j.command('09_mesh_v2', ['ReconstructMesh', '-i', root / 'scene.mvs', '-p', root / 'scene.ply',
        '-o', root / 'mesh.mvs', '--archive-type', 2, '--target-face-num', a.target_faces,
        '--free-space-support', int(a.free_space_support), '--close-holes', a.close_holes, '--max-threads', a.threads],
        [root / 'mesh.mvs', root / 'mesh.ply'])
    mesh = root / 'mesh.ply'
    if a.floor:
        def floor():
            save(root / 'floor.json', flatten_floor(dense / 'sparse' / 'images.bin', mesh, root / 'mesh_floor.ply'))
            return [root / 'mesh_floor.ply', root / 'floor.json']
        j.stage('09b_floor', floor)
        mesh = root / 'mesh_floor.ply'
    # Both seam-leveling solvers (global CG over patch interiors, local BiCGSTAB Poisson blending
    # over a 20px border strip) fail on these scenes; release builds skip their asserts, so NaN/huge
    # colors are written as black or saturated texels. Seam leveling is cosmetic, so disable both.
    # The stage name changed so --resume redoes texturing for jobs finished with the old settings
    # (v5: textures the floor-flattened mesh).
    j.command('10_texture_v5', ['TextureMesh', '-i', root / 'mesh.mvs', '-m', mesh,
        '-o', root / 'model.mvs', '--export-type', 'obj', '--max-texture-size', a.texture_size,
        '--global-seam-leveling', 0, '--local-seam-leveling', 0,
        '--max-threads', a.threads], lambda: texture_outputs(root)[0])
    # Runs on every invocation so --resume also repairs models textured by older versions.
    opaque_materials(root)
    outputs, vertices, triangles = texture_outputs(root)
    save(root / 'report.json', {'input': str(video), 'video': info, 'reconstruction': selection,
        'decode': json.loads((root / 'decode.json').read_text(encoding='utf-8')),
        'gpu_policy': {'gpus': a.gpus, 'bundle_adjustment': 'CUDA dense solver requested; Ceres selects by problem size',
                       'fusion_meshing_texturing_projection': 'CPU'},
        'vertices': vertices, 'faces': triangles, 'outputs': [str(x.relative_to(root)) for x in outputs],
        'notes': ['Arbitrary scale/orientation; measure and align before Unity.',
                  'Largest component only. Inspect missing areas and reconstruction quality.',
                  'Face budget is approximate. Colliders and VRChat setup are not generated.']})
    print(f'DONE: {root / "model.obj"}\nKeep OBJ, MTL and textures together.', flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--fps', type=float, default=1)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--face-size', type=int, default=1536)
    parser.add_argument('--faces', type=int, choices=(4, 6), default=6)
    parser.add_argument('--gpus', default='all', help='All NVIDIA GPUs by default; or CUDA indices such as 0,1,2,3')
    parser.add_argument('--decode', choices=('auto', 'cuda', 'cpu'), default='auto',
                        help='Prefer NVDEC; auto retries CPU with a logged reason if unavailable')
    parser.add_argument('--threads', type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument('--overlap', type=int, default=8, help='Temporal neighbor count before face splitting')
    parser.add_argument('--max-frames', type=int, default=2000, help='Time samples before face splitting')
    parser.add_argument('--loop-images', type=int, default=20,
                        help='Similar images matched per image anywhere in the video (0 disables loop matching)')
    parser.add_argument('--target-faces', type=int, default=500000)
    parser.add_argument('--texture-size', type=int, default=4096)
    parser.add_argument('--free-space-support', action=argparse.BooleanOptionalAction, default=True,
                        help='Keep weakly-supported surfaces such as white walls when meshing (--no-free-space-support disables)')
    parser.add_argument('--close-holes', type=int, default=300,
                        help='Close mesh holes up to this many boundary edges (0 disables; OpenMVS default 30)')
    parser.add_argument('--floor', action=argparse.BooleanOptionalAction, default=True,
                        help='Snap mesh vertices that sank below the walked floor back onto it (--no-floor disables)')
    parser.add_argument('--min-registered-ratio', type=float, default=.3)
    parser.add_argument('--sfm', choices=('global', 'incremental'), default='global',
                        help='Camera pose solver: global (all frames at once) or incremental (COLMAP mapper)')
    a = parser.parse_args(argv)
    for key in ('fps', 'face_size', 'threads', 'overlap', 'max_frames', 'target_faces', 'texture_size'):
        if not math.isfinite(getattr(a, key)) or getattr(a, key) <= 0:
            parser.error(f'--{key.replace("_", "-")} must be positive and finite')
    if not math.isfinite(a.start) or a.start < 0:
        parser.error('--start must be nonnegative and finite')
    if a.duration is not None and (not math.isfinite(a.duration) or a.duration <= 0):
        parser.error('--duration must be positive and finite')
    if a.loop_images < 0:
        parser.error('--loop-images must be nonnegative')
    if a.close_holes < 0:
        parser.error('--close-holes must be nonnegative')
    if not 0 < a.min_registered_ratio <= 1:
        parser.error('--min-registered-ratio must be in (0,1]')
    if a.gpus != 'all' and not re.fullmatch(r'\d+(,\d+)*', a.gpus):
        parser.error('--gpus must be all or comma-separated nonnegative indices')
    return a


def main(argv=None):
    a = parse_args(argv)
    video, root = a.video.expanduser().resolve(strict=True), a.output.expanduser().resolve()
    if not video.is_file() or video.is_relative_to(root):
        raise RuntimeError('Input must be a video file outside the output directory.')
    a.gpus = ','.join(str(x) for x in preflight(a.gpus))
    info = probe(video)
    if a.start >= info['duration']:
        raise RuntimeError('--start is beyond the end of the video.')
    seconds = min(a.duration or info['duration'], info['duration'] - a.start)
    if seconds * a.fps > a.max_frames + .01:
        raise RuntimeError('Frame limit exceeded. Increase --max-frames or reduce --duration / --fps.')
    config = {k: v for k, v in vars(a).items() if k not in ('video', 'output', 'resume', 'threads')}
    config.update(version=5, video=identity(video))
    root.mkdir(parents=True, exist_ok=True)
    with job_lock(root):
        j = Job(root, config, a.resume)
        save(root / 'video.json', info)
        reconstruct(j, a, video, info)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted. Rerun the same command with --resume.', file=sys.stderr)
        sys.exit(130)
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)

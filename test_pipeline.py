import contextlib
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import mp4_to_model as m


def images_bin(centres, faces=6):
    """COLMAP images.bin for a level rig (cam0 rotation = identity, so world up is -y)."""
    data = struct.pack('<Q', len(centres) * faces)
    for n, c in enumerate(centres, 1):
        for f in range(faces):
            data += struct.pack('<i7di', n * faces + f, 1, 0, 0, 0, *(-x for x in c), 1)
            data += f'cam{f}/{n:08d}.jpg'.encode() + b'\0' + struct.pack('<Q', 0)
    return data


def mesh_ply(vertices, faces=((0, 1, 2),)):
    header = (f'ply\nformat binary_little_endian 1.0\ncomment OpenMVS\nelement vertex {len(vertices)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              f'element face {len(faces)}\nproperty list uchar uint vertex_indices\nend_header\n')
    return (header.encode() + b''.join(struct.pack('<3f', *v) for v in vertices) +
            b''.join(struct.pack('<B3I', 3, *f) for f in faces))


# A corridor walked along x at eye height 1.5 (floor y=0, up = -y): a floor grid with a
# sunken patch, a table top, a ceiling, and one vertex far below (a lower storey).
WALK = [(x / 5, -1.5, 0) for x in range(51)]
FLOOR = [(x / 4, 0, z / 4) for x in range(41) for z in range(-8, 9)]
SUNKEN = [(5 + x / 4, .6, z / 4) for x in range(4) for z in range(-2, 3)]
TABLE = [(3, -.7, 1), (3.5, -.7, 1)]
CEILING = [(x / 2, -3, z / 2) for x in range(21) for z in range(-4, 5)]
DEEP = [(8, 5, 0)]
SCENE = FLOOR + SUNKEN + TABLE + CEILING + DEEP


class FakeJob(m.Job):
    def run(self, name, command):
        self.calls.append((name, command))
        r = self.root
        if self.fail == name:
            self.fail = None
            raise RuntimeError('simulated process failure')
        def write(path, data=b'x'):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if name in ('01_frames_cuda', '01_frames_cpu'):
            for i in range(command.count('-map')):
                for n in range(1, 11):
                    write(r / 'images' / f'cam{i}' / f'{n:08d}.jpg')
        elif name == '02_features':
            write(r / 'database.db')
        elif name == '04_sparse':
            out = Path(command[command.index('--output_path') + 1])
            for component, images in (('0', 12), ('1', self.registered)):
                for file, n in (('cameras.bin', 1), ('images.bin', images), ('points3D.bin', 1000)):
                    write(out / component / file, struct.pack('<Q', n))
        elif name == '05_undistort':
            for file in ('cameras.bin', 'points3D.bin'):
                write(r / 'dense' / 'sparse' / file)
            write(r / 'dense' / 'sparse' / 'images.bin', images_bin(WALK))
            write(r / 'dense' / 'images' / 'cam0' / '00000001.jpg')
            write(r / 'dense' / 'images' / 'cam5' / '00000001.jpg')
        elif name == '06_depth':
            write(r / 'dense' / 'stereo' / 'depth_maps' / 'cam0' / '00000001.jpg.geometric.bin')
        elif name == '07_fusion_v2':
            write(r / 'dense' / 'fused.ply')
            write(r / 'dense' / 'fused.ply.vis')
        elif name == '08_import_v2':
            write(r / 'scene.mvs')
            write(r / 'scene.ply')
        elif name == '09_mesh_v2':
            if command[command.index('--archive-type') + 1] != 2:
                raise AssertionError('Must save a non-interface MVS archive')
            write(r / 'mesh.mvs')
            write(r / 'mesh.ply', mesh_ply(SCENE))
        elif name == '10_texture_v5':
            write(r / 'model.obj', b'mtllib model.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
            # Same material header that OpenMVS v2.4.0 writes.
            write(r / 'model.mtl', b'newmtl material_00\nKa 1.000000 1.000000 1.000000\n'
                  b'Kd 1 1 1\nKs 0.000000 0.000000 0.000000\nTr 1.000000\nillum 1\n'
                  b'Ns 1.000000\nmap_Kd texture.jpg\n')
            write(r / 'texture.jpg')


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.info = {'width': 5760, 'height': 2880, 'duration': 10, 'codec': 'hevc', 'pix_fmt': 'yuv420p'}
        self.a = self.args()
        self.config = {'video': 'hash', 'gpus': self.a.gpus}

    def args(self, *extra):
        """Parsed arguments with the face size main() derives from the video."""
        a = m.parse_args(['video.mp4', '-o', str(self.root), '--gpus', '0,1,2,3', *extra])
        a.face_size = a.face_size or m.native_face_size(self.info['width'])
        return a

    def job(self, resume=False, fail=None, registered=40):
        j = FakeJob(self.root, self.config, resume)
        j.calls, j.fail, j.registered = [], fail, registered
        return j

    def reconstruct(self, job):
        with contextlib.redirect_stdout(io.StringIO()):
            m.reconstruct(job, self.a, Path('/input/video.mp4'), self.info)

    def test_pipeline_and_multigpu_forwarding(self):
        j = self.job()
        self.reconstruct(j)
        report = json.loads((self.root / 'report.json').read_text())
        self.assertEqual(report['reconstruction']['registered_views'], 40)
        self.assertEqual(Path(report['reconstruction']['path']).name, '1')
        self.assertEqual(report['faces'], 1)
        self.assertEqual(report['decode']['mode'], 'nvdec')
        self.assertEqual(sum(n.startswith('01_frames_') for n, _ in j.calls), 1)
        mapper = next(cmd for name, cmd in j.calls if name == '04_sparse')
        self.assertEqual(mapper[:2], ['colmap', 'global_mapper'])
        for option in ('--GlobalMapper.gp_use_gpu', '--GlobalMapper.ba_ceres_use_gpu'):
            self.assertEqual(mapper[mapper.index(option) + 1], 1)
        for option in ('--GlobalMapper.gp_gpu_index', '--GlobalMapper.ba_gpu_index'):
            self.assertEqual(mapper[mapper.index(option) + 1], '0')
        for name, cmd in j.calls:
            if name in ('02_features', '03_matches', '06_depth'):
                self.assertIn('0,1,2,3', cmd)
        importer = next(cmd for name, cmd in j.calls if name == '08_import_v2')
        self.assertEqual(Path(importer[importer.index('-i') + 1]), self.root / 'dense')

    def test_failure_and_resume(self):
        j = self.job(fail='06_depth')
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            self.reconstruct(j)
        state = json.loads((self.root / 'state.json').read_text())
        self.assertNotIn('06_depth', state['completed'])
        self.assertIn('05_undistort', state['completed'])
        resumed = self.job(resume=True)
        self.reconstruct(resumed)
        self.assertEqual(resumed.calls[0][0], '06_depth')
        self.assertFalse(any(n == '04_sparse' for n, _ in resumed.calls))

    def test_low_registration_stops_dense(self):
        j = self.job(registered=12)
        with self.assertRaisesRegex(RuntimeError, 'Insufficient reconstruction'):
            self.reconstruct(j)
        self.assertFalse(any(n == '06_depth' for n, _ in j.calls))

    def test_resume_ignores_thread_count_from_older_jobs(self):
        self.job()
        state = json.loads((self.root / 'state.json').read_text())
        state['config']['threads'] = 16
        m.save(self.root / 'state.json', state)
        self.assertEqual(m.Job(self.root, self.config, True).state['config'], self.config)

    def test_resume_rejects_changed_settings(self):
        self.job()
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            m.Job(self.root, {'video': 'different'}, True)

    def test_legacy_resume_preserves_completed_work_and_backup(self):
        old = {**self.config, 'version': 1, 'min_registered_ratio': .25}
        current = {**old, 'version': 2, 'decode': 'auto'}
        frame = self.root / 'frame.jpg'
        frame.write_bytes(b'cached-frame')
        state = {'config': old, 'completed': {'01_frames': ['frame.jpg'], '04_sparse': ['model.bin']}}
        m.save(self.root / 'state.json', state)
        original = (self.root / 'state.json').read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            j = m.Job(self.root, current, True)
        self.assertEqual(j.state['config'], current)
        self.assertEqual(j.state['completed']['04_sparse'], ['model.bin'])
        self.assertIn('decode.json', j.state['completed']['01_frames'])
        self.assertEqual(json.loads((self.root / 'decode.json').read_text())['mode'], 'legacy-cache')
        self.assertEqual(frame.read_bytes(), b'cached-frame')
        backups = list(self.root.glob('state.before-v2-*.json'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)
        m.Job(self.root, current, True)
        self.assertEqual(len(list(self.root.glob('state.before-v2-*.json'))), 1)

    def test_legacy_resume_rejects_processing_changes_without_mutation(self):
        old = {**self.config, 'version': 1, 'min_registered_ratio': .3}
        m.save(self.root / 'state.json', {'config': old, 'completed': {'01_frames': ['frame.jpg']}})
        original = (self.root / 'state.json').read_bytes()
        current = {**old, 'version': 2, 'decode': 'auto'}
        for change in ({'video': 'changed'}, {'min_registered_ratio': .25}, {'decode': 'cuda'}):
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, 'changed'):
                m.Job(self.root, {**current, **change}, True)
            self.assertEqual((self.root / 'state.json').read_bytes(), original)
            self.assertFalse((self.root / 'decode.json').exists())
        self.assertFalse(list(self.root.glob('state.before-v2-*.json')))

    def test_legacy_resume_rejects_missing_cached_frames(self):
        old = {**self.config, 'version': 1}
        m.save(self.root / 'state.json', {'config': old, 'completed': {'01_frames': ['missing.jpg']}})
        with self.assertRaisesRegex(RuntimeError, 'Missing'):
            m.Job(self.root, {**old, 'version': 2, 'decode': 'auto'}, True)
        self.assertFalse((self.root / 'decode.json').exists())

    def test_resume_detects_deleted_outputs(self):
        j = self.job()
        p = self.root / 'artifact'
        p.write_bytes(b'x')
        j.stage('a', lambda: [p])
        p.unlink()
        with self.assertRaisesRegex(RuntimeError, 'Missing'):
            self.job(resume=True).stage('a', lambda: [p])

    def test_texture_dependencies(self):
        j = self.job()
        self.reconstruct(j)
        (self.root / 'texture.jpg').unlink()
        with self.assertRaisesRegex(RuntimeError, 'Missing'):
            m.texture_outputs(self.root)

    def test_materials_are_opaque(self):
        j = self.job()
        self.reconstruct(j)
        mtl = (self.root / 'model.mtl').read_text().splitlines()
        self.assertNotIn('Tr 1.000000', mtl)
        self.assertEqual(mtl[:2], ['newmtl material_00', 'd 1.000000'])
        self.assertIn('map_Kd texture.jpg', mtl)
        m.opaque_materials(self.root)
        self.assertEqual((self.root / 'model.mtl').read_text().splitlines(), mtl)

    def test_resume_repairs_completed_transparent_material(self):
        j = self.job()
        self.reconstruct(j)
        (self.root / 'model.mtl').write_text('newmtl material_00\nTr 1.000000\nmap_Kd texture.jpg\n')
        resumed = self.job(resume=True)
        self.reconstruct(resumed)
        self.assertEqual(resumed.calls, [])
        self.assertEqual((self.root / 'model.mtl').read_text(),
                         'newmtl material_00\nd 1.000000\nmap_Kd texture.jpg\n')

    def test_texture_disables_seam_leveling(self):
        j = self.job()
        self.reconstruct(j)
        cmd = next(cmd for name, cmd in j.calls if name == '10_texture_v5')
        self.assertEqual(cmd[cmd.index('--global-seam-leveling') + 1], 0)
        self.assertEqual(cmd[cmd.index('--local-seam-leveling') + 1], 0)

    def test_resume_retextures_job_finished_by_old_version(self):
        j = self.job()
        self.reconstruct(j)
        state = json.loads((self.root / 'state.json').read_text())
        outputs = state['completed'].pop('10_texture_v5')
        for old in ('10_texture', '10_texture_v2', '10_texture_v3', '10_texture_v4'):
            with self.subTest(old=old):
                state['completed'].pop('10_texture_v5', None)
                state['completed'][old] = outputs
                m.save(self.root / 'state.json', state)
                resumed = self.job(resume=True)
                self.reconstruct(resumed)
                self.assertEqual([n for n, _ in resumed.calls], ['10_texture_v5'])
                state = json.loads((self.root / 'state.json').read_text())

    def test_fusion_masks_out_down_face(self):
        def pixels(path):
            data = path.read_bytes()
            width, height = struct.unpack('>II', data[16:24])
            (n,) = struct.unpack('>I', data[33:37])
            self.assertEqual(data[37:41], b'IDAT')
            rows = m.zlib.decompress(data[41:41 + n])
            return width, height, set(rows[1:width + 1])
        j = self.job()
        self.reconstruct(j)
        fusion = next(cmd for name, cmd in j.calls if name == '07_fusion_v2')
        self.assertEqual(Path(fusion[fusion.index('--StereoFusion.mask_path') + 1]), self.root / 'dense' / 'masks')
        masks = self.root / 'dense' / 'masks'
        self.assertEqual(pixels(masks / 'cam0' / '00000001.jpg.png'), (self.a.face_size, self.a.face_size, {255}))
        self.assertEqual(pixels(masks / 'cam5' / '00000001.jpg.png'), (self.a.face_size, self.a.face_size, {0}))

    def test_four_faces_fuse_without_masks(self):
        self.a.faces = 4
        j = self.job()
        self.reconstruct(j)
        fusion = next(cmd for name, cmd in j.calls if name == '07_fusion_v2')
        self.assertNotIn('--StereoFusion.mask_path', fusion)

    def test_resume_refuses_old_fusion_and_redoes_dense_onward(self):
        j = self.job()
        self.reconstruct(j)
        state = json.loads((self.root / 'state.json').read_text())
        for new, old in (('07_fusion_v2', '07_fusion'), ('08_import_v2', '08_import'),
                         ('09_mesh_v2', '09_mesh'), ('10_texture_v5', '10_texture_v3')):
            state['completed'][old] = state['completed'].pop(new)
        del state['completed']['07_masks']
        m.save(self.root / 'state.json', state)
        resumed = self.job(resume=True)
        self.reconstruct(resumed)
        self.assertEqual([n for n, _ in resumed.calls], ['07_fusion_v2', '08_import_v2', '09_mesh_v2', '10_texture_v5'])

    def test_rig_rotations_match_v360_directions(self):
        def rotate(q, v):
            w, x, y, z = m.quat_mul(m.quat_mul(q, (0, *v)), (q[0], -q[1], -q[2], -q[3]))
            return (x, y, z)
        cameras = m.rig_config(6)[0]['cameras']
        self.assertEqual([c['image_prefix'] for c in cameras], [f'cam{i}/' for i in range(6)])
        self.assertTrue(cameras[0]['ref_sensor'])
        # Forward axis of each v360 view, in the panorama (= cam0) frame: Ry(yaw) Rx(pitch) (0,0,1).
        for camera, (yaw, pitch) in zip(cameras[1:], m.FACES[1:]):
            y, p = math.radians(yaw), math.radians(pitch)
            forward = (math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p))
            with self.subTest(yaw=yaw, pitch=pitch):
                got = rotate(camera['cam_from_rig_rotation'], forward)
                for g, e in zip(got, (0, 0, 1)):
                    self.assertAlmostEqual(g, e, places=6)
                self.assertEqual(camera['cam_from_rig_translation'], [0, 0, 0])
        # Spot-check axes: yaw 90 looks along +x (right), pitch 90 along -y (up).
        self.assertAlmostEqual(rotate(cameras[1]['cam_from_rig_rotation'], (1, 0, 0))[2], 1, places=6)
        self.assertAlmostEqual(rotate(cameras[4]['cam_from_rig_rotation'], (0, -1, 0))[2], 1, places=6)

    def test_rig_is_configured_before_matching(self):
        j = self.job()
        self.reconstruct(j)
        names = [n for n, _ in j.calls]
        self.assertLess(names.index('02_features'), names.index('02_rig'))
        self.assertLess(names.index('02_rig'), names.index('03_matches'))
        features = next(cmd for name, cmd in j.calls if name == '02_features')
        self.assertEqual(features[features.index('--ImageReader.single_camera_per_folder') + 1], 1)
        self.assertNotIn('--ImageReader.single_camera', features)
        matcher = next(cmd for name, cmd in j.calls if name == '03_matches')
        self.assertEqual(matcher[matcher.index('--SequentialMatching.overlap') + 1], self.a.overlap)
        self.assertEqual(matcher[matcher.index('--SequentialMatching.expand_rig_images') + 1], 1)
        mapper = next(cmd for name, cmd in j.calls if name == '04_sparse')
        self.assertEqual(mapper[mapper.index('--GlobalMapper.refine_sensor_from_rig') + 1], 0)
        rig = json.loads((self.root / 'rig.json').read_text())
        self.assertEqual(len(rig[0]['cameras']), 6)

    def test_incremental_mapper_option(self):
        self.a = self.args('--sfm', 'incremental')
        j = self.job()
        self.reconstruct(j)
        mapper = next(cmd for name, cmd in j.calls if name == '04_sparse')
        self.assertEqual(mapper[:2], ['colmap', 'mapper'])
        self.assertEqual(mapper[mapper.index('--Mapper.ba_use_gpu') + 1], 1)
        self.assertEqual(mapper[mapper.index('--Mapper.ba_gpu_index') + 1], '0')
        self.assertEqual(mapper[mapper.index('--Mapper.ba_refine_sensor_from_rig') + 1], 0)

    def test_v3_resume_keeps_matches_and_redoes_sfm(self):
        self.config = {**self.config, 'version': 3}
        j = self.job()
        self.reconstruct(j)
        stale = self.root / 'dense' / 'stereo' / 'depth_maps' / 'cam0' / '00000001.jpg.geometric.bin'
        self.assertTrue(stale.exists())
        self.config = {**self.config, 'version': 5, 'sfm': 'global', 'loop_images': 20}
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = self.job(resume=True)
        self.assertEqual(list(resumed.state['completed']), ['01_frames', '02_features', '02_rig', '03_matches'])
        for name in ('images', 'database.db', 'rig.json', 'decode.json', 'logs'):
            self.assertTrue((self.root / name).exists(), name)
        for name in ('dense', 'sparse', 'selected_model.json', 'model.obj', 'state.json'):
            self.assertTrue((self.root / 'v3-backup' / name).exists(), name)
        self.assertFalse((self.root / 'dense').exists())
        self.reconstruct(resumed)
        self.assertEqual([n for n, _ in resumed.calls][:2], ['03_loops', '04_sparse'])
        self.assertTrue((self.root / 'model.obj').exists())
        self.job(resume=True)

    def test_v4_resume_adds_loop_matching(self):
        self.config = {**self.config, 'version': 4, 'sfm': 'global'}
        self.reconstruct(self.job())
        self.config = {**self.config, 'version': 5, 'loop_images': 20}
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = self.job(resume=True)
        self.assertEqual(list(resumed.state['completed']), list(m.MATCH_STAGES))
        self.assertTrue((self.root / 'v4-backup' / 'dense').exists())
        self.reconstruct(resumed)
        names = [n for n, _ in resumed.calls]
        self.assertEqual(names[:2], ['03_loops', '04_sparse'])
        loops = resumed.calls[0][1]
        self.assertEqual(loops[:2], ['colmap', 'vocab_tree_matcher'])
        self.assertEqual(loops[loops.index('--VocabTreeMatching.num_images') + 1], 20)
        self.assertIn('0,1,2,3', loops)

    def test_loop_matching_runs_between_sequential_matching_and_sfm(self):
        j = self.job()
        self.reconstruct(j)
        names = [n for n, _ in j.calls]
        self.assertLess(names.index('03_matches'), names.index('03_loops'))
        self.assertLess(names.index('03_loops'), names.index('04_sparse'))

    def test_loop_matching_can_be_disabled(self):
        self.a = self.args('--loop-images', '0')
        j = self.job()
        self.reconstruct(j)
        self.assertNotIn('03_loops', [n for n, _ in j.calls])

    def test_v2_resume_chains_to_current_version(self):
        old = {**self.config, 'version': 2}
        names = [f'images/{n:08d}_{i}.jpg' for n in range(1, 11) for i in range(6)]
        for name in names:
            (self.root / name).parent.mkdir(exist_ok=True)
            (self.root / name).write_bytes(b'x')
        m.save(self.root / 'state.json', {'config': old, 'completed': {'01_frames': names}})
        current = {**old, 'version': 5, 'sfm': 'global', 'loop_images': 20}
        with contextlib.redirect_stdout(io.StringIO()):
            j = m.Job(self.root, current, True)
        self.assertEqual(j.state['config'], current)
        self.assertEqual(list(j.state['completed']), ['01_frames'])
        self.assertTrue((self.root / 'images' / 'cam5' / '00000010.jpg').exists())

    def test_v3_resume_rejects_other_changes(self):
        m.save(self.root / 'state.json', {'config': {**self.config, 'version': 3}, 'completed': {}})
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            m.Job(self.root, {**self.config, 'version': 5, 'sfm': 'global', 'loop_images': 20, 'fps': 2}, True)
        self.assertFalse((self.root / 'v3-backup').exists())

    def test_v2_resume_reuses_frames_in_rig_layout(self):
        old = {**self.config, 'version': 2}
        names = [f'images/{n:08d}_{i}.jpg' for n in range(1, 11) for i in range(6)]
        for name in names:
            (self.root / name).parent.mkdir(exist_ok=True)
            (self.root / name).write_bytes(name.encode())
        (self.root / 'decode.json').write_text('{}')
        (self.root / 'database.db').write_bytes(b'old')
        (self.root / 'dense' / 'images').mkdir(parents=True)
        (self.root / 'model.obj').write_text('old')
        m.save(self.root / 'state.json', {'config': old, 'completed': {
            '01_frames': [*names, 'decode.json'], '02_features': ['database.db'], '10_texture_v5': ['model.obj']}})
        with contextlib.redirect_stdout(io.StringIO()):
            j = m.Job(self.root, {**old, 'version': 3}, True)
        self.assertEqual(list(j.state['completed']), ['01_frames'])
        self.assertEqual((self.root / 'images' / 'cam3' / '00000007.jpg').read_bytes(), b'images/00000007_3.jpg')
        self.assertFalse(list((self.root / 'images').glob('*.jpg')))
        for name in ('database.db', 'dense', 'model.obj', 'state.json'):
            self.assertTrue((self.root / 'v2-backup' / name).exists(), name)
            if name != 'state.json':
                self.assertFalse((self.root / name).exists(), name)
        self.assertTrue((self.root / 'decode.json').exists())
        m.Job(self.root, {**old, 'version': 3}, True)

    def test_texture_path_escape(self):
        (self.root / 'model.obj').write_text('v 0 0 0\nf 1 1 1\n')
        (self.root / 'model.mtl').write_text('map_Kd ../outside.jpg\n')
        with self.assertRaisesRegex(RuntimeError, 'outside'):
            m.texture_outputs(self.root)

    def test_input_hash_is_content_based(self):
        a, b = self.root / 'a', self.root / 'b'
        a.write_bytes(b'ab')
        b.write_bytes(b'cd')
        self.assertNotEqual(m.identity(a), m.identity(b))


    def test_decode_once_and_download_before_projection(self):
        cmd = m.frame_command(self.a, Path('video.mp4'), self.root, True, 'nv12')
        self.assertEqual(cmd.count('-i'), 1)
        self.assertEqual(cmd.count('-map'), 6)
        self.assertLess(cmd.index('-hwaccel'), cmd.index('-i'))
        self.assertEqual(cmd[cmd.index('-hwaccel_output_format') + 1], 'cuda')
        graph = cmd[cmd.index('-filter_complex') + 1]
        self.assertLess(graph.index('hwdownload'), graph.index('v360'))
        self.assertEqual(graph.count('v360='), 6)

    def test_nvdec_failure_logs_cpu_fallback(self):
        j = self.job(fail='01_frames_cuda')
        self.reconstruct(j)
        self.assertEqual([n for n, _ in j.calls[:2]], ['01_frames_cuda', '01_frames_cpu'])
        record = json.loads((self.root / 'decode.json').read_text())
        self.assertEqual(record['mode'], 'cpu')
        self.assertIn('simulated', record['fallback_reason'])

    def test_strict_cuda_decode_does_not_fall_back(self):
        self.a.decode = 'cuda'
        j = self.job(fail='01_frames_cuda')
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            self.reconstruct(j)
        self.assertFalse(any(n == '01_frames_cpu' for n, _ in j.calls))
        self.assertNotIn('01_frames', j.state['completed'])

    def test_explicit_cpu_decode(self):
        self.a.decode = 'cpu'
        j = self.job()
        self.reconstruct(j)
        self.assertEqual(j.calls[0][0], '01_frames_cpu')

    def test_unconfigured_pixel_format_skips_nvdec(self):
        self.info['pix_fmt'] = 'unsupported'
        j = self.job()
        self.reconstruct(j)
        self.assertEqual(j.calls[0][0], '01_frames_cpu')
        self.assertIn('not configured', json.loads((self.root / 'decode.json').read_text())['fallback_reason'])

    def test_face_size_defaults_to_panorama_resolution(self):
        self.assertIsNone(m.parse_args(['v.mp4', '-o', 'out']).face_size)
        self.assertEqual(m.parse_args(['v.mp4', '-o', 'out', '--face-size', '1024']).face_size, 1024)
        for width, size in ((3840, 1072), (5760, 1600), (7680, 2128)):
            self.assertEqual(m.native_face_size(width), size)

    def test_default_gpu_selection_is_all(self):
        a = m.parse_args(['v.mp4', '-o', 'out'])
        self.assertEqual(a.gpus, 'all')
        self.assertEqual(a.decode, 'auto')
        def run(command, **kwargs):
            output = '0\n1\n2\n3\n' if command[0] == 'nvidia-smi' else (
                'FeatureExtraction.gpu_index FeatureMatching.gpu_index PatchMatchStereo.gpu_index Mapper.ba_use_gpu GlobalMapper.gp_use_gpu VocabTreeMatching.num_images v360')
            return SimpleNamespace(stdout=output, stderr='', returncode=0)
        with patch.object(m.shutil, 'which', return_value='installed'), patch.object(m.subprocess, 'run', side_effect=run):
            self.assertEqual(m.preflight(a.gpus), [0, 1, 2, 3])
            self.assertEqual(m.preflight('1,3'), [1, 3])

    def test_floor_lifts_sunken_vertices_only(self):
        j = self.job()
        self.reconstruct(j)
        texture = next(cmd for name, cmd in j.calls if name == '10_texture_v5')
        self.assertEqual(Path(texture[texture.index('-m') + 1]), self.root / 'mesh_floor.ply')
        _, _, vertices, xyz, rest = m.ply_vertices(self.root / 'mesh_floor.ply')
        self.assertEqual(rest, mesh_ply(SCENE)[-len(rest):])
        got = [tuple(v[k] for k in xyz) for v in vertices]
        n = len(FLOOR)
        for v in got[n:n + len(SUNKEN)]:
            self.assertAlmostEqual(v[1], 0, places=4)
        self.assertEqual(got[:n], [tuple(map(float, v)) for v in FLOOR])
        for kept, original in zip(got[n + len(SUNKEN):], TABLE + CEILING + DEEP):
            self.assertEqual(kept, tuple(struct.unpack('<3f', struct.pack('<3f', *original))))
        report = json.loads((self.root / 'floor.json').read_text())
        self.assertAlmostEqual(report['eye_height'], 1.5, places=4)
        self.assertEqual(report['flattened'], n + len(SUNKEN))
        for a, b in zip(report['up'], (0, -1, 0)):
            self.assertAlmostEqual(a, b)

    def test_floor_can_be_disabled(self):
        self.a.floor = False
        j = self.job()
        self.reconstruct(j)
        texture = next(cmd for name, cmd in j.calls if name == '10_texture_v5')
        self.assertEqual(Path(texture[texture.index('-m') + 1]), self.root / 'mesh.ply')
        self.assertFalse((self.root / 'mesh_floor.ply').exists())

    def test_resume_with_new_face_count_redoes_mesh_onward(self):
        self.config.update(target_faces=500000, texture_size=4096, floor=True)
        self.reconstruct(self.job())
        cases = (({'target_faces': 3000000}, ['09_mesh_v2', '10_texture_v5'], True),
                 ({'texture_size': 8192}, ['10_texture_v5'], False),
                 ({'smooth': 2}, ['09_mesh_v2', '10_texture_v5'], True),
                 ({'virtual_face_images': 0}, ['10_texture_v5'], False),
                 ({'floor': False}, ['10_texture_v5'], False))
        for change, calls, floor_redone in cases:
            with self.subTest(change=change):
                (self.root / 'floor.json').write_text('{"stale": 1}')
                self.config.update(change)
                vars(self.a).update(change)
                resumed = self.job(resume=True)
                self.reconstruct(resumed)
                self.assertEqual([n for n, _ in resumed.calls], calls)
                self.assertEqual('stale' not in (self.root / 'floor.json').read_text(), floor_redone)
                self.assertEqual(json.loads((self.root / 'state.json').read_text())['config'], self.config)

    def test_mesh_keeps_weak_walls_and_closes_holes(self):
        j = self.job()
        self.reconstruct(j)
        mesh = next(cmd for name, cmd in j.calls if name == '09_mesh_v2')
        self.assertEqual(mesh[mesh.index('--free-space-support') + 1], 1)
        self.assertEqual(mesh[mesh.index('--close-holes') + 1], 300)
        self.assertEqual(mesh[mesh.index('--smooth') + 1], 8)
        texture = next(cmd for name, cmd in j.calls if name == '10_texture_v5')
        self.assertEqual(texture[texture.index('--virtual-face-images') + 1], 3)
        self.assertEqual(texture[texture.index('--max-texture-size') + 1], 8192)
        a = m.parse_args(['v.mp4', '-o', 'out', '--no-free-space-support', '--close-holes', '0'])
        self.assertEqual((a.free_space_support, a.close_holes), (False, 0))

    def test_resume_with_new_wall_settings_redoes_mesh_onward(self):
        self.reconstruct(self.job())
        for change in ({'free_space_support': True}, {'close_holes': 300}):
            with self.subTest(change=change):
                self.config.update(change)
                resumed = self.job(resume=True)
                self.reconstruct(resumed)
                self.assertEqual([n for n, _ in resumed.calls], ['09_mesh_v2', '10_texture_v5'])

    def test_resume_adds_floor_to_jobs_from_before_it(self):
        self.reconstruct(self.job())
        state = json.loads((self.root / 'state.json').read_text())
        state['completed'].pop('09b_floor')
        m.save(self.root / 'state.json', state)
        self.config['floor'] = True
        resumed = self.job(resume=True)
        self.reconstruct(resumed)
        self.assertEqual([n for n, _ in resumed.calls], ['10_texture_v5'])
        self.assertTrue((self.root / 'mesh_floor.ply').exists())

    def test_invalid_parameters(self):
        for extra in (['--fps', 'nan'], ['--duration', '0'], ['--gpus', '0;rm'], ['--start', '-1'],
                      ['--loop-images', '-1'], ['--close-holes', '-1'],
                      ['--smooth', '-1'], ['--virtual-face-images', '-1'], ['--face-size', '0']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    m.parse_args(['v.mp4', '-o', 'out', *extra])


if __name__ == '__main__':
    unittest.main()

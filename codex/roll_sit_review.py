"""Replay verification and evidence for the bounded frozen-model roll/sit study."""
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LP_NUM_THREADS'] = '2'
os.environ['MESA_SHADER_CACHE_DIR'] = '/tmp/roll_sit_shader'
from roll_sit_probe import *


def relaxed_low_upright(r):
    # Post-hoc sensitivity check; never substitutes for the registered criterion.
    return (r['tilt_deg'] <= 10 and .07 <= r['base_height'] <= .20
            and r['total_load'] >= .6 * r['weight_N']
            and r['com_speed'] <= .04 and r['angular_speed'] <= .2)


def verify():
    hashes = verify_frozen()
    w = GoalWorld()
    checks = []
    for task in ['roll', 'sit']:
        summary = json.loads((OUT2 / f'{task}_summary.json').read_text())
        for row in summary['results']:
            stem = f"{task}_{row['pose']}_{row['seed']}_{row['sign']:+d}"
            a = np.load(OUT2 / f'{stem}.npz')
            report = json.loads((OUT2 / f'{stem}.json').read_text())
            w.restore(a['initial'])
            frames, costs = [], []
            for action in a['actions']:
                _, cost, block = w.advance_full(action, trace=True)
                frames.extend(block)
                costs.append(cost)
            endpoint = w.snapshot()
            frames = np.array(frames)
            error = float(np.max(abs(frames[:, :109] - a['trace'][:, :109])))
            cost_error = float(np.max(abs(np.array(costs) - a['costs'])))
            assert error < 1e-8 and cost_error < 1e-7, (stem, error, cost_error)
            assert np.max(abs(frames[:, 85:109]) - w.torque_limit) < 1e-8
            r0 = np.array(report['initial_rotation'])
            rf = np.array(report['final_rotation'])
            tail, relaxed = [], []
            for frame in frames[-51:]:
                w.restore(frame)
                m = measurements(w, r0, rf)
                tail.append(valid(task, m))
                relaxed.append(relaxed_low_upright(m))
            w.restore(endpoint)
            m = measurements(w, r0, rf)
            success = bool(all(tail) and valid(task, m))
            assert success == report['success']
            checks.append({'trial': stem, 'success_recomputed': success,
                           'max_state_motor_replay_error': error,
                           'max_cost_replay_error': cost_error,
                           'torque_limits_verified': True,
                           'posthoc_low_upright_final_1s': bool(all(relaxed) and relaxed_low_upright(m))})
    # This is a continuation from an already observed state, not a new-start trial.
    a = np.load(OUT2 / 'sit_known_state_hold_diagnostic.npz')
    report = json.loads((OUT2 / 'sit_known_state_hold_diagnostic.json').read_text())
    w.restore(a['trigger'])
    target = w.d.qpos[7:].copy()
    frames = []
    for _ in range(7):
        action = np.clip((target - w.d.qpos[7:]) / ACTION_LIMIT, -1, 1)
        _, _, block = w.advance_full(action, trace=True)
        frames.extend(block)
    endpoint = w.snapshot()
    error = float(np.max(abs(np.array(frames)[:, :109] - a['hold_trace'][:, :109])))
    assert error < 1e-8, error
    w.restore(a['initial'])
    r0 = w.d.xmat[w.base].reshape(3, 3).copy()
    metrics = []
    for frame in frames[-51:]:
        w.restore(frame)
        metrics.append(measurements(w, r0, np.eye(3)))
    w.restore(endpoint)
    final = measurements(w, r0, np.eye(3))
    strict = bool(all(valid('sit', r) for r in metrics) and valid('sit', final))
    relaxed = bool(all(relaxed_low_upright(r) for r in metrics) and relaxed_low_upright(final))
    assert strict == report['success']
    diagnostic = {'scope': report['type'], 'strict_seated_success': strict,
                  'posthoc_low_upright_final_1s': relaxed,
                  'max_state_motor_replay_error': error, 'final': final,
                  'last_1s_ranges': {k: [min(r[k] for r in metrics), max(r[k] for r in metrics)]
                                      for k in ['tilt_deg', 'com_speed', 'angular_speed', 'seat_fraction']}}
    result = {'frozen_file_count': len(hashes), 'fresh_start_checks': checks,
              'known_state_continuation': diagnostic}
    (OUT2 / 'replay_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'fresh_trials_verified': len(checks), 'fresh_successes': sum(r['success_recomputed'] for r in checks),
                      'maximum_replay_error': max(r['max_state_motor_replay_error'] for r in checks),
                      'known_state_continuation': diagnostic}, indent=2), flush=True)


def render():
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    w = GoalWorld()
    renderer = mj.Renderer(w.m, 320, 480)
    camera = mj.MjvCamera()
    camera.distance = 1.3
    camera.azimuth = 120
    camera.elevation = -30
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    cases = [
        ('roll_prone_912_+1', 'ROLL: largest partial rotation | FAIL'),
        ('sit_supine_910_+1', 'SIT: fresh supine start | FAIL'),
        ('sit_known_state_hold_diagnostic', 'KNOWN-STATE HOLD: diagnostic only'),
    ]
    data = []
    for stem, label in cases:
        a = np.load(OUT2 / f'{stem}.npz')
        data.append((a, label))
    writer = imageio.get_writer(OUT2 / 'roll_sit_evidence.mp4', fps=12.5, quality=7)
    poster = None
    for index in range(0, max(len(a['trace']) for a, _ in data), 4):
        panels = []
        for a, label in data:
            j = min(index, len(a['trace']) - 1)
            w.restore(a['trace'][j])
            camera.lookat[:] = [w.d.qpos[0], w.d.qpos[1], .12]
            renderer.update_scene(w.d, camera)
            renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = False
            panel = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, 480, 65), fill=(18, 24, 31))
            tilt = np.rad2deg(np.arccos(np.clip(w.d.xmat[w.base].reshape(3, 3)[2, 2], -1, 1)))
            elapsed = w.d.time - a['initial'][-1]
            draw.text((7, 5), label, font=font, fill='white')
            draw.text((7, 25), f't={elapsed:.2f}s | trunk tilt from vertical={tilt:.1f} deg', font=font, fill='white')
            message = 'Recorded physics; frozen network + goal optimizer'
            if 'trigger' in a.files:
                message = 'Joint hold after 3.06s; excluded from fresh trials'
                if index >= len(a['trace']):
                    message = 'End of diagnostic (last frame held for display)'
            draw.text((7, 45), message, font=font, fill=(210, 220, 230))
            panels.append(np.asarray(panel))
        frame = np.concatenate(panels, axis=1)
        writer.append_data(frame)
        if index == 264:
            poster = frame.copy()
    writer.close()
    renderer.close()
    Image.fromarray(poster if poster is not None else frame).save(OUT2 / 'roll_sit_evidence.png')
    print(OUT2 / 'roll_sit_evidence.mp4', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['verify', 'render'])
    args = parser.parse_args()
    verify() if args.command == 'verify' else render()

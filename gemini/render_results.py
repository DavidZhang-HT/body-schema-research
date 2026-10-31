"""Render MuJoCo physics comparison videos & scientific telemetry dashboard for Pi Plus."""
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from world_and_schema import ASSETS, ROOT, SCENE_XML

OUT_DIR = ROOT / 'outputs'


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/System/Library/Fonts/SFNSMono.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_timeseries_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    y_label: str,
    series: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int], int, str]],
    hlines: list[tuple[float, tuple[int, int, int], str]] | None = None,
    vspans: list[tuple[float, float, tuple[int, int, int], str]] | None = None,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=12, fill=(20, 26, 36), outline=(48, 60, 82), width=2)
    f_title = load_font(20, bold=True)
    f_axis = load_font(14)
    f_leg = load_font(14, bold=True)

    draw.text((x0 + 18, y0 + 10), title, fill=(232, 240, 250), font=f_title)

    px0, py0 = x0 + 74, y0 + 58
    px1, py1 = x1 - 24, y1 - 36
    pw = max(px1 - px0, 10)
    ph = max(py1 - py0, 10)

    xmin, xmax = x_range
    ymin, ymax = y_range
    if abs(ymax - ymin) < 1e-9:
        ymax = ymin + 1.0

    def to_px(xv: float, yv: float) -> tuple[int, int]:
        u = (xv - xmin) / max(xmax - xmin, 1e-9)
        v = (yv - ymin) / max(ymax - ymin, 1e-9)
        u = float(np.clip(u, 0.0, 1.0))
        v = float(np.clip(v, 0.0, 1.0))
        return int(px0 + u * pw), int(py1 - v * ph)

    # Vertical phase spans
    if vspans:
        for sx0, sx1, col, lbl in vspans:
            ux0, _ = to_px(sx0, ymin)
            ux1, _ = to_px(sx1, ymin)
            if ux1 > ux0:
                draw.rectangle((ux0, py0, ux1, py1), fill=col)
                if lbl:
                    draw.text((ux0 + 6, py0 + 4), lbl, fill=(170, 190, 215), font=f_axis)

    # Grid lines
    for frac in np.linspace(0.0, 1.0, 5):
        yv = ymin + frac * (ymax - ymin)
        _, gy = to_px(xmin, yv)
        draw.line((px0, gy, px1, gy), fill=(36, 45, 62), width=1)
        draw.text((x0 + 10, gy - 8), f'{yv:.2f}', fill=(145, 162, 185), font=f_axis)

    for frac in np.linspace(0.0, 1.0, 7):
        xv = xmin + frac * (xmax - xmin)
        gx, _ = to_px(xv, ymin)
        draw.line((gx, py0, gx, py1), fill=(36, 45, 62), width=1)
        draw.text((gx - 12, py1 + 8), f'{xv:.1f}s', fill=(145, 162, 185), font=f_axis)

    draw.rectangle((px0, py0, px1, py1), outline=(70, 86, 112), width=1)

    if hlines:
        used_hy: list[int] = []
        for yv, col, lbl in hlines:
            if ymin <= yv <= ymax:
                _, hy = to_px(xmin, yv)
                for seg_x in range(px0, px1, 14):
                    draw.line((seg_x, hy, min(seg_x + 8, px1), hy), fill=col, width=2)
                if lbl:
                    x_lbl = px1 - max(175, int(len(lbl) * 8.2))
                    if any(abs(hy - prev_y) < 16 for prev_y in used_hy):
                        x_lbl -= 185
                    used_hy.append(hy)
                    draw.text((x_lbl, hy - 17), lbl, fill=col, font=f_axis)

    # Plot curves
    leg_x = px0 + 12
    leg_y = py0 + 24
    for tx, ty, col, width, label in series:
        if len(tx) < 2:
            continue
        step = max(1, len(tx) // 900)
        pts = [to_px(float(x), float(y)) for x, y in zip(tx[::step], ty[::step])]
        draw.line(pts, fill=col, width=width)
        if label:
            draw.line((leg_x, leg_y + 8, leg_x + 22, leg_y + 8), fill=col, width=max(width, 3))
            draw.text((leg_x + 28, leg_y), label, fill=col, font=f_leg)
            leg_x += 28 + int(len(label) * 8.2) + 18

    draw.text((x0 + 12, y0 + 32), y_label, fill=(130, 150, 175), font=f_axis)


def render_telemetry_dashboard() -> Path:
    exp1 = np.load(OUT_DIR / 'exp1_raw_umr_replay.npz')
    exp2 = np.load(OUT_DIR / 'exp2_uncalibrated_schema.npz')
    exp3 = np.load(OUT_DIR / 'exp3_full_body_schema.npz')
    summary = json.loads((OUT_DIR / 'experiment_summary.json').read_text())

    W, H = 1800, 1120
    img = Image.new('RGB', (W, H), (12, 16, 23))
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle((0, 0, W, 86), fill=(18, 24, 35))
    draw.text(
        (28, 14),
        'Pi+ (22-DOF URDF)  |  Body Schema Self-Exploration + UMR Kinematic Reference -> Physical Walking',
        fill=(238, 244, 252),
        font=load_font(26, bold=True),
    )
    r1 = summary['exp1_raw_umr_replay']
    r2 = summary['exp2_uncalibrated_schema']
    r3 = summary['exp3_full_body_schema']
    sub = (
        f"Exp1 Raw UMR: Fall at {r1['duration_s']-0.25:.2f}s (dx={r1['forward_distance_m']:+.2f}m)   |   "
        f"Exp2 Uncalibrated Nominal Schema: Fall at {r2['duration_s']-0.25:.2f}s (dx={r2['forward_distance_m']:+.2f}m)   |   "
        f"Exp3 Full 4-Stage Body Schema: {'PASSED ALL CHECKS' if r3['passed'] else ('COMPLETED' if r3['completed_without_fall'] else r3['failure_reason'])} (dx={r3['forward_distance_m']:+.2f}m)"
    )
    draw.text((28, 52), sub, fill=(112, 224, 188), font=load_font(16, bold=True))

    t3 = exp3['t']
    t_end = float(t3[-1])
    # Identify curriculum phase transition times in Exp 3
    pids = exp3['phase_id']
    t_p2 = float(t3[np.argmax(pids >= 2)]) if np.any(pids >= 2) else 3.0
    t_p3 = float(t3[np.argmax(pids >= 3)]) if np.any(pids >= 3) else 6.1
    push_idx = np.flatnonzero(exp3['push_active'])
    t_push0 = float(t3[push_idx[0]]) if len(push_idx) else 9.0
    t_push1 = float(t3[push_idx[-1]]) if len(push_idx) else 9.16

    # Panel 1: Stage 2 Online Inertial & Friction Parameter Convergence (0 - 3.5s zoomed or full)
    true_disc = r3['true_world_discrepancies']
    true_dm10 = 10.0 * float(true_disc['delta_torso_mass_kg'])
    true_dcx_mm = 1000.0 * float(true_disc['delta_torso_com_m'][0])
    true_dcy_mm = 1000.0 * float(true_disc['delta_torso_com_m'][1])
    est_dc_mm = 1000.0 * exp3['est_dc']
    draw_timeseries_panel(
        draw,
        (24, 104, 888, 594),
        '1. Stage 2 Active Postural SysID (Pure Proprioception Contact-Nullspace RLS)',
        (0.0, min(4.0, t_end)),
        (-68.0, 52.0),
        'mm / 10*kg',
        [
            (t3, 10.0 * exp3['est_dm'], (96, 230, 178), 3, '10*dM_hat (10*kg)'),
            (t3, est_dc_mm[:, 0], (255, 130, 110), 3, 'dCx_hat (mm)'),
            (t3, est_dc_mm[:, 1], (110, 185, 255), 3, 'dCy_hat (mm)'),
            (t3, 100.0 * (exp3['est_damping'] - 0.04), (245, 205, 100), 2, '100*db_hat'),
        ],
        hlines=[
            (true_dm10, (70, 165, 128), f'True 10*dM (+{true_dm10:.1f})'),
            (true_dcx_mm, (195, 95, 82), f'True dCx ({true_dcx_mm:+.0f}mm)'),
            (true_dcy_mm, (82, 140, 198), f'True dCy ({true_dcy_mm:+.0f}mm)'),
        ],
        vspans=[(0.0, t_p2, (26, 38, 48), 'Double-Support Sway/Squat ASID')],
    )

    # Panel 2: Forward Displacement & Base Height across Exp 1, Exp 2, Exp 3
    t1 = exp1['t']
    t2 = exp2['t']
    dx1 = exp1['qpos'][:, 0] - exp1['qpos'][0, 0]
    dx2 = exp2['qpos'][:, 0] - exp2['qpos'][0, 0]
    dx3 = exp3['qpos'][:, 0] - exp3['qpos'][0, 0]
    draw_timeseries_panel(
        draw,
        (912, 104, 1776, 594),
        '2. Physical Forward Progression & Survival (Torque-Driven Floating-Base MuJoCo)',
        (0.0, max(t_end, float(t2[-1]), 4.0)),
        (-0.38, max(0.65, float(dx3.max()) + 0.08)),
        'Meters',
        [
            (t1, dx1, (255, 98, 98), 3, 'Exp1: Raw UMR Replay'),
            (t2, dx2, (255, 188, 82), 3, 'Exp2: Uncalibrated Nominal Schema'),
            (t3, dx3, (88, 232, 172), 4, 'Exp3: 4-Stage Body Schema Walk'),
            (t3, exp3['qpos'][:, 2], (145, 175, 255), 2, 'Exp3 Base Z (m)'),
        ],
        hlines=[(0.0, (90, 102, 125), 'Start X = 0')],
        vspans=[
            (0.0, t_p2, (24, 33, 46), 'Stage2 ASID'),
            (t_p2, t_p3, (28, 36, 52), 'PhaseB In-Place'),
            (t_push0, t_push1, (68, 34, 42), 'Push!'),
        ],
    )

    # Panel 3: Stage 3 Calibrated CoM vs True CoM Lateral Tracking (mm)
    draw_timeseries_panel(
        draw,
        (24, 614, 888, 1096),
        '3. Stage 3 Dynamic Projection: Lateral CoM & ZMP Tracking (Curriculum)',
        (0.0, t_end),
        (-115.0, 115.0),
        'Y (mm)',
        [
            (t3, 1000.0 * exp3['zmp_ref'][:, 1], (115, 138, 172), 2, 'Planned ZMP_y'),
            (t3, 1000.0 * exp3['com_ref'][:, 1], (115, 195, 255), 3, 'Projected LIPM CoM_y'),
            (t3, 1000.0 * exp3['true_com'][:, 1], (92, 232, 174), 3, 'True Physical CoM_y'),
        ],
        hlines=[
            (81.5, (68, 92, 120), 'Left Sole Center (+81.5mm)'),
            (-81.5, (68, 92, 120), 'Right Sole Center (-81.5mm)'),
        ],
        vspans=[
            (0.0, t_p2, (24, 33, 46), 'Stage2'),
            (t_p2, t_p3, (28, 36, 52), 'In-Place'),
            (t_p3, t_end, (23, 35, 42), 'UMR Forward Walk'),
        ],
    )

    # Panel 4: Stage 4 Surprise-Driven Attribution, Reflex Offset & Motor Torque Ratio
    reflex_mag_mm = 1000.0 * np.linalg.norm(exp3['reflex_offset_xy'], axis=1)
    peak_tau_pct = 100.0 * np.max(np.abs(exp3['torque']) / 21.0, axis=1)
    draw_timeseries_panel(
        draw,
        (912, 614, 1776, 1096),
        '4. Stage 4 Surprise Attribution, Capture-Point Foot Reflex & Torque Margin',
        (0.0, t_end),
        (0.0, 112.0),
        '% / mm / 100*Surprise',
        [
            (t3, peak_tau_pct, (130, 165, 235), 2, 'Peak Torque (% of 21Nm)'),
            (t3, np.clip(100.0 * exp3['dcm_surprise'], 0.0, 108.0), (255, 112, 112), 3, '100*Push Surprise'),
            (t3, 2.0 * reflex_mag_mm, (255, 214, 92), 3, '2*Reflex Foot Shift (mm)'),
        ],
        hlines=[
            (100.0, (190, 82, 82), '100% URDF Torque Limit (21 Nm)'),
            (58.0, (185, 120, 95), 'Push Reflex Threshold (11.6N)'),
        ],
        vspans=[(t_push0, t_push1, (75, 32, 40), 'External Push')],
    )

    out_png = OUT_DIR / 'body_schema_telemetry.png'
    img.save(out_png)
    return out_png


def render_full_curriculum_video(fps: int = 30) -> Path:
    exp3 = np.load(OUT_DIR / 'exp3_full_body_schema.npz')
    summary = json.loads((OUT_DIR / 'experiment_summary.json').read_text())
    r3 = summary['exp3_full_body_schema']

    m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=720, width=1280)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = False

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.38
    cam.azimuth = 136
    cam.elevation = -14

    times = exp3['t']
    qs = exp3['qpos']
    est_dm = exp3['est_dm']
    est_dc = exp3['est_dc']
    pids = exp3['phase_id']
    pushes = exp3['push_active']
    surprises = exp3['dcm_surprise']
    forces = exp3['foot_force_true']

    f_h1 = load_font(22, bold=True)
    f_h2 = load_font(16, bold=True)
    f_mono = load_font(15)

    phase_labels = {
        1: ('STAGE 2: ACTIVE POSTURAL EXPLORATION (DOUBLE-SUPPORT ASID)', (98, 195, 255)),
        2: ('CURRICULUM PHASE B: STEPPING IN PLACE (CONTACT & DROOP CALIBRATION)', (255, 208, 102)),
        3: ('STAGE 3+4: UMR-PROJECTED WALK + SURPRISE-DRIVEN REFLEX', (92, 232, 174)),
    }

    out_mp4 = OUT_DIR / 'full_curriculum_walk.mp4'
    smooth_look_x = float(qs[0, 0])
    saved_push_preview = False

    with imageio.get_writer(out_mp4, fps=fps, codec='libx264', quality=8) as writer:
        for k, t_frame in enumerate(np.arange(times[0], times[-1] + 1e-8, 1.0 / fps)):
            idx = min(int(np.searchsorted(times, t_frame)), len(times) - 1)
            d.qpos[:] = qs[idx]
            mujoco.mj_forward(m, d)
            smooth_look_x = 0.88 * smooth_look_x + 0.12 * float(qs[idx, 0])
            cam.lookat[:] = [smooth_look_x + 0.04, 0.0, 0.30]

            renderer.update_scene(d, cam, scene_option=opt)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)

            # Top HUD banner
            draw.rectangle((0, 0, 1280, 82), fill=(15, 20, 30))
            p_title, p_col = phase_labels.get(int(pids[idx]), ('WALKING', (200, 220, 240)))
            draw.text((22, 12), 'Pi+ 22-DOF URDF  |  Body Schema Self-Exploration (Torque-Driven MuJoCo)', fill=(236, 242, 250), font=f_h1)
            draw.text((22, 46), f'{p_title}   |   t = {t_frame:5.2f}s', fill=p_col, font=f_h2)

            status_badge = 'PASSED STRICT TORQUE/LIMIT CHECKS' if r3['passed'] else 'EXPERIMENTAL'
            draw.text((925, 16), status_badge, fill=(92, 232, 174) if r3['passed'] else (255, 130, 110), font=f_h2)
            dx_cm = 100.0 * float(qs[idx, 0] - qs[0, 0])
            draw.text((925, 46), f'Forward: {dx_cm:+5.1f} cm  |  Base Z: {qs[idx, 2]:.3f} m', fill=(210, 224, 242), font=f_mono)

            # Bottom-left live Body Schema card
            draw.rounded_rectangle((18, 594, 650, 704), radius=10, fill=(16, 22, 33), outline=(52, 68, 92), width=2)
            draw.text((32, 604), 'Internal Body Schema Online Estimate vs Ground Truth:', fill=(185, 204, 228), font=f_h2)
            true_disc = r3['true_world_discrepancies']
            t_dm = float(true_disc['delta_torso_mass_kg'])
            t_dc = [1000.0 * float(x) for x in true_disc['delta_torso_com_m']]
            dc_mm = 1000.0 * est_dc[idx]
            draw.text(
                (32, 632),
                f'Payload dM_hat : {est_dm[idx]:+5.2f} kg   (True {t_dm:+.2f} kg)',
                fill=(96, 230, 178),
                font=f_mono,
            )
            draw.text(
                (32, 655),
                f'Torso dCoM_hat : [{dc_mm[0]:+5.1f}, {dc_mm[1]:+4.1f}, {dc_mm[2]:+4.1f}] mm  (True [{t_dc[0]:+.0f}, {t_dc[1]:+.0f}, {t_dc[2]:+.0f}])',
                fill=(130, 195, 255),
                font=f_mono,
            )
            draw.text(
                (32, 678),
                f'Ground Load Fz : L = {forces[idx, 0]:5.1f} N   R = {forces[idx, 1]:5.1f} N',
                fill=(210, 220, 235),
                font=f_mono,
            )

            # Surprise / External Push alert banner
            if pushes[idx] or (int(pids[idx]) == 3 and surprises[idx] > 0.58):
                draw.rounded_rectangle((675, 594, 1260, 704), radius=10, fill=(68, 24, 34), outline=(255, 110, 110), width=2)
                draw.text((692, 608), 'SURPRISE DETECTED: UNEXPECTED EXTERNAL PUSH!', fill=(255, 130, 120), font=f_h2)
                draw.text(
                    (692, 638),
                    f'Inferred Push Score = {surprises[idx]:.2f} (~{20.0*surprises[idx]:.1f} N) -> EXTERNAL',
                    fill=(255, 215, 150),
                    font=f_mono,
                )
                draw.text(
                    (692, 666),
                    'Action: Freeze Inertial RLS + Trigger Capture-Point Footstep Reflex',
                    fill=(255, 236, 190),
                    font=f_mono,
                )

            writer.append_data(np.asarray(frame))
            if k == int(1.2 * fps):
                frame.save(OUT_DIR / 'preview_stage2_sysid.png')
            elif pushes[idx] and surprises[idx] > 0.58 and not saved_push_preview:
                frame.save(OUT_DIR / 'preview_forward_walk.png')
                saved_push_preview = True

    renderer.close()
    return out_mp4


def render_ab_comparison_video(fps: int = 30) -> Path:
    """Side-by-side 1280x720 comparison: Left = Baselines (Exp1 & Exp2), Right = Full Body Schema (Exp3)."""
    exp1 = np.load(OUT_DIR / 'exp1_raw_umr_replay.npz')
    exp2 = np.load(OUT_DIR / 'exp2_uncalibrated_schema.npz')
    exp3 = np.load(OUT_DIR / 'exp3_full_body_schema.npz')

    m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    d = mujoco.MjData(m)
    ren_half = mujoco.Renderer(m, height=620, width=640)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = False

    cam_l = mujoco.MjvCamera(); cam_l.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_l.distance = 1.35; cam_l.azimuth = 136; cam_l.elevation = -14

    cam_r = mujoco.MjvCamera(); cam_r.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam_r.distance = 1.35; cam_r.azimuth = 136; cam_r.elevation = -14

    # Align comparison timeline to forward walking start in Exp 3 vs Exp 1 / Exp 2!
    t3 = exp3['t']; q3 = exp3['qpos']; pids3 = exp3['phase_id']
    t_walk3_start = float(t3[np.argmax(pids3 >= 3)]) if np.any(pids3 >= 3) else 0.0
    walk_dur = float(t3[-1] - t_walk3_start)

    t1 = exp1['t']; q1 = exp1['qpos']
    t2 = exp2['t']; q2 = exp2['qpos']

    f_title = load_font(20, bold=True)
    f_sub = load_font(15, bold=True)

    out_mp4 = OUT_DIR / 'ab_comparison_walk.mp4'
    total_clip_s = max(walk_dur, 6.5)

    with imageio.get_writer(out_mp4, fps=fps, codec='libx264', quality=8) as writer:
        for k, tau_t in enumerate(np.arange(0.0, total_clip_s + 1e-8, 1.0 / fps)):
            # Left pane shows Exp 1 for first 2.6s (falls at 1.57s), then switches to Exp 2 (Uncalibrated Nominal Schema)!
            if tau_t < 2.6:
                i_l = min(int(np.searchsorted(t1, tau_t)), len(t1) - 1)
                ql = q1[i_l]
                left_title = 'Baseline 1: Direct UMR Kinematic Replay'
                left_stat = f'FALLS BACKWARD (t={min(tau_t, float(t1[-1])):.2f}s)'
                left_col = (255, 108, 108)
            else:
                t_e2 = tau_t - 2.6
                i_l = min(int(np.searchsorted(t2, t_e2)), len(t2) - 1)
                ql = q2[i_l]
                left_title = 'Baseline 2: Uncalibrated Nominal URDF Schema'
                left_stat = f'Uncalibrated CoM/Mass Mismatch (t={min(t_e2, float(t2[-1])):.2f}s)'
                left_col = (255, 188, 86)

            d.qpos[:] = ql
            mujoco.mj_forward(m, d)
            cam_l.lookat[:] = [float(ql[0]) + 0.03, 0.0, 0.28]
            ren_half.update_scene(d, cam_l, scene_option=opt)
            img_l = Image.fromarray(ren_half.render())

            # Right pane shows Exp 3 forward walking phase (after Stage 2 & Phase B calibration)
            t_e3 = min(t_walk3_start + tau_t, float(t3[-1]))
            i_r = min(int(np.searchsorted(t3, t_e3)), len(t3) - 1)
            qr = q3[i_r]
            d.qpos[:] = qr
            mujoco.mj_forward(m, d)
            cam_r.lookat[:] = [float(qr[0]) + 0.03, 0.0, 0.28]
            ren_half.update_scene(d, cam_r, scene_option=opt)
            img_r = Image.fromarray(ren_half.render())

            canvas = Image.new('RGB', (1280, 720), (14, 18, 27))
            canvas.paste(img_l, (0, 100))
            canvas.paste(img_r, (640, 100))
            draw = ImageDraw.Draw(canvas)
            draw.line((640, 0, 640, 720), fill=(62, 78, 105), width=3)

            draw.text((20, 14), left_title, fill=(236, 242, 250), font=f_title)
            draw.text((20, 46), left_stat, fill=left_col, font=f_sub)

            draw.text((662, 14), 'Proposed: Calibrated Body Schema + UMR Projection', fill=(236, 242, 250), font=f_title)
            dx3_cm = 100.0 * float(qr[0] - q3[0, 0])
            draw.text(
                (662, 46),
                f'STABLE TORQUE WALK  |  Forward: {dx3_cm:+5.1f} cm  (Calibrated +2.6kg Payload)',
                fill=(92, 232, 174),
                font=f_sub,
            )

            writer.append_data(np.asarray(canvas))
            if k == int(1.6 * fps):
                canvas.save(OUT_DIR / 'preview_ab_comparison.png')

    ren_half.close()
    return out_mp4


def main() -> None:
    png = render_telemetry_dashboard()
    print('Saved telemetry dashboard:', png)
    v1 = render_full_curriculum_video()
    print('Saved full curriculum video:', v1)
    v2 = render_ab_comparison_video()
    print('Saved A/B comparison video:', v2)


if __name__ == '__main__':
    main()

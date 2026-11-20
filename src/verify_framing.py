"""
Exact 9:16 Blender HORIZONTAL-Fit NDC Projection Pre-Flight Gate for Scientist #8: Robert Hooke
Verifies ALL 6 Beats fit cleanly inside the 9:16 portrait safe frame (|ndcX| <= 0.86, ndcY in [-0.55, +0.86]).
"""
import math
import numpy as np

RES_X, RES_Y = 1080, 1920
ASPECT = RES_X / RES_Y
SENSOR_W = 36.0
MIC_POS = np.array([-0.88, 0.34, 0.0])
OVER_X, OVER_Y = 0.94, -0.02
WATCH_POS = np.array([0.02, 0.48, 0.42])


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def smoothstep(e0, e1, x):
    t = clamp01((x - e0) / max(1e-6, e1 - e0))
    return t * t * (3.0 - 2.0 * t)


def ease_out_quint(x):
    t = clamp01(x)
    return 1.0 - (1.0 - t) ** 5


def lerp(a, b, t):
    return a + (b - a) * t


def bezier_quad(p0, p1, p2, t):
    u = 1.0 - t
    return p0 * (u * u) + p1 * (2.0 * u * t) + p2 * (t * t)


def compute_camera_state(t):
    if t < 4.5:
        u = clamp01(t / 4.5)
        pos = bezier_quad(np.array([-0.44, -1.28, 0.48]), np.array([-0.30, -1.14, 0.44]), np.array([-0.18, -1.04, 0.40]), u)
        tgt = lerp(np.array([-0.26, -0.04, 0.36]), np.array([-0.24, -0.04, 0.34]), u)
        lens = lerp(40.0, 44.0, u)
    elif t < 9.5:
        pan_u = ease_out_quint(clamp01((t - 7.0) / 1.1))
        pos = lerp(np.array([0.26, -1.06, 0.38]), np.array([-0.02, -1.44, 0.46]), pan_u)
        tgt = lerp(np.array([0.25, -0.04, 0.22]), np.array([-0.04, -0.04, 0.32]), pan_u)
        lens = lerp(40.0, 36.0, pan_u)
    elif t < 15.5:
        u = clamp01((t - 9.5) / 6.0)
        ang = math.radians(-18.0 + 38.0 * u)
        rad = lerp(1.18, 1.04, math.sin(u * math.pi))
        pos = np.array([MIC_POS[0] + 0.06 + math.sin(ang) * rad * 0.65, MIC_POS[1] - math.cos(ang) * rad, lerp(0.52, 0.70, u)])
        tgt = lerp(np.array([MIC_POS[0] + 0.04, MIC_POS[1], 0.26]), np.array([MIC_POS[0] + 0.04, MIC_POS[1], 0.40]), smoothstep(10.5, 13.0, t))
        lens = lerp(39.0, 43.0, u)
    elif t < 20.5:
        u = clamp01((t - 15.5) / 5.0)
        ang = math.radians(-14.0 + 30.0 * u)
        rad = lerp(0.98, 1.06, u)
        pos = np.array([OVER_X + math.sin(ang) * rad * 0.55, OVER_Y - math.cos(ang) * rad, lerp(0.44, 0.50, u)])
        tgt = np.array([OVER_X, OVER_Y, 0.34])
        lens = lerp(40.0, 42.0, u)
    elif t < 25.5:
        u = clamp01((t - 20.5) / 5.0)
        pos = bezier_quad(np.array([-0.20, -0.76, 0.54]), np.array([0.02, -0.82, 0.62]), np.array([0.20, -0.88, 0.68]), u)
        tgt = lerp(np.array([WATCH_POS[0], WATCH_POS[1], 0.44]), np.array([WATCH_POS[0], WATCH_POS[1], 0.46]), u)
        lens = lerp(42.0, 40.0, u)
    else:
        u = clamp01((t - 25.5) / 4.5)
        pos = bezier_quad(np.array([0.10, -1.96, 0.80]), np.array([0.04, -2.22, 0.78]), np.array([0.0, -2.42, 0.74]), u)
        tgt = lerp(np.array([0.0, 0.34, 0.58]), np.array([0.0, 0.26, 0.56]), u)
        lens = lerp(34.0, 31.0, u)
    return pos, tgt, lens


def project(world_pt, cam_pos, cam_tgt, lens):
    fwd = cam_tgt - cam_pos
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)
    rel = world_pt - cam_pos
    cx, cy, cz = float(np.dot(rel, right)), float(np.dot(rel, up)), float(np.dot(rel, fwd))
    tan_hx = (SENSOR_W * 0.5) / lens
    tan_hy = tan_hx / ASPECT
    return cx / (cz * tan_hx), cy / (cz * tan_hy), cz


checks = [
    ("Beat 1 Spring Rig (t=2.6s)", 2.6, {
        "RigLeft": np.array([-0.55, -0.04, 0.36]),
        "RigTop": np.array([-0.34, -0.04, 0.72]),
        "RulerRight": np.array([-0.11, -0.04, 0.36]),
    }),
    ("Beat 2a Cipher (t=6.2s)", 6.2, {
        "FolioLeft": np.array([0.09, -0.04, 0.04]),
        "FolioRight": np.array([0.43, -0.04, 0.04]),
        "CipherText": np.array([0.26, -0.04, 0.26]),
    }),
    ("Beat 2b Formula+Spring (t=8.5s)", 8.5, {
        "SpringLeft": np.array([-0.48, -0.04, 0.36]),
        "FormulaRight": np.array([0.44, -0.04, 0.28]),
    }),
    ("Beat 3 Microscope+Cells (t=12.8s)", 12.8, {
        "MicLeft": np.array([-1.02, 0.34, 0.22]),
        "CondenserR": np.array([-0.62, 0.28, 0.16]),
        "CorkCellsTop": np.array([-0.84, 0.32, 0.54]),
    }),
    ("Beat 4 Overload Snap (t=17.8s)", 17.8, {
        "OverLeft": np.array([0.76, -0.02, 0.34]),
        "OverRight": np.array([1.12, -0.02, 0.34]),
        "OverTop": np.array([0.94, -0.02, 0.65]),
    }),
    ("Beat 5 Pocket Watch (t=23.2s)", 23.2, {
        "WatchLeft": np.array([-0.26, 0.48, 0.42]),
        "WatchRight": np.array([0.30, 0.48, 0.42]),
        "WatchCrown": np.array([0.02, 0.48, 0.76]),
    }),
    ("Beat 6 Grand Finale (t=28.2s)", 28.2, {
        "MicroscopeL": np.array([-0.92, 0.34, 0.30]),
        "OverloadR": np.array([0.96, -0.02, 0.34]),
        "DNATop": np.array([0.0, 0.20, 1.20]),
        "TitleRight": np.array([0.55, 0.58, 1.32]),
    }),
]

errors = []
for label, t, pts in checks:
    cpos, ctgt, lens = compute_camera_state(t)
    print(f"\n[{label}] cam={np.round(cpos,2)} tgt={np.round(ctgt,2)} lens={lens:.1f}mm")
    for name, p in pts.items():
        nx, ny, cz = project(p, cpos, ctgt, lens)
        status = "OK"
        if abs(nx) > 0.86 or ny > 0.86 or ny < -0.55:
            status = "CLIP!"
            errors.append(f"{label} {name}: ndc=({nx:+.2f}, {ny:+.2f})")
        print(f"  {name:14s}: ndcX={nx:+.2f}  ndcY={ny:+.2f}  depth={cz:.2f}m  [{status}]")

if errors:
    print(f"\nFAILED ({len(errors)} framing clips):")
    for e in errors:
        print(" -", e)
    raise SystemExit(1)
print("\nPASSED: 100% of all 6 Beats fit cleanly inside 9:16 portrait safe frame!")

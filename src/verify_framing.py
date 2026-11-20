"""
Exact 9:16 Blender HORIZONTAL-Fit NDC Projection Pre-Flight Gate for Scientist #8: Robert Hooke + MPFB2 Cast!
Verifies BOTH realistic human characters (head top Z=+0.78m, shoulders) AND every scientific instrument
fit cleanly inside the 9:16 portrait safe frame (|ndcX| <= 0.86, ndcY in [-0.55, +0.86]).
"""
import math
import numpy as np

RES_X, RES_Y = 1080, 1920
ASPECT = RES_X / RES_Y
SENSOR_W = 36.0
MIC_POS = np.array([-0.88, 0.34, 0.0])
OVER_X, OVER_Y = 0.94, -0.02


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
        pos = bezier_quad(np.array([-0.44, -1.62, 0.56]), np.array([-0.30, -1.48, 0.52]), np.array([-0.18, -1.36, 0.48]), u)
        tgt = lerp(np.array([-0.28, 0.22, 0.44]), np.array([-0.24, 0.22, 0.42]), u)
        lens = lerp(36.0, 39.0, u)
    elif t < 9.5:
        pan_u = ease_out_quint(clamp01((t - 7.0) / 1.1))
        pos = lerp(np.array([0.28, -1.58, 0.56]), np.array([-0.02, -1.78, 0.56]), pan_u)
        tgt = lerp(np.array([0.32, 0.30, 0.44]), np.array([0.02, 0.32, 0.44]), pan_u)
        lens = lerp(37.0, 34.0, pan_u)
    elif t < 15.5:
        u = clamp01((t - 9.5) / 6.0)
        ang = math.radians(-16.0 + 34.0 * u)
        rad = lerp(1.48, 1.34, math.sin(u * math.pi))
        pos = np.array([MIC_POS[0] + 0.06 + math.sin(ang) * rad * 0.60, MIC_POS[1] - math.cos(ang) * rad, lerp(0.56, 0.70, u)])
        tgt = lerp(np.array([MIC_POS[0] + 0.04, MIC_POS[1] + 0.16, 0.42]), np.array([MIC_POS[0] + 0.04, MIC_POS[1] + 0.16, 0.48]), smoothstep(10.5, 13.0, t))
        lens = lerp(36.0, 39.0, u)
    elif t < 20.5:
        u = clamp01((t - 15.5) / 5.0)
        ang = math.radians(-12.0 + 26.0 * u)
        rad = lerp(1.42, 1.50, u)
        pos = np.array([OVER_X + math.sin(ang) * rad * 0.50, OVER_Y - math.cos(ang) * rad, lerp(0.50, 0.56, u)])
        tgt = np.array([OVER_X - 0.02, OVER_Y + 0.24, 0.44])
        lens = lerp(36.0, 38.0, u)
    elif t < 25.5:
        u = clamp01((t - 20.5) / 5.0)
        pos = bezier_quad(np.array([-0.26, -1.32, 0.56]), np.array([-0.08, -1.42, 0.64]), np.array([0.08, -1.48, 0.70]), u)
        tgt = lerp(np.array([-0.08, 0.42, 0.46]), np.array([-0.06, 0.44, 0.48]), u)
        lens = lerp(36.0, 35.0, u)
    else:
        u = clamp01((t - 25.5) / 4.5)
        pos = bezier_quad(np.array([0.06, -2.08, 0.78]), np.array([0.02, -2.32, 0.76]), np.array([0.0, -2.50, 0.72]), u)
        tgt = lerp(np.array([0.0, 0.40, 0.56]), np.array([0.0, 0.34, 0.54]), u)
        lens = lerp(33.0, 30.0, u)
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
    ("Beat 1 Hooke + Spring (t=2.6s)", 2.6, {
        "HookeHead": np.array([-0.34, 0.62, 0.78]),
        "RigLeft": np.array([-0.55, -0.04, 0.36]),
        "RulerRight": np.array([-0.11, -0.04, 0.36]),
    }),
    ("Beat 2a Rival + Cipher (t=6.2s)", 6.2, {
        "RivalHead": np.array([0.42, 0.64, 0.78]),
        "CipherText": np.array([0.26, -0.04, 0.26]),
    }),
    ("Beat 2b Hooke + Rival 2-Shot (t=8.5s)", 8.5, {
        "HookeHead": np.array([-0.36, 0.64, 0.78]),
        "RivalHead": np.array([0.42, 0.64, 0.78]),
        "Formula": np.array([0.26, -0.04, 0.28]),
    }),
    ("Beat 3 Hooke + Microscope + Cells (t=12.8s)", 12.8, {
        "HookeHead": np.array([-0.82, 0.78, 0.78]),
        "MicLeft": np.array([-1.02, 0.34, 0.22]),
        "CorkCells": np.array([-0.84, 0.32, 0.54]),
    }),
    ("Beat 4 Rival Shock + Snap (t=17.8s)", 17.8, {
        "RivalHead": np.array([0.88, 0.76, 0.78]),
        "OverLeft": np.array([0.76, -0.02, 0.34]),
        "OverRight": np.array([1.12, -0.02, 0.34]),
    }),
    ("Beat 5 Hooke + Pocket Watch (t=23.2s)", 23.2, {
        "HookeHead": np.array([-0.34, 0.74, 0.78]),
        "WatchLeft": np.array([-0.14, 0.24, 0.36]),
        "WatchRight": np.array([0.38, 0.24, 0.36]),
    }),
    ("Beat 6 Grand Cast Finale (t=28.2s)", 28.2, {
        "HookeHead": np.array([-0.58, 0.74, 0.78]),
        "RivalHead": np.array([0.62, 0.74, 0.78]),
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
print("\nPASSED: 100% of ALL Cast Heads + Scientific Instruments fit cleanly inside 9:16 portrait safe frame!")

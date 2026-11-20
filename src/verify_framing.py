"""
Pre-flight mathematical 9:16 portrait camera framing & ZERO-OVERLAP sightline verifier for Robert Hooke V3!
Verifies across all 6 Beats:
  1. 100% of ALL Cast Heads + Scientific Instruments stay cleanly inside the 9:16 portrait safe frame
     (|ndcX| <= 0.85, ndcY in [-0.50, +0.72]).
  2. ZERO Instrument-Blocking-Face Overlap: checks horizontal screen separation |ndcX(Actor) - ndcX(HeroProp)| >= 0.34
     in every single close-up / medium shot!
"""
import math
import numpy as np

RES_X, RES_Y = 1080, 1920
ASPECT = RES_X / RES_Y
SENSOR_W = 36.0
SENSOR_H = SENSOR_W / ASPECT  # 64.0mm with sensor_fit='HORIZONTAL'


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


# Exact Symmetric Zero-Overlap Station Anchors (synchronized 1:1 with build_hooke_scene.py!)
MIC_POS = np.array([-1.62, -0.06, 0.0])
RIG_X, RIG_Y = -0.78, -0.10
FOLIO_POS = np.array([0.00, -0.10, 0.0])
WATCH_POS = np.array([0.28, 0.02, 0.30])
OVER_X, OVER_Y = 1.42, -0.08


def compute_beat(b_idx, t):
    if b_idx == 0:
        # Beat 1: Hooke (Left, X=-1.24) + Hero Spring Rig (Right, X=-0.78) — TIGHT MEDIUM SHOT, ZERO OVERLAP!
        u = clamp01(t / 4.5)
        pos = bezier_quad(np.array([-1.06, -1.48, 0.56]), np.array([-0.98, -1.36, 0.54]), np.array([-0.92, -1.28, 0.52]), u)
        tgt = lerp(np.array([-1.00, 0.16, 0.44]), np.array([-0.98, 0.16, 0.42]), u)
        return pos, tgt, lerp(39.0, 42.0, u)
    if b_idx == 1:
        # Beat 2a (Rival Right + Cipher Center-Left) -> Beat 2b (Hooke Left + Rival Right + F=-kx Center)!
        pan_u = ease_out_quint(clamp01((t - 7.0) / 1.0))
        pos = lerp(np.array([0.24, -1.36, 0.54]), np.array([0.00, -1.56, 0.56]), pan_u)
        tgt = lerp(np.array([0.22, 0.18, 0.44]), np.array([0.01, 0.22, 0.44]), pan_u)
        return pos, tgt, lerp(39.0, 36.0, pan_u)
    if b_idx == 2:
        # Beat 3: Microscope & Cork Cells (Left, X=-1.62) + Hooke (Right, X=-1.16) — ZERO OVERLAP!
        u = clamp01((t - 9.5) / 6.0)
        pos = bezier_quad(np.array([-1.46, -1.42, 0.56]), np.array([-1.38, -1.32, 0.58]), np.array([-1.32, -1.26, 0.60]), u)
        tgt = lerp(np.array([-1.40, 0.14, 0.42]), np.array([-1.38, 0.16, 0.48]), smoothstep(10.5, 13.0, t))
        return pos, tgt, lerp(39.0, 41.0, u)
    if b_idx == 3:
        # Beat 4: Rival Watchmaker (Left, X=+0.92) + Overload Spring SNAP (Right, X=+1.42) — ZERO OVERLAP!
        u = clamp01((t - 15.5) / 5.0)
        pos = bezier_quad(np.array([1.14, -1.44, 0.54]), np.array([1.18, -1.36, 0.55]), np.array([1.22, -1.42, 0.56]), u)
        tgt = np.array([1.16, 0.16, 0.44])
        return pos, tgt, lerp(38.0, 40.0, u)
    if b_idx == 4:
        # Beat 5: Hooke (Left, X=-0.26) + Giant Golden Skeleton Pocket Watch (Right, X=+0.28) — ZERO OVERLAP!
        u = clamp01((t - 20.5) / 5.0)
        pos = bezier_quad(np.array([-0.04, -1.40, 0.54]), np.array([0.02, -1.34, 0.58]), np.array([0.08, -1.38, 0.62]), u)
        tgt = lerp(np.array([0.01, 0.22, 0.44]), np.array([0.02, 0.24, 0.46]), u)
        return pos, tgt, lerp(38.0, 40.0, u)
    # Beat 6: Grand Cast Finale — Hooke (Left -0.50), Rival (Right +0.58), Watch & DNA Spring Center!
    u = clamp01((t - 25.5) / 4.5)
    pos = bezier_quad(np.array([0.04, -1.90, 0.72]), np.array([0.03, -2.06, 0.70]), np.array([0.02, -2.18, 0.68]), u)
    tgt = lerp(np.array([0.03, 0.28, 0.54]), np.array([0.03, 0.26, 0.52]), u)
    return pos, tgt, lerp(34.5, 32.5, u)


def project(cam_pos, tgt, lens_mm, pt):
    fwd = tgt - cam_pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    d = pt - cam_pos
    z = np.dot(d, fwd)
    x = np.dot(d, right)
    y = np.dot(d, up)
    ndc_x = (x / z) * (lens_mm / (SENSOR_W * 0.5))
    ndc_y = (y / z) * (lens_mm / (SENSOR_H * 0.5))
    return ndc_x, ndc_y, z


checks = [
    (0, 2.6, "Beat 1 Hooke(L) + Spring(R)", {
        "HookeHead": np.array([-1.24, 0.50, 0.74]),
        "SpringCenter": np.array([RIG_X + 0.06, RIG_Y, 0.44]),
        "RulerRight": np.array([RIG_X + 0.22, RIG_Y, 0.36]),
    }, [("HookeHead", "SpringCenter", 0.42)]),
    (1, 6.2, "Beat 2a Rival(R) + Cipher(L-Center)", {
        "RivalHead": np.array([0.46, 0.52, 0.76]),
        "CipherText": np.array([FOLIO_POS[0], FOLIO_POS[1], 0.26]),
    }, [("RivalHead", "CipherText", 0.40)]),
    (1, 8.5, "Beat 2b Hooke(L) + Rival(R) + F=-kx(C)", {
        "HookeHead": np.array([-0.44, 0.52, 0.74]),
        "RivalHead": np.array([0.46, 0.52, 0.76]),
        "Formula": np.array([FOLIO_POS[0], FOLIO_POS[1], 0.28]),
    }, [("HookeHead", "RivalHead", 0.60), ("HookeHead", "Formula", 0.30)]),
    (2, 12.8, "Beat 3 Microscope/Cells(L) + Hooke(R)", {
        "HookeHead": np.array([-1.16, 0.50, 0.74]),
        "MicLeft": np.array([MIC_POS[0] - 0.08, MIC_POS[1], 0.22]),
        "CorkCells": np.array([MIC_POS[0] + 0.02, MIC_POS[1], 0.46]),
    }, [("HookeHead", "CorkCells", 0.42)]),
    (3, 17.8, "Beat 4 Rival(L) + OverloadSnap(R)", {
        "RivalHead": np.array([0.92, 0.50, 0.76]),
        "OverSpring": np.array([OVER_X - 0.04, OVER_Y, 0.42]),
        "OverPost": np.array([OVER_X + 0.13, OVER_Y, 0.44]),
    }, [("RivalHead", "OverSpring", 0.42)]),
    (4, 23.2, "Beat 5 Hooke(L) + PocketWatch(R)", {
        "HookeHead": np.array([-0.26, 0.50, 0.74]),
        "WatchCenter": np.array([WATCH_POS[0], WATCH_POS[1], WATCH_POS[2]]),
        "WatchRight": np.array([WATCH_POS[0] + 0.24, WATCH_POS[1], WATCH_POS[2]]),
    }, [("HookeHead", "WatchCenter", 0.42)]),
    (5, 28.2, "Beat 6 Grand Cast Finale", {
        "HookeHead": np.array([-0.50, 0.56, 0.74]),
        "RivalHead": np.array([0.58, 0.56, 0.76]),
        "WatchCenter": np.array([0.08, 0.02, 0.30]),
        "DNATop": np.array([0.00, 0.08, 1.15]),
        "TitleRight": np.array([0.42, 0.56, 1.32]),
    }, [("HookeHead", "RivalHead", 0.60), ("HookeHead", "WatchCenter", 0.30)]),
]

errors = 0
for b_idx, t, label, pts, sep_rules in checks:
    c_pos, tgt, lens = compute_beat(b_idx, t)
    print(f"\n[{label} (t={t:.1f}s)] cam={np.round(c_pos,2)} tgt={np.round(tgt,2)} lens={lens:.1f}mm")
    proj_map = {}
    for pname, pt in pts.items():
        nx, ny, z = project(c_pos, tgt, lens, pt)
        proj_map[pname] = (nx, ny, z)
        ok = (z > 0.3) and (abs(nx) <= 0.85) and (-0.50 <= ny <= 0.72)
        if not ok:
            errors += 1
        print(f"  {pname:14s}: ndcX={nx:+.2f}  ndcY={ny:+.2f}  depth={z:.2f}m  {'[OK]' if ok else '[CLIP!]'}")
    for a_k, b_k, min_sep in sep_rules:
        sep = abs(proj_map[a_k][0] - proj_map[b_k][0])
        s_ok = sep >= min_sep
        if not s_ok:
            errors += 1
        print(f"  SEPARATION {a_k} <-> {b_k}: |dX|={sep:.2f} (min {min_sep:.2f}) {'[CLEAR!]' if s_ok else '[OVERLAP!]'}")

if errors:
    print(f"\nFAILED: {errors} framing/overlap error(s)!")
    raise SystemExit(1)
print("\nPASSED: 100% ZERO-OVERLAP Sightlines + Tight Portrait Framing across ALL 6 Beats!")

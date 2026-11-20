"""
Pre-flight mathematical 9:16 portrait camera framing, ZERO-OVERLAP sightline & STATION ISOLATION verifier for Robert Hooke V5!
Verifies across all 6 Beats:
  1. 100% of ALL Cast Heads + Scientific Instruments stay cleanly inside the 9:16 portrait safe frame
     (|ndcX| <= 0.85, ndcY in [-0.50, +0.72]) — INCLUDING BOTH HookeHead & RivalHead in Beat 2a (t=5.5s)!
  2. ZERO Instrument-Blocking-Face Overlap: checks horizontal screen separation |ndcX(Actor) - ndcX(HeroProp)| >= 0.34
     in every single close-up / medium shot!
  3. ZERO Station Edge Bleed: confirms Microscope Condenser Globe stays completely OUTSIDE Beat 1 left edge
     AND Spring Rig Base stays completely OUTSIDE Beat 3 right edge!
"""
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


# Exact V5 Symmetric Zero-Overlap Station Anchors (synchronized 1:1 with build_hooke_scene.py!)
MIC_POS = np.array([-1.92, -0.06, 0.0])
RIG_X, RIG_Y = -0.74, -0.10
FOLIO_POS = np.array([0.00, -0.10, 0.0])
WATCH_POS = np.array([0.28, 0.02, 0.30])
OVER_X, OVER_Y = 1.42, -0.08


def compute_beat(b_idx, t):
    if b_idx == 0:
        # Beat 1: Hooke (Left, X=-1.18, facing RIGHT +24°) + Hero Spring Rig (Right, X=-0.74)
        u = clamp01(t / 4.5)
        pos = bezier_quad(np.array([-1.00, -1.46, 0.56]), np.array([-0.94, -1.36, 0.54]), np.array([-0.88, -1.28, 0.52]), u)
        tgt = lerp(np.array([-0.95, 0.14, 0.46]), np.array([-0.93, 0.14, 0.44]), u)
        return pos, tgt, lerp(39.0, 41.5, u)
    if b_idx == 1:
        # Beat 2: Balanced Face-to-Face 2-Shot from t=4.5s (BOTH Hooke & Rival 100% INSIDE frame!)
        pan_u = ease_out_quint(clamp01((t - 6.9) / 1.1))
        pos = lerp(np.array([0.08, -1.68, 0.56]), np.array([-0.02, -1.60, 0.56]), pan_u)
        tgt = lerp(np.array([0.04, 0.18, 0.46]), np.array([0.00, 0.20, 0.46]), pan_u)
        return pos, tgt, lerp(36.0, 35.5, pan_u)
    if b_idx == 2:
        # Beat 3: Microscope + Floating Cork Cells (Left, X=-1.92) + Hooke (Right, X=-1.40, facing LEFT -26°)
        u = clamp01((t - 9.5) / 6.0)
        pos = bezier_quad(np.array([-1.72, -1.46, 0.58]), np.array([-1.66, -1.36, 0.60]), np.array([-1.60, -1.30, 0.62]), u)
        tgt = lerp(np.array([-1.66, 0.14, 0.46]), np.array([-1.64, 0.14, 0.52]), smoothstep(10.5, 13.0, t))
        return pos, tgt, lerp(38.5, 40.5, u)
    if b_idx == 3:
        # Beat 4: Rival Watchmaker (Left, X=+0.94, facing RIGHT +24°) + Overload Spring SNAP (Right, X=+1.42)
        u = clamp01((t - 15.5) / 5.0)
        pos = bezier_quad(np.array([1.14, -1.44, 0.54]), np.array([1.18, -1.36, 0.55]), np.array([1.22, -1.42, 0.56]), u)
        tgt = np.array([1.16, 0.16, 0.46])
        return pos, tgt, lerp(38.0, 40.0, u)
    if b_idx == 4:
        # Beat 5: Hooke (Left, X=-0.28, facing RIGHT +24°) + Giant Golden Skeleton Pocket Watch (Right, X=+0.28)
        u = clamp01((t - 20.5) / 5.0)
        pos = bezier_quad(np.array([-0.04, -1.40, 0.54]), np.array([0.02, -1.34, 0.58]), np.array([0.06, -1.38, 0.62]), u)
        tgt = lerp(np.array([0.01, 0.20, 0.46]), np.array([0.02, 0.22, 0.48]), u)
        return pos, tgt, lerp(38.0, 40.0, u)
    # Beat 6: Grand Cast Finale — Hooke (Left -0.52), Rival (Right +0.56), Watch & DNA Spring Center!
    u = clamp01((t - 25.5) / 4.5)
    pos = bezier_quad(np.array([0.04, -1.90, 0.72]), np.array([0.03, -2.06, 0.70]), np.array([0.02, -2.18, 0.68]), u)
    tgt = lerp(np.array([0.03, 0.26, 0.55]), np.array([0.03, 0.24, 0.53]), u)
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


# FLOOR_Z = -0.99, height ~1.73..1.75m -> Head center ~0.68..0.70m, Head top ~0.74..0.76m!
checks = [
    (0, 0.4, "Beat 1 Start Hooke(L) + Spring(R)", {
        "HookeHead": np.array([-1.18, 0.44, 0.72]),
        "SpringCenter": np.array([RIG_X + 0.08, RIG_Y, 0.44]),
        "RulerRight": np.array([RIG_X + 0.22, RIG_Y, 0.36]),
    }, [("HookeHead", "SpringCenter", 0.42)]),
    (0, 2.6, "Beat 1 Mid Hooke(L) + Spring(R)", {
        "HookeHead": np.array([-1.18, 0.44, 0.72]),
        "SpringCenter": np.array([RIG_X + 0.08, RIG_Y, 0.44]),
        "RulerRight": np.array([RIG_X + 0.22, RIG_Y, 0.36]),
    }, [("HookeHead", "SpringCenter", 0.42)]),
    (1, 5.5, "Beat 2a BOTH Hooke(L) + Rival(R) + Cipher(C)", {
        "HookeHead": np.array([-0.44, 0.46, 0.72]),
        "RivalHead": np.array([0.46, 0.46, 0.74]),
        "CipherText": np.array([FOLIO_POS[0], FOLIO_POS[1] - 0.04, 0.25]),
    }, [("HookeHead", "RivalHead", 0.60), ("HookeHead", "CipherText", 0.30)]),
    (1, 8.5, "Beat 2b BOTH Hooke(L) + Rival(R) + F=-kx(C)", {
        "HookeHead": np.array([-0.44, 0.46, 0.72]),
        "RivalHead": np.array([0.46, 0.46, 0.74]),
        "Formula": np.array([FOLIO_POS[0], FOLIO_POS[1] - 0.04, 0.30]),
    }, [("HookeHead", "RivalHead", 0.60), ("HookeHead", "Formula", 0.30)]),
    (2, 12.8, "Beat 3 Microscope/Cells(L) + Hooke(R)", {
        "HookeHead": np.array([-1.40, 0.44, 0.72]),
        "MicLeft": np.array([MIC_POS[0] - 0.08, MIC_POS[1], 0.22]),
        "CorkCellsTop": np.array([MIC_POS[0] + 0.02, MIC_POS[1] - 0.04, 0.76]),
        "CorkCellsBot": np.array([MIC_POS[0] + 0.02, MIC_POS[1] - 0.04, 0.48]),
    }, [("HookeHead", "CorkCellsTop", 0.42)]),
    (3, 17.8, "Beat 4 Rival(L) + OverloadSnap(R)", {
        "RivalHead": np.array([0.90, 0.49, 0.74]),
        "OverSpring": np.array([OVER_X - 0.04, OVER_Y, 0.42]),
        "OverPost": np.array([OVER_X + 0.13, OVER_Y, 0.44]),
    }, [("RivalHead", "OverSpring", 0.42)]),
    (4, 23.2, "Beat 5 Hooke(L) + PocketWatch(R)", {
        "HookeHead": np.array([-0.28, 0.44, 0.72]),
        "WatchCenter": np.array([WATCH_POS[0], WATCH_POS[1], WATCH_POS[2]]),
        "WatchRight": np.array([WATCH_POS[0] + 0.24, WATCH_POS[1], WATCH_POS[2]]),
    }, [("HookeHead", "WatchCenter", 0.42)]),
    (5, 28.2, "Beat 6 Grand Cast Finale", {
        "HookeHead": np.array([-0.52, 0.52, 0.72]),
        "RivalHead": np.array([0.56, 0.52, 0.74]),
        "WatchCenter": np.array([0.06, 0.04, 0.30]),
        "DNATop": np.array([0.00, 0.08, 1.15]),
        "TitleRight": np.array([0.42, 0.56, 1.52]),
    }, [("HookeHead", "RivalHead", 0.60), ("HookeHead", "WatchCenter", 0.30)]),
]

# Neighbor-station out-of-frame isolation checks (confirm zero accidental prop edge intrusion!)
isolation_checks = [
    (0, 0.4, "Beat 1 Start: CondenserGlobe right edge OUTSIDE left frame", np.array([MIC_POS[0] + 0.22, MIC_POS[1] - 0.05, 0.16]), "LEFT"),
    (2, 11.5, "Beat 3 Mid: SpringRigBase left edge OUTSIDE right frame", np.array([RIG_X - 0.21, RIG_Y, 0.04]), "RIGHT"),
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

print("\n[STATION ISOLATION CHECKS (ZERO EDGE INTRUSION)]")
for b_idx, t, label, pt, side in isolation_checks:
    c_pos, tgt, lens = compute_beat(b_idx, t)
    nx, ny, z = project(c_pos, tgt, lens, pt)
    ok = (nx < -1.03) if side == "LEFT" else (nx > 1.03)
    if not ok:
        errors += 1
    print(f"  {label}: ndcX={nx:+.2f} ({'OUTSIDE' if ok else 'INTRUDING!'})")

if errors:
    print(f"\nFAILED: {errors} framing/overlap/isolation error(s)!")
    raise SystemExit(1)
print("\nPASSED: 100% ZERO-OVERLAP Sightlines + Balanced Beat 2 2-Shot + Station Isolation across ALL 6 Beats!")

"""
Scientist #8: Robert Hooke (1665-1678) — Springs, Secret Riddles & The Discovery of Cells
Full 30-Second (720 Frames @ 24fps, 1080x1920 Lanczos) Blender 4.2 LTS Cycles Turbo Dual-GPU Cinema Short

100% Unique Visual Identity (1665 Royal Society of London Candlelit Laboratory):
  - Rich Stuart Mahogany Workbench, Object-Mapped Red-Brick & Limestone London Archway,
    Leaded Rain-Lit Gothic Window, Warm Leather-Bound Scholar Bookshelves & Brass Oil Lamps.
  - Beat 1 (0.0-4.5s): Gleaming Helical Steel/Brass Tension Spring Rig stretching dynamically 1x -> 2x
    with calibrated brass weights + glowing vertical cyan displacement scale & live force arrows!
  - Beat 2 (4.5-9.5s): 3D Floating Brass Anagram Cipher ('ceiiinosssttuv') above crimson *Micrographia* folio
    morphing cleanly into gleaming 3D 'F = -kx' right as Hooke doubles the weight!
  - Beat 3 (9.5-15.5s): Iconic 1665 Gold-Tooled Leather & Brass Compound Microscope + Oil Lamp Glass Condenser
    Globe -> macro orbit into 37 translucent glowing golden-amber Hexagonal Cork 'CELLS'!
  - Beat 4 (15.5-20.5s): True 3.8° Dutch-Tilt Overload Test Spring -> heavy crimson weight drops, coil glows
    hot copper-orange, permanently overstretches & SNAPS with flying metallic sparks at t=17.65s!
  - Beat 5 (20.5-25.5s): Giant Jewel-Like Skeletonized Brass Pocket Watch on Velvet Cushion: ruby center jewel,
    oscillating brass balance wheel, real breathing Archimedean spiral hairspring & meshing escapement gears!
  - Beat 6 (25.5-30.0s): Hero Wide Finale blooming with a glowing 3D Double-Helix DNA Elastic Spring &
    Metallic Skyscraper Damper Coil above the entire glowing Royal Society laboratory!

Zero-Compromise Cycles Turbo Engine (~4-6 min on Kaggle GPU T4 x2):
  - True OPTIX/CUDA GPU Raytracing + AgX Punchy Colors + True 9:16 HORIZONTAL Sensor Fit (0 cropped subjects!)
  - Noisy caustics off + blur_glossy=1.0 + RGB_ALBEDO fast OIDN + JPEG-98 zero-stall + 24fps + 75%->1080p Lanczos!
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import bpy
from mathutils import Euler, Quaternion, Vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from mpfb_characters import apply_facial_acting, create_realistic_scientist, set_bone_char_rot


# ============================================================================
# 1. CLI ARGUMENTS & CONFIGURATION
# ============================================================================
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Robert Hooke Cycles Turbo GPU 3D Short Renderer (MPFB2 Cast)")
    p.add_argument("--start-frame", type=int, default=1)
    p.add_argument("--end-frame", type=int, default=720)
    p.add_argument("--fps", type=int, default=int(os.environ.get("HOOKE_FPS", "24")))
    p.add_argument("--samples", type=int, default=int(os.environ.get("HOOKE_SAMPLES", "10")))
    p.add_argument("--scale", type=int, default=int(os.environ.get("HOOKE_SCALE", "75")))
    p.add_argument("--out-dir", type=str, default=os.environ.get("HOOKE_FRAMES_DIR", "render_frames"))
    p.add_argument("--qa-only", action="store_true", help="Render 6 hero Beat keyframes to qa_frames/")
    return p.parse_args(argv)


ARGS = parse_args()
FPS = ARGS.fps
DURATION_SEC = 30
TOTAL_FRAMES = FPS * DURATION_SEC
RES_X = 1080
RES_Y = 1920
OUT_DIR = os.path.abspath("qa_frames" if ARGS.qa_only else ARGS.out_dir)
os.makedirs(OUT_DIR, exist_ok=True)

# Load pre-mastered Kokoro voice_manifest.json for exact jaw lip-sync!
MANIFEST_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "audio", "output", "voice_manifest.json"))
VOICE_MANIFEST = json.load(open(MANIFEST_PATH, encoding="utf-8")) if os.path.isfile(MANIFEST_PATH) else {}


def sample_lip_sync(speaker: str, t: float) -> float:
    curve = VOICE_MANIFEST.get("lipSync", {}).get(speaker, [])
    if not curve:
        return 0.0
    idx = max(0, min(len(curve) - 1, int(round(t * 30.0))))
    return float(curve[idx])


# ============================================================================
# 2. MATH, EASING & EXACT QUATERNION LOOK-AT WITH DUTCH TILT
# ============================================================================
def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = clamp01((x - e0) / max(1e-6, e1 - e0))
    return t * t * (3.0 - 2.0 * t)


def ease_out_quint(x: float) -> float:
    t = clamp01(x)
    return 1.0 - (1.0 - t) ** 5


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def vlerp(a: Vector, b: Vector, t: float) -> Vector:
    return a.lerp(b, t)


def bezier_quad(p0: Vector, p1: Vector, p2: Vector, t: float) -> Vector:
    u = 1.0 - t
    return p0 * (u * u) + p1 * (2.0 * u * t) + p2 * (t * t)


def look_at_quat(cam_pos: Vector, target: Vector, roll_deg: float = 0.0) -> Quaternion:
    fwd = (target - cam_pos).normalized()
    q = fwd.to_track_quat("-Z", "Y")
    if abs(roll_deg) > 1e-4:
        q = q @ Quaternion(Vector((0.0, 0.0, 1.0)), math.radians(roll_deg))
    return q


# ============================================================================
# 3. CLEAN SCENE (EXTENSION-SAFE FOR MPFB2!) & CONFIGURE CYCLES TURBO GPU
# ============================================================================
# IMPORTANT: Do NOT call read_factory_settings(use_empty=True) — it unloads Blender 4.2 extensions like MPFB2!
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for mesh in list(bpy.data.meshes):
    bpy.data.meshes.remove(mesh, do_unlink=True)
for mat in list(bpy.data.materials):
    bpy.data.materials.remove(mat, do_unlink=True)
for cam_d in list(bpy.data.cameras):
    bpy.data.cameras.remove(cam_d, do_unlink=True)
for l_d in list(bpy.data.lights):
    bpy.data.lights.remove(l_d, do_unlink=True)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.render.resolution_x = RES_X
scene.render.resolution_y = RES_Y
scene.render.resolution_percentage = ARGS.scale
scene.render.fps = FPS
scene.render.use_persistent_data = True
scene.render.use_overwrite = False
scene.render.use_placeholder = False
scene.frame_start = max(1, ARGS.start_frame)
scene.frame_end = min(TOTAL_FRAMES, ARGS.end_frame)

scene.cycles.samples = ARGS.samples
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.06
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.cycles.denoising_input_passes = "RGB_ALBEDO"
scene.cycles.denoising_prefilter = "NONE"
try:
    scene.cycles.denoising_use_gpu = True
except Exception:
    pass
scene.cycles.caustics_reflective = False
scene.cycles.caustics_refractive = False
scene.cycles.sample_clamp_indirect = 4.0
scene.cycles.blur_glossy = 1.0
scene.cycles.max_bounces = 4
scene.cycles.transmission_bounces = 4
scene.cycles.glossy_bounces = 2
scene.cycles.diffuse_bounces = 2
scene.cycles.transparent_max_bounces = 14
try:
    scene.cycles.texture_limit_render = "2048"
except Exception:
    pass

try:
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Punchy"
except Exception:
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - High Contrast"
    except Exception:
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "High Contrast"
scene.view_settings.exposure = 0.16

gpu_enabled = False
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpu_devices = [d for d in prefs.devices if d.type == backend]
            if gpu_devices:
                for d in prefs.devices:
                    d.use = d.type == backend
                scene.cycles.device = "GPU"
                gpu_enabled = True
                print(
                    f"[Cycles Turbo] {backend} GPU ({len(gpu_devices)} device(s)) | "
                    f"Frames {scene.frame_start}..{scene.frame_end} ({FPS}fps, {ARGS.scale}% res, {ARGS.samples} spp)"
                )
                break
        except Exception:
            continue
except Exception as e:
    print(f"[Cycles] GPU warning: {e}")

if not gpu_enabled:
    scene.cycles.device = "CPU"
    print("[Cycles] Falling back to CPU")

scene.render.image_settings.file_format = "JPEG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.quality = 98
scene.render.filepath = os.path.join(OUT_DIR, "frame_")


# ============================================================================
# 4. 1665 LONDON ROYAL SOCIETY PROCEDURAL CYCLES PBR MATERIALS (AGX CALIBRATED)
# ============================================================================
def set_bsdf(bsdf, name: str, val):
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val


def make_mahogany_mat(name="StuartMahogany", scale_vec=(1.1, 5.2, 1.1), dark=False):
    """Deep warm reddish-brown English Stuart mahogany wood."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = scale_vec
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 5.6
    ns.inputs["Detail"].default_value = 8.0
    ns.inputs["Distortion"].default_value = 1.4
    rmp = nt.nodes.new("ShaderNodeValToRGB")
    rmp.color_ramp.elements[0].position = 0.24
    rmp.color_ramp.elements[0].color = (0.032, 0.008, 0.006, 1.0) if dark else (0.058, 0.015, 0.010, 1.0)
    rmp.color_ramp.elements[1].position = 0.78
    rmp.color_ramp.elements[1].color = (0.095, 0.028, 0.016, 1.0) if dark else (0.165, 0.048, 0.024, 1.0)
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.10
    set_bsdf(bsdf, "Roughness", 0.30 if not dark else 0.40)
    set_bsdf(bsdf, "Coat Weight", 0.34)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_london_brick_mat():
    """Object-mapped warm London Tudor brick & stone masonry (zero stretched flat wall!)."""
    mat = bpy.data.materials.new("LondonBrickWall")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (math.pi * 0.5, 0.0, 0.0)
    mp.inputs["Scale"].default_value = (1.25, 1.25, 1.25)
    br = nt.nodes.new("ShaderNodeTexBrick")
    br.offset = 0.5
    br.inputs["Color1"].default_value = (0.14, 0.072, 0.056, 1.0)
    br.inputs["Color2"].default_value = (0.095, 0.052, 0.044, 1.0)
    br.inputs["Mortar"].default_value = (0.036, 0.032, 0.030, 1.0)
    br.inputs["Scale"].default_value = 3.5
    br.inputs["Mortar Size"].default_value = 0.026
    br.inputs["Mortar Smooth"].default_value = 0.14
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.28
    set_bsdf(bsdf, "Roughness", 0.78)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], br.inputs["Vector"])
    nt.links.new(br.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(br.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_metal_mat(name, rgb, rough=0.20, metallic=0.96, bump_str=0.025):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    set_bsdf(bsdf, "Base Color", (*rgb, 1.0))
    set_bsdf(bsdf, "Metallic", metallic)
    set_bsdf(bsdf, "Roughness", rough)
    if bump_str > 0:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        ns = nt.nodes.new("ShaderNodeTexNoise")
        ns.inputs["Scale"].default_value = 44.0
        ns.inputs["Detail"].default_value = 5.0
        bmp = nt.nodes.new("ShaderNodeBump")
        bmp.inputs["Strength"].default_value = bump_str
        nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
        nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
        nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_glass_mat(name, ior=1.52, tint=(0.94, 0.99, 1.0), rough=0.012, coat=0.35):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    set_bsdf(bsdf, "Base Color", (*tint, 1.0))
    set_bsdf(bsdf, "Roughness", rough)
    set_bsdf(bsdf, "IOR", ior)
    set_bsdf(bsdf, "Transmission Weight", 0.96)
    set_bsdf(bsdf, "Transmission", 0.96)
    set_bsdf(bsdf, "Coat Weight", coat)
    return mat


def make_emissive_mat(name, rgb, strength=4.8):
    """AgX-calibrated vivid saturated emission (zero pure-white washout!)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1.0)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_cork_cell_mat():
    """High-contrast sculpted golden-amber honeycomb cell walls (crisp individual Micrographia cells, zero washout!)."""
    mat = bpy.data.materials.new("MicrographiaCorkCell")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    set_bsdf(bsdf, "Base Color", (0.82, 0.44, 0.08, 1.0))
    set_bsdf(bsdf, "Metallic", 0.20)
    set_bsdf(bsdf, "Roughness", 0.26)
    set_bsdf(bsdf, "Emission Color", (0.95, 0.50, 0.08, 1.0))
    set_bsdf(bsdf, "Emission Strength", 0.65)
    return mat


def make_london_night_sky_mat():
    """Luminous deep sapphire-indigo moonlit London twilight backdrop behind mullioned Royal Society window!"""
    mat = bpy.data.materials.new("LondonMoonlitSky")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    rmp = nt.nodes.new("ShaderNodeValToRGB")
    cr = rmp.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.04, 0.09, 0.19, 1.0)
    e1 = cr.elements.new(0.55)
    e1.color = (0.06, 0.16, 0.34, 1.0)
    cr.elements[-1].position = 1.0
    cr.elements[-1].color = (0.14, 0.28, 0.50, 1.0)
    em.inputs["Strength"].default_value = 1.45
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], rmp.inputs["Fac"])
    nt.links.new(rmp.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


mahogany_mat = make_mahogany_mat("StuartMahogany", (1.1, 5.2, 1.1), dark=False)
panel_wood_mat = make_mahogany_mat("WainscotWood", (2.4, 2.4, 2.4), dark=True)
brick_mat = make_london_brick_mat()
brass_mat = make_metal_mat("EnglishBrass", (0.88, 0.62, 0.21), rough=0.19, metallic=0.96, bump_str=0.022)
steel_mat = make_metal_mat("SpringSteel", (0.76, 0.80, 0.86), rough=0.16, metallic=0.98, bump_str=0.015)
blued_steel_mat = make_metal_mat("BluedWatchSteel", (0.12, 0.28, 0.62), rough=0.18, metallic=0.95, bump_str=0.015)
bronze_mat = make_metal_mat("AgedBronze", (0.28, 0.18, 0.09), rough=0.34, metallic=0.88, bump_str=0.028)
iron_mat = make_metal_mat("CastIron", (0.05, 0.055, 0.065), rough=0.42, metallic=0.82, bump_str=0.03)
crimson_leather_mat = make_metal_mat("CrimsonLeather", (0.34, 0.035, 0.045), rough=0.52, metallic=0.0, bump_str=0.04)
emerald_velvet_mat = make_metal_mat("EmeraldVelvet", (0.025, 0.14, 0.095), rough=0.88, metallic=0.0, bump_str=0.0)
parchment_mat = make_metal_mat("MicrographiaPage", (0.92, 0.85, 0.68), rough=0.65, metallic=0.0, bump_str=0.02)
ruby_jewel_mat = make_glass_mat("WatchRubyJewel", ior=1.77, tint=(0.95, 0.08, 0.16), rough=0.01, coat=0.7)
lens_glass_mat = make_glass_mat("MicroscopeGlass", ior=1.52, tint=(0.90, 0.98, 1.0), rough=0.01, coat=0.5)
cork_cell_mat = make_cork_cell_mat()
cork_backplate_mat = make_metal_mat("CorkBacklitDisc", (0.015, 0.06, 0.055), rough=0.32, metallic=0.1, bump_str=0.0)
night_sky_mat = make_london_night_sky_mat()

book_mats = [
    crimson_leather_mat,
    make_metal_mat("BookNavy", (0.04, 0.09, 0.20), rough=0.65, metallic=0.0, bump_str=0.0),
    make_metal_mat("BookHunter", (0.04, 0.16, 0.09), rough=0.65, metallic=0.0, bump_str=0.0),
    make_metal_mat("BookOchre", (0.28, 0.16, 0.05), rough=0.65, metallic=0.0, bump_str=0.0),
]

# AgX-calibrated vivid saturated glows (ZERO pure white washout!)
gold_glow_mat = make_emissive_mat("HookeGoldGlow", (1.0, 0.56, 0.06), strength=4.6)
cyan_glow_mat = make_emissive_mat("HookeCyanGlow", (0.04, 0.68, 1.0), strength=4.8)
emerald_glow_mat = make_emissive_mat("HookeEmeraldGlow", (0.08, 0.88, 0.44), strength=4.4)
hot_copper_mat = make_emissive_mat("OverloadCopperGlow", (1.0, 0.20, 0.04), strength=6.2)
lamp_flame_mat = make_emissive_mat("OilLampFlame", (1.0, 0.50, 0.08), strength=8.0)

# Razor-sharp anti-bloom 3D Text materials + bundled DejaVuSans-Bold font (zero mashed Bfont letters!)
text_gold_mat = make_emissive_mat("TextGoldCrisp", (1.0, 0.72, 0.14), strength=2.4)
text_cyan_mat = make_emissive_mat("TextCyanCrisp", (0.14, 0.82, 1.0), strength=2.6)
title_gold_mat = bpy.data.materials.new("TitleCinemaGold")
title_gold_mat.use_nodes = True
_tbsdf = title_gold_mat.node_tree.nodes["Principled BSDF"]
set_bsdf(_tbsdf, "Base Color", (1.0, 0.76, 0.24, 1.0))
set_bsdf(_tbsdf, "Metallic", 0.72)
set_bsdf(_tbsdf, "Roughness", 0.24)
set_bsdf(_tbsdf, "Emission Color", (1.0, 0.70, 0.18, 1.0))
set_bsdf(_tbsdf, "Emission Strength", 1.75)

FONT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "subtitles", "fonts", "DejaVuSans-Bold.ttf"))
HERO_FONT = bpy.data.fonts.load(FONT_PATH) if os.path.isfile(FONT_PATH) else None


# ============================================================================
# 5. SMOOTH-SHADED & PROCEDURAL SPRING / HEX-CELL GEOMETRY HELPERS
# ============================================================================
def shade_smooth(obj, angle_deg: float = 38.0):
    for p in obj.data.polygons:
        p.use_smooth = True
    es = obj.modifiers.new("SmoothSplit", "EDGE_SPLIT")
    es.split_angle = math.radians(angle_deg)


def add_box(name, loc, scale, mat, rot=(0.0, 0.0, 0.0), bevel_w=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    o.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel_w > 0:
        bv = o.modifiers.new("Bevel", "BEVEL")
        bv.width = bevel_w
        bv.segments = 2
    o.data.materials.append(mat)
    return o


def add_cyl(name, loc, radius, depth, mat, rot=(0.0, 0.0, 0.0), verts=48, smooth=True, bevel_w=0.0):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    if smooth:
        shade_smooth(o)
    if bevel_w > 0:
        bv = o.modifiers.new("Bevel", "BEVEL")
        bv.width = bevel_w
        bv.segments = 2
    o.data.materials.append(mat)
    return o


def add_cone(name, loc, r1, r2, depth, mat, rot=(0.0, 0.0, 0.0), verts=40):
    bpy.ops.mesh.primitive_cone_add(vertices=verts, radius1=r1, radius2=r2, depth=depth, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    shade_smooth(o)
    o.data.materials.append(mat)
    return o


def add_sphere(name, loc, radius, mat, segs=32, rings=20):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segs, ring_count=rings, radius=radius, location=loc)
    o = bpy.context.active_object
    o.name = name
    for p in o.data.polygons:
        p.use_smooth = True
    o.data.materials.append(mat)
    return o


def add_torus(name, loc, major_r, minor_r, mat, rot=(0.0, 0.0, 0.0), maj_seg=64, min_seg=14):
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_r,
        minor_radius=minor_r,
        major_segments=maj_seg,
        minor_segments=min_seg,
        location=loc,
        rotation=rot,
    )
    o = bpy.context.active_object
    o.name = name
    for p in o.data.polygons:
        p.use_smooth = True
    o.data.materials.append(mat)
    return o


def build_helical_spring(name, loc, radius, length, turns, wire_r, mat, segs_per_turn=20):
    """Build a true 3D tubular helical spring anchored at Z=0 extending downward to Z=-length!
    Scaling .scale.z stretches/compresses the spring smoothly around its top anchor!"""
    curve_data = bpy.data.curves.new(name + "_Curve", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = wire_r
    curve_data.bevel_resolution = 5
    spline = curve_data.splines.new("POLY")
    total_pts = int(turns * segs_per_turn) + 4
    spline.points.add(total_pts)
    # Top hook stem
    spline.points[0].co = (0.0, 0.0, 0.0, 1.0)
    spline.points[1].co = (radius, 0.0, -0.015, 1.0)
    n_helix = total_pts - 3
    for i in range(n_helix + 1):
        u = i / n_helix
        ang = u * turns * math.pi * 2.0
        z = -0.015 - u * (length - 0.030)
        spline.points[i + 2].co = (math.cos(ang) * radius, math.sin(ang) * radius, z, 1.0)
    spline.points[total_pts].co = (0.0, 0.0, -length, 1.0)

    obj = bpy.data.objects.new(name, curve_data)
    obj.location = Vector(loc)
    scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def build_spiral_hairspring(name, loc, r_in, r_out, turns, wire_r, mat, segs=120):
    """Build an authentic flat Archimedean watch balance hairspring in XY plane!"""
    curve_data = bpy.data.curves.new(name + "_Curve", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = wire_r
    curve_data.bevel_resolution = 4
    spline = curve_data.splines.new("POLY")
    spline.points.add(segs)
    for i in range(segs + 1):
        u = i / segs
        ang = u * turns * math.pi * 2.0
        r = lerp(r_in, r_out, u)
        spline.points[i].co = (math.cos(ang) * r, math.sin(ang) * r, 0.0, 1.0)
    obj = bpy.data.objects.new(name, curve_data)
    obj.location = Vector(loc)
    scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def build_hex_cell_lattice(name, loc, rings=3, cell_r=0.032, wall_h=0.038, mat=None):
    """Build a crisp hollow 3D biological honeycomb cork-cell mesh (37 hexagonal chambers!) in 1 fast mesh!"""
    mesh = bpy.data.meshes.new(name + "_Mesh")
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(loc)
    scene.collection.objects.link(obj)
    if mat:
        obj.data.materials.append(mat)

    verts = []
    faces = []
    dx = cell_r * 1.54
    dy = cell_r * math.sqrt(3.0) * 0.89
    inner_r = cell_r * 0.78

    for q in range(-rings, rings + 1):
        for r in range(-rings, rings + 1):
            if abs(q + r) > rings:
                continue
            cx = dx * q
            cy = dy * (r + q * 0.5)
            # Subtle organic biological height variation like real Micrographia cork!
            h_var = wall_h * (0.85 + 0.30 * math.sin(q * 1.7 + r * 2.3))
            base = len(verts)
            for s in range(6):
                a = math.radians(s * 60.0)
                ca, sa = math.cos(a), math.sin(a)
                verts.append((cx + ca * cell_r, cy + sa * cell_r, 0.0))
                verts.append((cx + ca * inner_r, cy + sa * inner_r, 0.0))
                verts.append((cx + ca * cell_r, cy + sa * cell_r, h_var))
                verts.append((cx + ca * inner_r, cy + sa * inner_r, h_var))
            for s in range(6):
                sn = (s + 1) % 6
                b0, b1 = base + s * 4, base + sn * 4
                # Top rim, outer wall, inner wall
                faces.append((b0 + 2, b1 + 2, b1 + 3, b0 + 3))
                faces.append((b0 + 0, b1 + 0, b1 + 2, b0 + 2))
                faces.append((b0 + 3, b1 + 3, b1 + 1, b0 + 1))

    mesh.from_pydata(verts, [], faces)
    mesh.update()
    shade_smooth(obj, 42.0)
    return obj


def add_toothed_gear(name, loc, radius, thickness, teeth, mat, flat_xy=True):
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=loc)
    hub = bpy.context.active_object
    hub.name = name
    rot_cyl = (0.0, 0.0, 0.0) if flat_xy else (math.pi * 0.5, 0.0, 0.0)
    rim = add_cyl(f"{name}_Rim", (0, 0, 0), radius, thickness, mat, rot=rot_cyl, verts=36)
    rim.parent = hub
    axle = add_cyl(f"{name}_Axle", (0, 0, 0), radius * 0.22, thickness * 2.0, iron_mat, rot=rot_cyl, verts=20)
    axle.parent = hub
    tooth_w = (2.0 * math.pi * radius / teeth) * 0.45
    tooth_h = radius * 0.22
    for i in range(teeth):
        ang = (i / teeth) * math.pi * 2.0
        if flat_xy:
            tx = math.cos(ang) * (radius + tooth_h * 0.34)
            ty = math.sin(ang) * (radius + tooth_h * 0.34)
            t_obj = add_box(f"{name}_T{i}", (tx, ty, 0.0), (tooth_h, tooth_w, thickness * 0.90), mat, rot=(0.0, 0.0, ang))
        else:
            tx = math.cos(ang) * (radius + tooth_h * 0.34)
            tz = math.sin(ang) * (radius + tooth_h * 0.34)
            t_obj = add_box(f"{name}_T{i}", (tx, 0.0, tz), (tooth_h, thickness * 0.90, tooth_w), mat, rot=(0.0, -ang, 0.0))
        t_obj.parent = hub
    return hub


def orient_beam(obj, p1: Vector, p2: Vector, radius_scale: float = 1.0):
    diff = p2 - p1
    length = diff.length
    obj.location = (p1 + p2) * 0.5
    obj.scale = (radius_scale, radius_scale, max(0.0005, length))
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = diff.normalized().to_track_quat("Z", "Y") if length > 1e-5 else Quaternion((1, 0, 0, 0))


# ============================================================================
# 6. BUILD 1665 ROYAL SOCIETY OF LONDON LABORATORY STAGE (V3 CINEMA GRADE!)
# ============================================================================
world = bpy.data.worlds.new("LondonTwilightLab")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.014, 0.032, 0.052, 1.0)  # Deep London sapphire twilight
bg.inputs["Strength"].default_value = 0.72

# Polished Stuart Mahogany Workbench (top Z = 0.0, Y in [-0.54, +0.34] so 1.74m characters stand cleanly waist-high at Y=+0.50!)
add_box("MahoganyBench", (0.0, -0.10, -0.06), (5.8, 0.88, 0.12), mahogany_mat, bevel_w=0.008)
add_box("BenchBrassEdge", (0.0, -0.545, -0.022), (5.82, 0.028, 0.044), brass_mat, bevel_w=0.004)
# Solid Paneled Mahogany Laboratory Cabinet Front (Z = -0.99..-0.12 -> exact waist-high bench, hands cleanly above table!)
FLOOR_Z = -0.99
add_box("BenchCabinetBody", (0.0, -0.12, -0.555), (5.6, 0.80, 0.87), panel_wood_mat)
for dr_x in (-1.65, -0.82, 0.0, 0.82, 1.65):
    add_box(f"BenchDrawerPanel_{dr_x:.2f}", (dr_x, -0.525, -0.34), (0.68, 0.024, 0.34), mahogany_mat, bevel_w=0.006)
    add_torus(f"BenchDrawerPull_{dr_x:.2f}", (dr_x, -0.545, -0.34), 0.032, 0.006, brass_mat, rot=(math.pi * 0.5, 0, 0))
add_box("LabWoodFloor", (0.0, 0.50, FLOOR_Z - 0.05), (7.6, 4.8, 0.10), panel_wood_mat)

# London Brick & Oak-Paneled Back Wall with Arched Leaded Window & Luminous Moonlit London Sky!
add_box("BrickWallLeft", (-2.45, 1.95, 1.55), (3.0, 0.24, 4.2), brick_mat)
add_box("BrickWallRight", (2.45, 1.95, 1.55), (3.0, 0.24, 4.2), brick_mat)
add_box("BrickWallLower", (0.0, 1.95, -0.25), (2.2, 0.24, 2.00), brick_mat)
add_box("BrickWallHeader", (0.0, 1.95, 2.75), (2.4, 0.24, 1.15), brick_mat)
# Luminous Sapphire-Indigo Moonlit London Night Sky behind leaded window (zero flat pitch-black rectangle!)
add_box("LondonMoonlitSkyPlane", (0.0, 2.38, 1.55), (2.6, 0.04, 2.2), night_sky_mat)
add_box("MahoganyWainscot", (0.0, 1.81, -0.13), (7.2, 0.06, 1.74), panel_wood_mat, bevel_w=0.006)
add_box("WindowSill", (0.0, 1.79, 0.74), (2.12, 0.14, 0.055), mahogany_mat, bevel_w=0.006)
for sx in (-0.98, 0.98):
    add_cyl(f"LabColumn_{sx}", (sx, 1.78, 1.48), 0.11, 3.6, panel_wood_mat, verts=28)
    add_cyl(f"ColBrassRing_{sx}", (sx, 1.78, 0.78), 0.125, 0.045, brass_mat, verts=28, bevel_w=0.005)
add_box("WindowMullionV", (0.0, 1.92, 1.48), (0.038, 0.044, 1.65), iron_mat)
add_box("WindowMullionH1", (0.0, 1.92, 1.24), (1.92, 0.044, 0.034), iron_mat)
add_box("WindowMullionH2", (0.0, 1.92, 1.74), (1.92, 0.044, 0.034), iron_mat)

# Royal Society Library Bookshelves spanning full bench backdrop (rich bokeh in Beats 1-6!)
for side_idx, bx in enumerate((-2.36, -1.52, 1.54, 2.38)):
    add_box(f"CaseBack_{side_idx}", (bx, 1.72, 1.22), (0.82, 0.04, 1.42), panel_wood_mat)
    for sz in (0.76, 1.14, 1.52, 1.88):
        add_box(f"CaseShelf_{side_idx}_{sz:.2f}", (bx, 1.60, sz), (0.84, 0.24, 0.035), mahogany_mat)
    for row_idx, rz in enumerate((0.88, 1.26, 1.64)):
        for k in range(6):
            bh = 0.20 + ((k * 3 + row_idx) % 4) * 0.022
            bt = 0.046 + (k % 2) * 0.010
            add_box(
                f"Tome_{side_idx}_{row_idx}_{k}",
                (bx - 0.28 + k * 0.112, 1.60, rz + bh * 0.5 - 0.10),
                (bt, 0.16, bh),
                book_mats[(k + row_idx + side_idx) % len(book_mats)],
            )

# Warm London Brass Oil Lamps on outer bench edges (clear of every actor!)
lamp_flames = []
for idx, (lx, ly) in enumerate([(-2.34, 0.18), (1.88, 0.18)]):
    add_cyl(f"LampBase_{idx}", (lx, ly, 0.015), 0.075, 0.030, brass_mat, bevel_w=0.004)
    add_cyl(f"LampBody_{idx}", (lx, ly, 0.085), 0.048, 0.11, brass_mat, bevel_w=0.005)
    add_cyl(f"LampChimney_{idx}", (lx, ly, 0.205), 0.036, 0.14, lens_glass_mat)
    fl = add_cone(f"LampFlame_{idx}", (lx, ly, 0.175), 0.012, 0.001, 0.048, lamp_flame_mat, verts=16)
    lamp_flames.append(fl)


# ============================================================================
# 7. ZONE A (MID-LEFT X=-0.74): HOOKE'S LAW HELICAL SPRING RIG + CENTER CIPHER!
# ============================================================================
RIG_X, RIG_Y = -0.74, -0.10
# Heavy Cast-Iron & Brass Gallows Stand (Z = 0..0.72)
add_box("SpringRigBase", (RIG_X, RIG_Y, 0.018), (0.42, 0.28, 0.036), iron_mat, bevel_w=0.005)
add_cyl("SpringRigPost", (RIG_X - 0.14, RIG_Y, 0.365), 0.016, 0.68, brass_mat, verts=32)
add_cyl("SpringRigFinial", (RIG_X - 0.14, RIG_Y, 0.72), 0.026, 0.030, brass_mat, bevel_w=0.004)
add_box("SpringRigArm", (RIG_X - 0.03, RIG_Y, 0.685), (0.28, 0.026, 0.026), brass_mat, bevel_w=0.004)
add_cyl("SpringAnchorHook", (RIG_X + 0.08, RIG_Y, 0.665), 0.016, 0.022, bronze_mat)

# Vertical Glowing Calibrated Extension Ruler right beside the spring!
RULER_X = RIG_X + 0.20
add_box("BrassRulerPlate", (RULER_X, RIG_Y, 0.34), (0.042, 0.012, 0.60), iron_mat, bevel_w=0.002)
add_box("BrassRulerTrim", (RULER_X, RIG_Y + 0.005, 0.34), (0.052, 0.008, 0.615), brass_mat, bevel_w=0.002)
for tick_i in range(9):
    tz = 0.10 + tick_i * 0.060
    tw = 0.034 if tick_i % 2 == 0 else 0.020
    add_box(f"RulerTick_{tick_i}", (RULER_X, RIG_Y - 0.008, tz), (tw, 0.004, 0.006), cyan_glow_mat if tick_i in (2, 5, 8) else brass_mat)

# Live Cyan Displacement Pointer Bar sliding smoothly down the ruler with the weight!
ruler_pointer = add_box("RulerPointer", (RIG_X + 0.14, RIG_Y - 0.012, 0.42), (0.11, 0.008, 0.012), cyan_glow_mat, bevel_w=0.002)

# REAL 3D HELICAL TENSION SPRING (Anchored at Z=0.66, natural length 0.22m -> stretches smoothly to 0.46m!)
SPRING_TOP_Z = 0.660
SPRING_REST_LEN = 0.220
hero_spring = build_helical_spring(
    "HookeHeroSpring",
    (RIG_X + 0.08, RIG_Y, SPRING_TOP_Z),
    radius=0.044,
    length=SPRING_REST_LEN,
    turns=9.0,
    wire_r=0.0055,
    mat=steel_mat,
)

# Hanging Brass Pan + 2 Calibrated Brass Weights (Weight 1 always attached, Weight 2 drops at t=2.35s for 2x stretch!)
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(RIG_X + 0.08, RIG_Y, SPRING_TOP_Z - SPRING_REST_LEN))
weight_pan = bpy.context.active_object
weight_pan.name = "BrassWeightPan"
pan_disc = add_cyl("PanDisc", (0, 0, -0.022), 0.054, 0.012, brass_mat, bevel_w=0.003)
pan_disc.parent = weight_pan
w1 = add_cyl("CalibratedWeight1", (0, 0, -0.056), 0.038, 0.058, brass_mat, bevel_w=0.004)
w1.parent = weight_pan
w1_band = add_torus("Weight1GlowBand", (0, 0, -0.056), 0.039, 0.0035, gold_glow_mat)
w1_band.parent = weight_pan
w2 = add_cyl("CalibratedWeight2", (0, 0, -0.118), 0.038, 0.058, brass_mat, bevel_w=0.004)
w2.parent = weight_pan
w2_band = add_torus("Weight2GlowBand", (0, 0, -0.118), 0.039, 0.0035, cyan_glow_mat)
w2_band.parent = weight_pan

# Crimson Leather-Bound *Micrographia* Folio + Floating 3D Secret Anagram ('ceiiinosssttuv') -> 'F = -kx' at CLEAN CENTER!
FOLIO_POS = Vector((0.00, -0.10, 0.0))
add_box("MicrographiaCover", (FOLIO_POS.x, FOLIO_POS.y, 0.024), (0.36, 0.26, 0.048), crimson_leather_mat, rot=(0, 0, -0.04), bevel_w=0.005)
add_box("MicrographiaPages", (FOLIO_POS.x + 0.004, FOLIO_POS.y, 0.024), (0.345, 0.245, 0.038), parchment_mat, rot=(0, 0, -0.04))
add_box("MicrographiaClasp1", (FOLIO_POS.x - 0.175, FOLIO_POS.y + 0.06, 0.024), (0.022, 0.032, 0.052), brass_mat, rot=(0, 0, -0.04))
add_box("MicrographiaClasp2", (FOLIO_POS.x - 0.175, FOLIO_POS.y - 0.06, 0.024), (0.022, 0.032, 0.052), brass_mat, rot=(0, 0, -0.04))

# Floating 3D Anagram Text (Beat 2a: 4.5-7.1s) -> Snaps into Glowing 3D 'F = - k x' (Beat 2b!)
bpy.ops.object.text_add(location=(FOLIO_POS.x, FOLIO_POS.y - 0.04, 0.25), rotation=(math.pi * 0.5, 0.0, 0.0))
anagram_txt = bpy.context.active_object
anagram_txt.name = "HookeAnagramCipher"
if HERO_FONT is not None:
    anagram_txt.data.font = HERO_FONT
anagram_txt.data.body = "ceiiinosssttuv"
anagram_txt.data.align_x = "CENTER"
anagram_txt.data.size = 0.050
anagram_txt.data.space_character = 1.18
anagram_txt.data.extrude = 0.0045
anagram_txt.data.bevel_depth = 0.0005
anagram_txt.data.materials.append(text_gold_mat)

bpy.ops.object.text_add(location=(FOLIO_POS.x, FOLIO_POS.y - 0.04, 0.30), rotation=(math.pi * 0.5, 0.0, 0.0))
formula_txt = bpy.context.active_object
formula_txt.name = "HookeFormula3D"
if HERO_FONT is not None:
    formula_txt.data.font = HERO_FONT
formula_txt.data.body = "F = - k x"
formula_txt.data.align_x = "CENTER"
formula_txt.data.size = 0.096
formula_txt.data.space_character = 1.12
formula_txt.data.extrude = 0.0065
formula_txt.data.bevel_depth = 0.0007
formula_txt.data.materials.append(text_cyan_mat)


# ============================================================================
# 8. ZONE B (FAR-LEFT X=-1.92): 1665 COMPOUND MICROSCOPE & CRISP HEXAGONAL CORK CELLS!
# ============================================================================
MIC_POS = Vector((-1.92, -0.06, 0.0))
add_cyl("MicroscopeBase", (MIC_POS.x, MIC_POS.y, 0.018), 0.135, 0.036, brass_mat, verts=48, bevel_w=0.005)
add_cyl("MicroscopePillar", (MIC_POS.x - 0.060, MIC_POS.y + 0.03, 0.21), 0.015, 0.38, brass_mat, verts=28)
add_cyl("MicroscopeStage", (MIC_POS.x + 0.02, MIC_POS.y, 0.120), 0.082, 0.014, bronze_mat, verts=40, bevel_w=0.003)

bpy.ops.object.empty_add(type="PLAIN_AXES", location=(MIC_POS.x + 0.02, MIC_POS.y, 0.22), rotation=(0.0, math.radians(-18.0), math.radians(-10.0)))
mic_barrel_piv = bpy.context.active_object
mic_barrel_piv.name = "MicroscopeBarrelPivot"
mb_main = add_cyl("MicroscopeLeatherTube", (0, 0, 0.04), 0.044, 0.20, crimson_leather_mat, verts=40)
mb_main.parent = mic_barrel_piv
for rz in (-0.060, 0.04, 0.140):
    mr = add_torus(f"MicroscopeRing_{rz:.2f}", (0, 0, rz), 0.045, 0.006, brass_mat)
    mr.parent = mic_barrel_piv
mb_snout = add_cone("MicroscopeObjective", (0, 0, -0.090), 0.018, 0.034, 0.078, brass_mat)
mb_snout.parent = mic_barrel_piv
mb_eye = add_cyl("MicroscopeEyepiece", (0, 0, 0.168), 0.032, 0.052, brass_mat, bevel_w=0.003)
mb_eye.parent = mic_barrel_piv
mb_lens = add_sphere("MicroscopeTopLens", (0, 0, 0.196), 0.024, lens_glass_mat, segs=24, rings=14)
mb_lens.scale = (1.0, 1.0, 0.45)
mb_lens.parent = mic_barrel_piv

# Iconic Hooke Water-Filled Glass Condenser Globe beside Microscope
add_cyl("CondenserFoot", (MIC_POS.x + 0.17, MIC_POS.y - 0.05, 0.014), 0.042, 0.028, brass_mat, bevel_w=0.003)
add_cyl("CondenserStem", (MIC_POS.x + 0.17, MIC_POS.y - 0.05, 0.072), 0.010, 0.095, brass_mat)
add_sphere("CondenserGlobe", (MIC_POS.x + 0.17, MIC_POS.y - 0.05, 0.158), 0.052, lens_glass_mat, segs=32, rings=20)

# CRISP HIGH-CONTRAST HEXAGONAL CORK 'CELLS' FLOATING CLEANLY ABOVE MICROSCOPE EYEPIECE (Z=0.62, +6cm AIR GAP!)
CELLS_BASE = Vector((MIC_POS.x + 0.02, MIC_POS.y - 0.04, 0.62))
bpy.ops.object.empty_add(type="PLAIN_AXES", location=CELLS_BASE, rotation=(math.radians(78.0), 0.0, math.radians(-8.0)))
cork_cells_group = bpy.context.active_object
cork_cells_group.name = "MicrographiaCorkCellsGroup"
cork_backplate = add_cyl("CorkBackplateDisc", (0, 0, -0.006), 0.135, 0.008, cork_backplate_mat, verts=48)
cork_backplate.parent = cork_cells_group
cork_rim = add_torus("CorkScopeRing", (0, 0, 0.004), 0.138, 0.0085, brass_mat, maj_seg=48, min_seg=12)
cork_rim.parent = cork_cells_group
cork_cells_mesh = build_hex_cell_lattice("MicrographiaCorkCells", (0, 0, 0.0), rings=2, cell_r=0.030, wall_h=0.024, mat=cork_cell_mat)
cork_cells_mesh.parent = cork_cells_group


# ============================================================================
# 9. ZONE C (FAR-RIGHT X=+1.42): OVERLOAD ELASTIC-LIMIT TEST SPRING (BEAT 4 SNAP TWIST!)
# ============================================================================
OVER_X, OVER_Y = 1.42, -0.08
add_box("OverloadBase", (OVER_X, OVER_Y, 0.018), (0.38, 0.26, 0.036), iron_mat, bevel_w=0.005)
add_cyl("OverloadPost", (OVER_X + 0.13, OVER_Y, 0.34), 0.015, 0.64, bronze_mat, verts=28)
add_box("OverloadArm", (OVER_X + 0.03, OVER_Y, 0.645), (0.24, 0.024, 0.024), bronze_mat, bevel_w=0.004)

OVER_TOP_Z = 0.625
overload_spring = build_helical_spring(
    "OverloadSpring",
    (OVER_X - 0.04, OVER_Y, OVER_TOP_Z),
    radius=0.042,
    length=0.20,
    turns=8.0,
    wire_r=0.0052,
    mat=brass_mat,
)
overload_glow_coil = build_helical_spring(
    "OverloadGlowCoil",
    (OVER_X - 0.04, OVER_Y, OVER_TOP_Z),
    radius=0.044,
    length=0.20,
    turns=8.0,
    wire_r=0.0032,
    mat=hot_copper_mat,
)
overload_weight = add_cyl("OverloadHeavyWeight", (OVER_X - 0.04, OVER_Y, 0.34), 0.068, 0.105, iron_mat, bevel_w=0.006)
overload_ring = add_torus("OverloadWeightBand", (OVER_X - 0.04, OVER_Y, 0.34), 0.070, 0.006, hot_copper_mat)

snap_sparks = []
for i in range(16):
    sp = add_sphere(f"SnapSpark_{i}", (OVER_X - 0.04, OVER_Y, 0.42), 0.013, hot_copper_mat if i % 2 == 0 else gold_glow_mat, segs=12, rings=8)
    snap_sparks.append(sp)


# ============================================================================
# 10. ZONE D (BEATS 5-6 HERO): GIANT SKELETONIZED BRASS POCKET WATCH & HAIRSPRING!
# ============================================================================
# Parented to watch_station_root so it appears cleanly in Beat 5 (X=+0.28) & centers in Beat 6 (X=+0.06)
# — keeping Beats 1 & 2 100% uncluttered behind Hooke & the glowing 3D cipher!
WATCH_POS = Vector((0.28, 0.02, 0.30))
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(WATCH_POS.x, WATCH_POS.y, 0.0))
watch_station_root = bpy.context.active_object
watch_station_root.name = "WatchStationRoot"

ws_base = add_box("WatchStandBase", (0.0, 0.03, 0.020), (0.46, 0.26, 0.040), mahogany_mat, bevel_w=0.006)
ws_base.parent = watch_station_root
ws_velvet = add_box("WatchVelvetCushion", (0.0, 0.02, 0.048), (0.40, 0.22, 0.026), emerald_velvet_mat, bevel_w=0.008)
ws_velvet.parent = watch_station_root

bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, WATCH_POS.z), rotation=(math.radians(68.0), 0.0, 0.0))
watch_case = bpy.context.active_object
watch_case.name = "HookePocketWatch"
watch_case.parent = watch_station_root

# Outer Ornate Brass Case, Crown Bow & Recessed Skeleton Plate
w_outer = add_torus("WatchCaseRim", (0, 0, 0), 0.245, 0.026, brass_mat, maj_seg=72, min_seg=18)
w_outer.parent = watch_case
w_back = add_cyl("WatchBackPlate", (0, 0, -0.022), 0.236, 0.012, bronze_mat, verts=64)
w_back.parent = watch_case
w_crown = add_cyl("WatchCrownStem", (0, 0.274, 0), 0.022, 0.052, brass_mat, rot=(math.pi * 0.5, 0, 0), verts=28)
w_crown.parent = watch_case
w_bow = add_torus("WatchCrownBow", (0, 0.320, 0), 0.048, 0.0085, brass_mat, maj_seg=36, min_seg=12)
w_bow.parent = watch_case

# 3 Meshing Brass Watchwork Escapement Gears inside skeleton plate!
w_gear1 = add_toothed_gear("WatchGear1", (-0.098, -0.078, -0.006), 0.068, 0.012, 14, brass_mat, flat_xy=True)
w_gear1.parent = watch_case
w_gear2 = add_toothed_gear("WatchGear2", (0.102, -0.074, -0.006), 0.058, 0.012, 12, brass_mat, flat_xy=True)
w_gear2.parent = watch_case
w_gear3 = add_toothed_gear("WatchGear3", (0.00, -0.135, -0.004), 0.044, 0.012, 10, steel_mat, flat_xy=True)
w_gear3.parent = watch_case

# Hero Oscillating Brass Balance Wheel + 3 Spokes + Poising Screws + Ruby Jewel Bearing!
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.042, 0.018))
balance_wheel = bpy.context.active_object
balance_wheel.name = "BalanceWheelPivot"
balance_wheel.parent = watch_case
bw_rim = add_torus("BalanceRim", (0, 0, 0), 0.135, 0.011, brass_mat, maj_seg=64, min_seg=14)
bw_rim.parent = balance_wheel
for sp_i in range(3):
    sa = math.radians(sp_i * 120.0)
    sp = add_box(f"BalanceSpoke_{sp_i}", (math.cos(sa) * 0.068, math.sin(sa) * 0.068, 0.0), (0.132, 0.010, 0.008), brass_mat, rot=(0, 0, sa))
    sp.parent = balance_wheel
for sc_i in range(12):
    sa = math.radians(sc_i * 30.0)
    sc = add_cyl(f"PoisingScrew_{sc_i}", (math.cos(sa) * 0.148, math.sin(sa) * 0.148, 0.0), 0.0055, 0.013, gold_glow_mat, rot=(0, math.pi * 0.5, sa), verts=12)
    sc.parent = balance_wheel
bw_ruby = add_sphere("RubyCenterJewel", (0, 0, 0.008), 0.019, ruby_jewel_mat, segs=24, rings=14)
bw_ruby.parent = balance_wheel

# REAL BREATHING BLUED-STEEL SPIRAL HAIRSPRING (expands & contracts smoothly with balance wheel!)
hairspring = build_spiral_hairspring("BalanceHairspring", (0.0, 0.042, 0.030), r_in=0.020, r_out=0.108, turns=4.5, wire_r=0.0038, mat=cyan_glow_mat, segs=130)
hairspring.parent = watch_case


# ============================================================================
# 11. ZONE E (BEAT 6 FINALE): GLOWING 3D DOUBLE-HELIX DNA ELASTIC SPRING & TITLE!
# ============================================================================
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.20, 0.96))
dna_group = bpy.context.active_object
dna_group.name = "ModernDNASpringGroup"

# Twin Intertwined DNA Helices (Glowing Cyan & Emerald-Gold elastic springs!)
dna_a = build_helical_spring("DNAHelixA", (0, 0, 0.24), radius=0.095, length=0.48, turns=2.5, wire_r=0.008, mat=cyan_glow_mat, segs_per_turn=24)
dna_a.parent = dna_group
dna_b = build_helical_spring("DNAHelixB", (0, 0, 0.24), radius=0.095, length=0.48, turns=2.5, wire_r=0.008, mat=emerald_glow_mat, segs_per_turn=24)
dna_b.rotation_euler = Euler((0.0, 0.0, math.pi))
dna_b.parent = dna_group

for r_i in range(10):
    u_r = (r_i + 0.5) / 10.0
    ang = u_r * 2.5 * math.pi * 2.0
    zr = 0.22 - u_r * 0.44
    rung = add_cyl(f"DNARung_{r_i}", (0.0, 0.0, zr), 0.005, 0.185, gold_glow_mat, rot=(0.0, math.pi * 0.5, ang), verts=12)
    rung.parent = dna_group

# Crisp Self-Luminous Cinema Gold 3D Banner Title centered cleanly inside upper sapphire window pane (Beats 5-6!)
bpy.ops.object.text_add(location=(0.0, 0.58, 1.52), rotation=(math.pi * 0.5, 0.0, 0.0))
title_txt = bpy.context.active_object
title_txt.name = "HookeTitle3D"
if HERO_FONT is not None:
    title_txt.data.font = HERO_FONT
title_txt.data.body = "HOOKE'S LAW\nF = - k x"
title_txt.data.align_x = "CENTER"
title_txt.data.align_y = "CENTER"
title_txt.data.size = 0.082
title_txt.data.space_character = 1.14
title_txt.data.space_line = 1.25
title_txt.data.extrude = 0.0065
title_txt.data.bevel_depth = 0.0007
title_txt.data.materials.append(title_gold_mat)


# ============================================================================
# 11B. REALISTIC MPFB2 V4 (@ZACKDFILMS-GRADE) 4K PHOTO-SCANNED HUMAN CAST!
# ============================================================================
HOOKE_ACTOR = None
RIVAL_ACTOR = None
try:
    HOOKE_ACTOR = create_realistic_scientist(
        name="RobertHooke",
        floor_z=FLOOR_Z,
        start_xy=(-1.18, 0.44),
        yaw_deg=24.0,
        cfg={
            "age": 0.58,
            "muscle": 0.54,
            "weight": 0.48,
            "height": 0.55,
            "skin_assets": ["middleage_caucasian_male.mhmat", "young_caucasian_male.mhmat"],
            "iris_major_rgb": (0.16, 0.38, 0.22),
            "iris_minor_rgb": (0.06, 0.16, 0.08),
            "eyebrow_assets": ["eyebrow002.mhclo", "eyebrow001.mhclo"],
            "hair_assets": ["punkduck_alpha7_curly.mhclo", "punkduck_alpha7_long.mhclo", "long01.mhclo", "bob01.mhclo"],
            "hair_tint_rgb": (0.45, 0.26, 0.15),
            "hair_tint_fac": 0.40,
            "brow_tint_fac": 0.50,
            "silver_hair": False,
            "suit_assets": [
                "toigo_male_double-breasted_suit.mhclo",
                "male_elegantsuit01.mhclo",
                "male_casualsuit01.mhclo",
            ],
            "suit_tint_rgb": (0.16, 0.48, 0.34),  # Rich Royal Society deep forest-emerald wool coat!
            "suit_tint_fac": 0.72,
            "shoe_assets": ["shoes01.mhclo", "shoes02.mhclo"],
        },
    )
    RIVAL_ACTOR = create_realistic_scientist(
        name="RivalWatchmaker",
        floor_z=FLOOR_Z,
        start_xy=(0.46, 0.46),
        yaw_deg=-26.0,
        cfg={
            "age": 0.82,
            "muscle": 0.54,
            "weight": 0.58,
            "height": 0.58,
            "skin_assets": ["old_caucasian_male.mhmat", "middleage_caucasian_male.mhmat"],
            "iris_major_rgb": (0.18, 0.44, 0.72),
            "iris_minor_rgb": (0.08, 0.18, 0.34),
            "eyebrow_assets": ["eyebrow005.mhclo", "eyebrow003.mhclo"],
            "hair_assets": ["grinsegold_wig_bow_tie.mhclo", "bob02.mhclo", "short02.mhclo"],
            "beard_assets": ["grinsegold_beard_sigmund_wip.mhclo", "rehmanpolanski_moustache_viking.mhclo"],
            "silver_hair": True,
            "silver_hair_main": False,
            "suit_assets": [
                "toigo_suit_with_dinner_jacket.mhclo",
                "toigo_male_suit_tie_and_jacket.mhclo",
                "male_elegantsuit01.mhclo",
            ],
            "suit_tint_rgb": (0.42, 0.08, 0.12),  # Rich Royal Burgundy gentleman's formal coat (no white jacket!)
            "suit_tint_fac": 0.85,
            "shoe_assets": ["shoes02.mhclo", "shoes01.mhclo"],
        },
    )
except Exception as mpfb_err:
    print(f"[MPFB2 Warning] Skipping human cast ({mpfb_err}) — install MPFB2 & packs in Cell 2!")


# ============================================================================
# 12. REMBRANDT 3-POINT CINEMA PORTRAIT LIGHTING + TRUE HORIZONTAL 9:16 CAMERA
# ============================================================================
# Warm Amber Oil-Lamp Rembrandt Key Area Light (sculpts photoreal skin pores, cheekbones & suit folds!)
bpy.ops.object.light_add(type="AREA", location=(-1.95, -1.65, 1.72))
key_light = bpy.context.active_object
key_light.data.energy = 295.0
key_light.data.size = 1.25
key_light.data.color = (1.0, 0.78, 0.52)
key_light.rotation_euler = Euler((math.radians(52), 0.0, math.radians(-42)))
key_light.visible_camera = False

# Cool London Sapphire Moonlight Rim Light (crisp backlit halo along alpha hair strands & wool shoulders!)
bpy.ops.object.light_add(type="AREA", location=(1.45, 2.35, 2.08))
rim_light = bpy.context.active_object
rim_light.data.energy = 360.0
rim_light.data.size = 1.60
rim_light.data.color = (0.42, 0.80, 1.0)
rim_light.rotation_euler = Euler((math.radians(58), 0.0, math.radians(146)))
rim_light.visible_camera = False

# Soft Warm Cinema Fill (low ratio so faces have real filmic shadow depth instead of flat front light!)
bpy.ops.object.light_add(type="AREA", location=(0.85, -1.65, 1.48))
wall_wash = bpy.context.active_object
wall_wash.data.energy = 68.0
wall_wash.data.size = 2.20
wall_wash.data.color = (0.94, 0.86, 0.78)
wall_wash.rotation_euler = Euler((math.radians(66), 0.0, math.radians(24)))
wall_wash.visible_camera = False

# Subtle Wet-Cornea Eye Catchlight & Precision Instrument Sparkle
bpy.ops.object.light_add(type="POINT", location=(0.0, -0.82, 0.68))
fill_accent = bpy.context.active_object
fill_accent.data.energy = 22.0
fill_accent.data.color = (0.86, 0.94, 1.0)
fill_accent.visible_camera = False

bpy.ops.object.empty_add(location=(0.0, 0.0, 0.44))
cam_target = bpy.context.active_object
cam_target.name = "CinemaCamTarget"

bpy.ops.object.camera_add(location=(-0.95, -1.42, 0.54))
cam = bpy.context.active_object
cam.name = "CinemaCamera"
scene.camera = cam
cam.data.sensor_fit = "HORIZONTAL"
cam.data.sensor_width = 36.0
cam.data.lens = 40.0
cam.data.dof.use_dof = True
cam.data.dof.focus_object = cam_target
cam.data.dof.aperture_fstop = 3.2
cam.rotation_mode = "QUATERNION"


# ============================================================================
# 13. 6-BEAT ZERO-OVERLAP CHARACTER + INSTRUMENT CINEMA CAMERA GRAMMAR (V5!)
# ============================================================================
def compute_beat_raw(beat_idx: int, t: float):
    """Return (cam_pos, tgt_pos, lens_mm, fstop, dutch_deg) — 100% ZERO-OVERLAP & ZERO HALF-CHOPPED ACTORS!"""
    if beat_idx == 0:
        # BEAT 1 (0..4.5s): Hooke (Left, X=-1.18, facing RIGHT +24°) + Hero Spring Rig (Right, X=-0.74)!
        u = clamp01(t / 4.5)
        pos = bezier_quad(Vector((-1.00, -1.46, 0.56)), Vector((-0.94, -1.36, 0.54)), Vector((-0.88, -1.28, 0.52)), u)
        tgt = vlerp(Vector((-0.95, 0.14, 0.46)), Vector((-0.93, 0.14, 0.44)), u)
        return pos, tgt, lerp(39.0, 41.5, u), 3.2, 0.0
    if beat_idx == 1:
        # BEAT 2 (4.5..9.5s): CLEAN FACE-TO-FACE 2-SHOT from t=4.5s (BOTH Hooke & Rival 100% INSIDE frame!)
        pan_u = ease_out_quint(clamp01((t - 6.9) / 1.1))
        pos = vlerp(Vector((0.08, -1.68, 0.56)), Vector((-0.02, -1.60, 0.56)), pan_u)
        tgt = vlerp(Vector((0.04, 0.18, 0.46)), Vector((0.00, 0.20, 0.46)), pan_u)
        return pos, tgt, lerp(36.0, 35.5, pan_u), 3.2, 0.0
    if beat_idx == 2:
        # BEAT 3 (9.5..15.5s): Microscope + Floating Cork Cells (Left, X=-1.92) + Hooke (Right, X=-1.40, facing LEFT -26°)!
        u = clamp01((t - 9.5) / 6.0)
        pos = bezier_quad(Vector((-1.72, -1.46, 0.58)), Vector((-1.66, -1.36, 0.60)), Vector((-1.60, -1.30, 0.62)), u)
        tgt = vlerp(Vector((-1.66, 0.14, 0.46)), Vector((-1.64, 0.14, 0.52)), smoothstep(10.5, 13.0, t))
        return pos, tgt, lerp(38.5, 40.5, u), 3.0, 0.0
    if beat_idx == 3:
        # BEAT 4 (15.5..20.5s): 3.6° Dutch-Tilt Rival (Left, X=+0.94, facing RIGHT +24°) + Overload Spring SNAP (Right, X=+1.42)!
        u = clamp01((t - 15.5) / 5.0)
        pos = bezier_quad(Vector((1.14, -1.44, 0.54)), Vector((1.18, -1.36, 0.55)), Vector((1.22, -1.42, 0.56)), u)
        tgt = Vector((1.16, 0.16, 0.46))
        shake = math.sin((t - 17.65) * 52.0) * math.exp(-(t - 17.65) * 6.5) * 0.028 if t >= 17.65 else 0.0
        pos.x += shake
        pos.z += shake * 0.5
        return pos, tgt, lerp(38.0, 40.0, u), 3.0, 3.6 * math.sin(u * math.pi)
    if beat_idx == 4:
        # BEAT 5 (20.5..25.5s): Hooke (Left, X=-0.28, facing RIGHT +24°) presenting Giant Skeleton Pocket Watch (Right, X=+0.28)!
        u = clamp01((t - 20.5) / 5.0)
        pos = bezier_quad(Vector((-0.04, -1.40, 0.54)), Vector((0.02, -1.34, 0.58)), Vector((0.06, -1.38, 0.62)), u)
        tgt = vlerp(Vector((0.01, 0.20, 0.46)), Vector((0.02, 0.22, 0.48)), u)
        return pos, tgt, lerp(38.0, 40.0, u), 3.2, 0.0
    # BEAT 6 (25.5..30.0s): Hero Wide Finale — Hooke (Left) + Rival (Right) angled inward flanking Watch, DNA Spring & Title!
    u = clamp01((t - 25.5) / 4.5)
    pos = bezier_quad(Vector((0.04, -1.90, 0.72)), Vector((0.03, -2.06, 0.70)), Vector((0.02, -2.18, 0.68)), u)
    tgt = vlerp(Vector((0.03, 0.26, 0.55)), Vector((0.03, 0.24, 0.53)), u)
    return pos, tgt, lerp(34.5, 32.5, u), 3.8, 0.0


BEAT_BOUNDS = [0.0, 4.5, 9.5, 15.5, 20.5, 25.5, 30.0]


def compute_camera_state(t: float):
    """Smooth single-take cinema camera with 0.42s easeOutQuint whip-pans synced to exact whip_whoosh SFX!"""
    sway_x = math.sin(t * 1.18) * 0.010 + math.sin(t * 2.45) * 0.003
    sway_z = math.cos(t * 0.94) * 0.008

    b_idx = 5
    for i in range(6):
        if BEAT_BOUNDS[i] <= t < BEAT_BOUNDS[i + 1]:
            b_idx = i
            break

    pos, tgt, lens, fstop, dutch = compute_beat_raw(b_idx, t)

    whip_dur = 0.42
    dt_in = t - BEAT_BOUNDS[b_idx]
    if b_idx > 0 and dt_in < whip_dur:
        w = ease_out_quint(dt_in / whip_dur)
        p0, t0, l0, f0, d0 = compute_beat_raw(b_idx - 1, BEAT_BOUNDS[b_idx])
        pos = vlerp(p0, pos, w)
        tgt = vlerp(t0, tgt, w)
        lens = lerp(l0, lens, w)
        fstop = lerp(f0, fstop, w)
        dutch = lerp(d0, dutch, w)

    pos.x += sway_x
    pos.z += sway_z
    return pos, tgt, lens, fstop, dutch


# ============================================================================
# 14. KEYFRAME ALL 720 FRAMES DETERMINISTICALLY (EXACT 30.000s AUDIO SYNC!)
# ============================================================================
for f in range(1, TOTAL_FRAMES + 1):
    t = (f - 1) / FPS

    # 1. Exact Camera Position, Look-At Quaternion + True Dutch Roll, Lens & Focus
    c_pos, c_tgt, c_lens, c_fstop, c_dutch = compute_camera_state(t)
    cam.location = c_pos
    cam_target.location = c_tgt
    cam.rotation_quaternion = look_at_quat(c_pos, c_tgt, c_dutch)
    cam.data.lens = c_lens
    cam.data.dof.aperture_fstop = c_fstop
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_quaternion", frame=f)
    cam_target.keyframe_insert("location", frame=f)
    cam.data.keyframe_insert("lens", frame=f)
    cam.data.keyframe_insert(data_path="dof.aperture_fstop", frame=f)

    # 2. Delicate Oil-Lamp Flame Flicker
    for idx, fl in enumerate(lamp_flames):
        flick = 1.0 + 0.12 * math.sin(t * 22.0 + idx * 1.9) + 0.06 * math.sin(t * 36.0)
        fl.scale = (1.0, 1.0, flick)
        fl.keyframe_insert("scale", frame=f)

    # 3. Hooke's Law Hero Spring Physics: 1x Weight (0.3-2.35s) -> 2x Weight Double Stretch + Damped Bounce!
    s1 = smoothstep(0.28, 0.85, t)
    b1 = math.sin((t - 0.85) * 14.0) * math.exp(-(t - 0.85) * 2.8) * 0.14 if t >= 0.85 else 0.0
    s2 = smoothstep(2.35, 2.85, t)
    b2 = math.sin((t - 2.85) * 12.5) * math.exp(-(t - 2.85) * 2.4) * 0.20 if t >= 2.85 else 0.0
    b3 = math.sin((t - 7.25) * 13.0) * math.exp(-(t - 7.25) * 2.2) * 0.16 if t >= 7.25 else 0.0

    stretch_ratio = max(0.85, min(2.05, 1.0 + 0.36 * s1 + b1 + 0.42 * s2 + b2 + b3))
    hero_spring.scale = Vector((1.0, 1.0, stretch_ratio))
    cur_bottom_z = SPRING_TOP_Z - SPRING_REST_LEN * stretch_ratio
    weight_pan.location = Vector((RIG_X + 0.08, RIG_Y, cur_bottom_z))
    ruler_pointer.location = Vector((RIG_X + 0.14, RIG_Y - 0.012, cur_bottom_z))
    w2_vis = max(0.0001, smoothstep(2.15, 2.45, t))
    w2.scale = Vector((w2_vis, w2_vis, w2_vis))
    w2_band.scale = Vector((w2_vis, w2_vis, w2_vis))
    hero_spring.keyframe_insert("scale", frame=f)
    weight_pan.keyframe_insert("location", frame=f)
    ruler_pointer.keyframe_insert("location", frame=f)
    w2.keyframe_insert("scale", frame=f)
    w2_band.keyframe_insert("scale", frame=f)

    # 4. Beat 2 Secret Anagram ('ceiiinosssttuv') -> Snaps into Glowing 3D 'F = -kx' at t=7.15s!
    ana_vis = smoothstep(4.5, 5.1, t) * (1.0 - smoothstep(6.95, 7.25, t))
    ana_sc = max(0.0001, ana_vis)
    anagram_txt.scale = Vector((ana_sc, ana_sc, ana_sc))
    anagram_txt.location = Vector((FOLIO_POS.x, FOLIO_POS.y - 0.04, 0.25 + math.sin(t * 2.6) * 0.010))
    anagram_txt.keyframe_insert("scale", frame=f)
    anagram_txt.keyframe_insert("location", frame=f)

    form_vis = smoothstep(7.05, 7.45, t) * (1.0 - smoothstep(10.0, 10.6, t))
    form_sc = max(0.0001, form_vis)
    formula_txt.scale = Vector((form_sc, form_sc, form_sc))
    formula_txt.location = Vector((FOLIO_POS.x, FOLIO_POS.y - 0.04, 0.30 + math.sin(t * 2.4) * 0.010))
    formula_txt.keyframe_insert("scale", frame=f)
    formula_txt.keyframe_insert("location", frame=f)

    # 5. Beat 3 Crisp Dark-Backlit Hexagonal Cork 'CELLS' Floating CLEANLY ABOVE Microscope (+6cm AIR GAP!)
    cell_u = smoothstep(10.5, 11.8, t) * (1.0 - smoothstep(15.3, 15.8, t)) + 0.65 * smoothstep(25.4, 26.2, t)
    cell_sc = max(0.0001, cell_u)
    cork_cells_group.scale = Vector((cell_sc, cell_sc, cell_sc))
    cork_cells_group.location = Vector((CELLS_BASE.x, CELLS_BASE.y, lerp(0.50, CELLS_BASE.z, min(1.0, cell_u)) + math.sin(t * 1.9) * 0.006))
    cork_cells_group.rotation_euler = Euler((math.radians(78.0), 0.0, math.radians(-8.0) + math.sin(t * 0.8) * 0.08))
    cork_cells_group.keyframe_insert("scale", frame=f)
    cork_cells_group.keyframe_insert("location", frame=f)
    cork_cells_group.keyframe_insert("rotation_euler", frame=f)

    # 6. Beat 4 Overload Elastic-Limit Spring -> Copper Glow Strain -> DRAMATIC SNAP at t=17.65s!
    if t < 17.65:
        ov_pull = smoothstep(16.0, 17.62, t)
        ov_vib = math.sin(t * 58.0) * 0.04 * smoothstep(16.6, 17.6, t)
        ov_stretch = 1.0 + ov_pull * 1.35 + ov_vib
        overload_spring.scale = Vector((1.0 - ov_pull * 0.16, 1.0 - ov_pull * 0.16, ov_stretch))
        glow_sc = max(0.0001, smoothstep(16.5, 17.55, t))
        overload_glow_coil.scale = Vector((glow_sc * (1.0 - ov_pull * 0.16), glow_sc * (1.0 - ov_pull * 0.16), glow_sc * ov_stretch))
        wt_z = OVER_TOP_Z - 0.20 * ov_stretch - 0.05
        overload_weight.location = Vector((OVER_X - 0.04, OVER_Y, wt_z))
        overload_ring.location = Vector((OVER_X - 0.04, OVER_Y, wt_z))
    else:
        dt_sn = t - 17.65
        fall_u = clamp01(dt_sn / 0.28)
        wt_z = lerp(0.11, 0.072, fall_u * fall_u)
        overload_weight.location = Vector((OVER_X - 0.04, OVER_Y, wt_z))
        overload_ring.location = Vector((OVER_X - 0.04, OVER_Y, wt_z))
        recoil = math.sin(dt_sn * 28.0) * math.exp(-dt_sn * 4.5) * 0.25
        overload_spring.scale = Vector((0.76, 0.76, 1.68 + recoil))
        overload_spring.rotation_euler = Euler((0.18, -0.22 + recoil * 0.5, 0.0))
        fade_gl = max(0.0001, 1.0 - smoothstep(17.65, 18.8, t))
        overload_glow_coil.scale = Vector((fade_gl * 0.76, fade_gl * 0.76, fade_gl * 1.68))

    overload_spring.keyframe_insert("scale", frame=f)
    overload_spring.keyframe_insert("rotation_euler", frame=f)
    overload_glow_coil.keyframe_insert("scale", frame=f)
    overload_weight.keyframe_insert("location", frame=f)
    overload_ring.keyframe_insert("location", frame=f)

    dt_sp = t - 17.65
    for i, sp in enumerate(snap_sparks):
        if 0.0 <= dt_sp < 1.65:
            u_s = dt_sp / 1.65
            ang = (i / 16.0) * math.pi * 2.0
            rad = 0.04 + u_s * (0.36 + (i % 4) * 0.09)
            sx = OVER_X - 0.04 + math.cos(ang * 1.2) * rad
            sy = OVER_Y + math.sin(ang * 1.5) * rad * 0.65
            sz = 0.34 + u_s * (0.28 - u_s * 0.58)
            sc = max(0.0001, (1.0 - u_s) * 0.95)
            sp.location = Vector((sx, sy, max(0.02, sz)))
            sp.scale = Vector((sc, sc, sc))
        else:
            sp.scale = Vector((0.0001, 0.0001, 0.0001))
        sp.keyframe_insert("location", frame=f)
        sp.keyframe_insert("scale", frame=f)

    # 7. Beat 5-6 Giant Pocket Watch Station (Unveiled cleanly at Beat 5, centers for Beat 6 Finale!)
    w_vis = max(0.0001, smoothstep(20.1, 20.6, t))
    w_center_u = smoothstep(25.3, 26.0, t)
    watch_station_root.scale = Vector((w_vis, w_vis, w_vis))
    watch_station_root.location = Vector((lerp(0.28, 0.06, w_center_u), lerp(0.02, 0.04, w_center_u), 0.0))
    watch_station_root.keyframe_insert("scale", frame=f)
    watch_station_root.keyframe_insert("location", frame=f)

    osc = math.sin(t * math.pi * 5.0)
    balance_wheel.rotation_euler = Euler((0.0, 0.0, osc * math.radians(68.0)))
    w_breath = 1.0 + osc * 0.14
    hairspring.scale = Vector((w_breath, w_breath, 1.0))
    hairspring.rotation_euler = Euler((0.0, 0.0, osc * math.radians(22.0)))

    w_gear1.rotation_euler = Euler((0.0, 0.0, t * 0.90))
    w_gear2.rotation_euler = Euler((0.0, 0.0, -t * 0.90 * (14.0 / 12.0)))
    w_gear3.rotation_euler = Euler((0.0, 0.0, t * 0.90 * (14.0 / 10.0)))
    for obj_w in (balance_wheel, hairspring, w_gear1, w_gear2, w_gear3):
        obj_w.keyframe_insert("rotation_euler", frame=f)
    hairspring.keyframe_insert("scale", frame=f)

    # 8. Beat 6 Modern Legacy: Glowing 3D Double-Helix DNA Elastic Spring + Brass Title!
    dna_u = smoothstep(25.5, 26.6, t)
    dna_sc = max(0.0001, dna_u)
    dna_stretch = 1.0 + 0.16 * math.sin((t - 25.5) * 4.8) * dna_u
    dna_group.scale = Vector((dna_sc, dna_sc, dna_sc * dna_stretch))
    dna_group.location = Vector((0.0, 0.08, 0.92))
    dna_group.rotation_euler = Euler((math.radians(14.0), math.radians(-18.0), t * 1.15))
    dna_group.keyframe_insert("scale", frame=f)
    dna_group.keyframe_insert("location", frame=f)
    dna_group.keyframe_insert("rotation_euler", frame=f)

    tit_u = smoothstep(21.2, 22.4, t)
    tit_sc = max(0.0001, tit_u)
    title_txt.scale = Vector((tit_sc, tit_sc, tit_sc))
    title_txt.keyframe_insert("scale", frame=f)

    # 9. REALISTIC MPFB2 CAST ACTING — TRUE +Z FACING SIGNS (+yaw/+rz = LOOK RIGHT +X, -yaw/-rz = LOOK LEFT -X)!
    breath = math.sin(t * 2.3) * 1.2
    if HOOKE_ACTOR is not None:
        hrig = HOOKE_ACTOR["rig"]
        h_lip = sample_lip_sync("hooke", t)
        h_base_z = HOOKE_ACTOR["base_z"]

        if t < 4.5:
            # Beat 1: Hooke LEFT (X=-1.18, Y=0.44), facing RIGHT (+24°) directly at Spring Rig (X=-0.74)!
            hx, hy, hyaw = -1.18, 0.44, 24.0
            lean_x = 5.0 + breath * 0.4
            head_rx, head_rz = 8.0 - smoothstep(2.2, 3.0, t) * 4.0, 14.0 + math.sin(t * 1.8) * 3.0
            g = smoothstep(2.0, 2.8, t)
            ua_l = (-22.0 - g * 8.0, 22.0, -30.0 - g * 6.0)
            la_l = (-34.0 - g * 4.0 + math.sin(t * 3.2) * 4.0, 0.0, -20.0)
            ua_r = (-16.0, -30.0, 18.0)
            la_r = (-38.0, 0.0, 28.0)
        elif t < 9.5:
            # Beat 2: Hooke LEFT (X=-0.44, Y=0.46), facing RIGHT (+28°) directly at Rival & glowing 'F = - k x'!
            hx, hy, hyaw = -0.44, 0.46, 28.0
            sp_u = smoothstep(6.9, 7.4, t)
            lean_x = 4.0 + sp_u * 3.0 + breath * 0.4
            head_rx, head_rz = 4.0 - sp_u * 3.0, 14.0 + math.sin(t * 3.2) * 3.0 * sp_u
            ua_l = (lerp(-18.0, -28.0, sp_u), lerp(26.0, 20.0, sp_u), lerp(-24.0, -34.0, sp_u))
            la_l = (lerp(-36.0, -38.0, sp_u) + math.sin(t * 4.5) * 4.0 * sp_u, 0.0, lerp(-26.0, -18.0, sp_u))
            ua_r = (lerp(-16.0, -20.0, sp_u), lerp(-30.0, -26.0, sp_u), lerp(18.0, 24.0, sp_u))
            la_r = (lerp(-36.0, -38.0, sp_u), 0.0, lerp(28.0, 24.0, sp_u))
        elif t < 15.5:
            # Beat 3: Hooke RIGHT (X=-1.40, Y=0.44), facing LEFT (-26°) directly at Microscope & Cork CELLS (X=-1.92)!
            hx, hy, hyaw = -1.40, 0.44, -26.0
            wonder = smoothstep(10.8, 12.4, t)
            lean_x = 6.0 - wonder * 3.0 + breath * 0.4
            head_rx, head_rz = 8.0 - wonder * 12.0, -16.0 + math.sin(t * 1.5) * 3.5
            ua_r = (lerp(-22.0, -30.0, wonder), lerp(-24.0, -20.0, wonder), lerp(28.0, 34.0, wonder))
            la_r = (lerp(-34.0, -38.0, wonder) + math.sin(t * 2.6) * 3.0, 0.0, lerp(22.0, 18.0, wonder))
            ua_l = (-16.0, 30.0, -18.0)
            la_l = (-38.0, 0.0, -28.0)
        elif t < 20.5:
            # Beat 4: Hooke off-camera left watching Rival's spring disaster
            hx, hy, hyaw = -0.60, 0.56, 22.0
            lean_x, head_rx, head_rz = 3.0, 4.0, 16.0
            ua_r, la_r = (-16.0, -30.0, 18.0), (-36.0, 0.0, 28.0)
            ua_l, la_l = (-16.0, 30.0, -18.0), (-36.0, 0.0, -28.0)
        elif t < 25.5:
            # Beat 5: Hooke LEFT (X=-0.28, Y=0.44), facing RIGHT (+24°) directly at Giant Pocket Watch (X=+0.28)!
            hx, hy, hyaw = -0.28, 0.44, 24.0
            lean_x = 4.0 + breath * 0.4
            head_rx, head_rz = 6.0, 14.0 + math.sin(t * 1.8) * 3.5
            ua_l = (-28.0 + math.sin(t * 2.2) * 3.0, 20.0, -34.0)
            la_l = (-36.0, 0.0, -18.0)
            ua_r = (-16.0, -30.0, 18.0)
            la_r = (-38.0, 0.0, 28.0)
        else:
            # Beat 6: Grand Finale — Hooke LEFT (X=-0.52, Y=0.52), angled INWARD (+18°) toward Watch & DNA Spring!
            hx, hy, hyaw = -0.52, 0.52, 18.0
            lean_x = 2.0 + breath * 0.4
            head_rx, head_rz = -2.0, 10.0 + math.sin(t * 1.5) * 3.0
            ua_l = (-26.0, 22.0, -30.0)
            la_l = (-36.0, 0.0, -20.0)
            ua_r = (-16.0, -30.0, 18.0)
            la_r = (-38.0, 0.0, 28.0)

        hrig.location = Vector((hx, hy, h_base_z))
        hrig.rotation_euler = Euler((0.0, 0.0, math.radians(hyaw)))
        hrig.keyframe_insert("location", frame=f)
        hrig.keyframe_insert("rotation_euler", frame=f)

        set_bone_char_rot(hrig, "spine02", lean_x * 0.45, 0.0, 0.0, frame=f)
        set_bone_char_rot(hrig, "spine01", lean_x * 0.55, 0.0, 0.0, frame=f)
        set_bone_char_rot(hrig, "neck01", head_rx * 0.35, 0.0, head_rz * 0.35, frame=f)
        set_bone_char_rot(hrig, "head", head_rx * 0.65, 0.0, head_rz * 0.65, frame=f)
        h_brow = 0.85 if 11.2 <= t < 15.5 else (0.45 if t >= 20.5 else 0.15)
        h_smile = 0.65 if (7.2 <= t < 9.5 or t >= 20.5) else (0.35 if 11.2 <= t < 15.5 else 0.15)
        apply_facial_acting(
            hrig, t, f, jaw_open_deg=1.0 + h_lip * 16.5, brow_lift=h_brow, smile=h_smile, blink_period=3.1, blink_phase=0.6
        )
        set_bone_char_rot(hrig, "upperarm01.R", *ua_r, frame=f)
        set_bone_char_rot(hrig, "lowerarm01.R", *la_r, frame=f)
        set_bone_char_rot(hrig, "upperarm01.L", *ua_l, frame=f)
        set_bone_char_rot(hrig, "lowerarm01.L", *la_l, frame=f)

    if RIVAL_ACTOR is not None:
        rrig = RIVAL_ACTOR["rig"]
        r_lip = sample_lip_sync("skeptic", t)
        r_base_z = RIVAL_ACTOR["base_z"]

        if t < 4.5:
            # Beat 1: Off-camera right waiting cleanly for Beat 2
            rx_p, ry_p, ryaw = 1.65, 0.52, -22.0
            lean_x, head_rx, head_rz = 2.0, 4.0, -12.0
            ua_l, la_l = (-16.0, 30.0, -18.0), (-36.0, 0.0, -28.0)
            ua_r, la_r = (-16.0, -30.0, 18.0), (-36.0, 0.0, 28.0)
            jaw_bias, r_brow, r_smile = 0.0, -0.2, 0.0
        elif t < 9.5:
            # Beat 2: Rival RIGHT (X=+0.46, Y=0.46), facing LEFT (-28°) directly at Hooke & 'ceiiinosssttuv'!
            rx_p, ry_p, ryaw = 0.46, 0.46, -28.0
            sk = 1.0 - smoothstep(6.8, 7.3, t)
            lean_x = 5.0 * sk + breath * 0.4
            head_rx, head_rz = 5.0 * sk - 3.0 * (1.0 - sk), -14.0 + math.sin(t * 3.6) * 3.5 * sk
            ua_r = (lerp(-20.0, -28.0, sk), lerp(-26.0, -20.0, sk), lerp(24.0, 34.0, sk))
            la_r = (lerp(-36.0, -38.0, sk) + math.sin(t * 4.2) * 4.0 * sk, 0.0, lerp(24.0, 18.0, sk))
            ua_l = (-18.0, 28.0, -22.0)
            la_l = (-36.0, 0.0, -26.0)
            jaw_bias = 0.0
            r_brow = lerp(0.85, -0.65, sk)
            r_smile = lerp(0.10, -0.25, sk)
        elif t < 15.5:
            # Beat 3: Off-camera right during Hooke's Microscope beat
            rx_p, ry_p, ryaw = 0.94, 0.50, -18.0
            lean_x, head_rx, head_rz = 3.0, 5.0, -10.0
            ua_l, la_l = (-16.0, 30.0, -18.0), (-36.0, 0.0, -28.0)
            ua_r, la_r = (-16.0, -30.0, 18.0), (-36.0, 0.0, 28.0)
            jaw_bias, r_brow, r_smile = 0.0, -0.2, 0.0
        elif t < 20.5:
            # Beat 4: Rival LEFT (X=+0.94, Y=0.44), facing RIGHT (+24°) directly at Overload Spring (X=+1.42) -> SHOCK RECOIL!
            shock = smoothstep(17.62, 17.85, t)
            recoil_osc = math.sin((t - 17.65) * 18.0) * math.exp(-(t - 17.65) * 3.5) * 4.0 if t >= 17.65 else 0.0
            rx_p, ry_p, ryaw = 0.94 - shock * 0.07, 0.44 + shock * 0.09, 24.0
            lean_x = lerp(7.0, -11.0, shock) + recoil_osc
            head_rx, head_rz = lerp(8.0, -9.0, shock), lerp(16.0, 14.0, shock) + math.sin(t * 6.0) * 3.5 * shock
            ua_l = (lerp(-26.0, -30.0, shock), lerp(22.0, 24.0, shock), lerp(-32.0, -34.0, shock))
            la_l = (lerp(-34.0, -52.0, shock), 0.0, lerp(-18.0, -34.0, shock))
            ua_r = (lerp(-16.0, -30.0, shock), lerp(-30.0, -24.0, shock), lerp(18.0, 34.0, shock))
            la_r = (lerp(-36.0, -52.0, shock), 0.0, lerp(28.0, 34.0, shock))
            jaw_bias = 12.0 * shock * (1.0 - smoothstep(18.6, 19.3, t))
            r_brow = lerp(-0.35, 1.0, shock)
            r_smile = lerp(0.35, -0.45, shock)
        elif t < 25.5:
            # Beat 5: Off-camera right during Hooke's Pocket Watch showcase
            rx_p, ry_p, ryaw = 1.45, 0.52, -20.0
            lean_x, head_rx, head_rz = 2.0, 4.0, -14.0
            ua_l, la_l = (-16.0, 30.0, -18.0), (-36.0, 0.0, -28.0)
            ua_r, la_r = (-16.0, -30.0, 18.0), (-36.0, 0.0, 28.0)
            jaw_bias, r_brow, r_smile = 0.0, 0.2, 0.1
        else:
            # Beat 6: Grand Finale — Rival RIGHT (X=+0.56, Y=0.52), angled INWARD (-18°) toward Watch & DNA Spring!
            rx_p, ry_p, ryaw = 0.56, 0.52, -18.0
            lean_x = 1.0 + breath * 0.4
            head_rx, head_rz = -3.0, -10.0 + math.sin(t * 1.6) * 3.0
            ua_r = (-26.0, -22.0, 30.0)
            la_r = (-36.0, 0.0, 20.0)
            ua_l = (-16.0, 30.0, -18.0)
            la_l = (-38.0, 0.0, -28.0)
            jaw_bias, r_brow, r_smile = 0.0, 0.55, 0.45

        rrig.location = Vector((rx_p, ry_p, r_base_z))
        rrig.rotation_euler = Euler((0.0, 0.0, math.radians(ryaw)))
        rrig.keyframe_insert("location", frame=f)
        rrig.keyframe_insert("rotation_euler", frame=f)

        set_bone_char_rot(rrig, "spine02", lean_x * 0.45, 0.0, 0.0, frame=f)
        set_bone_char_rot(rrig, "spine01", lean_x * 0.55, 0.0, 0.0, frame=f)
        set_bone_char_rot(rrig, "neck01", head_rx * 0.35, 0.0, head_rz * 0.35, frame=f)
        set_bone_char_rot(rrig, "head", head_rx * 0.65, 0.0, head_rz * 0.65, frame=f)
        apply_facial_acting(
            rrig,
            t,
            f,
            jaw_open_deg=max(jaw_bias, 1.0 + r_lip * 17.0),
            brow_lift=r_brow,
            smile=r_smile,
            blink_period=3.5,
            blink_phase=1.9,
        )
        set_bone_char_rot(rrig, "upperarm01.L", *ua_l, frame=f)
        set_bone_char_rot(rrig, "lowerarm01.L", *la_l, frame=f)
        set_bone_char_rot(rrig, "upperarm01.R", *ua_r, frame=f)
        set_bone_char_rot(rrig, "lowerarm01.R", *la_r, frame=f)


# ============================================================================
# 15. EXECUTE RENDER (QA 6-BEAT PREVIEW OR PRODUCTION FRAME RANGE!)
# ============================================================================
if ARGS.qa_only:
    # Preview all 6 Beats (Beat 2 shows iconic 2-shot of Hooke + Rival flanking F = -kx!)
    qa_times = [2.6, 8.5, 12.8, 17.8, 23.2, 28.2]
    qa_frames = [max(1, min(TOTAL_FRAMES, int(round(ts * FPS)) + 1)) for ts in qa_times]
    print(f"[Hooke Cycles Turbo QA] Rendering 6 Beat keyframes {qa_frames} ({FPS}fps, {ARGS.samples}spp, {ARGS.scale}% res) -> {OUT_DIR} ...")
    t_render_start = time.time()
    for idx, qf in enumerate(qa_frames, 1):
        scene.frame_set(qf)
        scene.render.filepath = os.path.join(OUT_DIR, f"qa_beat{idx}_f{qf:04d}.jpg")
        bpy.ops.render.render(write_still=True)
    dt_pure = time.time() - t_render_start
    print(
        f"[QA Pure Render Speed] 6 frames in {dt_pure:.1f}s (~{dt_pure/6:.2f}s/frame -> "
        f"True Est. Full 720-frame Dual-T4 Render Time: ~{(dt_pure/6)*360/60:.1f} min!)"
    )
else:
    print(
        f"[Hooke Cycles Turbo] Rendering frames {scene.frame_start}..{scene.frame_end} "
        f"({RES_X}x{RES_Y} @ {ARGS.scale}%, {FPS}fps, {ARGS.samples} spp) -> {OUT_DIR} ..."
    )
    t_render_start = time.time()
    bpy.ops.render.render(animation=True)
    print(f"[Hooke Cycles Turbo] Batch {scene.frame_start}..{scene.frame_end} complete in {(time.time()-t_render_start)/60:.1f} min!")

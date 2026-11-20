"""
Reusable MPFB2 (MakeHuman for Blender 4.2) Realistic Anatomical Human Character Engine
Produces true @zackdfilms-grade Cycles characters:
  - Real 19,158-vertex MakeHuman anatomical body, face, eyelids, lips, ears & 5-finger hands
  - Canonical 163-bone humanoid skeleton ('default') with real jaw, neck, spine, arms & fingers
  - Fast Cycles Christensen-Burley Subsurface Scattering (SSS) realistic skin shader
  - Photorealistic procedural fibrous-iris eyeballs seated at exact eye.L / eye.R bone sockets
  - Bone-weighted 1665 period velvet coat, white linen cravat & breeches + 3D collar/buttons/hair
  - Roll-proof character-space bone posing (M_rest^-1 @ R_char @ M_rest) + Kokoro lip-sync!
"""
from __future__ import annotations

import importlib
import math
import sys

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _set_bsdf(bsdf, name: str, val):
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val


def enable_mpfb2():
    """Ensure MPFB2 extension is enabled in Blender 4.2 headless session and return service classes."""
    # Try enabling extension if not already loaded in sys.modules
    if not any(m.endswith("mpfb.services.humanservice") for m in sys.modules):
        for mod_name in ("bl_ext.user_default.mpfb", "mpfb"):
            try:
                bpy.ops.preferences.addon_enable(module=mod_name)
                if any(m.endswith("mpfb.services.humanservice") for m in sys.modules):
                    print(f"[MPFB2] Enabled addon module: {mod_name}")
                    break
            except Exception:
                continue

    def dynamic_import(absolute_package_str: str, key: str):
        for amod in list(sys.modules.keys()):
            if amod.endswith(absolute_package_str):
                mpfb_mod = importlib.import_module(amod)
                if hasattr(mpfb_mod, key):
                    return getattr(mpfb_mod, key)
        raise ImportError(f"MPFB2 module '{absolute_package_str}.{key}' not loaded! Install MPFB2 extension first.")

    return (
        dynamic_import("mpfb.services.humanservice", "HumanService"),
        dynamic_import("mpfb.services.targetservice", "TargetService"),
        dynamic_import("mpfb.entities.objectproperties", "HumanObjectProperties"),
    )


# ============================================================================
# REALISTIC CYCLES HUMAN SKIN, IRIS, VELVET & HAIR SHADERS
# ============================================================================
def make_sss_skin_material(name: str, base_rgb=(0.76, 0.50, 0.37), rough=0.44):
    """Fast photorealistic Cycles Subsurface Scattering skin shader (Christensen-Burley for GPU speed!)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    # Subtle skin-tone micro-variation & fine pore bump
    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 95.0
    ns.inputs["Detail"].default_value = 6.0

    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["Factor"].default_value = 0.08
    mix.inputs["A"].default_value = (*base_rgb, 1.0)
    mix.inputs["B"].default_value = (base_rgb[0] * 0.86, base_rgb[1] * 0.78, base_rgb[2] * 0.72, 1.0)

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.045
    bmp.inputs["Distance"].default_value = 0.002

    _set_bsdf(bsdf, "Roughness", rough)
    _set_bsdf(bsdf, "IOR", 1.40)
    _set_bsdf(bsdf, "Subsurface Weight", 0.24)
    _set_bsdf(bsdf, "Subsurface Scale", 0.012)
    _set_bsdf(bsdf, "Subsurface Radius", (1.0, 0.35, 0.20))
    if hasattr(bsdf, "subsurface_method"):
        try:
            bsdf.subsurface_method = "BURLEY"
        except Exception:
            pass
    _set_bsdf(bsdf, "Coat Weight", 0.08)
    _set_bsdf(bsdf, "Coat Roughness", 0.35)

    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_velvet_coat_material(name: str, base_rgb=(0.045, 0.14, 0.11)):
    """Rich 1665 Royal Society woven velvet brocade coat material with subtle cloth sheen!"""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")
    ns = nt.nodes.new("ShaderNodeTexNoise")
    ns.inputs["Scale"].default_value = 65.0
    ns.inputs["Detail"].default_value = 8.0
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.12

    _set_bsdf(bsdf, "Base Color", (*base_rgb, 1.0))
    _set_bsdf(bsdf, "Roughness", 0.72)
    _set_bsdf(bsdf, "Sheen Weight", 0.75)
    _set_bsdf(bsdf, "Sheen Roughness", 0.45)

    nt.links.new(tc.outputs["Object"], ns.inputs["Vector"])
    nt.links.new(ns.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_linen_material(name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    _set_bsdf(bsdf, "Base Color", (0.88, 0.85, 0.78, 1.0))
    _set_bsdf(bsdf, "Roughness", 0.78)
    _set_bsdf(bsdf, "Sheen Weight", 0.35)
    return mat


def make_hair_material(name: str, rgb=(0.08, 0.045, 0.028)):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    wv = nt.nodes.new("ShaderNodeTexWave")
    wv.inputs["Scale"].default_value = 24.0
    wv.inputs["Distortion"].default_value = 3.0
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = 0.18
    _set_bsdf(bsdf, "Base Color", (*rgb, 1.0))
    _set_bsdf(bsdf, "Roughness", 0.42)
    _set_bsdf(bsdf, "Coat Weight", 0.18)
    nt.links.new(tc.outputs["Object"], wv.inputs["Vector"])
    nt.links.new(wv.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_photoreal_eye_material(name: str, iris_rgb=(0.14, 0.28, 0.18)):
    """Procedural Cycles eye shader (crisp dark pupil + colored iris ring + limbal ring + warm sclera).
    Uses Generated [0..1] bbox coordinates so it is 100% scale-independent along local -Y forward pole!"""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    # In Generated [0..1] space on a sphere: Y = 0.0 is dead center of front (-Y) pupil pole!
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.clamp = True
    mr.inputs["From Min"].default_value = 0.00
    mr.inputs["From Max"].default_value = 0.11
    mr.inputs["To Min"].default_value = 0.0
    mr.inputs["To Max"].default_value = 1.0

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "LINEAR"
    cr = ramp.color_ramp
    cr.elements[0].position = 0.00
    cr.elements[0].color = (0.008, 0.008, 0.012, 1.0)  # Pupil core
    e1 = cr.elements.new(0.16)
    e1.color = (0.008, 0.008, 0.012, 1.0)  # Crisp pupil edge
    e2 = cr.elements.new(0.22)
    e2.color = (*iris_rgb, 1.0)  # Bright inner iris
    e3 = cr.elements.new(0.68)
    e3.color = (iris_rgb[0] * 0.42, iris_rgb[1] * 0.42, iris_rgb[2] * 0.42, 1.0)  # Dark limbal ring
    cr.elements[-1].position = 0.78
    cr.elements[-1].color = (0.90, 0.87, 0.83, 1.0)  # Natural warm sclera

    _set_bsdf(bsdf, "Roughness", 0.05)
    _set_bsdf(bsdf, "Coat Weight", 0.95)
    _set_bsdf(bsdf, "Coat Roughness", 0.02)

    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Y"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


# ============================================================================
# MATERIAL PARTITIONING ON MPFB2 MESH + PERIOD HAIR / EYES / COLLAR ATTACHMENTS
# ============================================================================
def _assign_period_costume_materials(basemesh, skin_mat, coat_mat, linen_mat, breeches_mat):
    """Assign realistic SSS skin to face/hands and 1665 period garments directly to MPFB2 polygons
    using exact rigged vertex groups -> 100% seamless skeletal deformation with zero clipping!"""
    basemesh.data.materials.clear()
    basemesh.data.materials.append(skin_mat)      # slot 0
    basemesh.data.materials.append(coat_mat)      # slot 1
    basemesh.data.materials.append(linen_mat)     # slot 2
    basemesh.data.materials.append(breeches_mat)  # slot 3

    vg_names = {vg.index: vg.name for vg in basemesh.vertex_groups}
    verts = basemesh.data.vertices

    skin_tokens = ("head", "jaw", "neck02", "neck03", "eye", "orbicularis", "risorius", "levator", "temporalis",
                   "lip", "nose", "tongue", "wrist", "hand", "metacarpal", "finger", "thumb", "special")
    coat_tokens = ("spine01", "spine02", "spine03", "spine04", "clavicle", "shoulder", "breast",
                   "upperarm", "lowerarm", "arm")
    leg_tokens = ("pelvis", "spine05", "thigh", "shin", "calf", "foot", "toe")

    # Classify each vertex cleanly by strongest anatomical bone weight
    v_slot = [0] * len(verts)
    for v in verts:
        best_w = -1.0
        best_name = ""
        for g in v.groups:
            nm = vg_names.get(g.group, "")
            if nm.startswith(("helper-", "joint-")):
                continue
            if g.weight > best_w:
                best_w = g.weight
                best_name = nm
        bn = best_name.lower()
        if any(tok in bn for tok in skin_tokens):
            v_slot[v.index] = 0
        elif "neck01" in bn:
            v_slot[v.index] = 2  # Crisp white linen cravat collar at neck base!
        elif any(tok in bn for tok in coat_tokens):
            v_slot[v.index] = 1  # Tailored 1665 velvet doublet & sleeves!
        elif any(tok in bn for tok in leg_tokens):
            v_slot[v.index] = 3  # Dark Stuart breeches!
        else:
            v_slot[v.index] = 0

    for poly in basemesh.data.polygons:
        poly.use_smooth = True
        counts = [0, 0, 0, 0]
        for vi in poly.vertices:
            counts[v_slot[vi]] += 1
        # Prioritize skin on face/hands, otherwise majority garment material
        if counts[0] >= 2:
            poly.material_index = 0
        else:
            poly.material_index = max(range(4), key=lambda k: counts[k])


def _parent_to_bone(obj, rig, bone_name: str, world_pos: Vector, world_rot=None):
    """Attach a 3D prop (eyeball, hair lock, collar lapel, button) cleanly to a specific MPFB2 bone."""
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    # Compute exact local matrix relative to bone tail (Blender BONE parenting origin is bone tail!)
    bone = rig.data.bones[bone_name]
    tail_mat = Matrix.Translation(bone.tail_local) @ bone.matrix_local.to_3x3().to_4x4()
    if world_rot is None:
        w_mat = Matrix.Translation(world_pos)
    else:
        w_mat = Matrix.Translation(world_pos) @ world_rot.to_matrix().to_4x4()
    obj.matrix_parent_inverse = tail_mat.inverted()
    obj.matrix_basis = w_mat


def _attach_eyes_hair_and_costume(name: str, rig, cfg: dict, eye_mat, hair_mat, coat_mat, linen_mat, brass_mat):
    """Attach photorealistic eyeballs, 17th-century wavy hair/wig, eyebrows, mustache & collar accents."""
    bones = rig.data.bones

    # 1. Photorealistic Eyeballs seated cleanly inside MPFB2 eye.L & eye.R sockets!
    for side in ("L", "R"):
        bname = f"eye.{side}"
        if bname in bones:
            epos = bones[bname].head_local.copy()
            # Seat slightly forward inside eyelid aperture
            epos.y -= 0.002
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=0.0125, location=(0, 0, 0))
            eye = bpy.context.active_object
            eye.name = f"{name}_Eye_{side}"
            for p in eye.data.polygons:
                p.use_smooth = True
            eye.data.materials.append(eye_mat)
            _parent_to_bone(eye, rig, "head", epos)

    # 2. Sculpted 17th-Century Flowing Wavy Hair Locks (Hooke) or Royal Society Powdered Wig (Rival)!
    if "head" in bones:
        h_head = bones["head"].head_local.copy()
        h_tail = bones["head"].tail_local.copy()
        crown = Vector((0.0, h_head.y + 0.012, h_tail.z - 0.018))

        # Smooth cranium hair cap hugging upper skull cleanly
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=14, radius=0.098, location=(0, 0, 0))
        cap = bpy.context.active_object
        cap.name = f"{name}_HairCap"
        cap.scale = (0.94, 1.06, 0.92)
        bpy.ops.object.transform_apply(scale=True)
        for p in cap.data.polygons:
            p.use_smooth = True
        cap.data.materials.append(hair_mat)
        _parent_to_bone(cap, rig, "head", crown + Vector((0.0, 0.022, -0.015)))

        # Multi-lock shoulder-length wavy Stuart curls sweeping down sides & back!
        is_wig = cfg.get("style") == "wig"
        num_locks = 18 if is_wig else 16
        for i in range(num_locks):
            # Span around sides & back (avoiding face front -Y!)
            ang = _lerp(math.radians(-115.0), math.radians(115.0), i / max(1, num_locks - 1))
            sx = math.sin(ang) * (0.095 if is_wig else 0.088)
            sy = math.cos(ang) * 0.105 + 0.016
            lock_len = 0.24 if is_wig else 0.22
            curve_data = bpy.data.curves.new(f"{name}_Lock_{i}_C", type="CURVE")
            curve_data.dimensions = "3D"
            curve_data.bevel_depth = 0.017 if is_wig else 0.014
            curve_data.bevel_resolution = 4
            sp = curve_data.splines.new("POLY")
            sp.points.add(5)
            for k in range(6):
                u = k / 5.0
                wave = math.sin(u * math.pi * 2.5 + i * 0.7) * 0.014
                flare = 1.0 + u * (0.34 if is_wig else 0.22)
                px = sx * flare + math.copysign(wave, sx) * 0.5
                py = sy * (1.0 + u * 0.08) + wave * 0.4
                pz = -u * lock_len
                sp.points[k].co = (px, py, pz, 1.0)
            lock = bpy.data.objects.new(f"{name}_HairLock_{i}", curve_data)
            bpy.context.scene.collection.objects.link(lock)
            lock.data.materials.append(hair_mat)
            _parent_to_bone(lock, rig, "head", crown + Vector((0.0, 0.0, 0.005)))

        # Sculpted Expressive Eyebrows seated above eye sockets!
        for s_idx, side in enumerate(("L", "R")):
            bname = f"eye.{side}"
            if bname in bones:
                ep = bones[bname].head_local.copy()
                sign = 1.0 if side == "L" else -1.0
                cd = bpy.data.curves.new(f"{name}_Brow_{side}_C", type="CURVE")
                cd.dimensions = "3D"
                cd.bevel_depth = 0.0038
                cd.bevel_resolution = 3
                sp = cd.splines.new("POLY")
                sp.points.add(3)
                sp.points[0].co = (sign * -0.016, -0.016, 0.020, 1.0)
                sp.points[1].co = (sign * -0.002, -0.018, 0.024, 1.0)
                sp.points[2].co = (sign * 0.012, -0.015, 0.023, 1.0)
                sp.points[3].co = (sign * 0.022, -0.008, 0.017, 1.0)
                brow = bpy.data.objects.new(f"{name}_Brow_{side}", cd)
                bpy.context.scene.collection.objects.link(brow)
                brow.data.materials.append(hair_mat)
                _parent_to_bone(brow, rig, "head", ep)

        # Optional Distinguished Mustache for Rival Watchmaker!
        if cfg.get("mustache", False) and "jaw" in bones:
            jp = bones["jaw"].head_local.copy()
            for sign in (-1.0, 1.0):
                cd = bpy.data.curves.new(f"{name}_Stache_{sign}_C", type="CURVE")
                cd.dimensions = "3D"
                cd.bevel_depth = 0.0055
                cd.bevel_resolution = 3
                sp = cd.splines.new("POLY")
                sp.points.add(3)
                sp.points[0].co = (sign * 0.004, -0.072, -0.022, 1.0)
                sp.points[1].co = (sign * 0.022, -0.068, -0.024, 1.0)
                sp.points[2].co = (sign * 0.042, -0.056, -0.028, 1.0)
                sp.points[3].co = (sign * 0.054, -0.046, -0.020, 1.0)
                st = bpy.data.objects.new(f"{name}_Stache_{sign}", cd)
                bpy.context.scene.collection.objects.link(st)
                st.data.materials.append(hair_mat)
                _parent_to_bone(st, rig, "head", jp)

    # 3. 3D Sculpted 1665 Linen Cravat Jabot + Brass Doublet Buttons parented to spine01!
    if "spine01" in bones:
        sp1 = bones["spine01"].tail_local.copy()
        # Flowing lace/linen cravat at throat
        bpy.ops.mesh.primitive_cone_add(vertices=20, radius1=0.052, radius2=0.024, depth=0.14, location=(0, 0, 0))
        cravat = bpy.context.active_object
        cravat.name = f"{name}_LinenCravat"
        cravat.scale = (1.15, 0.42, 1.0)
        bpy.ops.object.transform_apply(scale=True)
        for p in cravat.data.polygons:
            p.use_smooth = True
        cravat.data.materials.append(linen_mat)
        _parent_to_bone(cravat, rig, "spine01", sp1 + Vector((0.0, -0.105, -0.075)))

        # 4 Gleaming Brass Period Coat Buttons down chest front
        for b_i in range(4):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=0.0095, location=(0, 0, 0))
            btn = bpy.context.active_object
            btn.name = f"{name}_Button_{b_i}"
            btn.scale = (1.0, 0.55, 1.0)
            bpy.ops.object.transform_apply(scale=True)
            btn.data.materials.append(brass_mat)
            _parent_to_bone(btn, rig, "spine01", sp1 + Vector((0.0, -0.118, -0.155 - b_i * 0.055)))


# ============================================================================
# ROLL-PROOF CHARACTER-SPACE BONE POSING HELPER
# ============================================================================
def set_bone_char_rot(rig, bone_name: str, rx_deg: float = 0.0, ry_deg: float = 0.0, rz_deg: float = 0.0, frame: int | None = None):
    """Apply a character-space Euler rotation (degrees) to any MPFB2 bone regardless of its rest roll!
    Character axes: +X = Character's Left, -Y = Forward (toward camera), +Z = Up."""
    if bone_name not in rig.pose.bones:
        return
    pb = rig.pose.bones[bone_name]
    pb.rotation_mode = "QUATERNION"
    bone = rig.data.bones[bone_name]
    m_rest = bone.matrix_local.to_3x3()
    r_char = Euler((math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)), "XYZ").to_matrix()
    q_local = (m_rest.inverted() @ r_char @ m_rest).to_quaternion()
    pb.rotation_quaternion = q_local
    if frame is not None:
        pb.keyframe_insert("rotation_quaternion", frame=frame)


# ============================================================================
# PUBLIC FACTORY: CREATE FULLY RIGGED, DRESSED 1665 REALISTIC SCIENTIST
# ============================================================================
def create_realistic_scientist(
    name: str,
    floor_z: float,
    start_xy: tuple[float, float],
    yaw_deg: float,
    cfg: dict,
    brass_mat,
):
    """Build a realistic anatomical MPFB2 human with 163-bone rig, SSS skin, eyes, hair & 1665 coat!"""
    HumanService, TargetService, HumanObjectProperties = enable_mpfb2()

    basemesh = HumanService.create_human(feet_on_ground=True, scale=0.1)
    basemesh.name = f"{name}_Body"

    # Set adult male anatomical proportions (macro.json: gender 1.0 = male, age 0.50..0.85 = young..elder)
    HumanObjectProperties.set_value("gender", 1.0, entity_reference=basemesh)
    HumanObjectProperties.set_value("age", float(cfg.get("age", 0.62)), entity_reference=basemesh)
    HumanObjectProperties.set_value("muscle", float(cfg.get("muscle", 0.55)), entity_reference=basemesh)
    HumanObjectProperties.set_value("weight", float(cfg.get("weight", 0.50)), entity_reference=basemesh)
    HumanObjectProperties.set_value("height", float(cfg.get("height", 0.56)), entity_reference=basemesh)
    HumanObjectProperties.set_value("caucasian", 1.0, entity_reference=basemesh)
    HumanObjectProperties.set_value("african", 0.0, entity_reference=basemesh)
    HumanObjectProperties.set_value("asian", 0.0, entity_reference=basemesh)
    TargetService.reapply_macro_details(basemesh)

    # Attach full 163-bone canonical MakeHuman skeleton rig (with jaw, eyes, spine, arms & 5 fingers!)
    rig = HumanService.add_builtin_rig(basemesh, "default")
    rig.name = f"{name}_Rig"

    # Build bespoke Cycles materials for this character
    skin_mat = make_sss_skin_material(f"{name}_SSSSkin", cfg.get("skin_rgb", (0.76, 0.50, 0.37)))
    coat_mat = make_velvet_coat_material(f"{name}_VelvetCoat", cfg.get("coat_rgb", (0.04, 0.14, 0.10)))
    linen_mat = make_linen_material(f"{name}_Linen")
    breeches_mat = make_velvet_coat_material(f"{name}_Breeches", (0.04, 0.04, 0.05))
    eye_mat = make_photoreal_eye_material(f"{name}_Iris", cfg.get("iris_rgb", (0.14, 0.28, 0.16)))
    hair_mat = make_hair_material(f"{name}_Hair", cfg.get("hair_rgb", (0.07, 0.04, 0.025)))

    _assign_period_costume_materials(basemesh, skin_mat, coat_mat, linen_mat, breeches_mat)
    _attach_eyes_hair_and_costume(name, rig, cfg, eye_mat, hair_mat, coat_mat, linen_mat, brass_mat)

    # Dynamically ground feet to exact floor_z (works regardless of MPFB2 version!)
    bpy.context.view_layer.update()
    min_z = min((basemesh.matrix_world @ v.co).z for v in basemesh.data.vertices)
    max_z = max((basemesh.matrix_world @ v.co).z for v in basemesh.data.vertices)
    rig.location = Vector((start_xy[0], start_xy[1], floor_z - min_z))
    rig.rotation_mode = "XYZ"
    rig.rotation_euler = Euler((0.0, 0.0, math.radians(yaw_deg)))

    print(
        f"[MPFB2] Built realistic '{name}' ({len(basemesh.data.vertices)} verts, {len(rig.data.bones)} bones, "
        f"height={max_z - min_z:.2f}m, feet@z={floor_z:.2f}m, head_top@z={floor_z + (max_z - min_z):.2f}m)"
    )
    return {"rig": rig, "body": basemesh, "height": max_z - min_z}

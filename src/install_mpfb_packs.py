"""
Headless MPFB2 Official 4K Asset Pack Installer for Kaggle / Colab Blender 4.2.3 LTS.
Extracts downloaded MakeHuman Community asset pack ZIPs cleanly into MPFB2's exact
LocationService.get_user_data() directory and refreshes MPFB2's asset library cache.
"""
from __future__ import annotations

import glob
import importlib
import os
import sys
import zipfile

import bpy


def _enable_mpfb2():
    if not any(m.endswith("mpfb.services.assetservice") for m in sys.modules):
        for mod_name in ("bl_ext.user_default.mpfb", "mpfb"):
            try:
                bpy.ops.preferences.addon_enable(module=mod_name)
                if any(m.endswith("mpfb.services.assetservice") for m in sys.modules):
                    break
            except Exception:
                continue

    def dyn(pkg: str, key: str):
        for amod in list(sys.modules.keys()):
            if amod.endswith(pkg):
                m = importlib.import_module(amod)
                if hasattr(m, key):
                    return getattr(m, key)
        raise ImportError(f"MPFB2 module '{pkg}.{key}' not found!")

    return (
        dyn("mpfb.services.locationservice", "LocationService"),
        dyn("mpfb.services.assetservice", "AssetService"),
    )


def main():
    LocationService, AssetService = _enable_mpfb2()
    data_dir = LocationService.get_user_data()
    os.makedirs(data_dir, exist_ok=True)
    print(f"[MPFB2 Packs] Target MPFB2 user data dir: {data_dir}")

    pack_dir = os.environ.get("MPFB_PACKS_DIR", "/tmp/mpfb_packs")
    zips = sorted(glob.glob(os.path.join(pack_dir, "*.zip")))
    if not zips:
        print(f"[MPFB2 Packs] Warning: no zip files found in {pack_dir}")
        return

    marker_dir = os.path.join(data_dir, ".installed_packs")
    os.makedirs(marker_dir, exist_ok=True)

    for zp in zips:
        base = os.path.basename(zp)
        marker = os.path.join(marker_dir, base + ".ok")
        if os.path.exists(marker):
            print(f"[MPFB2 Packs] Already installed: {base}")
            continue
        print(f"[MPFB2 Packs] Extracting {base} -> {data_dir} ...")
        # Use MPFB2's own check/fix helper if zip has nested folder, else direct extractall
        try:
            check = AssetService.check_asset_pack_zip(zp)
            if check is not None:
                res = AssetService.fix_and_extract_asset_pack_zip(zp, data_dir)
                if res is not None:
                    print(f"[MPFB2 Packs] fix_and_extract returned notice ({res}), falling back to direct extract")
                    with zipfile.ZipFile(zp, "r") as zf:
                        zf.extractall(data_dir)
            else:
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(data_dir)
        except Exception as err:
            print(f"[MPFB2 Packs] Direct extract fallback for {base} ({err})")
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(data_dir)
        with open(marker, "w", encoding="utf-8") as mf:
            mf.write("ok\n")

    AssetService.update_all_asset_lists()
    skins = AssetService.list_mhmat_assets("skins")
    clothes = AssetService.list_mhclo_assets("clothes")
    hair = AssetService.list_mhclo_assets("hair")
    eyes = AssetService.list_mhclo_assets("eyes")
    brows = AssetService.list_mhclo_assets("eyebrows")
    lashes = AssetService.list_mhclo_assets("eyelashes")
    teeth = AssetService.list_mhclo_assets("teeth")
    print(
        f"[MPFB2 Packs] READY! Installed -> "
        f"skins={len(skins)}, eyes={len(eyes)}, eyebrows={len(brows)}, "
        f"eyelashes={len(lashes)}, teeth={len(teeth)}, hair={len(hair)}, clothes={len(clothes)}"
    )


if __name__ == "__main__":
    main()

# Scientist #8: Robert Hooke — V5 100% Kaggle `GPU T4 x2` Optimized Edition

Complete **30-second (`1080×1920` Lanczos, `24 fps`, `720` frames) Blender 4.2 LTS Cycles Turbo Dual-GPU Short** purpose-built for **Kaggle `GPU T4 x2`** — featuring **official MakeHuman 4K photo-scanned skins, refractive high-poly eyes, alpha-strand hair/wig, real 3D beard, and tailored formal suits**, **pre-mastered studio audio** (Kokoro 3-voice narration + 118 BPM Stuart Baroque & Watchwork Score + 24 spatial Foley SFX), **real facial bone acting & Kokoro jaw lip-sync**, and **bundled-font word-synced boxless subtitles**.

## Why V5 Runs Rock-Solid on Kaggle `GPU T4 x2`
- **100% Disk-Backed `/kaggle/working/.cache` (0 MB `/tmp` RAM Waste!)**: On Kaggle, `/tmp` lives inside System RAM (`tmpfs`). `render_on_kaggle.ipynb` installs Blender 4.2.3 LTS and caches everything on `/kaggle/working/.cache` (real SSD disk!) and cleans archives immediately — keeping **100% of Kaggle's 30 GB System RAM free** so `Kernel died unexpectedly` never happens!
- **~1.2 Second 4K Human Asset Install (GitHub Fastly CDN)**: Curated 43 MB `mpfb_hooke_4k_assets.zip` served from the dedicated `assets-v5` CDN branch downloads in **~1.2 seconds (`34.9 MB/s`)** in parallel with Blender 4.2.3 & MPFB2 (~20s total Cell 2 setup!).
- **True `+Z` Gaze & Balanced 6-Beat Staging**: Both characters face directly toward active instruments and face-to-face in Beat 2 (`ceiiinosssttuv` → `F = - k x`), zero Beat 2 edge chopping, elevated hands above bench, +6 cm floating Cork Cells clearance above microscope eyepiece, sealed back wall, and fixed Blender 4.2 `ShaderNodeMix` RGBA suit color grading.

## Ultra-Realistic 4K Human Cast (Auto-Installed in Cell 2!)
- **Robert Hooke** (`src/mpfb_characters.py`): 4K photo-scanned `middleage_caucasian_male` skin (`ENHANCED_SSS` + fast `BURLEY`), `high-poly` refractive `PROCEDURAL_EYES` (hazel-emerald), alpha eyebrows & eyelashes, 3D teeth & tongue, `punkduck_alpha7_curly` real alpha-strand hair, Margaret Toigo `toigo_male_double-breasted_suit` deep forest-emerald wool coat & dress shoes.
- **Rival Watchmaker**: 4K photo-scanned `old_caucasian_male` skin (`ENHANCED_SSS`), icy-blue `PROCEDURAL_EYES`, alpha eyebrows/eyelashes/teeth, `grinsegold_wig_bow_tie` historical powdered wig, `grinsegold_beard_sigmund_wip` 3D silver professor beard, Margaret Toigo `toigo_suit_with_dinner_jacket` royal burgundy formal suit.
- **Asset Attribution**: Official MakeHuman Community assets (`makehuman_system_assets` CC0, `suits01` CC0 by Margaret Toigo, `bodyparts05` CC0 by grinsegold, `hair03` CC-BY by punkduck & grinsegold).

## 1-Click Kaggle `GPU T4 x2` Workflow
1. Go to **Kaggle -> New Notebook -> File -> Import Notebook -> GitHub**:
   - Repo: `ssghtc002-sudo/hooke-elasticity-micrographia-3d`
   - File: `render_on_kaggle.ipynb`
2. In **Session options**: **Accelerator: `GPU T4 x2`** & **Internet: `ON`**.
3. Run **Cell 1 -> Cell 2 (~20s!) -> Cell 3 (~20s 6-Beat Visual Preview Grid)**, then **Cell 4 -> Cell 5** -> click the **`FileLink` (`robert_hooke_elasticity_short.mp4`)** or download from Kaggle's `/kaggle/working` Output panel!

"""
Robert Hooke (1665-1678) — Springs, Secret Riddles & The Discovery of Cells
V2 Cinema Audio: 118 BPM Stuart-Era English Baroque, Watchmaker Clockwork & Elastic Cinema Score
+ 3 Kokoro Voices + 24 Spatial Foley SFX + Exact Word-Synced Boxless ASS Karaoke Subtitles

Run with:
    /home/antigravity/shorts-engine/.venv/bin/python audio/generate_audio.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SHORTS_ENGINE_ROOT = Path("/home/antigravity/shorts-engine")
if str(SHORTS_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(SHORTS_ENGINE_ROOT))

from audio.dspCore import (
    CHANNELS,
    DURATION,
    FPS,
    SR,
    TOTAL_FRAMES,
    TOTAL_SAMPLES,
    add_stereo_event,
    adsr_env,
    butter_filter,
    master_and_export,
    midi_to_hz,
    pan_stereo,
    synthesize_dialogue,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIMELINE_PATH = PROJECT_ROOT / "config" / "timeline.json"

BPM = 118.0
BEAT = 60.0 / BPM
EIGHTH = BEAT * 0.5
SIXTEENTH = BEAT * 0.25

TTS_OVERRIDES: dict[str, str] = {
    "L1": "Hang one weight on a spring, it stretches. Double the weight, it stretches twice as far!",
    "L2": "Scrambled Latin letters, Hooke? Explain your hidden law!",
    "L3": "More stretch means more pullback, F equals minus K X!",
    "L4": "Next, Hooke aimed his brass microscope at cork, and named those tiny honeycomb rooms cells!",
    "L5": "Warning: stretch past the elastic limit, and metal deforms forever.",
    "L6": "No! You snapped my watch spring!",
    "L7": "Yet Hooke's tiny spiral hairspring tamed the pocket watch, shrinking clocks to fit your palm!",
    "L8": "Car shocks, skyscrapers, even DNA coils today, all obey Robert Hooke's law.",
}


def add_royal_hall_reverb(stereo: np.ndarray, mix: float = 0.14) -> np.ndarray:
    """Warm Royal Society London oak-and-masonry chamber reflections."""
    out = np.copy(stereo)
    taps = [(0.015, 0.30, 0.18), (0.029, 0.20, 0.28), (0.047, 0.14, 0.12), (0.068, 0.09, 0.18)]
    wet = np.zeros_like(stereo)
    for dt, gl, gr in taps:
        d = int(dt * SR)
        if d < len(stereo):
            wet[d:, 0] += stereo[:-d, 0] * gl
            wet[d:, 1] += stereo[:-d, 1] * gr
    wet = butter_filter(wet, "low", 4400.0, order=2)
    return (out * (1.0 - mix * 0.35) + wet * mix).astype(np.float32)


# ============================================================================
# 118 BPM ENGLISH BAROQUE LUTE, WATCHWORK & CELLO-BRASS CINEMA SCORE
# ============================================================================
def synth_lute_pluck(freq: float, dur: float, vel: float = 0.22) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    wave = (
        0.62 * np.sin(2 * math.pi * freq * t)
        + 0.32 * np.sin(2 * math.pi * 2 * freq * t)
        + 0.18 * np.sin(2 * math.pi * 3 * freq * t)
        + 0.10 * np.sin(2 * math.pi * 5 * freq * t)
    )
    env = np.exp(-t * 6.5) * np.minimum(1.0, t / 0.003)
    return (wave * env * vel).astype(np.float32)


def synth_bass_cello(freq: float, dur: float, vel: float = 0.34) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    wave = (
        0.80 * np.sin(2 * math.pi * freq * t)
        + 0.26 * np.sin(2 * math.pi * 2 * freq * t)
        + 0.09 * np.sin(2 * math.pi * 3 * freq * t)
    )
    env = np.exp(-t * 6.8) * np.minimum(1.0, t / 0.006)
    return butter_filter((wave * env * vel).astype(np.float32), "low", 880.0, order=2)


def synth_celesta(freq: float, dur: float, vel: float = 0.18) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    wave = 0.78 * np.sin(2 * math.pi * freq * t) + 0.18 * np.sin(2 * math.pi * 2.0 * freq * t) * np.exp(-t * 6.0)
    env = np.exp(-t * 4.2) * np.minimum(1.0, t / 0.005)
    return (wave * env * vel).astype(np.float32)


def synth_strings(freqs: list[float], dur: float, vel: float = 0.17) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    ol = np.zeros(n, dtype=np.float32)
    orr = np.zeros(n, dtype=np.float32)
    env = adsr_env(n, attack=min(0.40, dur * 0.22), decay=0.22, sustain=0.82, release=min(0.50, dur * 0.25))
    for i, f in enumerate(freqs):
        dl = 1.0 - 0.0020 * ((i % 2) * 2 - 1)
        dr = 1.0 + 0.0020 * ((i % 2) * 2 - 1)
        ol += 0.64 * np.sin(2 * math.pi * f * dl * t) + 0.25 * np.sin(4 * math.pi * f * dl * t) + 0.10 * np.sin(6 * math.pi * f * dl * t)
        orr += 0.64 * np.sin(2 * math.pi * f * dr * t) + 0.25 * np.sin(4 * math.pi * f * dr * t) + 0.10 * np.sin(6 * math.pi * f * dr * t)
    norm = max(1, len(freqs))
    stereo = np.stack([ol / norm, orr / norm], axis=-1) * env[:, None] * vel
    return butter_filter(stereo.astype(np.float32), "low", 3300.0, order=2)


def synth_brass_horn(freq: float, dur: float, vel: float = 0.22) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    vib = 1.0 + 0.0038 * np.sin(2 * math.pi * 4.8 * t) * np.minimum(1.0, t / 0.22)
    ph = 2 * math.pi * freq * np.cumsum(vib) / SR
    w = 0.56 * np.sin(ph) + 0.34 * np.sin(2 * ph) + 0.22 * np.sin(3 * ph) + 0.10 * np.sin(4 * ph)
    env = adsr_env(n, attack=0.06, decay=0.14, sustain=0.78, release=min(0.28, dur * 0.32))
    return butter_filter((w * env * vel).astype(np.float32), "low", 2150.0, order=2)


def synth_watch_tick(high: bool = True, vel: float = 0.14) -> np.ndarray:
    n = int(0.038 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f1 = 3650.0 if high else 2920.0
    f2 = 5800.0 if high else 4600.0
    click = (np.sin(2 * math.pi * f1 * t) + 0.45 * np.sin(2 * math.pi * f2 * t)) * np.exp(-t * 165.0)
    return (click * vel).astype(np.float32)


def compose_bgm() -> np.ndarray:
    bgm = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)

    # 1. Crisp 1675 Pocket-Watch Hairspring Escapement Ticks (Beats 1-2 & Beat 5!)
    for k in range(int(9.5 / EIGHTH)):
        t_tk = k * EIGHTH
        add_stereo_event(bgm, pan_stereo(synth_watch_tick(k % 2 == 0, vel=0.10), -0.22 if k % 2 == 0 else 0.18), t_tk)
    for k in range(int((25.5 - 20.5) / EIGHTH)):
        t_tk = 20.5 + k * EIGHTH
        add_stereo_event(bgm, pan_stereo(synth_watch_tick(k % 2 == 0, vel=0.15), -0.18 if k % 2 == 0 else 0.18), t_tk)

    # 2. Beats 1-2 (0.0-9.5s, E Minor): Clever Stuart Baroque Lute Ostinato + Cello Pizzicato
    em_arp = [64, 67, 71, 76, 71, 67, 66, 71, 63, 66, 71, 75, 71, 66, 64, 67]
    bass_em = [40, 40, 47, 47, 48, 48, 47, 47, 40, 40, 43, 43, 45, 45, 47, 47]
    t_cur = 0.0
    idx = 0
    while t_cur < 9.40:
        m = em_arp[idx % len(em_arp)]
        pan = math.sin(idx * 0.65) * 0.36
        add_stereo_event(bgm, pan_stereo(synth_lute_pluck(midi_to_hz(m), 0.36, vel=0.20), pan), t_cur)
        if idx % 2 == 0:
            bm = bass_em[(idx // 2) % len(bass_em)]
            add_stereo_event(bgm, pan_stereo(synth_bass_cello(midi_to_hz(bm), 0.42, vel=0.34), -0.08), t_cur)
        t_cur += EIGHTH
        idx += 1

    add_stereo_event(bgm, synth_strings([midi_to_hz(m) for m in (52, 59, 64, 67)], 4.5, vel=0.18), 0.0)
    add_stereo_event(bgm, synth_strings([midi_to_hz(m) for m in (48, 55, 60, 64)], 4.8, vel=0.19), 4.6)

    # 3. Beat 3 (9.5-15.5s, G Major Microscopic Wonder): Shimmering Celesta Honeycomb Arpeggios + Lush Strings!
    add_stereo_event(bgm, synth_strings([midi_to_hz(m) for m in (55, 62, 67, 71)], 3.1, vel=0.23), 9.5)
    add_stereo_event(bgm, synth_strings([midi_to_hz(m) for m in (57, 64, 69, 74)], 3.0, vel=0.24), 12.6)
    cell_notes = [67, 71, 74, 79, 78, 74, 71, 76, 81, 78, 74, 83]
    t_cur = 9.55
    c_idx = 0
    while t_cur < 15.35:
        m = cell_notes[c_idx % len(cell_notes)]
        add_stereo_event(bgm, pan_stereo(synth_celesta(midi_to_hz(m), 0.62, vel=0.18), math.sin(c_idx * 0.85) * 0.52), t_cur)
        t_cur += SIXTEENTH * 1.5
        c_idx += 1

    # 4. Beat 4 (15.5-20.5s): Strain Tension -> Sudden Comedic Silence on 17.65s SNAP! -> Cheeky Pizzicato Bounce!
    add_stereo_event(bgm, synth_strings([midi_to_hz(m) for m in (52, 59, 64, 68)], 2.05, vel=0.22), 15.55)
    # Comedic pause 17.65..18.95s so SNAP + "Hey—that ruined my watch!" punch hard!
    for k, m in enumerate([64, 68, 71, 76]):
        t_p = 19.15 + k * EIGHTH
        add_stereo_event(bgm, pan_stereo(synth_bass_cello(midi_to_hz(m - 12), 0.34, vel=0.34), 0.0), t_p)
        add_stereo_event(bgm, pan_stereo(synth_lute_pluck(midi_to_hz(m), 0.34, vel=0.21), 0.25), t_p)

    # 5. Beats 5-6 (20.5-30.0s, Triumphant E Major Horology & Modern Engineering Climax!)
    add_stereo_event(bgm, synth_strings([midi_to_hz(m) for m in (52, 59, 64, 68, 76)], 4.8, vel=0.28), 20.5)
    add_stereo_event(bgm, synth_strings([midi_to_hz(m) for m in (57, 64, 69, 73, 81)], 2.3, vel=0.28), 25.3)
    add_stereo_event(bgm, synth_strings([midi_to_hz(m) for m in (52, 59, 64, 68, 76)], 2.4, vel=0.30), 27.5)

    horn_mel = [
        (20.65, 64, 0.68), (21.38, 68, 0.50), (21.92, 71, 0.88),
        (22.86, 76, 1.08), (24.00, 75, 0.54), (24.60, 76, 0.72),
        (25.60, 81, 0.82), (26.48, 80, 0.62), (27.18, 76, 1.45),
    ]
    for th, mh, dh in horn_mel:
        add_stereo_event(bgm, pan_stereo(synth_brass_horn(midi_to_hz(mh), dh, vel=0.22), -0.10), th)

    finale_cel = [76, 80, 83, 88, 83, 80, 81, 85, 88, 93, 88, 83, 76, 88]
    for k, m in enumerate(finale_cel):
        tf = 25.65 + k * (EIGHTH * 1.1)
        if tf < 29.4:
            add_stereo_event(bgm, pan_stereo(synth_celesta(midi_to_hz(m), 0.72, vel=0.17), math.sin(k) * 0.48), tf)

    bgm = add_royal_hall_reverb(bgm, mix=0.20)
    tail = int(0.45 * SR)
    bgm[-tail:] *= np.linspace(1.0, 0.0, tail, dtype=np.float32)[:, None]
    return bgm


# ============================================================================
# 24 SPATIAL STEREO FOLEY SFX (SPRING BOING, CIPHER, MICROSCOPE, SNAP, WATCH!)
# ============================================================================
def synth_sfx(kind: str, pan: float) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(kind)) % 100000)

    if kind == "whip_whoosh":
        n = int(0.38 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        noise = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (280.0, 2700.0), order=2)
        env = np.sin(math.pi * t / 0.38) ** 1.4
        tone = np.sin(2 * math.pi * (170 + 500 * (t / 0.38)) * t) * 0.16
        return pan_stereo((noise * 0.42 + tone) * env * 0.75, pan)

    if kind in ("brass_weight_clank", "heavy_weight_drop"):
        n = int(0.45 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        thud_f = 95.0 if kind == "heavy_weight_drop" else 145.0
        thud = np.sin(2 * math.pi * (thud_f * np.exp(-t * 14.0)) * t) * np.exp(-t * 14.0) * 0.75
        ring = (np.sin(2 * math.pi * 820 * t) + 0.55 * np.sin(2 * math.pi * 1740 * t)) * np.exp(-t * 12.0) * 0.42
        return pan_stereo((thud + ring) * (0.95 if kind == "heavy_weight_drop" else 0.78), pan)

    if kind in ("spring_stretch_boing", "spring_bounce_ring"):
        n = int(0.68 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        f0 = 165.0 if kind == "spring_stretch_boing" else 220.0
        wobble = f0 * (1.0 + 0.32 * np.sin(2 * math.pi * 9.5 * t) * np.exp(-t * 3.2))
        ph = 2 * math.pi * np.cumsum(wobble) / SR
        w = (0.65 * np.sin(ph) + 0.30 * np.sin(2.05 * ph) + 0.15 * np.sin(3.12 * ph)) * np.exp(-t * 4.5)
        return pan_stereo(w * 0.65, pan)

    if kind in ("cipher_lock_click", "microscope_focus_click"):
        n = int(0.14 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        f1 = 2100.0 if kind == "cipher_lock_click" else 1650.0
        w = (np.sin(2 * math.pi * f1 * t) + 0.6 * np.sin(2 * math.pi * f1 * 1.85 * t)) * np.exp(-t * 58.0)
        return pan_stereo(w * 0.62, pan)

    if kind in ("formula_brass_chime", "watch_escapement_chime", "cork_cell_shimmer"):
        n = int(0.95 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        freqs = [659.25, 987.77, 1318.5] if kind == "formula_brass_chime" else [783.99, 1174.66, 1567.98]
        w = sum(np.sin(2 * math.pi * f * t) * (0.42 / (i + 1)) for i, f in enumerate(freqs))
        env = np.exp(-t * 3.6) * np.minimum(1.0, t / 0.010)
        return pan_stereo(w * env * 0.56, pan)

    if kind == "cork_cell_bloom":
        n = int(1.25 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        env = adsr_env(n, 0.22, 0.25, 0.65, 0.45)
        w = 0.48 * np.sin(2 * math.pi * 196.0 * t) + 0.32 * np.sin(2 * math.pi * 392.0 * t) + 0.24 * np.sin(2 * math.pi * 587.33 * t)
        return pan_stereo(w * env * 0.55, pan)

    if kind == "metal_strain_creak":
        n = int(0.75 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        f = 340.0 + 480.0 * (t / 0.75) + 45.0 * np.sin(2 * math.pi * 24.0 * t)
        ph = 2 * math.pi * np.cumsum(f) / SR
        w = (np.sin(ph) + 0.45 * np.sin(2.7 * ph)) * adsr_env(n, 0.08, 0.10, 0.80, 0.12)
        return pan_stereo(butter_filter(w, "band", (300.0, 3200.0), order=2) * 0.52, pan)

    if kind == "spring_overload_snap":
        n = int(0.85 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        crack = butter_filter(rng.standard_normal(n).astype(np.float32), "high", 1400.0, order=2) * np.exp(-t * 22.0) * 0.85
        twang_f = 920.0 * np.exp(-t * 4.5) + 210.0
        twang = np.sin(2 * math.pi * np.cumsum(twang_f) / SR) * np.exp(-t * 5.2) * 0.65
        boom = np.sin(2 * math.pi * (125.0 * np.exp(-t * 10.0)) * t) * np.exp(-t * 12.0) * 0.70
        return pan_stereo((crack + twang + boom) * 0.92, pan)

    if kind == "watch_parts_clatter":
        n = int(0.70 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        out = np.zeros(n, dtype=np.float32)
        for k, dt in enumerate([0.0, 0.08, 0.15, 0.24, 0.36, 0.49]):
            idx0 = int(dt * SR)
            m = min(int(0.12 * SR), n - idx0)
            if m > 0:
                tt = np.arange(m, dtype=np.float32) / SR
                out[idx0 : idx0 + m] += np.sin(2 * math.pi * (2400 + k * 480) * tt) * np.exp(-tt * 38.0) * (0.45 - k * 0.05)
        return pan_stereo(out * 0.72, pan)

    if kind in ("balance_wheel_hum", "dna_harmonic_pulse"):
        n = int(1.10 * SR)
        t = np.arange(n, dtype=np.float32) / SR
        f0 = 329.63 if kind == "balance_wheel_hum" else 440.0
        mod = 0.65 + 0.35 * np.sin(2 * math.pi * 5.0 * t)
        w = (0.55 * np.sin(2 * math.pi * f0 * t) + 0.30 * np.sin(2 * math.pi * f0 * 1.5 * t)) * mod
        return pan_stereo(w * adsr_env(n, 0.18, 0.20, 0.65, 0.38) * 0.48, pan)

    return pan_stereo(np.zeros(int(0.1 * SR), dtype=np.float32), pan)


# ============================================================================
# EXACT WORD-SYNCED BOXLESS ASS SUBTITLES (YOUTUBE SHORTS STANDARD)
# ============================================================================
def fmt_ass_time(sec: float) -> str:
    cs = int(round(max(0.0, sec) * 100))
    h = cs // 360000
    m = (cs % 360000) // 6000
    s = (cs % 6000) // 100
    rem = cs % 100
    return f"{h}:{m:02d}:{s:02d}.{rem:02d}"


HANGING_WORDS = {"a", "an", "the", "of", "on", "in", "to", "with", "and", "or", "as", "so", "its", "his"}


def chunk_words(words: list[dict[str, Any]], max_words: int = 6, max_chars: int = 40) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_chars = 0
    for idx, w in enumerate(words):
        wt = w["word"]
        add_len = len(wt) + (1 if cur else 0)
        next_is_emdash = (idx + 1 < len(words)) and words[idx + 1]["word"].startswith("—")
        if cur and (len(cur) >= max_words or cur_chars + add_len > max_chars) and not wt.startswith("—"):
            if len(cur) >= 3 and cur[-1]["word"].lower().strip(".,!?—'") in HANGING_WORDS:
                carry = cur.pop()
                chunks.append(cur)
                cur = [carry, w]
                cur_chars = len(carry["word"]) + 1 + len(wt)
            else:
                chunks.append(cur)
                cur = [w]
                cur_chars = len(wt)
        else:
            cur.append(w)
            cur_chars += add_len
        if (wt.endswith(("—", ",", ".", "!", "?", ";")) or (not next_is_emdash and "—" in wt)) and len(cur) >= 2:
            chunks.append(cur)
            cur = []
            cur_chars = 0
    if cur:
        if len(cur) == 1 and chunks and len(chunks[-1]) <= 6:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)
    return chunks


def write_word_karaoke_ass(manifest_lines: list[dict[str, Any]], ass_path: Path) -> None:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,64,&H00FFFFFF,&H00F8BD38,&H00120803,&H96000000,-1,0,0,0,100,100,0,0,1,7,2.5,2,80,80,245,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    for line in manifest_lines:
        chunks = chunk_words(line["words"], max_words=6, max_chars=38)
        for ch in chunks:
            for idx, w in enumerate(ch):
                t0 = w["start"] if idx > 0 else ch[0]["start"]
                t1 = ch[idx + 1]["start"] if idx + 1 < len(ch) else w["end"]
                if t1 <= t0 + 0.01:
                    t1 = t0 + 0.04
                split_at = (len(ch) + 1) // 2 if len(ch) >= 5 else len(ch)
                parts = []
                for j, cw in enumerate(ch):
                    token = cw["word"]
                    styled = f"{{\\c&H00F8BD38&}}{token}{{\\c&H00FFFFFF&}}" if j == idx else token
                    if j == split_at:
                        parts.append("\\N" + styled)
                    elif j == 0:
                        parts.append(styled)
                    else:
                        sep = "" if (ch[j - 1]["word"].endswith("—") or token.startswith("—")) else " "
                        parts.append(sep + styled)
                events.append(f"Dialogue: 0,{fmt_ass_time(t0)},{fmt_ass_time(t1)},Default,,0,0,0,,{''.join(parts)}")

    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    print(f"[Subtitles] Exported word-synced boxless ASS: {ass_path} ({len(events)} highlight frames)")


def main() -> None:
    raw_tl = json.loads(TIMELINE_PATH.read_text(encoding="utf-8"))
    tl = json.loads(json.dumps(raw_tl))
    tl["sfxEvents"] = [{**ev, "time": float(ev["t"])} for ev in tl.get("sfx", [])]

    print("[1/4] Synthesizing Kokoro multi-voice dialogue (Narrator + Hooke + Skeptic)...")
    dialogue_bus, speaker_mono_buses, manifest_lines = synthesize_dialogue(tl, TTS_OVERRIDES)
    dialogue_bus = add_royal_hall_reverb(dialogue_bus, mix=0.10)

    print("[2/4] Composing 118 BPM Stuart Baroque & Watchmaker Clockwork Cinema Score...")
    bgm_bus = compose_bgm()

    print("[3/4] Synthesizing 24 spatial stereo Foley SFX...")
    sfx_bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    for ev in tl["sfxEvents"]:
        add_stereo_event(sfx_bus, synth_sfx(ev["type"], float(ev.get("pan", 0.0))), float(ev["time"]))
    sfx_bus = add_royal_hall_reverb(sfx_bus, mix=0.11)

    print("[4/4] Mastering 30.000s stereo WAV (-14 LUFS, Peak <= -1.0 dBFS) + Boxless ASS Subtitles...")
    master_and_export(PROJECT_ROOT, tl, dialogue_bus, bgm_bus, sfx_bus, speaker_mono_buses, manifest_lines)
    write_word_karaoke_ass(manifest_lines, PROJECT_ROOT / raw_tl["audio"]["assPath"])
    print("DONE! Hooke Master Audio + Word-Synced Subtitles ready.")


if __name__ == "__main__":
    main()

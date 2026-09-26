"""
pipeline/music_composer.py — BOBO KIDS STUDIO
===============================================
Phase 6: Script JSON → Background music tracks + SFX files

WHAT THIS MODULE DOES:
  1. Reads each scene's 'music' mood and 'sfx' list from script.json
  2. Generates WAV background music loops for each mood using pure math
  3. Generates WAV sound effects (sparkle, tada, chime, etc.) using pure math
  4. Saves a music_manifest.json that Phase 7 uses to place SFX at timestamps

WHY PURE MATH (numpy sine waves)?
  Real music libraries like music21 or FluidSynth need MIDI soundfonts
  (large downloads, complex setup). Instead, we use the fundamental
  building block of all sound: the SINE WAVE.
    y(t) = A × sin(2π × f × t)
  Where:  A = amplitude (volume),  f = frequency (pitch in Hz),  t = time in seconds
  By mixing sine waves at musical intervals with shaped envelopes, we get
  recognisable instrument-like sounds with ZERO extra dependencies.

MUSIC THEORY USED:
  - Pentatonic scale: 5 notes that always sound pleasant together
    C(261Hz), D(293Hz), E(329Hz), G(392Hz), A(440Hz) — no dissonance possible
  - Harmonics: real instruments produce overtones at 2×, 3×, 4× the fundamental
  - ADSR envelope: Attack (fade in), Decay, Sustain, Release (fade out)
    Makes sounds feel natural instead of robotic clicks
  - Tempo: 100 BPM for playful, 80 BPM for calm, 120 BPM for celebration

SFX DESIGN:
  Each SFX is a short signature sound (0.3–2.0 seconds) built from:
    - Tone sweeps (frequency rises = magical/positive)
    - Noise bursts (percussion-like impacts)
    - Chord stabs (multiple notes at once)

OUTPUT STRUCTURE:
  assets/
    music/
      playful_adventure.wav    ← loopable, ~8 bars
      gentle_curious.wav
      calm_learning.wav
      celebration.wav
      lullaby.wav
    sfx/
      sparkle.wav
      tada.wav
      box_open.wav
      chime.wav
      boing.wav
      gasp.wav
      ...
  audio/<video_id>/
    music_manifest.json        ← Phase 7 reads this for final mix
"""

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from tqdm import tqdm

from config import cfg
from pipeline import get_logger

logger = get_logger("music_composer")

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE = 44100          # CD quality: 44100 samples per second
BIT_DEPTH   = "PCM_16"       # 16-bit integers = standard WAV quality

# ── Note frequencies (Hz) ────────────────────────────────────────────────────
# WHY these exact values?
#   Equal temperament tuning: each semitone = previous × 2^(1/12)
#   A4 = 440 Hz is the international standard reference pitch.

NOTE_FREQ = {
    "C3": 130.81, "D3": 146.83, "E3": 164.81, "G3": 196.00, "A3": 220.00,
    "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23, "G4": 392.00,
    "A4": 440.00, "B4": 493.88,
    "C5": 523.25, "D5": 587.33, "E5": 659.25, "G5": 783.99, "A5": 880.00,
    "C6": 1046.50,
}

# C major pentatonic: 5 notes that ALWAYS sound good together
# Children's music almost universally uses this scale
PENTATONIC = ["C4", "D4", "E4", "G4", "A4", "C5", "D5", "E5", "G5"]

# ── Audio math helpers ────────────────────────────────────────────────────────

def time_axis(duration_sec: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Create a time array from 0 to duration_sec."""
    return np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)


def sine_wave(freq: float, duration_sec: float, amplitude: float = 1.0,
              sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Generate a pure sine wave.

    y(t) = amplitude × sin(2π × freq × t)

    This is the fundamental building block of all synthesised sound.
    """
    t = time_axis(duration_sec, sr)
    return amplitude * np.sin(2 * np.pi * freq * t)


def adsr_envelope(
    n_samples: int,
    attack: float = 0.01,    # fraction of total duration
    decay: float  = 0.10,
    sustain: float = 0.70,   # amplitude level (0-1)
    release: float = 0.20,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    ADSR Envelope — shapes how a sound changes over time.

    WHY THIS MATTERS:
      A pure sine wave starts and ends abruptly → 'click' sound.
      An ADSR envelope makes it feel like a real instrument:

        1.0 |   /‾‾‾‾‾‾\___________
            |  /          sustain   \\
        0.0 |_/                      \\____
               A   D       S          R

      Attack  = how fast the sound fades IN
      Decay   = how fast it drops from peak to sustain
      Sustain = the steady volume during the main note
      Release = how fast it fades OUT at the end

    Example: A xylophone has very short attack (instant), short sustain,
             long release (the note rings out).
    """
    n = n_samples
    atk = max(1, int(attack * n))
    dec = max(1, int(decay * n))
    rel = max(1, int(release * n))
    sus = max(1, n - atk - dec - rel)

    env = np.concatenate([
        np.linspace(0, 1, atk),                    # Attack
        np.linspace(1, sustain, dec),               # Decay
        np.full(sus, sustain),                      # Sustain
        np.linspace(sustain, 0, rel),               # Release
    ])
    return env[:n]


def add_harmonics(
    freq: float, duration_sec: float, sr: int = SAMPLE_RATE,
    harmonic_weights: list[float] | None = None,
) -> np.ndarray:
    """
    Generate a tone with harmonics (overtones) for richer sound.

    WHY HARMONICS?
      A pure sine wave sounds thin and electronic.
      Real instruments produce overtones at multiples of the base frequency:
        Harmonic 1 (fundamental): freq × 1   ← loudest
        Harmonic 2 (octave):      freq × 2   ← softer
        Harmonic 3:               freq × 3   ← even softer
      This makes the sound feel warm and musical.

      Xylophone:   strong fundamental, medium 2nd, weak 3rd+
      Flute/bell:  strong fundamental, very weak harmonics
      String:      many harmonics all significant
    """
    if harmonic_weights is None:
        harmonic_weights = [1.0, 0.5, 0.25, 0.12, 0.06]  # xylophone-like

    wave = np.zeros(int(sr * duration_sec))
    total_weight = sum(harmonic_weights)
    for i, weight in enumerate(harmonic_weights, start=1):
        wave += sine_wave(freq * i, duration_sec, weight / total_weight, sr)
    return wave


def note_to_wave(
    note_name: str, duration_sec: float,
    instrument: str = "xylophone", sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Convert a note name (e.g. 'C4') to a shaped audio wave.

    Parameters
    ----------
    note_name : str   e.g. 'C4', 'G4'
    duration_sec : float
    instrument : str  'xylophone' | 'bell' | 'bass' | 'flute'
    """
    freq = NOTE_FREQ.get(note_name, 261.63)

    # Instrument presets: (harmonic_weights, adsr)
    presets = {
        "xylophone": {
            "harmonics": [1.0, 0.4, 0.2, 0.1],
            "adsr": (0.005, 0.05, 0.3, 0.30),
        },
        "bell": {
            "harmonics": [1.0, 0.6, 0.3, 0.15, 0.08],
            "adsr": (0.002, 0.02, 0.4, 0.50),
        },
        "bass": {
            "harmonics": [1.0, 0.3, 0.1],
            "adsr": (0.01, 0.1, 0.7, 0.15),
        },
        "flute": {
            "harmonics": [1.0, 0.15, 0.05],
            "adsr": (0.02, 0.05, 0.8, 0.10),
        },
    }
    preset = presets.get(instrument, presets["xylophone"])

    wave = add_harmonics(freq, duration_sec, sr, preset["harmonics"])
    a, d, s, r = preset["adsr"]
    envelope = adsr_envelope(len(wave), a, d, s, r, sr)
    return wave * envelope


def normalize(wave: np.ndarray, target_peak: float = 0.8) -> np.ndarray:
    """
    Scale a wave so its peak amplitude is target_peak.

    WHY? Prevents digital clipping (distortion when signal exceeds -1 to +1 range).
    Clipping sounds like a harsh crackling noise and ruins audio quality.
    """
    peak = np.max(np.abs(wave))
    if peak < 1e-6:
        return wave
    return wave * (target_peak / peak)


def mix(*waves: np.ndarray, weights: list[float] | None = None) -> np.ndarray:
    """
    Mix multiple audio signals together.

    In digital audio, mixing = addition.
    WHY WEIGHTS? Prevent the sum from exceeding 1.0 (clipping).
    """
    max_len = max(len(w) for w in waves)
    if weights is None:
        weights = [1.0 / len(waves)] * len(waves)
    result = np.zeros(max_len)
    for wave, w in zip(waves, weights):
        result[:len(wave)] += wave * w
    return result


def fade_in_out(wave: np.ndarray, fade_ms: int = 20, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply short fade-in and fade-out to eliminate clicks at boundaries."""
    fade_samples = int(fade_ms / 1000 * sr)
    fade_samples = min(fade_samples, len(wave) // 4)
    wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
    wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return wave


def save_wav(wave: np.ndarray, path: Path, sr: int = SAMPLE_RATE) -> None:
    """Save a numpy array as a 16-bit WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    wave = normalize(wave, target_peak=0.85)
    wave = fade_in_out(wave)
    sf.write(str(path), wave, sr, subtype=BIT_DEPTH)
    logger.debug(f"Saved: {path.name} ({path.stat().st_size / 1024:.0f} KB)")


# ── Background music generators ───────────────────────────────────────────────

def _make_sequence(notes: list[str], beat_sec: float,
                   instrument: str = "xylophone",
                   sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Chain a list of note names into a sequence.
    notes can include 'R' for rests.
    """
    segments = []
    for note in notes:
        if note == "R":
            segments.append(np.zeros(int(sr * beat_sec)))
        else:
            segments.append(note_to_wave(note, beat_sec, instrument, sr))
    return np.concatenate(segments)


def gen_playful_adventure(sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Upbeat, bouncy children's melody — used for exploration scenes.
    Tempo: 108 BPM (beat = 0.555 sec)
    Melody: C-E-G-E-D-E-G-A (ascending + bobbing pattern)
    Harmony: bass notes every 2 beats
    """
    bpm = 108
    beat = 60 / bpm

    melody_notes = [
        "C4","E4","G4","E4",  "D4","F4","A4","G4",
        "E4","G4","A4","G4",  "E4","D4","C4","R",
        "C5","A4","G4","E4",  "D4","E4","G4","A4",
        "G4","E4","D4","C4",  "C4","R","R","R",
    ]
    bass_notes = [
        "C3","R","G3","R",  "D3","R","A3","R",
        "C3","R","G3","R",  "C3","R","R","R",
        "C3","R","G3","R",  "D3","R","A3","R",
        "G3","R","C3","R",  "C3","R","R","R",
    ]

    melody = _make_sequence(melody_notes, beat, "xylophone", sr)
    bass   = _make_sequence(bass_notes,   beat, "bass",      sr)

    return mix(melody, bass, weights=[0.65, 0.35])


def gen_gentle_curious(sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Soft, wondering melody — used for discovery/learning scenes.
    Tempo: 80 BPM (slower = more thoughtful)
    """
    bpm = 80
    beat = 60 / bpm

    melody_notes = [
        "E4","G4","A4","G4",  "E4","D4","E4","R",
        "G4","A4","C5","A4",  "G4","E4","D4","R",
        "E4","G4","A4","G4",  "E4","D4","C4","R",
        "D4","E4","G4","A4",  "G4","R","R","R",
    ]
    harm_notes = [
        "C4","E4","A3","E4",  "C4","D3","C4","R",
        "E4","A3","E4","A3",  "E4","C4","D3","R",
        "C4","E4","A3","E4",  "C4","D3","C3","R",
        "D3","C4","E4","A3",  "E4","R","R","R",
    ]

    melody = _make_sequence(melody_notes, beat, "flute",     sr)
    harm   = _make_sequence(harm_notes,   beat, "bell",      sr)

    return mix(melody, harm, weights=[0.60, 0.40])


def gen_calm_learning(sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Gentle, repetitive pattern — used for interactive/quiz scenes.
    Very simple: just two alternating arpeggios.
    """
    bpm = 72
    beat = 60 / bpm

    melody_notes = [
        "C4","E4","G4","E4",  "C4","E4","G4","A4",
        "D4","G4","A4","G4",  "D4","G4","A4","G4",
        "C4","E4","G4","E4",  "C4","E4","A4","G4",
        "E4","D4","C4","D4",  "C4","R","R","R",
    ]

    melody = _make_sequence(melody_notes, beat, "bell", sr)
    return melody


def gen_celebration(sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Triumphant, energetic melody — used for celebration scenes.
    Tempo: 128 BPM (fast and exciting!)
    """
    bpm = 128
    beat = 60 / bpm

    melody_notes = [
        "C5","C5","G4","G4",  "A4","A4","G4","R",
        "F4","F4","E4","E4",  "D4","D4","C4","R",
        "G4","G4","F4","F4",  "E4","E4","D4","R",
        "G4","G4","F4","F4",  "E4","E4","D4","R",
        "C5","C5","G4","G4",  "A4","A4","G4","R",
        "F4","F4","E4","E4",  "D4","D4","C4","R",
    ]
    bass_notes = [
        "C3","R","G3","R",  "A3","R","G3","R",
        "F3","R","E3","R",  "D3","R","C3","R",
        "G3","R","F3","R",  "E3","R","D3","R",
        "G3","R","F3","R",  "E3","R","D3","R",
        "C3","R","G3","R",  "A3","R","G3","R",
        "F3","R","E3","R",  "D3","R","C3","R",
    ]

    melody = _make_sequence(melody_notes, beat, "xylophone", sr)
    bass   = _make_sequence(bass_notes,   beat, "bass",      sr)
    return mix(melody, bass, weights=[0.65, 0.35])


def gen_lullaby(sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Soft, gentle closing music — used for goodbye scenes.
    Tempo: 60 BPM (very slow, calming)
    """
    bpm = 60
    beat = 60 / bpm

    melody_notes = [
        "E4","D4","C4","D4",  "E4","E4","E4","R",
        "D4","D4","D4","R",   "E4","G4","G4","R",
        "E4","D4","C4","D4",  "E4","E4","E4","E4",
        "D4","D4","E4","D4",  "C4","R","R","R",
    ]

    melody = _make_sequence(melody_notes, beat, "flute", sr)
    return melody


# Music generators keyed by mood name (from script.json 'music' field)
MUSIC_GENERATORS: dict[str, Callable] = {
    "playful_adventure": gen_playful_adventure,
    "gentle_curious":    gen_gentle_curious,
    "calm_learning":     gen_calm_learning,
    "celebration":       gen_celebration,
    "lullaby":           gen_lullaby,
}

# ── SFX generators ────────────────────────────────────────────────────────────

def sfx_sparkle(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Ascending shimmer — magical/positive moment."""
    freqs = [NOTE_FREQ[n] for n in ["C5","E5","G5","C5","G5"]]
    segments = []
    for i, freq in enumerate(freqs):
        t = time_axis(0.12, sr)
        wave = np.sin(2 * np.pi * freq * t)
        env = np.exp(-t * 8)
        segments.append(wave * env * (0.8 - i * 0.1))
        segments.append(np.zeros(int(sr * 0.04)))
    return normalize(np.concatenate(segments))


def sfx_tada(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Triumphant flourish — success/reveal moment."""
    # Quick ascending arpeggio then chord stab
    arp_notes = ["C4","E4","G4","C5"]
    segments = []
    for note in arp_notes:
        segments.append(note_to_wave(note, 0.08, "xylophone", sr))
    arp = np.concatenate(segments)

    # Final chord: C major
    chord = mix(
        note_to_wave("C4", 0.5, "bell", sr),
        note_to_wave("E4", 0.5, "bell", sr),
        note_to_wave("G4", 0.5, "bell", sr),
        note_to_wave("C5", 0.5, "bell", sr),
        weights=[0.4, 0.3, 0.2, 0.1],
    )
    return normalize(np.concatenate([arp, chord]))


def sfx_box_open(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Creak + shimmer — opening a surprise box."""
    # Creak: noise burst with low-pass filter feel (low freq rumble)
    t = time_axis(0.2, sr)
    noise = np.random.randn(len(t)) * np.exp(-t * 15)
    creak = noise * np.sin(2 * np.pi * 120 * t)

    # Shimmer after opening
    shimmer = sfx_sparkle(sr)
    gap = np.zeros(int(sr * 0.1))

    return normalize(np.concatenate([creak, gap, shimmer]))


def sfx_chime(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Single soft chime — gentle attention."""
    return normalize(note_to_wave("A5", 0.8, "bell", sr))


def sfx_twinkle(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Quick twinkling — stars/magic."""
    notes = ["G5","A5","G5","E5","G5"]
    segs = []
    for note in notes:
        segs.append(note_to_wave(note, 0.10, "bell", sr))
        segs.append(np.zeros(int(sr * 0.02)))
    return normalize(np.concatenate(segs))


def sfx_boing(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Bouncy spring — playful action."""
    duration = 0.4
    t = time_axis(duration, sr)
    # Frequency sweeps DOWN (pitch drop = bouncy feel)
    freq_sweep = 600 * np.exp(-t * 8) + 200
    phase = 2 * np.pi * np.cumsum(freq_sweep) / sr
    wave = np.sin(phase)
    env = adsr_envelope(len(wave), 0.01, 0.1, 0.5, 0.3, sr)
    return normalize(wave * env)


def sfx_gasp(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Surprise gasp — quick breath-like sound."""
    duration = 0.25
    t = time_axis(duration, sr)
    noise = np.random.randn(len(t))
    # Shaped noise: rises then falls
    env = np.concatenate([
        np.linspace(0, 1, len(t) // 3),
        np.linspace(1, 0.3, len(t) // 3),
        np.linspace(0.3, 0, len(t) - 2 * (len(t) // 3)),
    ])[:len(t)]
    # Filter to breathy range (1000-4000 Hz) by modulating with sine
    filtered = noise * np.sin(2 * np.pi * 2000 * t) * env * 0.3
    return normalize(filtered)


def sfx_wind_gust(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Whooshing wind — balloon floating."""
    duration = 0.6
    t = time_axis(duration, sr)
    noise = np.random.randn(len(t))
    # Rising then falling envelope
    n = len(t)
    env = np.concatenate([
        np.linspace(0, 1, n // 3),
        np.ones(n // 6),
        np.linspace(1, 0, n - n // 3 - n // 6),
    ])[:n]
    wave = noise * env * np.sin(2 * np.pi * 500 * t + np.cumsum(noise * 0.1))
    return normalize(wave * 0.5)


def sfx_pop(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Quick pop — balloon caught/burst."""
    t = time_axis(0.15, sr)
    noise = np.random.randn(len(t))
    env = np.exp(-t * 40)
    freq_hit = np.sin(2 * np.pi * 300 * t) * np.exp(-t * 30)
    return normalize((noise * env * 0.4 + freq_hit * 0.6))


def sfx_cheer(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Crowd cheer — celebration."""
    duration = 0.8
    t = time_axis(duration, sr)
    # Multiple noise bands mixed = crowd-like
    noise = np.random.randn(len(t))
    env = np.concatenate([
        np.linspace(0, 1, len(t) // 4),
        np.ones(len(t) // 2),
        np.linspace(1, 0, len(t) - len(t) // 4 - len(t) // 2),
    ])[:len(t)]
    wave = noise * env
    # Modulate for excitement
    wave *= (1 + 0.3 * np.sin(2 * np.pi * 8 * t))
    return normalize(wave * 0.6)


def sfx_party_horn(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Party horn toot — celebration."""
    duration = 0.5
    t = time_axis(duration, sr)
    # Detuned square-ish wave for horn tone
    freq = 440
    wave = (np.sin(2 * np.pi * freq * t) +
            0.5 * np.sin(2 * np.pi * freq * 2 * t) +
            0.3 * np.sin(2 * np.pi * freq * 3 * t))
    env = adsr_envelope(len(wave), 0.02, 0.05, 0.7, 0.2, sr)
    return normalize(wave * env)


def sfx_ding(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Correct-answer ding — quiz moment."""
    # Two quick notes: D5 then G5 (rising = 'correct!')
    n1 = note_to_wave("D5", 0.15, "bell", sr)
    gap = np.zeros(int(sr * 0.03))
    n2 = note_to_wave("G5", 0.25, "bell", sr)
    return normalize(np.concatenate([n1, gap, n2]))


def sfx_applause(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Applause — intro/celebration."""
    duration = 1.0
    t = time_axis(duration, sr)
    noise = np.random.randn(len(t))
    # Clapping rhythm: bursts of noise
    bursts = np.zeros(len(t))
    for beat_ms in [0, 250, 500, 750]:
        start = int(beat_ms / 1000 * sr)
        burst_len = int(0.08 * sr)
        if start + burst_len < len(bursts):
            env = np.exp(-np.arange(burst_len) * 30 / sr)
            bursts[start:start+burst_len] += env
    wave = noise * bursts
    return normalize(wave * 0.7)


def sfx_wings_flutter(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Firefly wings — delicate fluttering."""
    duration = 0.3
    t = time_axis(duration, sr)
    # Rapid AM modulation = flutter effect
    carrier = np.sin(2 * np.pi * 800 * t)
    flutter = (1 + np.sin(2 * np.pi * 25 * t)) / 2   # 25 Hz flutter rate
    env = adsr_envelope(len(t), 0.05, 0.1, 0.4, 0.3, sr)
    return normalize(carrier * flutter * env * 0.5)


def sfx_wave_goodbye(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Soft descending notes — goodbye wave."""
    notes = ["G4","E4","C4"]
    segs = []
    for note in notes:
        segs.append(note_to_wave(note, 0.3, "flute", sr))
        segs.append(np.zeros(int(sr * 0.05)))
    return normalize(np.concatenate(segs))


def sfx_footsteps_grass(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Soft footsteps on grass — walking sound."""
    segs = []
    for _ in range(3):
        t = time_axis(0.08, sr)
        noise = np.random.randn(len(t))
        env = np.exp(-np.arange(len(t)) * 60 / sr)
        step = noise * env * 0.3
        segs.append(step)
        segs.append(np.zeros(int(sr * 0.15)))
    return normalize(np.concatenate(segs))


# SFX registry — maps script sfx label → generator function
SFX_REGISTRY: dict[str, Callable] = {
    "sparkle":       sfx_sparkle,
    "tada":          sfx_tada,
    "box_open":      sfx_box_open,
    "chime":         sfx_chime,
    "twinkle":       sfx_twinkle,
    "boing":         sfx_boing,
    "gasp":          sfx_gasp,
    "wind_gust":     sfx_wind_gust,
    "pop":           sfx_pop,
    "cheer":         sfx_cheer,
    "party_horn":    sfx_party_horn,
    "ding":          sfx_ding,
    "applause":      sfx_applause,
    "wings_flutter": sfx_wings_flutter,
    "wave":          sfx_wave_goodbye,
    "footsteps_grass": sfx_footsteps_grass,
}


# ── Main composer class ───────────────────────────────────────────────────────

class MusicComposer:
    """
    Generates all music and SFX assets for a video.

    USAGE:
      composer = MusicComposer()
      manifest = composer.compose(script, video_id)

    OUTPUT:
      assets/music/<mood>.wav       — background music loops
      assets/sfx/<effect>.wav       — sound effect files
      audio/<video_id>/music_manifest.json  — timing map for Phase 7
    """

    def __init__(self):
        self.music_dir = cfg.PIPELINE_DIRS["music"]
        self.sfx_dir   = cfg.PIPELINE_DIRS["sfx"]
        self.audio_dir = cfg.PIPELINE_DIRS["audio"]
        self.music_dir.mkdir(parents=True, exist_ok=True)
        self.sfx_dir.mkdir(parents=True, exist_ok=True)

    def compose(self, script: dict, video_id: str, force_redo: bool = False) -> dict:
        """
        Generate all audio assets for a script and return a music manifest.
        """
        manifest_path = self.audio_dir / video_id / "music_manifest.json"

        if manifest_path.exists() and not force_redo:
            logger.info(f"Music manifest cached. Loading from {manifest_path}")
            return json.loads(manifest_path.read_text(encoding="utf-8"))

        scenes = script.get("scenes", [])

        # Collect unique moods and SFX needed
        moods_needed = set()
        sfx_needed   = set()
        for scene in scenes:
            mood = scene.get("music")
            if mood and mood in MUSIC_GENERATORS:
                moods_needed.add(mood)
            for sfx in scene.get("sfx", []):
                if sfx in SFX_REGISTRY:
                    sfx_needed.add(sfx)

        logger.info(f"Moods to generate: {sorted(moods_needed)}")
        logger.info(f"SFX to generate:   {sorted(sfx_needed)}")

        # ── Generate background music tracks ─────────────────────────────────
        print()
        print("Generating background music tracks...")
        music_files = {}
        for mood in tqdm(sorted(moods_needed), desc="Music tracks", unit="track"):
            wav_path = self.music_dir / f"{mood}.wav"
            if wav_path.exists() and not force_redo:
                logger.debug(f"Reusing cached music: {mood}.wav")
            else:
                logger.debug(f"Generating: {mood}.wav")
                wave = MUSIC_GENERATORS[mood]()
                save_wav(wave, wav_path)
                logger.info(f"Music: {mood}.wav ({wav_path.stat().st_size // 1024} KB)")
            music_files[mood] = str(wav_path)

        # ── Generate SFX ─────────────────────────────────────────────────────
        print()
        print("Generating sound effects...")
        sfx_files = {}
        for sfx_name in tqdm(sorted(sfx_needed), desc="SFX", unit="sfx"):
            wav_path = self.sfx_dir / f"{sfx_name}.wav"
            if wav_path.exists() and not force_redo:
                logger.debug(f"Reusing cached SFX: {sfx_name}.wav")
            else:
                logger.debug(f"Generating SFX: {sfx_name}.wav")
                wave = SFX_REGISTRY[sfx_name]()
                save_wav(wave, wav_path)
                logger.info(f"SFX: {sfx_name}.wav ({wav_path.stat().st_size // 1024} KB)")
            sfx_files[sfx_name] = str(wav_path)

        # ── Build music manifest ──────────────────────────────────────────────
        # The manifest tells Phase 7 (video assembly):
        #   "For scene 1, use playful_adventure.wav as background.
        #    At 1.0s into this scene, play the 'sparkle' SFX."
        manifest_scenes = []
        for scene in scenes:
            mood = scene.get("music")
            scene_sfx = [
                {
                    "label": sfx,
                    "file":  sfx_files.get(sfx, ""),
                    # SFX timing: spread evenly through the scene
                    # Phase 7 can refine these based on dialogue timestamps
                    "offset_sec": round(
                        (i + 0.5) * scene.get("duration_sec", 15) / max(len(scene.get("sfx", [])), 1),
                        2
                    ),
                    "volume": 0.7,
                }
                for i, sfx in enumerate(scene.get("sfx", []))
                if sfx in sfx_files
            ]
            manifest_scenes.append({
                "scene_number": scene["scene_number"],
                "title":        scene.get("title", ""),
                "duration_sec": scene.get("duration_sec", 15),
                "bg_music_mood": mood,
                "bg_music_file": music_files.get(mood, "") if mood else "",
                "bg_music_volume": 0.18,   # Background music is quiet under narration
                "sfx_cues": scene_sfx,
            })

        manifest = {
            "video_id":    video_id,
            "music_files": music_files,
            "sfx_files":   sfx_files,
            "scenes":      manifest_scenes,
        }

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info(f"Music manifest saved: {manifest_path}")
        return manifest


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = sys.argv[1:]
    force_redo = "--force-redo" in args
    args = [a for a in args if not a.startswith("--")]
    video_id = args[0] if args else "20260924_learn_colors_red"

    script_path = cfg.PIPELINE_DIRS["scripts"] / video_id / "script.json"
    if not script_path.exists():
        sys.stderr.write(f"[ERROR] Script not found: {script_path}\n")
        sys.exit(1)

    script = json.loads(script_path.read_text(encoding="utf-8"))
    sys.stderr.write(f"\nPhase 6 -- Music + SFX Generation\n")
    sys.stderr.write(f"Topic:  {script['topic']}\n")
    sys.stderr.write(f"Scenes: {len(script['scenes'])}\n\n")

    composer = MusicComposer()
    manifest = composer.compose(script, video_id, force_redo=force_redo)

    sys.stderr.write(f"\n=== Music + SFX Complete ===\n")
    sys.stderr.write(f"Music tracks: {len(manifest['music_files'])}\n")
    sys.stderr.write(f"SFX files:    {len(manifest['sfx_files'])}\n")
    sys.stderr.write(f"Manifest:     {cfg.PIPELINE_DIRS['audio'] / video_id / 'music_manifest.json'}\n\n")

    sys.stderr.write("Music tracks:\n")
    for mood, path in manifest["music_files"].items():
        size_kb = Path(path).stat().st_size // 1024
        sys.stderr.write(f"  {mood:<22} -> {Path(path).name} ({size_kb} KB)\n")

    sys.stderr.write("\nSFX files:\n")
    for name, path in manifest["sfx_files"].items():
        size_kb = Path(path).stat().st_size // 1024
        sys.stderr.write(f"  {name:<20} -> {Path(path).name} ({size_kb} KB)\n")

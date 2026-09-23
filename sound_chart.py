from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_RATE = 44_100
DEFAULT_GAIN = 1.35
DEFAULT_PERCENTILE = 99.5
DENSITY_CENTER = 0.5
DENSITY_SPAN = 0.49
PCM_CHUNK_SAMPLES = 1_000_000
ACTION_CHUNK = 50_000
ANGLE_BLOCK = 200_000


def run_ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def decode_mono_f32(source: Path, raw_path: Path, rate: int) -> int:
    run_ffmpeg([
        "-i", str(source),
        "-ac", "1",
        "-ar", str(rate),
        "-f", "f32le",
        str(raw_path),
    ])
    size = raw_path.stat().st_size
    if size % 4:
        raise RuntimeError("Decoded PCM size is not aligned to float32 samples")
    return size // 4


def estimate_abs_percentile(pcm: np.memmap, percentile: float) -> float:
    # Keep normalization cheap even for very large source files.
    # Up to ~2 million evenly spaced samples are enough for this use case.
    n = len(pcm)
    if n == 0:
        return 1.0
    step = max(1, n // 2_000_000)
    sampled = np.abs(np.asarray(pcm[::step], dtype=np.float64))
    value = float(np.percentile(sampled, percentile))
    return max(value, 1e-12)


def encode_pdm_hits(
    pcm: np.memmap,
    hit_path: Path,
    *,
    scale: float,
    gain: float,
) -> int:
    """Encode mono PCM into first-order 1-bit PDM pulse positions.

    Hit sample indices are written as raw int64 values so the program does not
    need to keep millions of Python integers in memory.
    """
    total_hits = 0
    accumulator = 0.0

    with hit_path.open("wb") as out:
        for start in range(0, len(pcm), PCM_CHUNK_SAMPLES):
            end = min(start + PCM_CHUNK_SAMPLES, len(pcm))
            x = np.asarray(pcm[start:end], dtype=np.float64)
            normalized = np.tanh(gain * x / scale)
            density = DENSITY_CENTER + DENSITY_SPAN * np.clip(normalized, -1.0, 1.0)

            cumulative = accumulator + np.cumsum(density, dtype=np.float64)
            levels = np.floor(cumulative).astype(np.int64)

            pulse = np.empty(len(levels), dtype=bool)
            pulse[0] = levels[0] > 0
            pulse[1:] = levels[1:] > levels[:-1]

            local = np.flatnonzero(pulse).astype(np.int64)
            if len(local):
                (local + start).tofile(out)
                total_hits += len(local)

            if len(cumulative):
                accumulator = float(cumulative[-1] - math.floor(cumulative[-1]))

    return total_hits


def bpm_for_gap(rate: int, gap: int) -> float:
    return 60.0 * rate / max(1, int(gap))


def make_settings(
    *,
    source: Path,
    title: str,
    artist: str,
    hitsound: str,
    silence_name: str,
    rate: int,
    floors: int,
    initial_bpm: float,
) -> dict:
    return {
        "version": 15,
        "artist": artist,
        "specialArtistType": "None",
        "artistPermission": "",
        "song": title,
        "author": "ADOFAISoundCharting",
        "separateCountdownTime": "Enabled",
        "previewImage": "",
        "previewIcon": "",
        "previewIconColor": "003f52",
        "previewSongStart": 0,
        "previewSongDuration": 10,
        "seizureWarning": "Disabled",
        "levelDesc": (
            f"{rate} Hz 1-bit PDM Sound Charting reconstruction, "
            f"{floors:,} straight tiles. Background audio is silent."
        ),
        "levelTags": "Sound Charting, Hz Chart, 1-bit, PDM, straight",
        "artistLinks": "",
        "speedTrialAim": 0,
        "difficulty": 1,
        "requiredMods": [],
        "songFilename": silence_name,
        "bpm": initial_bpm,
        "volume": 100,
        "offset": 0,
        "pitch": 100,
        "hitsound": hitsound,
        "hitsoundVolume": 100,
        "countdownTicks": 0,
        "trackColorType": "Single",
        "trackColor": "debb7b",
        "secondaryTrackColor": "ffffff",
        "trackColorAnimDuration": 2,
        "trackColorPulse": "None",
        "trackPulseLength": 10,
        "trackStyle": "Standard",
        "trackTexture": "",
        "trackTextureScale": 1,
        "trackGlowIntensity": 100,
        "trackAnimation": "None",
        "beatsAhead": 3,
        "trackDisappearAnimation": "None",
        "beatsBehind": 4,
        "backgroundColor": "000000",
        "showDefaultBGIfNoImage": "Enabled",
        "showDefaultBGTile": "Enabled",
        "defaultBGTileColor": "101010",
        "defaultBGShapeType": "Default",
        "defaultBGShapeColor": "ffffff",
        "bgImage": "",
        "bgImageColor": "ffffff",
        "parallax": [100, 100],
        "bgDisplayMode": "FitToScreen",
        "lockRot": "Disabled",
        "loopBG": "Disabled",
        "unscaledSize": 100,
        "relativeTo": "Player",
        "position": [0, 0],
        "rotation": 0,
        "zoom": 100,
        "pulseOnFloor": "Enabled",
        "bgVideo": "",
        "loopVideo": "Disabled",
        "vidOffset": 0,
        "floorIconOutlines": "Disabled",
        "stickToFloors": "Disabled",
    }


def write_zero_angles(f, count: int) -> None:
    block = ",".join(["0"] * ANGLE_BLOCK)
    full, rem = divmod(count, ANGLE_BLOCK)
    for i in range(full):
        if i:
            f.write(",")
        f.write(block)
    if rem:
        if full:
            f.write(",")
        f.write(",".join(["0"] * rem))


def write_actions(f, hits: np.memmap, rate: int) -> int:
    """Write SetSpeed events only where pulse spacing changes."""
    if len(hits) < 2:
        return 0

    event_count = 0
    first_action = True
    previous_gap: int | None = None
    previous_hit = int(hits[0])

    # Chunked scan keeps the temporary arrays bounded.
    for start in range(1, len(hits), PCM_CHUNK_SAMPLES):
        end = min(start + PCM_CHUNK_SAMPLES, len(hits))
        current = np.asarray(hits[start:end], dtype=np.int64)
        if len(current) == 0:
            continue

        with_prev = np.empty(len(current) + 1, dtype=np.int64)
        with_prev[0] = previous_hit
        with_prev[1:] = current
        gaps = np.diff(with_prev)

        changed = np.empty(len(gaps), dtype=bool)
        changed[0] = previous_gap is None or int(gaps[0]) != previous_gap
        if len(gaps) > 1:
            changed[1:] = gaps[1:] != gaps[:-1]

        indices = np.flatnonzero(changed)
        for chunk_start in range(0, len(indices), ACTION_CHUNK):
            idx = indices[chunk_start:chunk_start + ACTION_CHUNK]
            parts: list[str] = []
            for local_i in idx:
                # hits[start + local_i] is the pulse reached by this gap.
                # This matches the experimental chart format used by this tool.
                floor = start + int(local_i) + 1
                gap = int(gaps[local_i])
                parts.append(
                    '{"floor":%d,"eventType":"SetSpeed","speedType":"Bpm",'
                    '"beatsPerMinute":%.12g,"bpmMultiplier":1,"angleOffset":0}'
                    % (floor, bpm_for_gap(rate, gap))
                )

            if parts:
                if not first_action:
                    f.write(",")
                f.write(",".join(parts))
                first_action = False
                event_count += len(parts)

        previous_gap = int(gaps[-1])
        previous_hit = int(current[-1])

    return event_count


def write_chart(
    output: Path,
    hits: np.memmap,
    settings: dict,
    rate: int,
) -> int:
    floors = len(hits)
    with output.open("w", encoding="utf-8", buffering=1024 * 1024) as f:
        f.write('{"angleData":[')
        write_zero_angles(f, floors + 1)
        f.write('],"settings":')
        f.write(json.dumps(settings, ensure_ascii=False, separators=(",", ":")))
        f.write(',"actions":[')
        actions = write_actions(f, hits, rate)
        f.write('],"decorations":[]}')
    return actions


def create_silence(path: Path, duration: float) -> None:
    run_ffmpeg([
        "-f", "lavfi",
        "-i", "anullsrc=r=48000:cl=stereo",
        "-t", f"{duration + 1.0:.6f}",
        "-c:a", "libvorbis",
        "-q:a", "0",
        str(path),
    ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Encode audio as an ultra-dense ADOFAI hitsound chart using 1-bit PDM."
    )
    p.add_argument("input", type=Path, help="Input audio file understood by FFmpeg")
    p.add_argument("-o", "--output", type=Path, help="Output .adofai path")
    p.add_argument("--rate", type=int, default=DEFAULT_RATE, help="PCM/PDM sample rate")
    p.add_argument("--hitsound", default="Sidestick", help="ADOFAI hitsound name")
    p.add_argument("--title", help="ADOFAI song title")
    p.add_argument("--artist", help="ADOFAI artist field")
    p.add_argument("--gain", type=float, default=DEFAULT_GAIN, help="Soft-clip gain")
    p.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PERCENTILE,
        help="Absolute-amplitude percentile used for normalization",
    )
    p.add_argument("--silence-name", default="silence.ogg", help="Silent OGG file name")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg was not found on PATH")
    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")
    if args.rate <= 0:
        raise SystemExit("--rate must be positive")
    if args.gain <= 0:
        raise SystemExit("--gain must be positive")
    if not 0 < args.percentile <= 100:
        raise SystemExit("--percentile must be in (0, 100]")

    output = args.output or args.input.with_name(args.input.stem + "_sound_chart.adofai")
    if output.suffix.lower() != ".adofai":
        output = output.with_suffix(".adofai")
    output.parent.mkdir(parents=True, exist_ok=True)
    silence = output.parent / args.silence_name

    if not args.force:
        existing = [p for p in (output, silence) if p.exists()]
        if existing:
            names = ", ".join(str(p) for p in existing)
            raise SystemExit(f"Output already exists: {names} (use --force to overwrite)")

    with tempfile.TemporaryDirectory(prefix="adofai_sound_chart_") as tmp:
        tmpdir = Path(tmp)
        raw_path = tmpdir / "audio.f32le"
        hit_path = tmpdir / "hits.i64"

        print(f"[1/5] Decoding mono PCM at {args.rate:,} Hz...")
        sample_count = decode_mono_f32(args.input, raw_path, args.rate)
        if sample_count == 0:
            raise SystemExit("Decoded audio is empty")
        duration = sample_count / args.rate
        pcm = np.memmap(raw_path, dtype="<f4", mode="r", shape=(sample_count,))

        print("[2/5] Estimating normalization...")
        scale = estimate_abs_percentile(pcm, args.percentile)
        print(f"      abs p{args.percentile:g} = {scale:.8g}")

        print("[3/5] Encoding 1-bit PDM pulses...")
        floors = encode_pdm_hits(pcm, hit_path, scale=scale, gain=args.gain)
        if floors == 0:
            raise SystemExit("PDM encoder produced no hits")
        del pcm

        hits = np.memmap(hit_path, dtype="<i8", mode="r", shape=(floors,))
        initial_gap = max(1, int(hits[0]))
        initial_bpm = bpm_for_gap(args.rate, initial_gap)

        title = args.title or f"{args.input.stem} [Sound Charting]"
        artist = args.artist or args.input.stem
        settings = make_settings(
            source=args.input,
            title=title,
            artist=artist,
            hitsound=args.hitsound,
            silence_name=silence.name,
            rate=args.rate,
            floors=floors,
            initial_bpm=initial_bpm,
        )

        print(f"[4/5] Writing {floors:,} straight tiles...")
        actions = write_chart(output, hits, settings, args.rate)
        del hits

        print("[5/5] Creating silent background OGG...")
        create_silence(silence, duration)

    print()
    print(f"duration   : {duration:.3f} s")
    print(f"sample rate: {args.rate:,} Hz")
    print(f"tiles      : {floors:,}")
    print(f"SetSpeed   : {actions:,}")
    print(f"max BPM    : {60 * args.rate:,.0f}")
    print(f"chart      : {output}")
    print(f"silence    : {silence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

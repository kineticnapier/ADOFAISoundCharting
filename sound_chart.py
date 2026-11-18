from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_PDM_RATE = 44_100
DEFAULT_PLAYSOUND_RATE = 11_025
DEFAULT_TILE_MS = 100.0
DEFAULT_VOLUME_MAX = 100
DEFAULT_GAIN = 1.35
DEFAULT_PERCENTILE = 99.5
DENSITY_CENTER = 0.5
DENSITY_SPAN = 0.49
DEFAULT_MAX_DENSITY = 0.98
DEFAULT_ENV_MS = 10.0
DEFAULT_GATE_DB = -60.0
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


class Envelope:
    """Look-ahead peak envelope of the soft-clipped signal.

    Block peaks are max-filtered with their neighbours and linearly
    interpolated between block centres, which guarantees env[n] >= |y[n]|.
    """

    def __init__(self, centers: np.ndarray, values: np.ndarray) -> None:
        self.centers = centers
        self.values = values

    def at(self, start: int, end: int) -> np.ndarray:
        idx = np.arange(start, end, dtype=np.float64)
        return np.interp(idx, self.centers, self.values)


def soft_clip(x: np.ndarray, *, scale: float, gain: float) -> np.ndarray:
    return np.tanh(gain * np.asarray(x, dtype=np.float64) / scale)


def compute_envelope(pcm: np.memmap, *, scale: float, gain: float, block: int) -> Envelope:
    n = len(pcm)
    blocks = -(-n // block)
    peaks = np.zeros(blocks, dtype=np.float64)
    chunk = max(block, (PCM_CHUNK_SAMPLES // block) * block)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        y = np.abs(soft_clip(pcm[start:end], scale=scale, gain=gain))
        pad = (-len(y)) % block
        if pad:
            y = np.concatenate([y, np.zeros(pad)])
        b0 = start // block
        peaks[b0:b0 + len(y) // block] = y.reshape(-1, block).max(axis=1)

    held = peaks.copy()
    held[1:] = np.maximum(held[1:], peaks[:-1])
    held[:-1] = np.maximum(held[:-1], peaks[1:])
    centers = np.arange(blocks, dtype=np.float64) * block + (block - 1) / 2.0
    return Envelope(centers, held)


def base_density(
    pcm: np.memmap,
    start: int,
    end: int,
    *,
    scale: float,
    gain: float,
    env: Envelope | None,
    gate: float,
) -> np.ndarray:
    """Pulse density before max-density scaling.

    classic  (env is None): 0.5 + 0.49*y      -> silence = 50% density
    adaptive               : (env + y) / 2    -> silence = 0 density
    """
    y = soft_clip(pcm[start:end], scale=scale, gain=gain)
    if env is None:
        return DENSITY_CENTER + DENSITY_SPAN * np.clip(y, -1.0, 1.0)
    e = env.at(start, end)
    d = 0.5 * (e + y)
    d[e < gate] = 0.0
    return np.clip(d, 0.0, 1.0)


def expected_pulses(pcm: np.memmap, **kw) -> float:
    total = 0.0
    for start in range(0, len(pcm), PCM_CHUNK_SAMPLES):
        end = min(start + PCM_CHUNK_SAMPLES, len(pcm))
        total += float(base_density(pcm, start, end, **kw).sum())
    return total


def encode_pdm_hits(
    pcm: np.memmap,
    hit_path: Path,
    *,
    scale: float,
    gain: float,
    env: Envelope | None,
    gate: float,
    density_scale: float,
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
            density = base_density(
                pcm, start, end, scale=scale, gain=gain, env=env, gate=gate
            )
            if env is not None:
                density *= density_scale

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
    hitsound_volume: int = 100,
    description: str | None = None,
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
        "levelDesc": description or (
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
        "hitsoundVolume": hitsound_volume,
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


def write_playsound_chart(
    output: Path,
    pcm: np.memmap,
    settings: dict,
    *,
    tiles: int,
    tile_len: int,
    start_floor: int,
    hitsound: str,
    volume_max: int,
    scale: float,
    gain: float,
    env: "Envelope | None",
    gate: float,
    compact: bool = False,
) -> tuple[int, int]:
    """Few straight tiles at a constant BPM; the audio is PlaySound events.

    Every sample n is a potential PlaySound at time n/rate. Its volume is a
    multi-level, first-order noise-shaped quantization of the target amplitude
    (0..volume_max). Zero-volume samples emit no event, so silence is free.

    Sample n lives on floor start_floor + n // tile_len with
    angleOffset = 180 * (n % tile_len) / tile_len (straight tile = 180 deg = 1 beat).
    """
    events = 0
    volume_sum = 0
    acc = 0.0
    prev_level = 0
    first = True
    # Field set/order copied from an editor-saved PlaySound event.
    # compact=True drops the fields left at their editor defaults.
    if compact:
        head = (
            '{"floor":%d,"eventType":"PlaySound","hitsound":'
            + json.dumps(hitsound)
            + ',"hitsoundVolume":%d,"angleOffset":%.7g}'
        )
    else:
        head = (
            '{"floor":%d,"eventType":"PlaySound","hitsound":'
            + json.dumps(hitsound)
            + ',"offset":0,"playDuration":0,"pitch":100'
            + ',"hitsoundVolume":%d,"angleOffset":%.7g,"eventTag":""}'
        )
    deg_per_sample = 180.0 / tile_len

    with output.open("w", encoding="utf-8", buffering=1024 * 1024) as f:
        f.write('{"angleData":[')
        write_zero_angles(f, tiles)
        f.write('],"settings":')
        f.write(json.dumps(settings, ensure_ascii=False, separators=(",", ":")))
        f.write(',"actions":[')

        for start in range(0, len(pcm), PCM_CHUNK_SAMPLES):
            end = min(start + PCM_CHUNK_SAMPLES, len(pcm))
            target = base_density(
                pcm, start, end, scale=scale, gain=gain, env=env, gate=gate
            ) * volume_max
            cumulative = acc + np.cumsum(target, dtype=np.float64)
            levels = np.floor(cumulative + 0.5).astype(np.int64)
            volume = np.diff(levels, prepend=prev_level)
            shift = math.floor(cumulative[-1])
            acc = float(cumulative[-1] - shift)
            prev_level = int(levels[-1] - shift)

            idx = np.flatnonzero(volume > 0)
            if not len(idx):
                continue
            n = idx + start
            floors = start_floor + n // tile_len
            offsets = (n % tile_len) * deg_per_sample
            vols = volume[idx]
            volume_sum += int(vols.sum())

            for c in range(0, len(idx), ACTION_CHUNK):
                part = ",".join(
                    head % (fl, v, off)
                    for fl, v, off in zip(
                        floors[c:c + ACTION_CHUNK].tolist(),
                        vols[c:c + ACTION_CHUNK].tolist(),
                        offsets[c:c + ACTION_CHUNK].tolist(),
                    )
                )
                if not first:
                    f.write(",")
                f.write(part)
                first = False
            events += len(idx)

        f.write('],"decorations":[]}')
    return events, volume_sum


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
    p.add_argument(
        "--engine",
        choices=("playsound", "pdm"),
        default="playsound",
        help="playsound: few tiles, audio carried by PlaySound events with angleOffset "
        "and per-event volume. pdm: one tile per 1-bit pulse (huge tile counts)",
    )
    p.add_argument(
        "--rate",
        type=int,
        help=f"Sample rate (default {DEFAULT_PLAYSOUND_RATE} for playsound, "
        f"{DEFAULT_PDM_RATE} for pdm). For playsound this is the max events/s",
    )
    p.add_argument(
        "--tile-ms",
        type=float,
        default=DEFAULT_TILE_MS,
        help="playsound: duration of one straight tile in milliseconds",
    )
    p.add_argument(
        "--compact-events",
        action="store_true",
        help="playsound: omit offset/playDuration/pitch/eventTag (smaller JSON; "
        "only if the game accepts events without them)",
    )
    p.add_argument(
        "--volume-max",
        type=int,
        default=DEFAULT_VOLUME_MAX,
        help="playsound: PlaySound volume used for full-scale amplitude",
    )
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
    p.add_argument(
        "--mode",
        choices=("adaptive", "classic"),
        default="adaptive",
        help="adaptive: pulse density follows loudness (silence = no tiles). "
        "classic: fixed 50%% density bias (silence = Fs/2 tiles per second)",
    )
    p.add_argument(
        "--max-density",
        type=float,
        default=DEFAULT_MAX_DENSITY,
        help="adaptive: pulse density at full-scale envelope (0-1]",
    )
    p.add_argument(
        "--max-tiles",
        type=int,
        help="adaptive: lower the density so the chart has at most ~this many tiles",
    )
    p.add_argument(
        "--env-ms",
        type=float,
        default=DEFAULT_ENV_MS,
        help="adaptive: envelope block length in milliseconds",
    )
    p.add_argument(
        "--gate-db",
        type=float,
        default=DEFAULT_GATE_DB,
        help="adaptive: envelope below this level (dBFS after normalization) emits no tiles",
    )
    p.add_argument("--silence-name", default="silence.ogg", help="Silent OGG file name")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.rate is None:
        args.rate = DEFAULT_PLAYSOUND_RATE if args.engine == "playsound" else DEFAULT_PDM_RATE

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
    if not 0 < args.max_density <= 1:
        raise SystemExit("--max-density must be in (0, 1]")
    if args.max_tiles is not None and args.max_tiles <= 0:
        raise SystemExit("--max-tiles must be positive")
    if args.env_ms <= 0:
        raise SystemExit("--env-ms must be positive")
    if args.tile_ms <= 0:
        raise SystemExit("--tile-ms must be positive")
    if args.volume_max <= 0:
        raise SystemExit("--volume-max must be positive")

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

        env = None
        gate = 10.0 ** (args.gate_db / 20.0)

        if args.engine == "playsound":
            if args.mode == "adaptive":
                block = max(1, round(args.rate * args.env_ms / 1000.0))
                env = compute_envelope(pcm, scale=scale, gain=args.gain, block=block)
            tile_len = max(1, round(args.rate * args.tile_ms / 1000.0))
            start_floor = 1
            tiles = start_floor + -(-sample_count // tile_len) + 1
            bpm = 60.0 * args.rate / tile_len
            title = args.title or f"{args.input.stem} [Sound Charting]"
            artist = args.artist or args.input.stem
            settings = make_settings(
                source=args.input,
                title=title,
                artist=artist,
                hitsound=args.hitsound,
                silence_name=silence.name,
                rate=args.rate,
                floors=tiles,
                initial_bpm=bpm,
                hitsound_volume=0,
                description=(
                    f"{args.rate} Hz PlaySound Sound Charting reconstruction, "
                    f"{tiles:,} straight tiles. Background audio is silent."
                ),
            )
            print(f"[3/4] Writing {tiles:,} tiles + PlaySound events ({args.mode})...")
            events, volume_sum = write_playsound_chart(
                output,
                pcm,
                settings,
                tiles=tiles,
                tile_len=tile_len,
                start_floor=start_floor,
                hitsound=args.hitsound,
                volume_max=args.volume_max,
                scale=scale,
                gain=args.gain,
                env=env,
                gate=gate,
                compact=args.compact_events,
            )
            del pcm
            print("[4/4] Creating silent background OGG...")
            create_silence(silence, duration + args.tile_ms / 1000.0 * (start_floor + 1))
            size = output.stat().st_size
            print()
            print(f"duration   : {duration:.3f} s")
            print(f"sample rate: {args.rate:,} Hz")
            print(f"tiles      : {tiles:,} (BPM {bpm:.10g})")
            print(f"PlaySound  : {events:,} ({events / duration:,.0f}/s, "
                  f"avg volume {volume_sum / max(events, 1):.1f})")
            print(f"file size  : {size / 1e6:,.1f} MB")
            print(f"chart      : {output}")
            print(f"silence    : {silence}")
            return 0

        density_scale = 1.0
        if args.mode == "adaptive":
            block = max(1, round(args.rate * args.env_ms / 1000.0))
            env = compute_envelope(pcm, scale=scale, gain=args.gain, block=block)
            density_scale = args.max_density
            if args.max_tiles is not None:
                base = expected_pulses(
                    pcm, scale=scale, gain=args.gain, env=env, gate=gate
                )
                if base * density_scale > args.max_tiles:
                    density_scale = args.max_tiles / base
                    print(f"      --max-tiles: density scaled to {density_scale:.4g}")

        print(f"[3/5] Encoding 1-bit PDM pulses ({args.mode})...")
        floors = encode_pdm_hits(
            pcm,
            hit_path,
            scale=scale,
            gain=args.gain,
            env=env,
            gate=gate,
            density_scale=density_scale,
        )
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
    print(f"tiles      : {floors:,} ({floors / duration:,.0f}/s)")
    print(f"SetSpeed   : {actions:,}")
    print(f"max BPM    : {60 * args.rate:,.0f}")
    print(f"chart      : {output}")
    print(f"silence    : {silence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

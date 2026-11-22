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
DEFAULT_MAX_TILES = 100_000
DEFAULT_ENV_MS = 10.0
DEFAULT_GATE_DB = -60.0
PCM_CHUNK_SAMPLES = 1_000_000
ACTION_CHUNK = 50_000
ANGLE_BLOCK = 200_000


def run_ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def decode_mono_f32(
    source: Path, raw_path: Path, rate: int, highpass: float = 0.0, lowpass: float = 0.0
) -> int:
    # Each filter is applied twice: two 2-pole stages = 4th order (24 dB/oct).
    chain = []
    if highpass > 0:
        chain += [f"highpass=f={highpass:g}"] * 2
    if lowpass > 0:
        chain += [f"lowpass=f={lowpass:g}"] * 2
    filters = ["-af", ",".join(chain)] if chain else []
    run_ffmpeg([
        "-i", str(source),
        "-ac", "1",
        "-ar", str(rate),
        *filters,
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


def drop_isolated_hits(
    hit_path: Path, vol_path: Path | None, count: int, max_gap: int
) -> int:
    """Remove hits that are more than max_gap samples away from both neighbours.

    Such a pulse repeats slower than the lowest audible frequency, so it is
    heard as a lone click instead of part of a tone. Rewrites the files in
    place and returns the new hit count.
    """
    if count == 0:
        return 0
    hits = np.fromfile(hit_path, dtype="<i8", count=count)
    before = np.empty(count, dtype=np.int64)
    after = np.empty(count, dtype=np.int64)
    before[0] = hits[0] + 1  # time 0 counts as the previous neighbour
    before[1:] = np.diff(hits)
    after[:-1] = before[1:]
    after[-1] = max_gap + 1
    keep = ~((before > max_gap) & (after > max_gap))
    hits[keep].tofile(hit_path)
    if vol_path is not None:
        vols = np.fromfile(vol_path, dtype="<i2", count=count)
        vols[keep].tofile(vol_path)
    return int(keep.sum())


def bpm_for_gap_f(rate: int, gap: float) -> float:
    return 60.0 * rate / gap


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


ANGLE_MIN = 90.0
ANGLE_MAX = 270.0


def angle_path(
    gaps: np.ndarray, base_gap: float, base_angle: float = 180.0
) -> tuple[np.ndarray, np.ndarray]:
    """Turn angle (deg) and speed multiplier for each gap at the base BPM.

    At the base BPM a gap of g samples is 180 * g / base_gap degrees.
    While the angle is < 90 the speed and the angle are doubled; while it is
    > 270 both are halved. Travel time = angle/180 * 60/(bpm*mult) is unchanged.
    """
    theta = base_angle * gaps.astype(np.float64) / base_gap
    mult = np.ones_like(theta)
    small = theta < ANGLE_MIN
    if small.any():
        k = np.ceil(np.log2(ANGLE_MIN / theta[small]))
        mult[small] = 2.0 ** k
    big = theta > ANGLE_MAX
    if big.any():
        k = np.ceil(np.log2(theta[big] / ANGLE_MAX))
        mult[big] = 2.0 ** -k
    theta = theta * mult
    # Guard against log2 rounding at the boundaries.
    for _ in range(2):
        lo = theta < ANGLE_MIN
        theta[lo] *= 2.0
        mult[lo] *= 2.0
        hi = theta > ANGLE_MAX
        theta[hi] *= 0.5
        mult[hi] *= 0.5
    return theta, mult


def alternative_angles(theta: np.ndarray, mult: np.ndarray):
    """Second valid (angle, multiplier) per floor, NaN where there is none.

    90..135 deg can also be played as 180..270 at double speed, and 180..270
    as 90..135 at half speed; 135..180 has only one form.
    """
    alt_t = np.full_like(theta, np.nan)
    alt_m = np.full_like(mult, np.nan)
    lo = theta <= 135.0
    hi = theta >= 180.0
    alt_t[lo], alt_m[lo] = theta[lo] * 2.0, mult[lo] * 2.0
    alt_t[hi], alt_m[hi] = theta[hi] * 0.5, mult[hi] * 0.5
    return alt_t, alt_m


def tile_positions(theta: np.ndarray) -> np.ndarray:
    """Positions of floors 1.. for an angle path (floor 0 at the origin)."""
    a = np.radians(np.concatenate([[0.0], np.cumsum(180.0 - theta)]))
    return np.cumsum(np.c_[np.cos(a), np.sin(a)], axis=0)


def overlap_ratio(theta: np.ndarray) -> float:
    """Fraction of floors that land on a grid cell another floor already used."""
    pos = tile_positions(theta)
    cells = np.round(pos).astype(np.int64)
    key = cells[:, 0] * 4_294_967_296 + cells[:, 1]
    return 1.0 - len(np.unique(key)) / max(1, len(key))


STEER_GRID_BITS = 12          # hashed occupancy grid 4096 x 4096 (wraps around)
STEER_RECENT = 4              # the last few floors never count as overlap
STEER_SPEED_PENALTY = 0.05    # cost of one extra SetSpeed vs. one overlap
STEER_SPREAD = 0.0            # reward per tile of distance from the start


def steer_angles(
    theta: np.ndarray,
    mult: np.ndarray,
    depth: int,
    progress: bool = True,
    spread: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick, at every floor that has two valid angles, the one whose future is
    least overlapping: all 2^depth combinations of the next `depth` choice
    floors are tried (the look-ahead is the cut-off) and the first half of the
    best combination is committed before searching again.

    Cost = floors landing on an already-used cell (ignoring the last few
    floors) + a small penalty per speed change.
    """
    spread_w = STEER_SPREAD if spread is None else spread
    theta = theta.copy()
    mult = mult.copy()
    alt_t, alt_m = alternative_angles(theta, mult)
    choices = np.flatnonzero(~np.isnan(alt_t))
    n = len(theta)
    if not len(choices) or depth <= 0:
        return theta, mult

    size = 1 << STEER_GRID_BITS
    mask = size - 1
    grid = np.full((size, size), -1_000_000, dtype=np.int64)
    combos = {}

    # State before gap i: heading of floor i, position of floor i+1.
    heading = 0.0
    pos = np.array([1.0, 0.0])
    grid[0, 0] = 0
    grid[1, 0] = 1
    prev_logm = 0.0
    done = 0  # next gap index not yet committed
    ci = 0
    commit = max(1, depth // 2)
    total = len(choices)
    next_report = 0

    def walk(g0: int, g1: int) -> None:
        nonlocal heading, pos, prev_logm
        if g1 <= g0:
            return
        hd = heading + np.cumsum(180.0 - theta[g0:g1])
        r = np.radians(hd)
        pts = pos + np.cumsum(np.c_[np.cos(r), np.sin(r)], axis=0)
        cells = np.round(pts).astype(np.int64) & mask
        grid[cells[:, 0], cells[:, 1]] = np.arange(g0 + 2, g1 + 2)
        heading = float(np.mod(hd[-1], 360.0))
        pos = pts[-1]
        prev_logm = float(np.log2(mult[g1 - 1]))

    while ci < total:
        if progress and ci >= next_report:
            print(f"      steering {ci / total:6.1%}", end="\r", flush=True)
            next_report = ci + max(1, total // 200)
        c = choices[ci:ci + depth]
        k = len(c)
        g0 = int(c[0])
        g1 = min(n, int(c[-1]) + 1 + STEER_RECENT)
        walk(done, g0)
        done = g0

        if k not in combos:
            combos[k] = ((np.arange(1 << k)[:, None] >> np.arange(k)) & 1).astype(bool)
        bits = combos[k]
        L = g1 - g0
        th = np.broadcast_to(theta[g0:g1], (len(bits), L)).copy()
        lm = np.broadcast_to(np.log2(mult[g0:g1]), (len(bits), L)).copy()
        cols = c - g0
        th[:, cols] = np.where(bits, alt_t[c], theta[c])
        lm[:, cols] = np.where(bits, np.log2(alt_m[c]), np.log2(mult[c]))

        hd = heading + np.cumsum(180.0 - th, axis=1)
        r = np.radians(hd)
        pts = pos + np.stack(
            [np.cumsum(np.cos(r), axis=1), np.cumsum(np.sin(r), axis=1)], axis=2
        )
        cells = np.round(pts).astype(np.int64) & mask
        last = grid[cells[..., 0], cells[..., 1]]
        idx = np.arange(g0 + 2, g1 + 2)
        overlaps = ((idx - last) > STEER_RECENT).sum(axis=1)
        changes = (np.diff(np.c_[np.full(len(bits), prev_logm), lm], axis=1) != 0).sum(axis=1)
        spread = np.hypot(pts[:, -1, 0], pts[:, -1, 1])
        best = int(np.argmin(
            overlaps + STEER_SPEED_PENALTY * changes - spread_w * spread
        ))

        m = min(commit, k)
        pick = bits[best, :m]
        sel = c[:m][pick]
        theta[sel], mult[sel] = alt_t[sel], alt_m[sel]
        g_end = int(c[m]) if m < k else int(c[-1]) + 1
        walk(g0, g_end)
        done = g_end
        ci += m

    walk(done, n)
    if progress:
        print(" " * 40, end="\r")
    return theta, mult


# Triangular lattice: direction d (0..5) = 60*d degrees; axial unit steps.
LATTICE_DIRS = np.array([(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)], dtype=np.int64)
LATTICE_SAMPLES_PER_BEAT = 3   # 1 sample = 60 deg at the base BPM
LATTICE_OUTSIDE_COST = 3.0     # per floor outside the target shape
LATTICE_TWIRL_COST = 0.05      # per Twirl event


def lattice_plan(gaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn k (in 60 deg units, 1..5) and speed multiplier per gap.

    At the base BPM one sample is 60 deg, so gaps of 1..5 samples are played
    as k = gap at the base speed. Longer gaps go straight (k = 3) with the speed
    lowered to 3/gap, which keeps every floor on the lattice.
    """
    g = gaps.astype(np.int64)
    k = np.where(g <= 5, g, 3).astype(np.int64)
    mult = np.where(g <= 5, 1.0, 3.0 / np.maximum(g, 1))
    return k, mult


def koch_order(radius: float) -> int:
    """Deepest order whose smallest edge is still >= ~4 lattice steps."""
    side = radius * math.sqrt(3)
    return max(0, int(math.floor(math.log(max(side / 4.0, 1.0), 3))))


def koch_polygon(radius: float, order: int) -> np.ndarray:
    """Koch snowflake vertices (x, y), circumradius of the base triangle = radius."""
    ang = np.radians([90.0, 210.0, 330.0])
    pts = np.c_[radius * np.cos(ang), radius * np.sin(ang)]
    rot = np.array([[0.5, math.sqrt(3) / 2], [-math.sqrt(3) / 2, 0.5]])  # -60 deg
    for _ in range(order):
        a = pts
        b = np.roll(pts, -1, axis=0)
        d = (b - a) / 3.0
        p1 = a + d
        p3 = a + 2 * d
        p2 = p1 + d @ rot.T  # bump outward for clockwise-ordered vertices
        pts = np.stack([a, p1, p2, p3], axis=1).reshape(-1, 2)
    return pts


class _LatticeMask:
    """Precomputed inside/outside table on the lattice (scanline fill)."""

    def __init__(self, poly: np.ndarray) -> None:
        h = math.sqrt(3) / 2
        ymin, ymax = poly[:, 1].min(), poly[:, 1].max()
        xmin, xmax = poly[:, 0].min(), poly[:, 0].max()
        r0 = int(math.floor(ymin / h)) - 1
        r1 = int(math.ceil(ymax / h)) + 1
        q0 = int(math.floor(xmin - r1 / 2)) - 2
        q1 = int(math.ceil(xmax - r0 / 2)) + 2
        self.q0, self.r0 = q0, r0
        self.mask = np.zeros((q1 - q0 + 1, r1 - r0 + 1), dtype=bool)
        a = poly
        b = np.roll(poly, -1, axis=0)
        qs = np.arange(q0, q1 + 1)
        for r in range(r0, r1 + 1):
            y = r * h
            cross = ((a[:, 1] <= y) & (b[:, 1] > y)) | ((b[:, 1] <= y) & (a[:, 1] > y))
            if not cross.any():
                continue
            aa, bb = a[cross], b[cross]
            xs = np.sort(aa[:, 0] + (y - aa[:, 1]) * (bb[:, 0] - aa[:, 0]) / (bb[:, 1] - aa[:, 1]))
            x = qs + 0.5 * r
            self.mask[:, r - r0] = (np.searchsorted(xs, x) % 2) == 1

    def contains(self, q: np.ndarray, r: np.ndarray) -> np.ndarray:
        i = q - self.q0
        j = r - self.r0
        ok = (i >= 0) & (i < self.mask.shape[0]) & (j >= 0) & (j < self.mask.shape[1])
        out = np.zeros(np.shape(q), dtype=bool)
        out[ok] = self.mask[i[ok], j[ok]]
        return out


_MASKS: dict = {}


def in_shape(q: np.ndarray, r: np.ndarray, shape: str, radius: float) -> np.ndarray:
    if shape == "koch":
        key = (shape, round(radius, 6))
        if key not in _MASKS:
            _MASKS[key] = _LatticeMask(koch_polygon(radius, koch_order(radius)))
        return _MASKS[key].contains(np.asarray(q), np.asarray(r))
    x = q + 0.5 * r
    y = r * (math.sqrt(3) / 2)
    if shape == "circle":
        return x * x + y * y <= radius * radius
    h = radius / 2
    c30, s30 = math.sqrt(3) / 2, 0.5
    up = (-y <= h) & (c30 * x + s30 * y <= h) & (-c30 * x + s30 * y <= h)
    down = (y <= h) & (c30 * x - s30 * y <= h) & (-c30 * x - s30 * y <= h)
    return up | down


def shape_radius(n: int, shape: str, fill: float) -> float:
    """Radius whose shape holds about n / fill lattice points."""
    pts = n / max(fill, 1e-6)
    if shape == "circle":  # lattice density 2/sqrt(3) per unit area
        return math.sqrt(pts * math.sqrt(3) / (2 * math.pi))
    if shape == "koch":  # area 8/5 of the base triangle -> 2.4 R^2 points
        return math.sqrt(pts / 2.4)
    return math.sqrt(pts / 2)  # hexagram area sqrt(3) R^2 -> 2 R^2 points


def steer_lattice(
    k: np.ndarray, depth: int, shape: str, radius: float, progress: bool = True
) -> np.ndarray:
    """Choose where to Twirl so the lattice walk stays inside the shape and
    avoids revisiting cells. Twirling flips the rotation, which turns k into
    6 - k with the same timing. All 2^depth toggle combinations of the next
    `depth` turning floors are simulated; the best first half is committed.

    Returns a bool array: Twirl on the floor of gap i.
    """
    n = len(k)
    twirl = np.zeros(n, dtype=bool)
    choices = np.flatnonzero(k != 3)
    if not len(choices) or depth <= 0:
        return twirl

    bits_n = STEER_GRID_BITS
    mask = (1 << bits_n) - 1
    grid = np.full((1 << bits_n, 1 << bits_n), -1_000_000, dtype=np.int64)
    combos = {}
    turn3 = 3 - k  # direction change for clockwise rotation

    # State before gap i: direction of floor i, cell of floor i+1, rotation.
    d = 0
    cell = LATTICE_DIRS[0].copy()
    spin = 1
    grid[0, 0] = 0
    grid[cell[0] & mask, cell[1] & mask] = 1
    done = 0
    ci = 0
    total = len(choices)
    commit = max(1, depth // 2)
    next_report = 0

    def walk(g0: int, g1: int) -> None:
        nonlocal d, cell, spin
        if g1 <= g0:
            return
        flips = np.cumsum(twirl[g0:g1])
        sp = spin * np.where(flips % 2 == 1, -1, 1)
        dirs = np.mod(d + np.cumsum(sp * turn3[g0:g1]), 6)
        pts = cell + np.cumsum(LATTICE_DIRS[dirs], axis=0)
        grid[pts[:, 0] & mask, pts[:, 1] & mask] = np.arange(g0 + 2, g1 + 2)
        d = int(dirs[-1])
        cell = pts[-1]
        spin = int(sp[-1])

    while ci < total:
        if progress and ci >= next_report:
            print(f"      steering {ci / total:6.1%}", end="\r", flush=True)
            next_report = ci + max(1, total // 200)
        c = choices[ci:ci + depth]
        kk = len(c)
        g0 = int(c[0])
        g1 = min(n, int(c[-1]) + 1 + STEER_RECENT)
        walk(done, g0)
        done = g0
        if kk not in combos:
            combos[kk] = ((np.arange(1 << kk)[:, None] >> np.arange(kk)) & 1).astype(bool)
        bits = combos[kk]
        L = g1 - g0
        tog = np.zeros((len(bits), L), dtype=np.int64)
        tog[:, c - g0] = bits
        sp = spin * np.where(np.cumsum(tog, axis=1) % 2 == 1, -1, 1)
        dirs = np.mod(d + np.cumsum(sp * turn3[g0:g1], axis=1), 6)
        steps = LATTICE_DIRS[dirs]
        q = cell[0] + np.cumsum(steps[..., 0], axis=1)
        r = cell[1] + np.cumsum(steps[..., 1], axis=1)
        last = grid[q & mask, r & mask]
        idx = np.arange(g0 + 2, g1 + 2)
        overlaps = ((idx - last) > STEER_RECENT).sum(axis=1)
        outside = (~in_shape(q, r, shape, radius)).sum(axis=1)
        cost = overlaps + LATTICE_OUTSIDE_COST * outside + LATTICE_TWIRL_COST * bits.sum(axis=1)
        best = int(np.argmin(cost))
        m = min(commit, kk)
        twirl[c[:m][bits[best, :m]]] = True
        g_end = int(c[m]) if m < kk else int(c[-1]) + 1
        walk(g0, g_end)
        done = g_end
        ci += m

    walk(done, n)
    if progress:
        print(" " * 40, end="\r")
    return twirl


def lattice_angles(k: np.ndarray, twirl: np.ndarray) -> np.ndarray:
    """Absolute angleData (multiples of 60) for floors 0..n (n = len(k))."""
    spin = np.where(np.cumsum(twirl) % 2 == 1, -1, 1)
    dirs = np.mod(np.cumsum(spin * (3 - k)), 6)
    return np.concatenate([[0], dirs * 60]).astype(np.int64)


def lattice_cells(k: np.ndarray, twirl: np.ndarray) -> np.ndarray:
    dirs = lattice_angles(k, twirl) // 60
    return np.cumsum(LATTICE_DIRS[dirs], axis=0)


def write_angles(f, n: int, theta: np.ndarray | None, absolute: np.ndarray | None = None) -> None:
    """angleData: floor 0 plus one entry per hit floor (len(hits) values).

    Straight path: all zeros. Angle path: the game computes the turn on floor j
    as mod(a[j-1] - a[j] - 180, 360) (exit = 90 - angleData, entry = previous
    exit + 180, clockwise), so a[j] = a[j-1] + 180 - theta[j].
    """
    if absolute is not None:
        for c in range(0, n, ANGLE_BLOCK):
            if c:
                f.write(",")
            f.write(",".join(map(str, absolute[c:c + ANGLE_BLOCK].tolist())))
        return
    if theta is None:
        write_zero_angles(f, n)
        return
    f.write("0")
    carry = 0.0
    for start in range(0, n - 1, PCM_CHUNK_SAMPLES):
        end = min(start + PCM_CHUNK_SAMPLES, n - 1)
        a = np.mod(carry + np.cumsum(180.0 - theta[start:end]), 360.0)
        carry = float(a[-1])
        for c in range(0, len(a), ANGLE_BLOCK):
            f.write(",")
            f.write(",".join("%.4f" % x for x in a[c:c + ANGLE_BLOCK].tolist()))


def write_actions(
    f,
    hits: np.memmap,
    rate: int,
    *,
    volumes: np.memmap | None = None,
    hitsound: str = "Sidestick",
    base_gap: float | None = None,
    mult: np.ndarray | None = None,
    twirl: np.ndarray | None = None,
) -> tuple[int, int]:
    """Write SetSpeed (and optionally SetHitsound) events for a hit timeline.

    Floor j (j >= 1) is the hit at sample hits[j-1]. In ADOFAI the speed of
    floor j governs the travel floor j -> j+1 (scrLevelMaker
    .CalculateFloorEntryTimes), so the gap hits[j] - hits[j-1] goes on floor j.
    A SetHitsound on floor j sets the volume of the hit *on* floor j
    (scrConductor.PlayHitTimes), so volumes[j-1] goes on floor j.
    Events are only written where the value changes.
    """
    n = len(hits)
    speed_count = 0
    volume_count = 0
    first = True
    base_bpm = 60.0 * rate / base_gap if base_gap is not None else None
    prev_speed = base_bpm if base_bpm is not None else -1.0
    prev_vol = int(volumes[0]) if volumes is not None and n else -1
    speed_fmt = (
        '{"floor":%d,"eventType":"SetSpeed","speedType":"Bpm",'
        '"beatsPerMinute":%.12g,"bpmMultiplier":1,"angleOffset":0}'
    )
    vol_fmt = (
        '{"floor":%d,"eventType":"SetHitsound","gameSound":"Hitsound","hitsound":'
        + json.dumps(hitsound)
        + ',"hitsoundVolume":%d}'
    )

    for start in range(0, n, PCM_CHUNK_SAMPLES):
        end = min(start + PCM_CHUNK_SAMPLES, n)
        floors = np.arange(start + 1, end + 1, dtype=np.int64)  # floor of each hit

        # Speed for floor j (travel to j+1): hits[j] - hits[j-1], for j < n.
        h = np.asarray(hits[start:min(end + 1, n)], dtype=np.int64)
        gaps = np.diff(h)  # gaps[i] belongs to floor start+1+i
        if base_bpm is None:
            speeds = 60.0 * rate / gaps
        else:
            speeds = base_bpm * mult[start:start + len(gaps)]
        s_prev = np.concatenate([[prev_speed], speeds[:-1]]) if len(speeds) else speeds
        sp_idx = np.flatnonzero(speeds != s_prev)
        if len(speeds):
            prev_speed = float(speeds[-1])

        if volumes is not None:
            v = np.asarray(volumes[start:end], dtype=np.int64)
            v_prev = np.concatenate([[prev_vol], v[:-1]])
            vo_idx = np.flatnonzero(v != v_prev)
            prev_vol = int(v[-1])
        else:
            vo_idx = np.empty(0, dtype=np.int64)
        tw_idx = (
            np.flatnonzero(twirl[start:start + len(gaps)])
            if twirl is not None else np.empty(0, dtype=np.int64)
        )

        for c0 in range(0, len(floors), ACTION_CHUNK):
            c1 = c0 + ACTION_CHUNK
            sp = sp_idx[(sp_idx >= c0) & (sp_idx < c1)]
            vo = vo_idx[(vo_idx >= c0) & (vo_idx < c1)]
            items = [(int(i), 0, speed_fmt % (floors[i], speeds[i]))
                     for i in sp.tolist()]
            items += [(int(i), 1, vol_fmt % (floors[i], int(v[i]))) for i in vo.tolist()]
            tw = tw_idx[(tw_idx >= c0) & (tw_idx < c1)]
            items += [(int(i), 2, '{"floor":%d,"eventType":"Twirl"}' % floors[i])
                      for i in tw.tolist()]
            if not items:
                continue
            items.sort()
            if not first:
                f.write(",")
            f.write(",".join(t for _, _, t in items))
            first = False
            speed_count += len(sp)
            volume_count += len(vo)

    return speed_count, volume_count


def write_chart(
    output: Path,
    hits: np.memmap,
    settings: dict,
    rate: int,
    *,
    volumes: np.memmap | None = None,
    hitsound: str = "Sidestick",
    base_gap: float | None = None,
    theta: np.ndarray | None = None,
    mult: np.ndarray | None = None,
    twirl: np.ndarray | None = None,
    absolute: np.ndarray | None = None,
) -> tuple[int, int]:
    with output.open("w", encoding="utf-8", buffering=1024 * 1024) as f:
        f.write('{"angleData":[')
        write_angles(f, len(hits), theta, absolute)  # floor 0 + one floor per hit
        f.write('],"settings":')
        f.write(json.dumps(settings, ensure_ascii=False, separators=(",", ":")))
        f.write(',"actions":[')
        counts = write_actions(
            f, hits, rate, volumes=volumes, hitsound=hitsound, base_gap=base_gap,
            mult=mult, twirl=twirl,
        )
        f.write('],"decorations":[]}')
    return counts


def encode_multilevel_hits(
    pcm: np.memmap,
    hit_path: Path,
    vol_path: Path,
    *,
    volume_max: int,
    scale: float,
    gain: float,
    env: "Envelope | None",
    gate: float,
) -> int:
    """Multi-level first-order noise-shaped quantization to integer volumes.

    Writes the sample index (int64) and volume (int16) of every non-zero
    sample. Silence (volume 0) produces no hit.
    """
    total = 0
    acc = 0.0
    prev_level = 0
    with hit_path.open("wb") as hout, vol_path.open("wb") as vout:
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
            if len(idx):
                (idx + start).astype("<i8").tofile(hout)
                volume[idx].astype("<i2").tofile(vout)
                total += len(idx)
    return total


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
    play_duration: int = 0,
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
    # playDuration (ms, int) makes the game call SetScheduledEndTime, so each
    # voice dies after that time. Concurrent voices ~= events/s * duration;
    # with the full clip length thousands of AudioSources overlap and Unity
    # culls most of them.
    if compact:
        head = (
            '{"floor":%d,"eventType":"PlaySound","hitsound":'
            + json.dumps(hitsound)
            + (',"playDuration":%d' % play_duration if play_duration else "")
            + ',"hitsoundVolume":%d,"angleOffset":%.7g}'
        )
    else:
        head = (
            '{"floor":%d,"eventType":"PlaySound","hitsound":'
            + json.dumps(hitsound)
            + ',"offset":0,"playDuration":%d,"pitch":100' % play_duration
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


def install_silence(path: Path, duration: float, tmpdir: Path) -> None:
    """Render the silent OGG in tmpdir, then copy it over the target.

    The target is often still open in ADOFAI / an editor (it is the chart's
    song file), which locks it on Windows. That must not throw away a chart
    that took minutes to write, so a locked target only produces a warning.
    """
    tmp = tmpdir / "silence.ogg"
    create_silence(tmp, duration)
    try:
        shutil.copyfile(tmp, path)
    except PermissionError:
        print()
        print(f"WARNING: could not write {path} (it is probably open in ADOFAI or an editor).")
        if path.exists():
            print("         Keeping the existing file; it may be shorter than this chart.")
        else:
            print("         The chart has no song file yet.")
        print("         Close the program and run again, or use a different -o name.")


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
        choices=("tiles", "playsound", "pdm"),
        default="pdm",
        help="pdm: one tile per 1-bit pulse, constant hitsound volume (default). "
        "tiles: one tile per sample, volume via SetHitsound (multi-level). "
        "playsound: few tiles, audio in PlaySound events (needs a renderer that "
        "handles PlaySound)",
    )
    p.add_argument(
        "--min-freq",
        type=float,
        default=20.0,
        help="Lowest audible frequency. The input is high-passed (4th order) here, "
        "and pdm/tiles pulses farther than 1/min-freq s from both neighbours "
        "are dropped. 0 = off",
    )
    p.add_argument(
        "--max-freq",
        type=float,
        default=20000.0,
        help="Highest audible frequency. The input is low-passed (4th order) here; "
        "skipped when >= rate/2 (the resampler already band-limits). 0 = off",
    )
    p.add_argument(
        "--path",
        choices=("angle", "lattice", "straight"),
        default="angle",
        help="angle: constant average BPM, gaps encoded as turn angles kept in "
        "90-270 deg by doubling/halving the speed. straight: all-straight route, "
        "every gap change is a SetSpeed (pdm/tiles engines)",
    )
    p.add_argument(
        "--steer-depth",
        type=int,
        default=4,
        help="angle path: look-ahead depth (choice floors) when choosing between "
        "the two valid angle/speed forms to avoid overlaps; tries 2^depth "
        "combinations per step. 0 = off",
    )
    p.add_argument(
        "--shape",
        choices=("hexagram", "circle", "koch"),
        default="hexagram",
        help="lattice path: region the Twirl steering keeps the path inside",
    )
    p.add_argument(
        "--fill",
        type=float,
        default=3.0,
        help="lattice path: tiles per lattice point inside the shape (sets its size; "
        "higher = smaller, denser shape with a crisper outline)",
    )
    p.add_argument(
        "--base-angle",
        type=float,
        default=180.0,
        help="angle path: turn angle (deg) of an average-length gap at the base BPM. "
        "180 = average gap goes straight; e.g. 120 curls into loops, 240 curls the "
        "other way. Timing is unchanged",
    )
    p.add_argument(
        "--steer-spread",
        type=float,
        default=0.0,
        help="angle path: reward for moving away from the start while steering. "
        "Even 0.002 turns the path into a near-straight line; 0 keeps it a cloud",
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
        "--play-duration",
        type=int,
        default=1,
        help="playsound: cut each PlaySound after this many ms (0 = full clip). "
        "Keeps concurrent voices ~= events/s * duration below Unity's voice limit",
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
        default=DEFAULT_MAX_TILES,
        help=f"pdm: lower the density so the chart has at most ~this many tiles "
        f"(default {DEFAULT_MAX_TILES:,}; 0 = no limit)",
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
    p.add_argument(
        "--silence-name",
        help="Silent OGG file name (default: <chart name>_silence.ogg)",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.rate is None:
        args.rate = DEFAULT_PDM_RATE if args.engine == "pdm" else DEFAULT_PLAYSOUND_RATE

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
    if args.max_tiles < 0:
        raise SystemExit("--max-tiles must be >= 0")
    if args.max_tiles == 0 or args.engine != "pdm":
        args.max_tiles = None
    if args.env_ms <= 0:
        raise SystemExit("--env-ms must be positive")
    if args.tile_ms <= 0:
        raise SystemExit("--tile-ms must be positive")
    if args.volume_max <= 0:
        raise SystemExit("--volume-max must be positive")
    if args.volume_max > 100:
        raise SystemExit("--volume-max must be <= 100 (AudioSource volume is clamped to 1)")
    if args.play_duration < 0:
        raise SystemExit("--play-duration must be >= 0")

    output = args.output or args.input.with_name(args.input.stem + "_sound_chart.adofai")
    if output.suffix.lower() != ".adofai":
        output = output.with_suffix(".adofai")
    output.parent.mkdir(parents=True, exist_ok=True)
    silence = output.parent / (args.silence_name or f"{output.stem}_silence.ogg")

    if not args.force:
        existing = [p for p in (output, silence) if p.exists()]
        if existing:
            names = ", ".join(str(p) for p in existing)
            raise SystemExit(f"Output already exists: {names} (use --force to overwrite)")

    steps = 4 if args.engine == "playsound" else 5
    vol_path = None

    with tempfile.TemporaryDirectory(prefix="adofai_sound_chart_") as tmp:
        tmpdir = Path(tmp)
        raw_path = tmpdir / "audio.f32le"
        hit_path = tmpdir / "hits.i64"

        band = []
        if args.min_freq > 0:
            band.append(f"high-pass {args.min_freq:g} Hz")
        if 0 < args.max_freq < args.rate / 2:
            band.append(f"low-pass {args.max_freq:g} Hz")
        band_text = (", " + ", ".join(band)) if band else ""
        print(f"[1/{steps}] Decoding mono PCM at {args.rate:,} Hz{band_text}...")
        lowpass = args.max_freq if 0 < args.max_freq < args.rate / 2 else 0.0
        sample_count = decode_mono_f32(
            args.input, raw_path, args.rate, args.min_freq, lowpass
        )
        if sample_count == 0:
            raise SystemExit("Decoded audio is empty")
        duration = sample_count / args.rate
        pcm = np.memmap(raw_path, dtype="<f4", mode="r", shape=(sample_count,))

        print(f"[2/{steps}] Estimating normalization...")
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
                play_duration=args.play_duration,
            )
            del pcm
            print("[4/4] Creating silent background OGG...")
            install_silence(silence, duration + args.tile_ms / 1000.0 * (start_floor + 1), tmpdir)
            size = output.stat().st_size
            print()
            print(f"duration   : {duration:.3f} s")
            print(f"sample rate: {args.rate:,} Hz")
            print(f"tiles      : {tiles:,} (BPM {bpm:.10g})")
            print(f"PlaySound  : {events:,} ({events / duration:,.0f}/s, "
                  f"avg volume {volume_sum / max(events, 1):.1f})")
            if args.play_duration:
                print(f"voices     : ~{events / duration * args.play_duration / 1000:,.1f} "
                      f"concurrent (playDuration {args.play_duration} ms)")
            print(f"file size  : {size / 1e6:,.1f} MB")
            print(f"chart      : {output}")
            print(f"silence    : {silence}")
            return 0

        density_scale = 1.0
        if args.mode == "adaptive":
            block = max(1, round(args.rate * args.env_ms / 1000.0))
            env = compute_envelope(pcm, scale=scale, gain=args.gain, block=block)
            density_scale = args.max_density
            if args.max_tiles is not None and args.engine == "pdm":
                base = expected_pulses(
                    pcm, scale=scale, gain=args.gain, env=env, gate=gate
                )
                if base * density_scale > args.max_tiles:
                    density_scale = args.max_tiles / base
                    print(f"      --max-tiles: density scaled to {density_scale:.4g}")

        if args.engine == "tiles":
            vol_path = tmpdir / "vols.i16"
            print(f"[3/5] Encoding multi-level hits ({args.mode})...")
            floors = encode_multilevel_hits(
                pcm,
                hit_path,
                vol_path,
                volume_max=args.volume_max,
                scale=scale,
                gain=args.gain,
                env=env,
                gate=gate,
            )
        else:
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

        if args.min_freq > 0 and floors:
            max_gap = int(args.rate / args.min_freq)
            kept = drop_isolated_hits(hit_path, vol_path, floors, max_gap)
            if kept < floors:
                print(f"      dropped {floors - kept:,} isolated pulses "
                      f"(> {max_gap:,} samples from both neighbours, < {args.min_freq:g} Hz)")
            floors = kept
            if floors == 0:
                raise SystemExit("No pulses left after --min-freq filtering")

        hits = np.memmap(hit_path, dtype="<i8", mode="r", shape=(floors,))
        volumes = (
            np.memmap(vol_path, dtype="<i2", mode="r", shape=(floors,))
            if vol_path is not None else None
        )
        initial_gap = max(1, int(hits[0]))
        initial_bpm = bpm_for_gap(args.rate, initial_gap)
        base_gap = None
        theta = mult = twirl = absolute = None
        if args.path == "lattice":
            base_gap = float(LATTICE_SAMPLES_PER_BEAT)
            initial_bpm = bpm_for_gap_f(args.rate, base_gap)
            k, mult = lattice_plan(np.diff(np.asarray(hits, dtype=np.int64)))
            radius = shape_radius(floors, args.shape, args.fill)
            depth = args.steer_depth if args.steer_depth > 0 else 0
            extra = f", Koch order {koch_order(radius)}" if args.shape == "koch" else ""
            print(f"[4/5] Steering lattice path inside a {args.shape} "
                  f"(radius {radius:,.0f}{extra}, depth {depth})...")
            twirl = steer_lattice(k, depth, args.shape, radius)
            absolute = lattice_angles(k, twirl)
            cells = lattice_cells(k, twirl)
            key = cells[:, 0] * 4_294_967_296 + cells[:, 1]
            inside = in_shape(cells[:, 0], cells[:, 1], args.shape, radius).mean()
            print(f"      overlapping floors: {1 - len(np.unique(key)) / len(key):.1%}, "
                  f"inside shape: {inside:.1%}, Twirl: {int(twirl.sum()):,}")
        elif args.path == "angle":
            avg_gap = (
                (int(hits[-1]) - int(hits[0])) / (floors - 1) if floors > 1 else float(initial_gap)
            )
            # An average gap turns base_angle degrees, so one beat (180 deg)
            # lasts avg_gap * 180 / base_angle samples.
            base_gap = avg_gap * 180.0 / args.base_angle
            initial_bpm = bpm_for_gap_f(args.rate, base_gap)
            theta, mult = angle_path(np.diff(np.asarray(hits, dtype=np.int64)), base_gap)
            if args.steer_depth > 0 and len(theta):
                before = overlap_ratio(theta)
                print(f"[4/5] Steering path (depth {args.steer_depth}, "
                      f"2^{args.steer_depth} futures per step)...")
                theta, mult = steer_angles(
                    theta, mult, args.steer_depth, spread=args.steer_spread
                )
                print(f"      overlapping floors: {before:.1%} -> {overlap_ratio(theta):.1%}")
        first_volume = int(volumes[0]) if volumes is not None else 100

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
            hitsound_volume=first_volume,
            description=(
                f"{args.rate} Hz {'multi-level SetHitsound' if volumes is not None else '1-bit PDM'} "
                f"Sound Charting reconstruction, {floors:,} tiles ({args.path} path). "
                "Background audio is silent."
            ),
        )

        print(f"[4/5] Writing {floors:,} tiles ({args.path} path)...")
        speed_events, volume_events = write_chart(
            output, hits, settings, args.rate, volumes=volumes, hitsound=args.hitsound,
            base_gap=base_gap,
            theta=theta,
            mult=mult,
            twirl=twirl,
            absolute=absolute,
        )
        del hits, volumes

        print("[5/5] Creating silent background OGG...")
        install_silence(silence, duration, tmpdir)

    print()
    print(f"duration   : {duration:.3f} s")
    print(f"sample rate: {args.rate:,} Hz")
    print(f"tiles      : {floors:,} ({floors / duration:,.0f}/s)")
    print(f"SetSpeed   : {speed_events:,}")
    if args.engine == "tiles":
        print(f"SetHitsound: {volume_events:,}")
    print(f"file size  : {output.stat().st_size / 1e6:,.1f} MB")
    if base_gap is not None:
        print(f"base BPM   : {60 * args.rate / base_gap:,.3f} "
              f"(1 beat = {base_gap:.4f} samples, avg gap = {args.base_angle:g} deg)")
    else:
        print(f"max BPM    : {60 * args.rate:,.0f}")
    print(f"chart      : {output}")
    print(f"silence    : {silence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

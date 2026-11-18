# ADOFAI Sound Charting

Turn an audio waveform into an absurdly dense A Dance of Fire and Ice level whose **hitsounds reconstruct the source audio**.

The generated chart uses a silent background track. If hitsounds are disabled, it becomes silent.

## How it works

```text
Audio file
  -> mono PCM
  -> normalize / soft-clip
  -> 1-bit pulse-density modulation (PDM)
  -> pulse positions
  -> straight ADOFAI tiles
  -> SetSpeed events encode pulse timing
  -> hitsounds reconstruct the waveform
```

For a sample rate `Fs`, each PDM pulse becomes a tile. If two pulses are `g` samples apart, a straight 180-degree tile interval is encoded as:

```text
BPM = 60 * Fs / g
```

At 44.1 kHz, the maximum BPM for a one-sample gap is therefore:

```text
2,646,000 BPM
```

The route is intentionally a perfectly straight line (`angleData` is all zeroes). The audio is carried by **hit timing**, not by geometric turns.

## Why this is interesting

This is effectively using ADOFAI's hitsound system as a very strange 1-bit DAC.

At sufficiently high sample rates, surprisingly recognizable speech, melodies, bass and distorted kick textures can be reconstructed from millions of hitsounds. In `classic` mode a 3-minute song at 44.1 kHz produces roughly 4 million tiles. The default `adaptive` mode scales tile density with loudness and drops silence, and `--max-tiles` caps the total.

## Requirements

- Python 3.10+
- `numpy`
- `ffmpeg` available on `PATH`
- ADOFAI

Install the Python dependency:

```bash
pip install -r requirements.txt
```

Check FFmpeg:

```bash
ffmpeg -version
```

## Usage

```bash
python sound_chart.py input.ogg -o output.adofai
```

MP3/WAV/FLAC/etc. are also accepted as long as FFmpeg can decode them.

Useful options:

```bash
python sound_chart.py input.mp3 \
  -o output.adofai \
  --rate 44100 \
  --hitsound Sidestick \
  --title "My Sound Chart"
```

For a larger / higher-rate experiment:

```bash
python sound_chart.py input.ogg -o output_66k.adofai --engine pdm --rate 66000
```

To keep the tile count under a limit (e.g. for the vanilla game):

```bash
python sound_chart.py input.ogg -o small.adofai --engine pdm --max-tiles 500000 --gate-db -45
```

The program also creates a silent `.ogg` beside the chart. Keep both files in the same folder.

## Output

For `song.ogg`, the default output is:

```text
song_sound_chart.adofai
song_sound_chart_silence.ogg
```

The generated level contains:

- a perfectly straight route
- one tile per PDM pulse
- `SetSpeed` events when the pulse spacing changes
- a global hitsound (default: `Sidestick`)
- a silent background OGG

Large inputs can create **hundreds of megabytes of JSON** and millions of tiles/events. Loading the result may stress both ADOFAI and third-party editors.

## Engines

### `--path angle` (default) / `--path straight`

`angle`: the whole chart runs at the **average BPM** (`60·Fs / mean gap`) and each
gap is a turn angle `180·g / mean_gap`. If the angle is < 90° the speed and the
angle are doubled, if it is > 270° both are halved (repeated as needed), so every
turn stays in 90–270° and SetSpeed only appears when that ×2ⁿ factor changes.
Turn on floor j follows the game code: `mod(a[j-1] − a[j] − 180, 360)`
(exit = 90 − angleData, entry = previous exit + 180, clockwise), so
`a[j] = a[j-1] + 180 − θ[j]`.

10 s clip, 100k tiles: straight 65,402 SetSpeed / 7.7 MB → angle 37,491 SetSpeed /
5.4 MB, same timing (simulated SNR 6.5 → 6.8 dB).

**Steering** (`--steer-depth`, default 4): floors at 90–135° can also be played
at 180–270° with double speed (and vice versa), same timing. At each such floor
all 2^depth combinations of the next `depth` choice floors are simulated (the
depth is the cut-off) and the first half of the best one is committed. Cost =
floors landing on an already-used cell + 0.05 per speed change
− `--steer-spread` × distance from the start.

10 s clip, 100k tiles (overlap = floors on an already-used grid cell):

| setting | overlap | SetSpeed |
|---|---|---|
| `--steer-depth 0` | 64% | 36,606 |
| default (depth 4) | 48% | 17,838 |
| `--steer-spread 0.002` | 23% | 17,309 (near-straight line) |
| `--steer-spread 0.1` | 7% | 41,714 (straight line) |

`straight`: every tile is 180°, every gap change is a SetSpeed (previous behaviour).

### `--engine tiles`

One straight tile per non-zero sample; the hit **volume** carries the amplitude.

```text
Audio -> mono PCM at --rate (default 11025 Hz)
      -> normalize / soft-clip / envelope bias (silence = 0)
      -> multi-level noise-shaped quantization to volume 0..--volume-max
      -> one tile per non-zero sample
         SetSpeed    on floor j: gap to the next hit   (only when it changes)
         SetHitsound on floor j: volume of the hit on j (only when it changes)
```

Event semantics were checked against the game code: a floor's speed governs the
travel *from* that floor to the next (`scrLevelMaker.CalculateFloorEntryTimes`),
and a `SetHitsound` on floor j sets the volume of the hit on floor j
(`scrConductor.PlayHitTimes`). Tile hitsounds are what renderers such as
HzHitSoundRenderer capture (`scrConductor.hitSoundsData`).

10 s clip, simulated with the game's timing rules (40 Hz–5 kHz):

| engine | --rate | tiles | events | size | SNR |
|---|---|---|---|---|---|
| tiles | 6000 | 59,569 | 57.5k | 6.3 MB | 10.5 dB |
| tiles | 8000 | 79,468 | 75.5k | 8.3 MB | 17.4 dB |
| tiles | 11025 | 109,565 | 101.6k | 11.2 MB | 18.2 dB |
| pdm `--max-tiles 80000` | 44100 | 80,000 | 48.8k | 5.8 MB | 3.9 dB |

Tile count ≈ `--rate` × seconds of non-silent audio, so for a ~100k-tile budget
use about 8 kHz for 10 s.

### `--engine playsound`

Tile count is what makes vanilla ADOFAI choke, so this engine uses **very few
tiles** and carries the audio in `PlaySound` events instead:

```text
Audio -> mono PCM at --rate (default 11025 Hz)
      -> normalize / soft-clip / envelope bias (silence = 0)
      -> multi-level noise-shaped quantization to volume 0..--volume-max
      -> one PlaySound event per non-zero sample
         floor       = 1 + n // tile_len
         angleOffset = 180 * (n % tile_len) / tile_len
```

- Tiles are straight, constant BPM, `--tile-ms` long (default 100 ms → 10 tiles/s,
  ~1,800 tiles for a 3-minute song). No `SetSpeed` events.
- Each event carries its own `hitsoundVolume`, so one event is a multi-level
  sample instead of a 1-bit pulse. That needs far fewer events than PDM for the
  same quality.
- The global tile hitsound volume is set to 0; only PlaySound is audible.
- Needs a hitsound renderer that also captures PlaySound; HzHitSoundRenderer
  currently only renders tile hitsounds.
- Every hit (tile or PlaySound) is a separate Unity `AudioSource.PlayScheduled`;
  the game does not pre-mix hitsounds. With full-length clips thousands of
  voices overlap and most get culled. `--play-duration` (default 1 ms) sets
  PlaySound's `playDuration`, which ends each voice early, so concurrent voices
  ≈ events/s × duration (≈8 at 8 kHz). Tile hitsounds have no such control.
- Events/s ≈ `--rate` during loud passages; silence emits nothing.
- `--rate` sets the bandwidth (≈ rate/2). Below ~8 kHz the sample-rate images
  fall inside the audible band.

30 s test clip (simulated with ideal impulses, 40 Hz–5 kHz):

| --rate | tiles | events | size | SNR |
|---|---|---|---|---|
| 8000 | 302 | 238,108 | 24 MB | 12.9 dB |
| 11025 | 303 | 328,750 | 34 MB | 18.4 dB |
| 22050 | 302 | 658,093 | 68 MB | 18.6 dB |
| (pdm, 300k tiles) | 300,001 | 211,047 SetSpeed | 25 MB | 6.3 dB |

A 3-minute song at 11025 Hz is roughly 2M events / 200 MB of JSON; at 8000 Hz
roughly 1.4M events / 145 MB. Lower `--rate` or cut the song if parsing is too slow.

Events use the editor's field set:

```json
{"floor":1,"eventType":"PlaySound","hitsound":"Sidestick","offset":0,"playDuration":0,"pitch":100,"hitsoundVolume":47,"angleOffset":0.225,"eventTag":""}
```

`--compact-events` drops `offset`/`playDuration`/`pitch`/`eventTag` (~35% smaller
JSON) if the game accepts events without them.

### `--engine pdm` (default)

One tile per 1-bit pulse at a constant hitsound volume; timing encoded with
`SetSpeed` (placed on the floor the travel starts from). `--max-tiles` defaults
to 100,000 (`--max-tiles 0` disables it). 10 s clip at the default cap:
99,999 tiles, 65k SetSpeed, 7.7 MB, simulated SNR 6.5 dB (12.6 dB uncapped with
189k tiles).

## PDM mapping

Audio is first normalized and soft-clipped: `y[n] = tanh(gain * x[n] / scale)`,
where `scale` is the `--percentile` absolute amplitude.

### `--mode adaptive` (default)

```text
e[n] = look-ahead peak envelope of |y|   (e[n] >= |y[n]|)
d[n] = max_density * (e[n] + y[n]) / 2
```

Pulse density follows loudness, so **silence produces (almost) no tiles** and quiet
noise produces very few. Loud passages approach `max_density * e` peak density.
The envelope is the max of neighbouring `--env-ms` blocks, linearly interpolated,
so `d` never goes negative. Envelope levels below `--gate-db` emit nothing.

`--max-tiles N` lowers the density uniformly so the chart stays at roughly N
tiles. Fewer tiles means a lower signal-to-noise ratio, but it is the knob for
making a chart small enough for the vanilla game to open.

### `--mode classic`

```text
d[n] = 0.5 + 0.49 * y[n]
```

Silence is 50% density, i.e. a constant `Fs/2` tiles per second regardless of
content (carrier near 22.05 kHz at 44.1 kHz).

### Comparison (14 s test clip: tone / digital silence / -54 dB noise / tone)

| mode | tiles | tiles in silent 3 s | SNR 40 Hz–5 kHz (ideal impulses) |
|---|---|---|---|
| classic | 308,692 | 66,150 | 9.4 dB |
| adaptive | 136,593 | 199 | 11.8 dB |
| adaptive `--max-tiles 50000` | 49,999 | – | 4.1 dB |

## Audible band only (`--min-freq` 20 Hz / `--max-freq` 20 kHz)

The audible range is roughly 20 Hz–20 kHz. The input is high-passed at
`--min-freq` and low-passed at `--max-freq` (both 4th order, 24 dB/oct) before
encoding, so nothing outside it is encoded. The low-pass is skipped when
`--max-freq` ≥ rate/2, since resampling already band-limits. In addition, a pulse more than `1/min-freq` s
(50 ms at 20 Hz) away from **both** neighbours is heard as a lone click rather
than part of a tone, so it is dropped. With the default adaptive mode and
`--gate-db -60` such pulses are rare (only just above the gate). Long gaps that
span real silence are kept: they carry the silence's duration.

## Notes / limitations

- This is experimental, not a normal charting workflow.
- The generated charts can be enormous.
- ADOFAI/editor playback engines may merge or drop extremely dense same-frame hits for performance reasons.
- The reconstructed sound is filtered by the chosen hitsound's own spectrum.
- `Sidestick` has worked well in experiments, but other hitsounds produce very different timbres.
- This project does **not** include or redistribute source music.

## 日本語メモ

音源を44.1 kHzなどでモノラルPCM化し、1-bit PDMに変換して、`1`になった時刻だけ床を置きます。デフォルトの`--engine playsound`では床は10枚/秒程度だけ置き、音は「サウンド再生」イベント（角度オフセット＋音量）で鳴らします。`--engine pdm`の`adaptive`モードでは音量エンベロープに比例してパルス密度を変えるので、無音・小さいノイズ部分はほぼ床が出ません。`--max-tiles`で総タイル数の上限を指定できます。床はすべて一直線で、床間隔は`SetSpeed`のBPMに変換します。背景音源は無音なので、聞こえる音はADOFAIのヒットサウンドだけです。

要するに：

```text
音声 -> 1-bit PDM -> ヒット時刻 -> 数百万タイル -> ヒットサウンドで音声再構成
```

## Status

Experimental. The current implementation focuses on the simple high-density PDM approach that produced the clearest results in testing.

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

At sufficiently high sample rates, surprisingly recognizable speech, melodies, bass and distorted kick textures can be reconstructed from millions of hitsounds. A 3-minute song at 44.1 kHz typically produces roughly 4 million tiles with the default settings.

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
python sound_chart.py input.ogg -o output_66k.adofai --rate 66000
```

The program also creates a silent `.ogg` beside the chart. Keep both files in the same folder.

## Output

For `song.ogg`, the default output is:

```text
song_sound_chart.adofai
silence.ogg
```

The generated level contains:

- a perfectly straight route
- one tile per PDM pulse
- `SetSpeed` events when the pulse spacing changes
- a global hitsound (default: `Sidestick`)
- a silent background OGG

Large inputs can create **hundreds of megabytes of JSON** and millions of tiles/events. Loading the result may stress both ADOFAI and third-party editors.

## PDM mapping

The current encoder maps normalized audio `x[n]` to pulse density with:

```text
d[n] = 0.5 + 0.49 * tanh(gain * x[n])
```

A first-order accumulator then emits the 1-bit pulse stream.

Silence is therefore near 50% density. At 44.1 kHz this creates a carrier near 22.05 kHz; increasing the sample rate can move more of the carrier/noise above the audible range.

## Notes / limitations

- This is experimental, not a normal charting workflow.
- The generated charts can be enormous.
- ADOFAI/editor playback engines may merge or drop extremely dense same-frame hits for performance reasons.
- The reconstructed sound is filtered by the chosen hitsound's own spectrum.
- `Sidestick` has worked well in experiments, but other hitsounds produce very different timbres.
- This project does **not** include or redistribute source music.

## 日本語メモ

音源を44.1 kHzなどでモノラルPCM化し、1-bit PDMに変換して、`1`になった時刻だけ床を置きます。床はすべて一直線で、床間隔は`SetSpeed`のBPMに変換します。背景音源は無音なので、聞こえる音はADOFAIのヒットサウンドだけです。

要するに：

```text
音声 -> 1-bit PDM -> ヒット時刻 -> 数百万タイル -> ヒットサウンドで音声再構成
```

## Status

Experimental. The current implementation focuses on the simple high-density PDM approach that produced the clearest results in testing.

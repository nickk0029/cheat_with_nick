"""
Day 5: detect WHEN someone starts and stops talking (voice activity detection).
Prints "speech STARTED" / "speech ENDED" live. Does not call any AI.

Uses the Silero VAD model that already comes with faster-whisper,
so there is nothing new to install.

Run (listen to your microphone):
    py vad_test.py
Run (listen to what your speakers play, e.g. a YouTube video):
    py vad_test.py system

Stop with Ctrl + C.
"""
import queue
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from faster_whisper.vad import VadOptions, get_speech_timestamps

SOURCE = sys.argv[1].lower() if len(sys.argv) > 1 else "mic"
RATE = 16000
BLOCK = 4096                 # samples per check = 0.256 seconds
SILENCE_END = 1.0            # seconds of silence that count as "speech ended"

opts = VadOptions(
    threshold=0.5,
    min_speech_duration_ms=100,
    min_silence_duration_ms=100,
    speech_pad_ms=0,
)


def has_speech(block):
    return len(get_speech_timestamps(block, opts)) > 0


def mic_blocks():
    import sounddevice as sd

    q = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata.copy().flatten())

    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                        blocksize=BLOCK, callback=cb):
        while True:
            yield q.get()


def system_blocks():
    import soundcard as sc

    speaker = sc.default_speaker()
    print(f"Listening to: {speaker.name}")
    loop = sc.get_microphone(id=str(speaker.name), include_loopback=True)
    with loop.recorder(samplerate=48000) as rec:
        while True:
            data = rec.record(numframes=BLOCK * 3)           # 48000 Hz
            mono = data.mean(axis=1).astype(np.float32)
            yield mono[: BLOCK * 3].reshape(-1, 3).mean(axis=1)   # -> 16000 Hz


blocks = system_blocks() if SOURCE == "system" else mic_blocks()

print(f"Source: {SOURCE}. Talk (or play a video). Press Ctrl + C to stop.\n")

speaking = False
silent_for = 0.0
start = 0.0
t0 = time.time()

try:
    for block in blocks:
        now = time.time() - t0
        if has_speech(block):
            silent_for = 0.0
            if not speaking:
                speaking = True
                start = now
                print(f"[{now:6.1f}s] speech STARTED")
        elif speaking:
            silent_for += BLOCK / RATE
            if silent_for >= SILENCE_END:
                speaking = False
                print(f"[{now:6.1f}s] speech ENDED   (talked for {now - start - silent_for:.1f}s)")
except KeyboardInterrupt:
    print("\nStopped.")
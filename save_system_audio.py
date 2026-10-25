"""
Day 3: record what my computer is playing (system audio) for 10 seconds
and save it as system_audio.wav.

Install:  py -m pip install soundcard
Run:      py save_system_audio.py   (while a YouTube video is playing)
"""
import wave

import numpy as np
import soundcard as sc

SECONDS = 60
RATE = 48000

speaker = sc.default_speaker()
print(f"Recording from: {speaker.name}")

# A loopback "microphone" lets us record whatever the speaker is playing
loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)

print(f"Recording {SECONDS} seconds... play a video now")
with loopback.recorder(samplerate=RATE) as rec:
    data = rec.record(numframes=RATE * SECONDS)

mono = data.mean(axis=1)                      # stereo -> mono
peak = float(np.abs(mono).max())
print(f"Loudest point: {peak:.3f}  (0.000 means silence = nothing was playing)")

pcm = (np.clip(mono, -1, 1) * 32767).astype(np.int16)
with wave.open("system_audio.wav", "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(RATE)
    f.writeframes(pcm.tobytes())

print("Saved system_audio.wav")
"""
AI Overlay Assistant - Day 6: Auto mode.
Press "Auto" and the app listens continuously (Mic or System audio),
detects when someone finishes talking, transcribes it, and answers if it is a question.

Install:
    py -m pip install google-genai faster-whisper sounddevice soundcard numpy python-dotenv mss pillow pynput
.env file:
    GEMINI_API_KEY=your_key_here
Run:
    py ai_overlay.py

Buttons:
    Auto     start/stop continuous listening (uses the Mic/System dropdown)
    Voice    old manual mode: click to start, click again to stop and answer
    Screen   read the screen
Hotkeys:
    Ctrl+Shift+H show/hide   Ctrl+Shift+S screenshot   Ctrl+Shift+A auto on/off
"""
import io
import os
import queue
import re
import threading
import time
import tkinter as tk
import warnings
from collections import deque
from tkinter import ttk

warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types
from faster_whisper.vad import VadOptions, get_speech_timestamps

try:
    import soundcard as sc
    HAS_LOOPBACK = True
except Exception:
    HAS_LOOPBACK = False

try:
    import mss
    from PIL import Image
    HAS_SHOT = True
except ImportError:
    HAS_SHOT = False

try:
    import pynput.keyboard as pk
    HAS_HOTKEYS = True
except ImportError:
    HAS_HOTKEYS = False

# ---------------- Config ----------------
MODEL = "gemini-3.5-flash-lite"
MAX_TOKENS = 800
MAX_TURNS = 8
WHISPER_MODEL = "base"
RATE = 16000
LOOPBACK_RATE = 48000
BLOCK = 4096                   # samples per VAD check = 0.256 s
SILENCE_END = 0.7              # seconds of silence = utterance finished
MIN_UTTERANCE = 0.8            # ignore sounds shorter than this (seconds)
MAX_UTTERANCE = 15.0           # force-cut very long speech (seconds)
MIN_SPEECH_BLOCKS = 3          # need at least 3 speech blocks (~0.75 s) or it is treated as noise
LANGUAGE = "en"                # fixed language is much faster than auto-detect; use "hi" for Hindi, None = auto
SYSTEM_PROMPT = (
    "You are an ultra-fast, precise assistant displayed in a compact overlay window. "
    "Provide direct, highly concise answers. For multiple-choice questions, state the correct "
    "option first, followed by a single line of clear reasoning. For code or technical questions, "
    "provide clean code snippets and a brief one-line note. Plain text only, no markdown symbols."
)

WAKE_RE = re.compile(r"^\W*(hey|hi|hello|ok|okay)\W+(assist\w*|a\s*sistant)\W*", re.IGNORECASE)
WHISPER_HINT = "Hey assistant. Questions about Python, coding, maths and general knowledge."
JUNK = {"thank you", "thanks", "thanks for watching", "you", "bye", "okay", "ok", "yeah", "hmm", "um", "uh"}

vad_opts = VadOptions(threshold=0.5, min_speech_duration_ms=100,
                      min_silence_duration_ms=100, speech_pad_ms=0)

QUESTION_STARTS = tuple(w + " " for w in [
    "what", "why", "how", "when", "where", "who", "which", "whose", "explain", "define",
    "describe", "write", "solve", "find", "calculate", "compare", "list", "tell me",
    "can you", "could you", "is", "are", "do", "does", "did", "will", "would", "should",
    "give me",
])

# ---------------- State ----------------
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

history = []
recent = deque(maxlen=8)       # last things heard (rolling context)
mic_frames = []
sys_frames = []
mic_stream = None
sys_thread = None
recording = False              # manual Voice mode
auto_on = False                # Auto mode
busy = False
stt = None
stt_lock = threading.Lock()
ui_q = queue.Queue()
utt_q = queue.Queue()          # finished utterances waiting to be transcribed


def ui(fn, *args):
    ui_q.put((fn, args))


def pump():
    while True:
        try:
            fn, args = ui_q.get_nowait()
        except queue.Empty:
            break
        fn(*args)
    root.after(40, pump)


# ---------------- UI helpers ----------------
def append(text, tag=None):
    out.config(state="normal")
    out.insert("end", text, tag)
    out.see("end")
    out.config(state="disabled")


def set_status(text):
    status.config(text=text)


def clear_chat():
    history.clear()
    recent.clear()
    out.config(state="normal")
    out.delete("1.0", "end")
    out.config(state="disabled")
    set_status("Cleared")


def copy_last_response():
    last = next((h["text"] for h in reversed(history) if h["role"] == "model"), "")
    if last:
        root.clipboard_clear()
        root.clipboard_append(last)
        set_status("Copied to clipboard!")
    else:
        set_status("Nothing to copy")


def toggle_window():
    if root.state() == "withdrawn":
        root.deiconify()
        root.attributes("-topmost", True)
        entry.focus()
    else:
        root.withdraw()


def start_com():
    """Windows needs COM started in every thread that records audio."""
    try:
        import ctypes
        ctypes.windll.ole32.CoInitializeEx(None, 0)
        return True
    except Exception:
        return False


def stop_com():
    try:
        import ctypes
        ctypes.windll.ole32.CoUninitialize()
    except Exception:
        pass


# ---------------- Gemini ----------------
def ask_ai(display_text, new_parts, history_text):
    global busy
    if busy:
        ui(set_status, "Busy processing...")
        return
    if client is None:
        ui(append, "\n[Error] GEMINI_API_KEY missing. Check your .env file.\n", "err")
        return

    busy = True
    ui(set_status, "Thinking...")
    ui(append, f"\nYou: {display_text}\n\n", "you")
    ui(append, "AI: ", "ai_label")

    contents = [
        types.Content(role=h["role"], parts=[types.Part.from_text(text=h["text"])])
        for h in history[-MAX_TURNS * 2:]
    ]
    contents.append(types.Content(role="user", parts=new_parts))

    reply = ""
    try:
        for chunk in client.models.generate_content_stream(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, max_output_tokens=MAX_TOKENS
            ),
        ):
            if chunk.text:
                reply += chunk.text
                ui(append, chunk.text, "ai")
        history.append({"role": "user", "text": history_text})
        history.append({"role": "model", "text": reply})
        ui(append, "\n", None)
        ui(set_status, "Listening (auto)..." if auto_on else "Ready")
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            msg = "Rate limit reached. Please wait a moment."
        ui(append, f"\n[Error] {msg}\n", "err")
        ui(set_status, "Error")
    finally:
        busy = False


def send_typed(_event=None):
    text = entry.get().strip()
    if not text:
        return
    entry.delete(0, "end")
    parts = [types.Part.from_text(text=text)]
    threading.Thread(target=ask_ai, args=(text, parts, text), daemon=True).start()


# ---------------- Speech to text ----------------
def get_stt():
    global stt
    with stt_lock:
        if stt is None:
            from faster_whisper import WhisperModel
            ui(set_status, "Loading Whisper model...")
            stt = WhisperModel(WHISPER_MODEL, compute_type="int8", cpu_threads=min(8, os.cpu_count() or 4))
    return stt


def transcribe(audio, trim_silence=False):
    """Returns (text, seconds it took)."""
    t0 = time.time()
    segments, _ = get_stt().transcribe(
        audio.astype(np.float32),
        language=LANGUAGE,
        beam_size=1,                          # fastest decoding
        vad_filter=trim_silence,              # auto mode already cut the audio with VAD
        condition_on_previous_text=False,
        initial_prompt=WHISPER_HINT,          # helps it recognise "hey assistant" and tech words
    )
    # drop segments Whisper itself is unsure about (usually noise it imagined as words)
    good = [s.text for s in segments if s.no_speech_prob < 0.6 and s.avg_logprob > -1.2]
    text = " ".join(good).strip()
    if re.sub(r"[^a-z ]", "", text.lower()).strip() in JUNK:
        text = ""
    return text, time.time() - t0


def is_question(text):
    t = text.strip().lower()
    if len(t.split()) < 3:
        return False
    return t.endswith("?") or t.startswith(QUESTION_STARTS)


# ---------------- Auto mode ----------------
def has_speech(block):
    return len(get_speech_timestamps(block, vad_opts)) > 0


def mic_blocks():
    q = queue.Queue()

    def cb(indata, frames, t, status_flags):
        q.put(indata.copy().flatten())

    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                        blocksize=BLOCK, callback=cb):
        while auto_on:
            try:
                yield q.get(timeout=0.5)
            except queue.Empty:
                continue


def system_blocks():
    speaker = sc.default_speaker()
    loop = sc.get_microphone(id=str(speaker.name), include_loopback=True)
    with loop.recorder(samplerate=LOOPBACK_RATE) as rec:
        while auto_on:
            data = rec.record(numframes=BLOCK * 3)
            mono = data.mean(axis=1).astype(np.float32)
            yield mono[: BLOCK * 3].reshape(-1, 3).mean(axis=1)    # 48k -> 16k


def listen_loop(source):
    """Background thread: cut the audio stream into utterances using VAD."""
    com = start_com()
    gen = None
    try:
        gen = system_blocks() if source == "System" else mic_blocks()
        speaking = False
        silent_for = 0.0
        speech_blocks = 0
        buf = []
        pre = None
        for block in gen:
            if not auto_on:
                break
            level = float(np.sqrt(np.mean(block ** 2)))          # how loud this block is
            if has_speech(block):
                if not speaking:
                    speaking = True
                    speech_blocks = 0
                    buf = [pre] if pre is not None else []
                speech_blocks += 1
                silent_for = 0.0
                buf.append(block)
                ui(set_status, f"Hearing speech...  level {level:.3f}")
            elif speaking:
                buf.append(block)
                silent_for += BLOCK / RATE
                if silent_for >= SILENCE_END:
                    speaking = False
                    if speech_blocks >= MIN_SPEECH_BLOCKS:
                        flush(buf)
                    buf = []
            else:
                ui(set_status, f"Listening (auto)...  level {level:.3f}")
            if speaking and len(buf) * BLOCK / RATE > MAX_UTTERANCE:
                flush(buf)
                buf = []
                speaking = False
            pre = block
    except Exception as e:
        ui(append, f"\n[Auto mode error] {e}\n", "err")
    finally:
        if gen is not None:
            gen.close()
        if com:
            stop_com()
        ui(set_status, "Auto off")


def flush(buf):
    audio = np.concatenate(buf)
    if len(audio) / RATE >= MIN_UTTERANCE:
        utt_q.put(audio)


def utterance_worker():
    """Background thread: transcribe each finished utterance, answer if it is a question."""
    while True:
        audio = utt_q.get()
        ui(set_status, "Transcribing...")
        try:
            text, secs = transcribe(audio)
        except Exception as e:
            ui(append, f"\n[STT Error] {e}\n", "err")
            continue
        if not text:
            ui(set_status, "Listening (auto)...")
            continue
        ui(append, f"\n(speech to text: {secs:.1f}s for {len(audio) / RATE:.1f}s of audio)", "heard")
        wake = WAKE_RE.match(text)
        if wake:
            text = text[wake.end():].strip()
            if not text:
                ui(append, "\nAssistant: I'm listening, ask your question.\n", "heard")
                ui(set_status, "Listening (auto)...")
                continue
        if wake or is_question(text):
            context = "\n".join(list(recent)[-4:])
            prompt = f"Recent conversation:\n{context}\n\nLatest question: {text}" if context else text
            recent.append(text)
            ask_ai(f"[Auto] {text}", [types.Part.from_text(text=prompt)], text)
        else:
            recent.append(text)
            ui(append, f"\nHeard: {text}\n", "heard")


def toggle_auto():
    global auto_on
    if not auto_on:
        if recording:
            set_status("Stop Voice first")
            return
        source = source_var.get()
        if source == "System" and not HAS_LOOPBACK:
            append("\n[Error] Run: py -m pip install soundcard\n", "err")
            return
        auto_on = True
        auto_btn.config(text="Auto: ON", bg="#16a34a")
        threading.Thread(target=get_stt, daemon=True).start()       # preload Whisper
        threading.Thread(target=listen_loop, args=(source,), daemon=True).start()
    else:
        auto_on = False
        auto_btn.config(text="Auto: OFF", bg="#27272a")
        set_status("Stopping auto...")


# ---------------- Manual voice (click start, click stop) ----------------
def mic_cb(indata, frame_count, time_info, status_flags):
    mic_frames.append(indata.copy().flatten())


def system_record_loop():
    com = start_com()
    try:
        speaker = sc.default_speaker()
        loop = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        with loop.recorder(samplerate=LOOPBACK_RATE) as rec:
            while recording:
                data = rec.record(numframes=LOOPBACK_RATE // 10)
                sys_frames.append(data.mean(axis=1).astype(np.float32))
    except Exception as e:
        ui(append, f"\n[System audio error] {e}\n", "err")
    finally:
        if com:
            stop_com()


def toggle_record():
    global mic_stream, sys_thread, recording
    if auto_on:
        set_status("Turn Auto off first")
        return
    source = source_var.get()

    if not recording:
        mic_frames.clear()
        sys_frames.clear()
        if source == "System":
            if not HAS_LOOPBACK:
                append("\n[Error] Run: py -m pip install soundcard\n", "err")
                return
            recording = True
            sys_thread = threading.Thread(target=system_record_loop, daemon=True)
            sys_thread.start()
            set_status("Listening to system audio...")
        else:
            try:
                mic_stream = sd.InputStream(samplerate=RATE, channels=1,
                                            dtype="float32", callback=mic_cb)
                mic_stream.start()
            except Exception as e:
                append(f"\n[Mic Error] {e}\n", "err")
                return
            recording = True
            set_status("Listening to mic...")
        rec_btn.config(text="Stop", bg="#ef4444")
    else:
        recording = False
        if mic_stream is not None:
            mic_stream.stop()
            mic_stream.close()
            mic_stream = None
        if sys_thread is not None:
            sys_thread.join(timeout=2)
            sys_thread = None
        rec_btn.config(text="Voice", bg="#27272a")
        set_status("Transcribing...")
        threading.Thread(target=manual_transcribe_and_ask, args=(source,), daemon=True).start()


def manual_transcribe_and_ask(source):
    frames = sys_frames if source == "System" else mic_frames
    if not frames:
        ui(set_status, "No audio captured")
        return
    try:
        audio = np.concatenate(frames).astype(np.float32)
        if source == "System":
            usable = len(audio) // 3 * 3
            audio = audio[:usable].reshape(-1, 3).mean(axis=1)
        text, _ = transcribe(audio, trim_silence=True)
    except Exception as e:
        ui(append, f"\n[STT Error] {e}\n", "err")
        return
    if not text:
        ui(set_status, "Heard nothing")
        return
    ask_ai(f"[{source}] {text}", [types.Part.from_text(text=text)], text)


# ---------------- Screenshot ----------------
def screenshot_ask():
    if not HAS_SHOT:
        append("\n[Error] Install mss and pillow first.\n", "err")
        return
    root.withdraw()
    root.after(300, _do_shot)


def _do_shot():
    try:
        with mss.mss() as m:
            shot = m.grab(m.monitors[1])
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    except Exception as e:
        root.deiconify()
        append(f"\n[Screenshot Error] {e}\n", "err")
        return
    root.deiconify()
    root.attributes("-topmost", True)

    img.thumbnail((1280, 1280))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)

    prompt = entry.get().strip() or "Analyze the content on my screen."
    entry.delete(0, "end")
    parts = [
        types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
        types.Part.from_text(text=prompt),
    ]
    label = f"[Screen] {prompt}"
    threading.Thread(target=ask_ai, args=(label, parts, label), daemon=True).start()


# ---------------- Hotkeys ----------------
def start_hotkeys():
    if not HAS_HOTKEYS:
        return
    try:
        hk = pk.GlobalHotKeys({
            "<ctrl>+<shift>+h": lambda: ui(toggle_window),
            "<ctrl>+<shift>+s": lambda: ui(screenshot_ask),
            "<ctrl>+<shift>+a": lambda: ui(toggle_auto),
        })
        hk.daemon = True
        hk.start()
    except Exception:
        pass


# ---------------- Window ----------------
root = tk.Tk()
root.title("AI Overlay Assistant")
root.geometry("460x580+40+40")
root.attributes("-topmost", True)
root.attributes("-alpha", 0.94)
root.configure(bg="#09090b")

top = tk.Frame(root, bg="#09090b")
top.pack(fill="x", padx=10, pady=(10, 4))
status = tk.Label(top, text="Ready", bg="#09090b", fg="#71717a", font=("Segoe UI", 9))
status.pack(side="left")
tk.Button(top, text="Clear", command=clear_chat, bg="#18181b", fg="#a1a1aa", relief="flat",
          font=("Segoe UI", 9), padx=8, pady=2, cursor="hand2").pack(side="right", padx=(4, 0))
tk.Button(top, text="Copy", command=copy_last_response, bg="#18181b", fg="#a1a1aa", relief="flat",
          font=("Segoe UI", 9), padx=8, pady=2, cursor="hand2").pack(side="right", padx=(4, 0))
alpha_slider = ttk.Scale(top, from_=0.4, to=1.0, value=0.94, length=60,
                         command=lambda v: root.attributes("-alpha", float(v)))
alpha_slider.pack(side="right", padx=6)

text_frame = tk.Frame(root, bg="#09090b")
text_frame.pack(fill="both", expand=True, padx=10, pady=4)
scrollbar = tk.Scrollbar(text_frame, width=8)
scrollbar.pack(side="right", fill="y")
out = tk.Text(text_frame, wrap="word", bg="#09090b", fg="#f4f4f5", font=("Segoe UI", 10),
              relief="flat", state="disabled", padx=8, pady=8, spacing3=4,
              yscrollcommand=scrollbar.set)
out.pack(side="left", fill="both", expand=True)
scrollbar.config(command=out.yview)
out.tag_config("you", foreground="#60a5fa")
out.tag_config("ai_label", foreground="#4ade80")
out.tag_config("ai", foreground="#f4f4f5")
out.tag_config("heard", foreground="#71717a")
out.tag_config("err", foreground="#f87171")

bottom = tk.Frame(root, bg="#09090b")
bottom.pack(fill="x", padx=10, pady=10)
entry = tk.Entry(bottom, bg="#18181b", fg="#f4f4f5", insertbackground="#fff", relief="flat",
                 font=("Segoe UI", 10))
entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 6))
entry.bind("<Return>", send_typed)

tk.Button(bottom, text="Screen", command=screenshot_ask, bg="#27272a", fg="#f4f4f5", relief="flat",
          font=("Segoe UI", 9, "bold"), padx=8, pady=6, cursor="hand2").pack(side="right", padx=(4, 0))
rec_btn = tk.Button(bottom, text="Voice", command=toggle_record, bg="#27272a", fg="#f4f4f5",
                    relief="flat", font=("Segoe UI", 9, "bold"), padx=8, pady=6, cursor="hand2")
rec_btn.pack(side="right", padx=(4, 0))
auto_btn = tk.Button(bottom, text="Auto: OFF", command=toggle_auto, bg="#27272a", fg="#f4f4f5",
                     relief="flat", font=("Segoe UI", 9, "bold"), padx=8, pady=6, cursor="hand2")
auto_btn.pack(side="right", padx=(4, 0))
source_var = tk.StringVar(value="Mic")
ttk.Combobox(bottom, textvariable=source_var, values=["Mic", "System"], width=7,
             state="readonly").pack(side="right", padx=(4, 0))

if client is None:
    append("[Error] GEMINI_API_KEY not found. Create a .env file with your key.\n", "err")
if not HAS_LOOPBACK:
    append("[Note] System audio disabled. Run: py -m pip install soundcard\n", "err")
if not HAS_SHOT:
    append("[Note] Screenshots disabled. Run: py -m pip install mss pillow\n", "err")
if not HAS_HOTKEYS:
    append("[Note] Hotkeys disabled. Run: py -m pip install pynput\n", "err")

threading.Thread(target=utterance_worker, daemon=True).start()
start_hotkeys()
pump()
entry.focus()
root.mainloop()
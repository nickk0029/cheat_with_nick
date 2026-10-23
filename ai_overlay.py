import io
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types

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
SAMPLE_RATE = 16000
SYSTEM_PROMPT = (
    "You are an ultra-fast, precise assistant displayed in a compact overlay window. "
    "Provide direct, highly concise answers. For multiple-choice questions, state the correct "
    "option first, followed by a single line of clear reasoning. For code or technical questions, "
    "provide clean code snippets and a brief one-line note. Plain text only, no markdown symbols."
)

# ---------------- State ----------------
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

history = []
frames = []
stream = None
recording = False
busy = False
stt = None
stt_lock = threading.Lock()
ui_q = queue.Queue()


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


# ---------------- UI Helpers ----------------
def append(text, tag=None):
    out.config(state="normal")
    out.insert("end", text, tag)
    out.see("end")
    out.config(state="disabled")


def set_status(text):
    status.config(text=text)


def clear_chat():
    history.clear()
    out.config(state="normal")
    out.delete("1.0", "end")
    out.config(state="disabled")
    set_status("Cleared")


def copy_last_response():
    try:
        # Find the last model response from history
        last_model_msg = next((h["text"] for h in reversed(history) if h["role"] == "model"), "")
        if last_model_msg:
            root.clipboard_clear()
            root.clipboard_append(last_model_msg)
            set_status("Copied to clipboard!")
        else:
            set_status("Nothing to copy")
    except Exception:
        set_status("Copy failed")


def toggle_window():
    if root.state() == "withdrawn":
        root.deiconify()
        root.attributes("-topmost", True)
        entry.focus()
    else:
        root.withdraw()


# ---------------- Gemini Inference ----------------
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
        ui(set_status, "Ready")
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            msg = "Rate limit reached. Please wait a moment."
        ui(append, f"\n[Error] {msg}\n", "err")
        ui(set_status, "Error")
    finally:
        busy = False
        ui(lambda: entry.focus())


def send_typed(_event=None):
    text = entry.get().strip()
    if not text:
        return
    entry.delete(0, "end")
    parts = [types.Part.from_text(text=text)]
    threading.Thread(target=ask_ai, args=(text, parts, text), daemon=True).start()


# ---------------- Voice Recording ----------------
def audio_cb(indata, frame_count, time_info, status_flags):
    frames.append(indata.copy())


def toggle_record():
    global stream, recording
    if not recording:
        try:
            frames.clear()
            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_cb)
            stream.start()
        except Exception as e:
            append(f"\n[Mic Error] {e}\n", "err")
            return
        recording = True
        rec_btn.config(text="Stop", bg="#ef4444")
        set_status("Listening...")
    else:
        stream.stop()
        stream.close()
        recording = False
        rec_btn.config(text="Voice", bg="#27272a")
        set_status("Transcribing...")
        threading.Thread(target=transcribe_and_ask, daemon=True).start()


def get_stt():
    global stt
    with stt_lock:
        if stt is None:
            from faster_whisper import WhisperModel
            ui(set_status, "Loading Whisper model...")
            stt = WhisperModel(WHISPER_MODEL, compute_type="int8")
    return stt


def transcribe_and_ask():
    if not frames:
        ui(set_status, "No audio captured")
        return
    try:
        audio = np.concatenate(frames).flatten()
        segments, _ = get_stt().transcribe(audio, vad_filter=True)
        text = " ".join(s.text for s in segments).strip()
    except Exception as e:
        ui(append, f"\n[STT Error] {e}\n", "err")
        return
    if not text:
        ui(set_status, "Heard nothing")
        return
    ask_ai(text, [types.Part.from_text(text=text)], text)


# ---------------- Screenshot Capture ----------------
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


# ---------------- Global Hotkeys ----------------
def start_hotkeys():
    if not HAS_HOTKEYS:
        return
    try:
        hk = pk.GlobalHotKeys({
            "<ctrl>+<shift>+h": lambda: ui(toggle_window),
            "<ctrl>+<shift>+s": lambda: ui(screenshot_ask),
        })
        hk.daemon = True
        hk.start()
    except Exception:
        pass


# ---------------- Main Window UI Setup ----------------
root = tk.Tk()
root.title("AI Overlay Assistant")
root.geometry("460x580+40+40")
root.attributes("-topmost", True)
root.attributes("-alpha", 0.94)
root.configure(bg="#09090b")

# Top Control Bar
top = tk.Frame(root, bg="#09090b")
top.pack(fill="x", padx=10, pady=(10, 4))

status = tk.Label(top, text="Ready", bg="#09090b", fg="#71717a", font=("Segoe UI", 9))
status.pack(side="left")

tk.Button(top, text="Clear", command=clear_chat, bg="#18181b", fg="#a1a1aa", relief="flat", font=("Segoe UI", 9), padx=8, pady=2, cursor="hand2").pack(side="right", padx=(4, 0))
tk.Button(top, text="Copy", command=copy_last_response, bg="#18181b", fg="#a1a1aa", relief="flat", font=("Segoe UI", 9), padx=8, pady=2, cursor="hand2").pack(side="right", padx=(4, 0))

alpha_slider = ttk.Scale(top, from_=0.4, to=1.0, value=0.94, length=60,
                         command=lambda v: root.attributes("-alpha", float(v)))
alpha_slider.pack(side="right", padx=6)

# Main Output Text Area with Scrollbar
text_frame = tk.Frame(root, bg="#09090b")
text_frame.pack(fill="both", expand=True, padx=10, pady=4)

scrollbar = tk.Scrollbar(text_frame, bg="#18181b", troughcolor="#09090b", activebackground="#27272a", width=8)
scrollbar.pack(side="right", fill="y")

out = tk.Text(text_frame, wrap="word", bg="#09090b", fg="#f4f4f5", font=("Segoe UI", 10),
              relief="flat", state="disabled", padx=8, pady=8, spacing3=4,
              yscrollcommand=scrollbar.set)
out.pack(side="left", fill="both", expand=True)
scrollbar.config(command=out.yview)

# Tag Styling
out.tag_config("you", foreground="#60a5fa")     # Blue
out.tag_config("ai_label", foreground="#4ade80") # Green label
out.tag_config("ai", foreground="#f4f4f5")       # White text
out.tag_config("err", foreground="#f87171")      # Soft red

# Bottom Input & Action Bar
bottom = tk.Frame(root, bg="#09090b")
bottom.pack(fill="x", padx=10, pady=10)

entry = tk.Entry(bottom, bg="#18181b", fg="#f4f4f5", insertbackground="#fff", relief="flat", font=("Segoe UI", 10))
entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 6))
entry.bind("<Return>", send_typed)

tk.Button(bottom, text="Screen", command=screenshot_ask, bg="#27272a", fg="#f4f4f5", relief="flat", font=("Segoe UI", 9, "bold"), padx=10, pady=6, cursor="hand2").pack(side="right", padx=(4, 0))
rec_btn = tk.Button(bottom, text="Voice", command=toggle_record, bg="#27272a", fg="#f4f4f5", relief="flat", font=("Segoe UI", 9, "bold"), padx=10, pady=6, cursor="hand2")
rec_btn.pack(side="right", padx=(4, 0))

# Startup Checks
if client is None:
    append("[Error] GEMINI_API_KEY not found. Create a .env file with your key.\n", "err")
if not HAS_SHOT:
    append("[Note] Screenshots disabled. Run: pip install mss pillow\n", "err")
if not HAS_HOTKEYS:
    append("[Note] Hotkeys disabled. Run: pip install pynput\n", "err")

start_hotkeys()
pump()
entry.focus()
root.mainloop()
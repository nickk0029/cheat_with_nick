import queue, threading, collections, time
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from google import genai
from google.genai import types

SR = 16000
FRAME_MS = 30
FRAME_LEN = SR * FRAME_MS // 1000
SILENCE_END_MS = 900
PRE_ROLL_MS = 500
MIN_RMS = 300

vad = webrtcvad.Vad(2)
audio_q = queue.Queue()
utterance_q = queue.Queue()
paused = threading.Event()

print("Loading Whisper model...")
model = WhisperModel("small", device="cpu", compute_type="int8")
print("Model ready.")

JUNK_PHRASES = ["thank you", "thanks for watching", "end of the video",
                "see you next time", "subscribe", "bye"]
PROMPT = "Technical interview about Python, decorators, generators, REST API, SQL, DSA, Django, Flask, system design."

# ---------- Gemini ----------
client = genai.Client()   # reads GEMINI_API_KEY automatically

SYSTEM = ("You are a live assistant for a technical interview. Answer in 3-5 short "
          "sentences, direct and simple, with a tiny code example only if needed. "
          "The question comes from speech-to-text, so fix obvious mishearings "
          "(e.g. 'decoder' means 'decorator' in a Python context).")

MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
history = []   # remembers earlier turns for follow-up questions

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM,
    max_output_tokens=500,
)


def ask_llm(question):
    history.append(types.Content(role="user", parts=[types.Part(text=question)]))
    for attempt in range(8):
        model_name = MODELS[attempt % len(MODELS)]
        answer = ""
        try:
            print("AI: ", end="", flush=True)
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=history[-8:],
                config=CONFIG,
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    answer += chunk.text
            print("\n")
            history.append(types.Content(role="model", parts=[types.Part(text=answer)]))
            return
        except Exception as e:
            msg = str(e)
            if ("503" in msg or "429" in msg) and attempt < 7:
                print(f"\n(busy, retrying with {MODELS[(attempt + 1) % len(MODELS)]}...)")
                time.sleep(1.5 * (attempt + 1))
                continue
            history.pop()   # drop the unanswered question
            print(f"\nLLM error: {e}")
            return


# ---------- Audio ----------
def callback(indata, frames, time_info, status):
    if not paused.is_set():
        audio_q.put(bytes(indata))


def vad_loop():
    pre_roll = collections.deque(maxlen=PRE_ROLL_MS // FRAME_MS)
    voiced, silence_frames = [], 0
    in_speech = False
    max_silence = SILENCE_END_MS // FRAME_MS

    while True:
        frame = audio_q.get()
        speech = vad.is_speech(frame, SR)

        if not in_speech:
            pre_roll.append(frame)
            if speech:
                in_speech = True
                voiced = list(pre_roll)
                silence_frames = 0
        else:
            voiced.append(frame)
            silence_frames = 0 if speech else silence_frames + 1
            if silence_frames > max_silence:
                audio = np.frombuffer(b"".join(voiced), dtype=np.int16)
                if len(audio) > SR * 0.4:
                    utterance_q.put(audio)
                in_speech, voiced = False, []
                pre_roll.clear()


def dedupe(text):
    parts = [p.strip() for p in text.replace("?", "?.").replace("!", "!.").split(".") if p.strip()]
    out = []
    for p in parts:
        if not out or p.lower() != out[-1].lower():
            out.append(p)
    return ". ".join(out)


def transcribe(audio_int16):
    audio = audio_int16.astype(np.float32) / 32768.0
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=2,
        vad_filter=False,
        condition_on_previous_text=False,
        initial_prompt=PROMPT,
        repetition_penalty=1.1,
    )
    return dedupe(" ".join(s.text.strip() for s in segments).strip())


def worker():
    while True:
        audio = utterance_q.get()
        paused.set()
        try:
            rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
            if rms < MIN_RMS:
                continue
            t0 = time.time()
            text = transcribe(audio)
            dt = time.time() - t0
            low = text.lower()
            if len(text) > 1 and not any(j in low for j in JUNK_PHRASES):
                print(f"[{dt:.2f}s] You: {text}")
                ask_llm(text)
        finally:
            with audio_q.mutex:
                audio_q.queue.clear()
            paused.clear()


threading.Thread(target=vad_loop, daemon=True).start()
threading.Thread(target=worker, daemon=True).start()

with sd.RawInputStream(samplerate=SR, blocksize=FRAME_LEN,
                       dtype="int16", channels=1, callback=callback):
    print("Listening... (Ctrl+C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("Stopped.")
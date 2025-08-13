import os
import sys
import time
import wave
import json
import queue
import zipfile
import tempfile
import shutil
import urllib.request
from pathlib import Path

import yaml
import numpy as np
import sounddevice as sd
import webrtcvad
import pyttsx3
from vosk import Model, KaldiRecognizer
from openai import OpenAI
import subprocess

CONFIG_PATH = Path(__file__).with_name("config.yaml")
MODELS_DIR = Path(__file__).with_name("models")


def load_config(path: Path) -> dict:
    if not path.exists():
        print(f"[ERROR] Config not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_vosk_model(model_name: str, model_url: str) -> Path:
    target_dir = MODELS_DIR / model_name
    if target_dir.exists() and (target_dir / "am/final.mdl").exists():
        return target_dir

    print(f"[INFO] Downloading Vosk model: {model_name}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / f"{model_name}.zip"
        urllib.request.urlretrieve(model_url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(Path(tmpdir))
        # Some archives contain a top-level folder with same name
        extracted_root = Path(tmpdir) / model_name
        if not extracted_root.exists():
            # fallback: find first directory
            for item in Path(tmpdir).iterdir():
                if item.is_dir():
                    extracted_root = item
                    break
        shutil.move(str(extracted_root), str(target_dir))

    print(f"[INFO] Vosk model ready at: {target_dir}")
    return target_dir


def ensure_piper_model(model_filename: str, model_url: str) -> Path:
    """Ensure Piper model file exists under models/piper/. Returns path to the model file."""
    piper_dir = MODELS_DIR / "piper"
    piper_dir.mkdir(parents=True, exist_ok=True)
    target_path = piper_dir / model_filename
    if target_path.exists():
        return target_path
    if not model_url:
        print("[ERROR] Piper model_url is not set in config.")
        return target_path
    print(f"[INFO] Downloading Piper model: {model_filename}")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / model_filename
        urllib.request.urlretrieve(model_url, tmp_path)
        shutil.move(str(tmp_path), str(target_path))
    print(f"[INFO] Piper model ready at: {target_path}")
    return target_path


class VADRecorder:
    def __init__(self, sample_rate: int, frame_ms: int, aggressiveness: int, silence_ms_to_end: int, max_record_ms: int):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000)
        self.aggressiveness = aggressiveness
        self.silence_ms_to_end = silence_ms_to_end
        self.max_record_ms = max_record_ms
        self.vad = webrtcvad.Vad(aggressiveness)

    def record(self) -> np.ndarray:
        print("[INFO] Recording... Speak now")
        frames = []
        silence_ms = 0
        total_ms = 0

        def callback(indata, frames_count, time_info, status):
            pass  # unused, we will use blocking read()

        with sd.RawInputStream(samplerate=self.sample_rate, blocksize=self.frame_bytes,
                               dtype='int16', channels=1):
            while True:
                data = sd.RawInputStream.read  # type: ignore[attr-defined]
                # RawInputStream.read is not directly accessible like this in context manager,
                # so we use sd.RawInputStream as ctx but read via sd.rec? Simpler: use sd.RawInputStream and read from sd._last_callback? No.
                # Switch to using sd.InputStream which provides read().
                break
        # Implement with InputStream for clarity
        audio_chunks = []
        with sd.InputStream(samplerate=self.sample_rate, blocksize=self.frame_bytes,
                            dtype='int16', channels=1) as stream:
            while True:
                indata, overflowed = stream.read(self.frame_bytes)
                if overflowed:
                    print("[WARN] Input overflow")
                pcm16 = indata.reshape(-1).tobytes()
                is_speech = self.vad.is_speech(pcm16, self.sample_rate)
                audio_chunks.append(pcm16)

                if is_speech:
                    silence_ms = 0
                else:
                    silence_ms += self.frame_ms

                total_ms += self.frame_ms

                if silence_ms >= self.silence_ms_to_end:
                    break
                if total_ms >= self.max_record_ms:
                    print("[INFO] Max utterance length reached")
                    break

        # Concatenate bytes and return as numpy int16
        audio_bytes = b''.join(audio_chunks)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        print("[INFO] Recording stopped")
        return audio_np


def asr_transcribe_vosk(model_dir: Path, sample_rate: int, audio_pcm16: np.ndarray) -> str:
    model = Model(str(model_dir))
    rec = KaldiRecognizer(model, sample_rate)
    rec.SetWords(True)

    # feed in chunks
    chunk_size = int(sample_rate * 0.2)  # 200 ms chunks
    offset = 0
    while offset < len(audio_pcm16):
        chunk = audio_pcm16[offset:offset+chunk_size]
        if len(chunk) == 0:
            break
        rec.AcceptWaveform(chunk.tobytes())
        offset += chunk_size
    final = rec.FinalResult()
    try:
        j = json.loads(final)
        text = j.get("text", "").strip()
        return text
    except Exception:
        return ""


def tts_speak_pyttsx3(engine: pyttsx3.Engine, text: str, rate: int | None = None, volume: float | None = None, voice_id: str | None = None):
    if rate is not None:
        engine.setProperty('rate', rate)
    if volume is not None:
        engine.setProperty('volume', volume)
    if voice_id:
        engine.setProperty('voice', voice_id)
    engine.say(text)
    engine.runAndWait()


def tts_speak_piper(text: str, bin_path: str, model_path: str,
                    speaker: str | None = None, length_scale: float = 1.0,
                    noise_scale: float = 0.667, noise_w: float = 0.8):
    """Synthesize speech with Piper CLI and play it back. Requires piper binary installed."""
    if not Path(bin_path).exists():
        print(f"[ERROR] Piper binary not found at {bin_path}")
        return
    if not Path(model_path).exists():
        print(f"[ERROR] Piper model not found at {model_path}")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "out.wav"
        cmd = [bin_path, "-m", model_path, "-f", "wav", "--output_file", str(wav_path)]
        if speaker:
            cmd += ["--speaker", str(speaker)]
        cmd += ["--length_scale", str(length_scale), "--noise_scale", str(noise_scale), "--noise_w", str(noise_w)]
        try:
            p = subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.returncode != 0:
                print(f"[ERROR] Piper failed: {p.stderr.decode('utf-8', 'ignore')}")
                return
        except Exception as e:
            print(f"[ERROR] Piper execution error: {e}")
            return
        # Play wav
        try:
            with wave.open(str(wav_path), 'rb') as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                frames = wf.readframes(wf.getnframes())
                data = np.frombuffer(frames, dtype=np.int16)
                if n_channels == 2:
                    data = data.reshape(-1, 2)
                sd.play(data, sr)
                sd.wait()
        except Exception as e:
            print(f"[WARN] Could not play Piper WAV via sounddevice: {e}")
            # fallback to system player if available
            try:
                subprocess.run(["aplay", str(wav_path)])
            except Exception:
                pass


def pick_tts_voice(engine: pyttsx3.Engine, lang_hint: str, cfg_voice: str | None) -> str | None:
    """Pick a voice id for pyttsx3.
    - If cfg_voice provided, return it.
    - Else try to find a voice matching lang_hint (e.g., 'ru' -> Milena/Yuri/Russian/ru).
    """
    if cfg_voice:
        return cfg_voice
    try:
        voices = engine.getProperty('voices') or []
    except Exception:
        return None
    targets = []
    if lang_hint.lower().startswith('ru'):
        targets = ['ru', 'russian', 'milena', 'yuri']
    elif lang_hint.lower().startswith('en'):
        targets = ['en', 'english', 'samantha', 'alex', 'victoria']
    else:
        targets = [lang_hint.lower()]

    def voice_fields(v):
        parts = [str(getattr(v, 'id', '')), str(getattr(v, 'name', ''))]
        langs = getattr(v, 'languages', [])
        try:
            parts.extend([x.decode('utf-8', 'ignore') if isinstance(x, (bytes, bytearray)) else str(x) for x in langs])
        except Exception:
            pass
        return ' '.join(parts).lower()

    best = None
    for v in voices:
        fields = voice_fields(v)
        if any(t in fields for t in targets):
            best = getattr(v, 'id', None) or getattr(v, 'name', None)
            break
    return best


def chat_openai(client: OpenAI, model: str, system_prompt: str, user_text: str,
                temperature: float = 0.2, max_tokens: int = 120) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] OpenAI request failed: {e}")
        return "Извините, произошла ошибка при формировании ответа."


def main():
    config = load_config(CONFIG_PATH)

    asr_cfg = config.get('asr', {})
    tts_cfg = config.get('tts', {})
    assistant_cfg = config.get('assistant', {})
    ui_cfg = config.get('ui', {})

    sample_rate = int(asr_cfg.get('sample_rate', 16000))

    # Prepare Vosk model
    if asr_cfg.get('engine') != 'vosk':
        print("[ERROR] Only 'vosk' ASR is implemented in this version of the app.")
        sys.exit(1)
    model_dir = ensure_vosk_model(
        asr_cfg.get('vosk_model_name', 'vosk-model-small-ru-0.22'),
        asr_cfg.get('vosk_model_url', 'https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip'),
    )

    # Prepare TTS engine(s)
    tts_engine = pyttsx3.init()

    # Prepare OpenAI client (ENV first, then config)
    openai_cfg = config.get('openai', {})
    api_key = os.getenv('OPENAI_API_KEY') or openai_cfg.get('api_key')
    if not api_key:
        print("[ERROR] OpenAI API key is not set. Set OPENAI_API_KEY env var or openai.api_key in config.yaml")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    push_to_talk = bool(ui_cfg.get('push_to_talk', True))

    recorder = VADRecorder(
        sample_rate=sample_rate,
        frame_ms=int(asr_cfg.get('vad', {}).get('frame_ms', 30)),
        aggressiveness=int(asr_cfg.get('vad', {}).get('aggressiveness', 2)),
        silence_ms_to_end=int(asr_cfg.get('vad', {}).get('silence_ms_to_end', 700)),
        max_record_ms=int(asr_cfg.get('vad', {}).get('max_record_ms', 12000)),
    )

    system_prompt = assistant_cfg.get('system_prompt', 'You are a helpful assistant.')
    model = assistant_cfg.get('model', 'gpt-4o-mini')
    temperature = float(assistant_cfg.get('temperature', 0.2))
    max_tokens = int(assistant_cfg.get('max_tokens', 120))

    # Configure TTS voice: if not specified in config, auto-pick based on ASR language
    desired_voice = tts_cfg.get('voice') or ""
    auto_voice = pick_tts_voice(tts_engine, asr_cfg.get('language', 'ru'), desired_voice if desired_voice else None)
    if auto_voice:
        try:
            tts_engine.setProperty('voice', auto_voice)
            print(f"[INFO] TTS voice: {auto_voice}")
        except Exception as e:
            print(f"[WARN] Cannot set TTS voice '{auto_voice}': {e}")

    # Prepare Piper if selected / or auto
    tts_engine_name = (tts_cfg.get('engine') or 'pyttsx3').lower()
    piper_cfg = tts_cfg.get('piper', {})
    piper_bin = piper_cfg.get('bin_path')
    piper_model_filename = piper_cfg.get('model_filename')
    piper_model_url = piper_cfg.get('model_url', '')
    piper_model_path_cfg = piper_cfg.get('model_path')  # absolute path to existing model
    piper_speaker = piper_cfg.get('speaker')
    piper_length = float(piper_cfg.get('length_scale', 1.0))
    piper_noise = float(piper_cfg.get('noise_scale', 0.667))
    piper_noise_w = float(piper_cfg.get('noise_w', 0.8))
    piper_model_path: Path | None = None
    # If engine is auto, prefer Piper on Linux/ARM when configured
    if tts_engine_name == 'auto':
        try:
            import platform
            is_linux = sys.platform.startswith('linux')
            machine = platform.machine().lower()
            is_arm = 'arm' in machine or 'aarch64' in machine
            if is_linux and is_arm and piper_bin and piper_model_filename:
                tts_engine_name = 'piper'
            else:
                tts_engine_name = 'pyttsx3'
        except Exception:
            tts_engine_name = 'pyttsx3'
    if tts_engine_name == 'piper':
        if piper_model_path_cfg and Path(piper_model_path_cfg).exists():
            piper_model_path = Path(piper_model_path_cfg)
        elif piper_model_filename:
            piper_model_path = ensure_piper_model(piper_model_filename, piper_model_url)

    print("\n=== RobotTutor Voice Assistant ===")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            if push_to_talk:
                input("\n[Enter] — говорить...")
            else:
                print("[INFO] Listening...")
            audio = recorder.record()
            if audio.size == 0:
                print("[WARN] Пустая запись, попробуйте ещё раз.")
                continue

            text = asr_transcribe_vosk(model_dir, sample_rate, audio)
            if not text:
                print("[INFO] Не удалось распознать речь.")
                continue
            print(f"[YOU]: {text}")

            reply = chat_openai(client, model, system_prompt, text, temperature=temperature, max_tokens=max_tokens)
            print(f"[BOT]: {reply}")

            if tts_engine_name == 'piper' and piper_bin and piper_model_path and piper_model_path.exists():
                tts_speak_piper(
                    reply,
                    piper_bin,
                    str(piper_model_path),
                    speaker=piper_speaker,
                    length_scale=piper_length,
                    noise_scale=piper_noise,
                    noise_w=piper_noise_w,
                )
            else:
                tts_speak_pyttsx3(
                    tts_engine,
                    reply,
                    rate=int(tts_cfg.get('rate', 180)),
                    volume=float(tts_cfg.get('volume', 1.0)),
                    voice_id=tts_cfg.get('voice') or None,
                )

    except KeyboardInterrupt:
        print("\n[INFO] Exiting...")
    finally:
        try:
            tts_engine.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()

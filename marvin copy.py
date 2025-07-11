import pvporcupine
import pyaudio
import struct
import os
import requests
import time
import wave
import openai
from datetime import datetime

# === НАСТРОЙКИ ===
KEYWORD_PATH = "marvin.ppn"
ACCESS_KEY = ""

# ⛔ ВСТАВЬ СВОЙ НАСТОЯЩИЙ OPENAI API-КЛЮЧ!
OPENAI_API_KEY =   # ← сюда твой ключ
openai.api_key = OPENAI_API_KEY

# === ЛОГ ===
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")

# === Запись речи после wakeword ===
def record_audio(filename="input.wav", record_seconds=5, rate=16000):
    log("🎤 Запись запроса пользователя...")
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate,
                     input=True, frames_per_buffer=1024)

    frames = []
    for _ in range(int(rate / 1024 * record_seconds)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    pa.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# === Распознавание речи (Whisper API)
def transcribe(filename="input.wav"):
    log("🧠 Распознаю аудио через Whisper (новый API)...")
    with open(filename, "rb") as f:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    log(f"📄 Распознан текст: {transcript.text}")
    return transcript.text

# === GPT-ответ
def ask_gpt(prompt):
    log("🤖 Отправляю запрос в GPT...")
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ты голосовой ассистент в стиле Марвина из 'Автостопом по галактике'."},
            {"role": "user", "content": prompt}
        ]
    )
    reply = response.choices[0].message.content
    log(f"🧠 Ответ GPT: {reply}")
    return reply

# === ИНИЦИАЛИЗАЦИЯ Porcupine
log("🔧 Инициализация Porcupine...")
porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keyword_paths=[KEYWORD_PATH]
)

pa = pyaudio.PyAudio()
audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

log("🕵️ Готов слушать “марвин”...")

# === ОСНОВНОЙ ЦИКЛ ===
try:
    last_log_time = 0
    while True:
        now = time.time()
        if now - last_log_time > 5.0:
            log("🎧 Слушаю...")
            last_log_time = now

        pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:
            log("🤖 Обнаружено ключевое слово: “марвин”")
            record_audio("input.wav", record_seconds=5)
            question = transcribe("input.wav")
            answer = ask_gpt(question)

except KeyboardInterrupt:
    log("⛔ Завершение по Ctrl+C")
finally:
    audio_stream.stop_stream()
    audio_stream.close()
    porcupine.delete()
    log("🧹 Очистка завершена. Пока!")

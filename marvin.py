import asyncio
import os
import struct
import pyaudio
import pvporcupine
import vertexai
from google import genai
from google.genai import types

# --- ОБЯЗАТЕЛЬНЫЕ НАСТРОЙКИ ---
GOOGLE_CLOUD_PROJECT = "useful-gearbox-464618-v3"
GOOGLE_CLOUD_LOCATION = "us-central1"
PICOVOICE_ACCESS_KEY = "VIHgWR3xcpv04Xlccis/3+RFLA/RWQw+EJFUOxb8eHr/56WVQ4Cu3g=="
KEYWORD_PATHS = ["marvin.ppn"] 
GEMINI_MODEL_NAME = "gemini-live-2.5-flash-preview-native-audio" 

# Настройки аудио
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 
CHUNK = 1024
PORCUPINE_FRAME_LENGTH = 512
# --- КОНЕЦ НАСТРОЕК ---


async def play_audio_from_queue(queue: asyncio.Queue):
    """Асинхронно воспроизводит аудио из очереди."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=24000, output=True)
    print("🎧 Аудиоплеер готов (настроен на 24кГц для ответа Gemini).")
    while True:
        try:
            audio_chunk = await asyncio.wait_for(queue.get(), timeout=1.0)
            if audio_chunk is None: break
            print(f"  -> 🔊 [Плеер]: Воспроизвожу {len(audio_chunk)} байт аудио...")
            stream.write(audio_chunk)
        except asyncio.TimeoutError:
            continue
    stream.stop_stream(); stream.close(); p.terminate()
    print("🎧 Аудиоплеер остановлен.")


async def start_gemini_dialogue(mic_stream, audio_queue):
    """
    Реализует диалог по правильной схеме "Запись-Отправка-Получение".
    """
    print("\n🚀 Подключаюсь к Gemini через Live API...")

    client = genai.Client(
        vertexai=True,
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
    )

    config = {
        "response_modalities": ["AUDIO"]
    }

    try:
        async with client.aio.live.connect(model=GEMINI_MODEL_NAME, config=config) as session:
            print("✅ Соединение установлено. Говорите, я запишу вашу фразу целиком.")

            # --- Фаза 1: Запись ---
            audio_chunks = []
            is_speaking = False
            silence_frames = 0
            SILENCE_THRESHOLD = 35 

            while True:
                audio_chunk = mic_stream.read(CHUNK, exception_on_overflow=False)
                unpacked_chunk = struct.unpack(f'{CHUNK}h', audio_chunk)
                is_currently_speaking = any(abs(sample) > 500 for sample in unpacked_chunk)
                if is_currently_speaking:
                    if not is_speaking: print("🎤 Обнаружена речь, идет запись...")
                    is_speaking = True
                    silence_frames = 0
                    audio_chunks.append(audio_chunk)
                elif is_speaking:
                    silence_frames += 1
                    audio_chunks.append(audio_chunk) 
                    if silence_frames > SILENCE_THRESHOLD:
                        print("🔇 Обнаружена тишина. Запись завершена.")
                        break
                if not is_speaking and not audio_chunks and silence_frames > 50:
                    print("🤷‍♂️ Не было произнесено ни слова. Возврат в режим ожидания.")
                    return
                silence_frames +=1
            if not audio_chunks: return

            # --- Фаза 2: Отправка ---
            print("📦 Упаковываю аудио и инструкцию в один запрос...")
            full_audio_data = b''.join(audio_chunks)

            content = types.Content(role="user", parts=[
                types.Part(text="Ты — голосовой ассистент по имени Марвин. Ответь на следующий аудио-запрос коротко и по делу."),
                types.Part(inline_data=types.Blob(mime_type="audio/pcm;rate=16000", data=full_audio_data))
            ])
            
            await session.send_client_content(turns=content, turn_complete=True)
            print("✅ Запрос отправлен. Ожидаю ответ...")

            # --- Фаза 3: Получение ---
            async for response in session.receive():
                print("  <- 📦 [Получатель]: Получен пакет от сервера!")
                if response.server_content and response.server_content.model_turn:
                    print("    - ✅ Пакет содержит контент от модели.")
                    for part in response.server_content.model_turn.parts:
                        
                        # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
                        # Правильно извлекаем аудио из part.inline_data.data
                        if part.inline_data and part.inline_data.data:
                            audio_data = part.inline_data.data
                            print(f"      - ✅✅✅ Найден аудио-фрагмент! Размер: {len(audio_data)} байт.")
                            await audio_queue.put(audio_data)
                        else:
                            print("      - ❌ В этой части нет аудио.")


    except Exception as e:
        print(f"⚠️ Произошла ошибка во время диалога с Gemini: {e}")

    print("🔚 Диалог завершен. Возвращаюсь в режим ожидания.")


async def main():
    porcupine = None
    p_audio = None
    mic_stream = None
    player_task = None
    audio_queue = None

    try:
        porcupine = pvporcupine.create(access_key=PICOVOICE_ACCESS_KEY, keyword_paths=KEYWORD_PATHS)
        print(f"✅ Движок Porcupine для '{os.path.basename(KEYWORD_PATHS[0])}' запущен.")
        
        p_audio = pyaudio.PyAudio()

        print(f"⚙️  Инициализация Vertex AI для проекта '{GOOGLE_CLOUD_PROJECT}'...")
        vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)
        print("✅ Инициализация Vertex AI завершена.")

        audio_queue = asyncio.Queue()
        player_task = asyncio.create_task(play_audio_from_queue(audio_queue))

        while True:
            print("\n" + "="*40)
            print("🤖 Состояние: Жду активационную команду 'Марвин'...")
            
            mic_stream = p_audio.open(
                rate=porcupine.sample_rate, channels=CHANNELS, format=pyaudio.paInt16,
                input=True, frames_per_buffer=PORCUPINE_FRAME_LENGTH
            )
            
            while True:
                pcm = mic_stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)
                keyword_index = porcupine.process(pcm_unpacked)
                if keyword_index >= 0:
                    print("✅ 'Марвин' обнаружен!")
                    break
            
            mic_stream.close()

            mic_stream = p_audio.open(
                rate=RATE, channels=CHANNELS, format=FORMAT,
                input=True, frames_per_buffer=CHUNK
            )
            
            await start_gemini_dialogue(mic_stream, audio_queue)
            
            mic_stream.close()

    except pvporcupine.PorcupineError as e:
        print(f"❌ Ошибка инициализации Porcupine: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Программа остановлена пользователем.")
    finally:
        if porcupine: porcupine.delete()
        if mic_stream and mic_stream.is_active(): mic_stream.close()
        if p_audio: p_audio.terminate()
        if audio_queue: await audio_queue.put(None)
        if player_task: await player_task
        print("🤖 Все системы выключены. До свидания!")

if __name__ == "__main__":
    asyncio.run(main())
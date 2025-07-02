import asyncio
import os
import struct
import string # Импортируем для работы с пунктуацией
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
# Ключевые фразы для завершения диалога
STOP_PHRASES = ["стоп", "хватит", "до свидания", "спасибо хватит", "завершить диалог"]
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
            stream.write(audio_chunk)
        except asyncio.TimeoutError:
            continue
    stream.stop_stream(); stream.close(); p.terminate()
    print("🎧 Аудиоплеер остановлен.")


async def start_gemini_dialogue(mic_stream, audio_queue):
    """
    Реализует полноценный диалог, который продолжается до команды "стоп".
    """
    print("\n🚀 Подключаюсь к Gemini через Live API...")

    client = genai.Client(
        vertexai=True,
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
    )

    # Включаем транскрипцию для распознавания стоп-слов
    config = {
        "response_modalities": ["AUDIO"],
        "input_audio_transcription": {}, 
    }

    try:
        async with client.aio.live.connect(model=GEMINI_MODEL_NAME, config=config) as session:
            print("✅ Соединение установлено.")

            # --- Фаза 1: Отправка системной инструкции ---
            print("🗣️  Отправляю системную инструкцию...")
            system_prompt = types.Content(role="user", parts=[
                types.Part(text="Ты — голосовой ассистент по имени Марвин. Отвечай на запросы коротко и по делу.")
            ])
            await session.send_client_content(turns=system_prompt, turn_complete=True)

            # --- Фаза 2: Диалоговый цикл ---
            while True: 
                print("\n🎙️  Слушаю ваш следующий вопрос... (Скажите 'стоп' или 'хватит' для завершения)")

                # --- Запись очередного вопроса ---
                audio_chunks = []
                is_speaking = False
                silence_frames = 0
                SILENCE_THRESHOLD = 35 

                while True:
                    audio_chunk = mic_stream.read(CHUNK, exception_on_overflow=False)
                    unpacked_chunk = struct.unpack(f'{CHUNK}h', audio_chunk)
                    is_currently_speaking = any(abs(sample) > 500 for sample in unpacked_chunk)
                    if is_currently_speaking:
                        if not is_speaking: print("   -> 🎤 Обнаружена речь, идет запись...")
                        is_speaking = True
                        silence_frames = 0
                        audio_chunks.append(audio_chunk)
                    elif is_speaking:
                        silence_frames += 1
                        audio_chunks.append(audio_chunk) 
                        if silence_frames > SILENCE_THRESHOLD:
                            print("   -> 🔇 Обнаружена тишина. Запись завершена.")
                            break
                    if not is_speaking and not audio_chunks and silence_frames > 50:
                        print("   -> 🤷‍♂️ Не было произнесено ни слова.")
                        continue
                    silence_frames +=1
                
                if not audio_chunks: continue

                # --- Отправка записанного аудио ---
                full_audio_data = b''.join(audio_chunks)
                audio_part = types.Part(inline_data=types.Blob(mime_type="audio/pcm;rate=16000", data=full_audio_data))
                await session.send_client_content(turns=types.Content(role="user", parts=[audio_part]), turn_complete=True)
                print("   -> ✅ Запрос отправлен. Ожидаю ответ...")

                # --- Получение ответа и проверка на стоп-слово ---
                stop_conversation = False
                full_transcript = ""
                async for response in session.receive():
                    
                    # --- ИСПРАВЛЕНИЕ ЗДЕСЬ: ПРАВИЛЬНЫЙ ПУТЬ К ТРАНСКРИПЦИИ ---
                    if response.server_content and response.server_content.input_transcription:
                        # Собираем полный транскрипт, т.к. он может приходить по частям
                        full_transcript += response.server_content.input_transcription.text
                    
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                await audio_queue.put(part.inline_data.data)

                # Проверяем стоп-слова после получения всех частей транскрипта
                if full_transcript:
                    print(f"   -> [Вы сказали]: '{full_transcript}'")
                    translator = str.maketrans('', '', string.punctuation)
                    clean_transcript = full_transcript.lower().translate(translator)
                    if any(phrase in clean_transcript for phrase in STOP_PHRASES):
                        print("   -> 🛑 Обнаружено стоп-слово. Завершаю диалог.")
                        stop_conversation = True
                
                if stop_conversation:
                    break # Выходим из диалогового цикла

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
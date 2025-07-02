import asyncio
import os
import struct
import pyaudio
import pvporcupine
import vertexai
from google import genai
# ИСПРАВЛЕНИЕ: Мы импортируем types, чтобы получить доступ к types.Blob
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
            stream.write(audio_chunk)
        except asyncio.TimeoutError:
            continue
    stream.stop_stream(); stream.close(); p.terminate()
    print("🎧 Аудиоплеер остановлен.")


async def start_gemini_dialogue(mic_stream, audio_queue):
    """
    Основная логика диалога с Gemini, используя Live API с правильными объектами.
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
            print("✅ Соединение установлено. Говорите. Для завершения фразы просто замолчите.")

            is_dialogue_active = True

            async def send_audio():
                nonlocal is_dialogue_active
                is_speaking = False
                silence_frames = 0
                SILENCE_THRESHOLD = 35

                while is_dialogue_active:
                    audio_chunk = mic_stream.read(CHUNK, exception_on_overflow=False)

                    unpacked_chunk = struct.unpack(f'{CHUNK}h', audio_chunk)
                    is_currently_speaking = any(abs(sample) > 500 for sample in unpacked_chunk)

                    if is_currently_speaking:
                        if not is_speaking: print("🎤 Обнаружена речь...")
                        is_speaking = True
                        silence_frames = 0
                    elif is_speaking:
                        silence_frames += 1
                    
                    # --- ИСПРАВЛЕНИЕ #1: ПРАВИЛЬНАЯ УПАКОВКА АУДИО ---
                    # Мы создаем объект Part, передавая ему inline_data,
                    # который содержит объект Blob с нашими аудио-байтами и MIME-типом.
                    part = types.Part(inline_data=types.Blob(mime_type="audio/pcm;rate=16000", data=audio_chunk))
                    
                    await session.send_client_content(
                        turns=types.Content(role="user", parts=[part])
                    )
                    
                    if is_speaking and silence_frames > SILENCE_THRESHOLD:
                        print("🔇 Обнаружена тишина. Завершаю реплику...")
                        await session.send_client_content(turn_complete=True)
                        is_dialogue_active = False
                
            async def receive_audio():
                async for response in session.receive():
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.audio:
                                await audio_queue.put(part.audio.data)

            send_task = asyncio.create_task(send_audio())
            receive_task = asyncio.create_task(receive_audio())
            await asyncio.gather(send_task, receive_task)

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

        # --- ИСПРАВЛЕНИЕ #2: ПРАВИЛЬНЫЙ ЦИКЛ УПРАВЛЕНИЯ МИКРОФОНОМ ---
        while True:
            print("\n" + "="*40)
            print("🤖 Состояние: Жду активационную команду 'Марвин'...")
            
            # 1. Открываем микрофон для Porcupine
            mic_stream = p_audio.open(
                rate=porcupine.sample_rate, channels=CHANNELS, format=pyaudio.paInt16,
                input=True, frames_per_buffer=PORCUPINE_FRAME_LENGTH
            )
            
            # 2. Слушаем, пока не услышим "Марвин"
            while True:
                pcm = mic_stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)
                keyword_index = porcupine.process(pcm_unpacked)
                if keyword_index >= 0:
                    print("✅ 'Марвин' обнаружен!")
                    break
            
            # 3. Закрываем микрофон Porcupine
            mic_stream.close()

            # 4. Открываем микрофон заново с настройками для Gemini
            mic_stream = p_audio.open(
                rate=RATE, channels=CHANNELS, format=FORMAT,
                input=True, frames_per_buffer=CHUNK
            )
            
            # 5. Запускаем диалог (даже если он упадет, мы вернемся в начало цикла)
            await start_gemini_dialogue(mic_stream, audio_queue)
            
            # 6. Закрываем микрофон Gemini
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
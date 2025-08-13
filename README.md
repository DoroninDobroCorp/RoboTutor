# RobotTutor: Голосовой ассистент (Mac → Raspberry Pi)

Надёжный минимальный офлайн ASR (Vosk) + локальный TTS (pyttsx3) + ответы ИИ (OpenAI). Подходит для разработки на macOS и запуска на Raspberry Pi.

## Возможности
- Автодетект речи с VAD (webrtcvad) и авто-стопом по тишине.
- Распознавание речи офлайн (Vosk, ru/en).
- Ответы от LLM (OpenAI, модель по умолчанию `gpt-4o-mini`).
- Озвучка ответа локально (pyttsx3: macOS — NSSpeechSynthesizer, Raspberry Pi — eSpeak).
- Режим push-to-talk: нажмите Enter, говорите, авто-стоп по тишине.

## Требования
- Python 3.10+
- Аккаунт OpenAI и переменная окружения `OPENAI_API_KEY`.

## Установка (macOS)
1) Установите PortAudio (для sounddevice):
   - Если есть Homebrew: `brew install portaudio`
2) Установите зависимости Python:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
3) Экспортируйте ключ:
```
export OPENAI_API_KEY=sk-...  # замените на свой
```
4) Запустите:
```
python main.py
```
Первый запуск скачает и распакует Vosk-модель (если её нет).

## Установка (Raspberry Pi OS)
1) Обновите систему и установите системные зависимости:
```
sudo apt update && sudo apt install -y python3-venv python3-dev portaudio19-dev espeak ffmpeg unzip
```
2) Установите Python-зависимости:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
3) Экспортируйте ключ:
```
export OPENAI_API_KEY=sk-...  # замените на свой
```
4) Запустите:
```
python main.py
```

## Настройка
Правьте параметры в `config.yaml`:
- `assistant.model` — модель OpenAI (например, `gpt-4o` для лучшего качества).
- `asr.language` — `ru` или `en`. Подберите подходящую Vosk-модель.
- `asr.vad.*` — чувствительность VAD и таймауты.
- `tts.rate`, `tts.voice` — скорость и голос TTS.

## Советы по задержке
- Оставьте `sample_rate: 16000` и `frame_ms: 30`.
- `silence_ms_to_end: 700` обычно даёт естественную остановку.
- На слабых платформах используйте краткие реплики ассистента.

## Возможные улучшения
- Заменить TTS на Piper для лучшего русского голоса на ARM.
- Добавить streaming-инференс (ASR/LLM/TTS) для сверхнизкой задержки.
- Кешировать Vosk-модель и выбирать язык автоматически.

## Запуск
- Нажмите Enter, говорите.
- По окончании пауза — авто-стоп. Текст распознается, отправляется в LLM, ответ озвучивается.
- `Ctrl+C` — выход.

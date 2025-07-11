import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
import google.generativeai as genai
import os
import subprocess
import time
import re
import platform
import sys
import hashlib
import json

# --- Класс для цветов в консоли (Палитра "Nordic Calm") ---
class Colors:
    # --- Основные цвета ---
    FAIL = '\033[38;2;191;97;106m'      # Красный (Aurora Red)
    OKGREEN = '\033[38;2;163;190;140m'   # Зеленый (Aurora Green)
    WARNING = '\033[38;2;235;203;139m'   # Желтый (Aurora Yellow)
    OKBLUE = '\033[38;2;94;129;172m'     # Голубой (Polar Night Blue)
    HEADER = '\033[38;2;180;142;173m'   # Пурпурный (Aurora Purple)
    CYAN = '\033[38;2;136;192;208m'     # Бирюзовый (Aurora Cyan)
    
    # --- Стили ---
    ENDC = '\033[0m'                    # Сброс
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # --- Дополнительные оттенки для разнообразия (если понадобятся) ---
    GREY = '\033[38;2;106;114;128m'      # Серый для второстепенной информации

# --- НАСТРОЙКИ ---
GOOGLE_API_KEY = "AIzaSyCGVITo4g7NqkNXpv2JjgiIguPWvqnbnWM"

GOOGLE_CLOUD_PROJECT = "useful-gearbox-464618-v3"
GOOGLE_CLOUD_LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-pro"

CONTEXT_SCRIPT = 'AskGpt.py'
CONTEXT_FILE = 'message_1.txt'
HISTORY_FILE = 'sloth_history.json'
ALLOWED_COMMANDS = (
    "sed", "rm", "mv", "touch", "mkdir", "npm", "npx", "yarn", "pnpm", "git", "echo", "./", "cat"
)
MAX_ITERATIONS = 15
API_TIMEOUT_SECONDS = 600

# --- Глобальные переменные для модели ---
model = None
ACTIVE_API_SERVICE = "N/A"
# Флаг для запоминания отказа Google AI API В ТЕКУЩЕЙ СЕССИИ. Сбрасывается при каждом запуске.
GOOGLE_AI_HAS_FAILED_THIS_SESSION = False

def initialize_model():
    """
    Инициализирует модель Gemini.
    Приоритет: Google API Key. Запасной вариант: Vertex AI.
    Запоминает, если Google API Key отказал в текущей сессии.
    """
    global model, ACTIVE_API_SERVICE, GOOGLE_AI_HAS_FAILED_THIS_SESSION

    print(f"{Colors.CYAN}⚙️  ЛОГ: Начинаю конфигурацию. Модель: {MODEL_NAME}{Colors.ENDC}")

    generation_config = {
        "temperature": 1, "top_p": 1, "top_k": 1, "max_output_tokens": 32768
    }

    # Проверяем флаг. Если API ключ уже отказал в этой сессии, не пытаемся его использовать снова.
    if not GOOGLE_AI_HAS_FAILED_THIS_SESSION and GOOGLE_API_KEY and "ВАШ_API_КЛЮЧ" not in GOOGLE_API_KEY:
        # <<< ИЗМЕНЕНИЕ: Более явное логирование >>>
        print(f"{Colors.CYAN}🔑 ЛОГ: Пробую приоритетный сервис: Google AI (API Key)...{Colors.ENDC}")
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            genai_safety_settings = {
                'HARM_CATEGORY_HARASSMENT': 'block_medium_and_above',
                'HARM_CATEGORY_HATE_SPEECH': 'block_medium_and_above',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'block_medium_and_above',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'block_none'
            }
            model = genai.GenerativeModel(
                model_name=MODEL_NAME,
                generation_config=generation_config,
                safety_settings=genai_safety_settings
            )
            # Короткий тестовый запрос для проверки работоспособности ключа
            model.generate_content("test", request_options={"timeout": 60})
            ACTIVE_API_SERVICE = "Google AI (API Key)"
            print(f"{Colors.OKGREEN}✅ ЛОГ: Успешно инициализировано через {ACTIVE_API_SERVICE}.{Colors.ENDC}")
            return
        except Exception as e:
            print(f"{Colors.WARNING}⚠️  ПРЕДУПРЕЖДЕНИЕ: Не удалось инициализировать через Google AI API Key: {e}{Colors.ENDC}")
            print(f"{Colors.CYAN}🔄 ЛОГ: Переключаюсь на запасной вариант (Vertex AI) для этой сессии.{Colors.ENDC}")
            GOOGLE_AI_HAS_FAILED_THIS_SESSION = True # Запоминаем отказ на время этой сессии
            model = None
    elif GOOGLE_AI_HAS_FAILED_THIS_SESSION:
        print(f"{Colors.CYAN}🔑 ЛОГ: Google AI API ранее отказал в этой сессии. Сразу использую Vertex AI.{Colors.ENDC}")
    else:
        print(f"{Colors.CYAN}🔑 ЛОГ: API ключ не указан. Использую Vertex AI.{Colors.ENDC}")

    try:
        print(f"{Colors.CYAN}🔩 ЛОГ: Попытка инициализации через Vertex AI...{Colors.ENDC}")
        vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)
        vertex_safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        model = GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
            safety_settings=vertex_safety_settings
        )
        ACTIVE_API_SERVICE = "Vertex AI"
        print(f"{Colors.OKGREEN}✅ ЛОГ: Vertex AI SDK успешно инициализирован. Используется {ACTIVE_API_SERVICE}.{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}❌ ЛОГ: КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать модель ни одним из способов.{Colors.ENDC}")
        print(f"{Colors.FAIL}   - Ошибка Vertex AI: {e}{Colors.ENDC}")
        sys.exit(1)


# --- БЛОК ПРОМПТ-ШАБЛОНОВ ---

def get_command_rules():
    return f"""
Ты — AI-ассистент в автоматизированной системе. Твоя задача — анализировать код и генерировать shell-команды для его изменения.

**КЛЮЧЕВЫЕ ПРАВИЛА:**

1.  **СТРАТЕГИЯ ИЗМЕНЕНИЙ:**
    *   **Точечные правки (`sed`):** Используй `sed` **только для простых, однострочных** замен, которые не содержат сложных спецсимволов.
    *   **Полная перезапись (`cat`):** Используй `cat <<'EOF' > path/to/file.txt ... EOF`, если нужно изменить **сигнатуру функции, JSX-разметку, несколько строк кода подряд или строки со сложными кавычками/символами**. Это предпочтительный метод для сложного рефакторинга. **Не пытайся использовать `sed` для этих целей!**

2.  **ПЕРЕЗАПИСЬ ФАЙЛОВ (Высокий риск!):**
    *   **ВНИМАНИЕ:** При использовании `cat` ты должен быть ПРЕДЕЛЬНО АККУРАТЕН. Всегда включай в блок `EOF` **полное и корректное** содержимое файла, сохраняя исходное форматирование.
    *   **СТРАТЕГИЯ "ОДИН БОЛЬШОЙ ЗА РАЗ":** Если твоя задача требует полной перезаписи **нескольких** больших файлов, изменяй **только один файл за одну итерацию**.

3.  **ФОРМАТ ОТВЕТА — ЭТО ЗАКОН:**
    *   **Действия:** Если нужны правки, твой ответ **ОБЯЗАН** содержать ДВА блока:
        1. Блок команд, обернутый в ```bash ... ```.
        2. Сразу после него — блок с кратким описанием твоей стратегии, обернутый в ```summary ... ```.
    *   **Завершение:**
        *   Если задача полностью решена и **не требует ручных действий от человека**, напиши **только** `ГОТОВО`. После этого слова добавь блок ```done_summary ... ``` с кратким перечнем ключевых шагов, которые привели к решению.
        *   Если после твоих правок **человеку нужно выполнить команды** (например, `npm start`), сначала напиши `ГОТОВО`, затем добавь блок ```done_summary ... ```, и только потом — блок ```manual ... ``` с инструкциями.

4.  **ФОКУС И ПРАГМАТИЗМ:**
    *   Твоя главная цель — решить **исходную задачу** пользователя. Как только функциональность заработает, напиши `ГОТОВО`.
    *   **Не занимайся перфекционизмом:** не исправляй стиль кода, не делай рефакторинг и не исправляй другие проблемы, не связанные с задачей, если они не являются прямой причиной сбоя.

5.  **РАЗРЕШЕННЫЕ КОМАНДЫ:** `{', '.join(ALLOWED_COMMANDS)}`. Команды, не входящие в этот список, должны быть помещены в блок ```manual```.

6.  **ПОЛНОТА КОДА:** **ЗАПРЕЩЕНО** использовать плейсхолдеры, многоточия (...) или комментарии (`// ... остальной код`) для сокращения блоков кода. Всегда предоставляй полный, готовый к выполнению код.
"""

def get_initial_prompt(context, task, fix_history=None):
    history_prompt_section = ""
    if fix_history:
        history_prompt_section = f"""
--- ИСТОРИЯ ПРЕДЫДУЩЕГО РЕШЕНИЯ, КОТОРОЕ ОКАЗАЛОСЬ НЕВЕРНЫМ ---
Ты уже пытался решить эту задачу и сообщил 'ГОТОВО', но это было ошибкой.
Вот краткое изложение твоих предыдущих действий:
{fix_history}
--- КОНЕЦ ИСТОРИИ ---
Проанализируй свою прошлую ошибку, новую информацию от пользователя и начни заново.
"""
    return f"{get_command_rules()}\n{history_prompt_section}\n--- КОНТЕКСТ ПРОЕКТА ---\n{context}\n--- КОНЕЦ КОНТЕКСТА ---\nЗадача: {task}\nПроанализируй задачу и предоставь ответ, строго следуя правилам."


def get_review_prompt(context, goal, iteration_count, attempt_history):
    iteration_info = ""
    if iteration_count >= 4:
        iteration_info = f"""
**ОСОБОЕ ВНИМАНИЕ (Итерация {iteration_count}):**
Ты уже сделал несколько шагов. Пожалуйста, проанализируй проблему более комплексно.
"""
    history_info = ""
    if attempt_history:
        history_info = (
            "--- ИСТОРИЯ ПРЕДЫДУЩИХ ПОПЫТОК ---\n"
            "Вот что ты уже сделал. Проанализируй всю историю (и успехи, и неудачи), чтобы выработать следующий шаг.\n\n"
            + "\n---\n".join(f"**На итерации {i+1}:**\n{s}" for i, s in enumerate(attempt_history)) +
            "\n\n--- КОНЕЦ ИСТОРИИ ---\n"
        )
    return f"""{get_command_rules()}
{iteration_info}
{history_info}
**ВАЖНО:** Предыдущий шаг выполнен. Код ниже — это **обновленное состояние** проекта.

**Твоя задача — ВЕРИФИКАЦИЯ:**
1.  Проанализируй **текущий** код, учитывая **всю историю твоих действий**.
2.  Если исходная цель достигнута, напиши `ГОТОВО`. Не ищи дополнительных улучшений.
3.  Если цель НЕ достигнута, предоставь следующий блок команд и описание (`summary`) для следующего шага.

--- КОНТЕКСТ ПРОЕКТА (ОБНОВЛЕННЫЙ) ---
{context}
--- КОНЕЦ КОНТЕКСТА ---

Напоминаю ИСХОДНУЮ ЦЕЛЬ: {goal}
"""

def get_error_fixing_prompt(failed_command, error_message, goal, context, iteration_count, attempt_history):
    iteration_info = ""
    if iteration_count >= 4:
        iteration_info = f"""
**ОСОБОЕ ВНИМАНИЕ (Итерация {iteration_count}):**
Ты уже сделал несколько шагов, и сейчас произошла ошибка. Пожалуйста, проанализируй проблему более комплексно.
"""
    history_info = ""
    if attempt_history:
        history_info = (
            "--- ИСТОРИЯ ПРЕДЫДУЩИХ ПОПЫТОК ---\n"
            "Вот что ты уже сделал. Проанализируй всю историю (и успехи, и неудачи), чтобы понять, почему текущая команда провалилась.\n\n"
            + "\n---\n".join(f"**На итерации {i+1}:**\n{s}" for i, s in enumerate(attempt_history)) +
            "\n--- КОНЕЦ ИСТОРИИ ---\n"
        )

    return f"""{get_command_rules()}
{iteration_info}
{history_info}
**ВАЖНО:** Твоя задача — исправить ошибку, которая только что произошла. Не пиши 'ГОТОВО'.

--- ДАННЫЕ О ТЕКУЩЕЙ ОШИБКЕ ---
КОМАНДА: {failed_command}
СООБЩЕНИЕ (stderr): {error_message}
--- КОНЕЦ ДАННЫХ ОБ ОШИБКЕ ---

Исходная ЦЕЛЬ была: {goal}

Проанализируй **текущую ошибку в контексте всей истории** и предоставь **исправленный блок команд** и описание (`summary`).

--- КОНТЕКСТ, ГДЕ ПРОИЗОШЛА ОШИБКА ---
{context}
--- КОНЕЦ КОНТЕКСТА ---
"""


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def notify_user(message):
    print(f"{Colors.OKBLUE}📢 ЛОГ: Отправляю уведомление: {message.replace(Colors.ENDC, '')}{Colors.ENDC}")
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(['afplay', '/System/Library/Sounds/Sosumi.aiff'], check=True, timeout=5)
        elif system == "Linux":
            subprocess.run(['zenity', '--info', '--text', message, '--title', 'Sloth Script', '--timeout=10', '--window-icon=info'], check=True, timeout=10)
            subprocess.run(['aplay', '/usr/share/sounds/alsa/Front_Center.wav'], check=True)
        elif system == "Windows":
            command = f'powershell -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show(\'{message}\', \'Sloth Script\');"'
            subprocess.run(command, shell=True, check=True, timeout=30)
    except Exception as e:
        print(f"{Colors.WARNING}⚠️  ПРЕДУПРЕЖДЕНИЕ: Не удалось отправить системное уведомление. Ошибка: {e}.{Colors.ENDC}")

def get_project_context():
    print(f"{Colors.CYAN}🔄 ЛОГ: Обновляю контекст проекта...{Colors.ENDC}")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_to_run_path = os.path.join(script_dir, CONTEXT_SCRIPT)
        context_file_path = os.path.join(script_dir, CONTEXT_FILE)

        if os.path.exists(context_file_path): os.remove(context_file_path)

        subprocess.run(['python3', script_to_run_path], check=True, capture_output=True, text=True, encoding='utf-8')

        with open(context_file_path, 'r', encoding='utf-8') as f:
            context_data = f.read()

        print(f"{Colors.OKGREEN}✅ ЛОГ: Контекст успешно обновлен. Размер: {len(context_data)} символов.{Colors.ENDC}")
        return context_data
    except Exception as e:
        print(f"{Colors.FAIL}❌ ЛОГ: КРИТИЧЕСКАЯ ОШИБКА в get_project_context: {e}{Colors.ENDC}")
        return None

def extract_todo_block(text):
    match = re.search(r"```bash\s*(.*?)\s*```", text, re.DOTALL)
    if match: return match.group(1).strip()
    return None

def extract_summary_block(text):
    match = re.search(r"```summary\s*(.*?)\s*```", text, re.DOTALL)
    if match: return match.group(1).strip()
    return None

def extract_manual_steps_block(text):
    match = re.search(r"```manual\s*(.*?)\s*```", text, re.DOTALL)
    if match: return match.group(1).strip()
    return None

def extract_done_summary_block(text):
    match = re.search(r"```done_summary\s*(.*?)\s*```", text, re.DOTALL)
    if match: return match.group(1).strip()
    return None

def get_file_hash(filepath):
    if not os.path.exists(filepath) or os.path.isdir(filepath): return None
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

def apply_shell_commands(commands_str):
    print(f"{Colors.OKBLUE}  [Детали] Вход в apply_shell_commands().{Colors.ENDC}")

    filepaths = re.findall(r'[\w\/\-\.]+\.[\w]+', commands_str)
    hashes_before = {fp: get_file_hash(fp) for fp in filepaths if os.path.exists(fp) and not os.path.isdir(fp)}

    try:
        is_macos = platform.system() == "Darwin"
        commands_str_adapted = re.sub(r"sed -i ", "sed -i '.bak' ", commands_str) if is_macos else commands_str

        full_command = f"set -e\n{commands_str_adapted}"

        print(f"{Colors.WARNING}⚡️ ЛОГ: Выполняю блок команд (с set -e):\n---\n{full_command}\n---{Colors.ENDC}")
        result = subprocess.run(['bash', '-c', full_command], capture_output=True, text=True, encoding='utf-8')

        if result.returncode != 0:
            error_msg = f"Команда завершилась с ненулевым кодом выхода ({result.returncode}).\nОшибка (STDERR): {result.stderr.strip()}"
            print(f"{Colors.FAIL}❌ ЛОГ: КРИТИЧЕСКАЯ ОШИБКА при выполнении блока команд.\n{error_msg}{Colors.ENDC}")
            return False, commands_str, result.stderr.strip() or "Команда провалилась без вывода в stderr."

        if result.stderr:
            print(f"{Colors.WARNING}⚠️  ПРЕДУПРЕЖДЕНИЕ (STDERR от успешной команды):\n{result.stderr.strip()}{Colors.ENDC}")

        if is_macos: subprocess.run("find . -name '*.bak' -delete", shell=True, check=True)

        hashes_after = {fp: get_file_hash(fp) for fp in hashes_before.keys()}

        if hashes_before and all(hashes_before.get(fp) == hashes_after.get(fp) for fp in hashes_before):
            error_msg = "Команда выполнилась успешно, но не изменила ни одного из целевых файлов. Вероятно, шаблон (например, в sed) не был найден или путь к файлу неверен."
            final_error_message = result.stderr.strip() if result.stderr else error_msg
            print(f"{Colors.FAIL}❌ ЛОГ: ОШИБКА ЛОГИКИ: {error_msg}{Colors.ENDC}")
            if result.stderr: print(f"Причина из STDERR: {final_error_message}")
            return False, commands_str, final_error_message

        print(f"{Colors.OKGREEN}✅ ЛОГ: Блок команд успешно выполнен и изменил файлы.{Colors.ENDC}")
        return True, None, None
    except Exception as e:
        print(f"{Colors.FAIL}❌ ЛОГ: Непредвиденная ОШИБКА в apply_shell_commands: {e}{Colors.ENDC}")
        return False, commands_str, str(e)

def extract_filepath_from_command(command):
    parts = command.split()
    for part in reversed(parts):
        if part in ['-c', '-e', '<<']: continue
        clean_part = part.strip("'\"")
        if ('/' in clean_part or '.' in clean_part) and os.path.exists(clean_part):
            return clean_part
    return None

def save_prompt_for_debugging(prompt_text):
    try:
        with open("sloth_debug_prompt.txt", "w", encoding='utf-8') as f:
            f.write(prompt_text)
        print(f"{Colors.OKBLUE}   - Отладочная информация: Полный промпт сохранен в 'sloth_debug_prompt.txt'.{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.WARNING}   - ВНИМАНИЕ: Не удалось сохранить отладочный файл промпта: {e}{Colors.ENDC}")

def send_request_to_model(prompt_text, iteration_count):
    global model, GOOGLE_AI_HAS_FAILED_THIS_SESSION
    try:
        prompt_size = len(prompt_text)
        print(f"{Colors.CYAN}🧠 ЛОГ: [Итерация {iteration_count}] Готовлю запрос в модель ({ACTIVE_API_SERVICE}).{Colors.ENDC}")
        print(f"{Colors.OKBLUE}   - Общий размер промпта: {prompt_size} символов.{Colors.ENDC}")
        save_prompt_for_debugging(prompt_text)

        if prompt_size > 100000:
             print(f"{Colors.WARNING}   - ВНИМАНИЕ: Размер промпта очень большой. Ответ может занять несколько минут.{Colors.ENDC}")

        request_options = {"timeout": API_TIMEOUT_SECONDS}
        print(f"{Colors.CYAN}⏳ ЛОГ: Отправляю запрос и ожидаю ответ... (таймаут: {API_TIMEOUT_SECONDS} секунд){Colors.ENDC}")
        response = model.generate_content(prompt_text, request_options=request_options)
        
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            reason_name = response.prompt_feedback.block_reason.name
            print(f"{Colors.FAIL}❌ ЛОГ: ЗАПРОС БЫЛ ЗАБЛОКИРОВАН API.{Colors.ENDC}")
            print(f"{Colors.FAIL}   - Причина блокировки: {reason_name}{Colors.ENDC}")
            print(f"{Colors.FAIL}   - Рейтинги безопасности: {response.prompt_feedback.safety_ratings}{Colors.ENDC}")
            raise ValueError(f"Промпт заблокирован из-за настроек безопасности. Причина: {reason_name}")

        if not response.candidates or response.candidates[0].finish_reason.name != "STOP":
            reason = response.candidates[0].finish_reason.name if response.candidates else "Кандидат не сгенерирован"
            safety_ratings = response.candidates[0].safety_ratings if response.candidates else "Нет данных"
            print(f"{Colors.FAIL}❌ ЛОГ: Ответ от модели был прерван или неполный.{Colors.ENDC}")
            print(f"{Colors.FAIL}   - Причина завершения: {reason}{Colors.ENDC}")
            print(f"{Colors.FAIL}   - Рейтинги безопасности ответа: {safety_ratings}{Colors.ENDC}")
            raise ValueError(f"Ответ от модели был прерван. Причина: {reason}")
        
        print(f"{Colors.OKGREEN}✅ ЛОГ: Ответ от модели получен успешно.{Colors.ENDC}")
        return response.text
        
    except Exception as e:
        print(f"{Colors.FAIL}❌ ЛОГ: ОШИБКА при запросе к API ({ACTIVE_API_SERVICE}): {e}{Colors.ENDC}")
        
        error_str = str(e).lower()
        is_quota_error = "quota" in error_str or "rate limit" in error_str or "exceeded" in error_str

        if ACTIVE_API_SERVICE == "Google AI (API Key)" and is_quota_error:
            print(f"{Colors.FAIL}🚨 ЛОГ: ОБНАРУЖЕНА ОШИБКА КВОТЫ В GOOGLE AI API!{Colors.ENDC}")
            print(f"{Colors.CYAN}   - Устанавливаю флаг отказа и перманентно (на эту сессию) переключаюсь на Vertex AI...{Colors.ENDC}")
            GOOGLE_AI_HAS_FAILED_THIS_SESSION = True
            model = None
            initialize_model()
        
        print(f"{Colors.CYAN}   - Проверьте файл 'sloth_debug_prompt.txt', чтобы увидеть точный промпт, вызвавший ошибку.{Colors.ENDC}")
        return None

def _read_multiline_input(prompt):
    print(prompt)
    lines = []
    empty_line_count = 0
    while empty_line_count < 3:
        try:
            line = input()
            if line:
                lines.append(line)
                empty_line_count = 0
            else:
                empty_line_count += 1
        except EOFError:
            break
    return '\n'.join(lines).strip()

def get_user_input():
    goal_prompt = (
        f"{Colors.HEADER}{Colors.BOLD}👋 Привет! Опиши свою основную цель.{Colors.ENDC}\n"
        f"{Colors.CYAN}💡 Пожалуйста, будь максимально точен и детален. "
        f"Чем лучше ты опишешь проблему и желаемый результат, тем быстрее я смогу помочь.\n"
        f"(Для завершения ввода, нажми Enter 3 раза подряд){Colors.ENDC}"
    )
    user_goal = _read_multiline_input(goal_prompt)

    if not user_goal:
        return None, None

    log_prompt = f"\n{Colors.HEADER}{Colors.BOLD}👍 Отлично. Теперь, если есть лог ошибки, вставь его. Если нет, просто нажми Enter 3 раза.{Colors.ENDC}"
    error_log = _read_multiline_input(log_prompt)

    return user_goal, error_log

def save_completion_history(goal, summary):
    history_data = {"previous_attempts": []}
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
        except json.JSONDecodeError:
            print(f"{Colors.WARNING}⚠️  ПРЕДУПРЕЖДЕНИЕ: Файл истории {HISTORY_FILE} поврежден. Создаю новый.{Colors.ENDC}")

    new_entry = {
        "initial_goal": goal,
        "solution_summary": summary
    }
    history_data.get("previous_attempts", []).insert(0, new_entry)

    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        print(f"{Colors.OKGREEN}💾 ЛОГ: История решения сохранена в {HISTORY_FILE}.{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}❌ ЛОГ: Не удалось сохранить историю решения: {e}{Colors.ENDC}")

def load_fix_history():
    if not os.path.exists(HISTORY_FILE):
        return None
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
        
        attempts = history_data.get("previous_attempts", [])
        if not attempts:
            return None
        
        last_attempt = attempts[0]
        
        text_history = (
            f"Это твоя самая последняя попытка решения, которая оказалась неверной:\n"
            f"  - Поставленная задача: {last_attempt.get('initial_goal', 'N/A')}\n"
            f"  - Твое 'решение': {last_attempt.get('solution_summary', 'N/A')}"
        )
        return text_history
    except Exception as e:
        print(f"{Colors.FAIL}❌ ЛОГ: Не удалось загрузить или прочитать файл истории {HISTORY_FILE}: {e}{Colors.ENDC}")
        return None


# --- ГЛАВНЫЙ ЦИКЛ ---
def main(is_fix_mode=False):
    """Основной рабочий цикл скрипта."""
    if is_fix_mode:
        print(f"{Colors.WARNING}🔧 ЛОГ: Запуск в режиме исправления (--fix).{Colors.ENDC}")

    user_goal, error_log = get_user_input()

    if not user_goal:
        raise ValueError("Цель не может быть пустой.")

    initial_task = user_goal
    if error_log:
        initial_task += "\n\n--- ЛОГ ОШИБКИ ДЛЯ АНАЛИЗА ---\n" + error_log

    project_context = get_project_context()
    if not project_context: raise ConnectionError("Не удалось получить первоначальный контекст проекта.")
    
    fix_history_content = None
    if is_fix_mode:
        fix_history_content = load_fix_history()
        if fix_history_content:
            print(f"{Colors.CYAN} historial ЛОГ: Загружена история последнего неудачного решения.{Colors.ENDC}")

    current_prompt = get_initial_prompt(project_context, initial_task, fix_history=fix_history_content)
    attempt_history = []

    for iteration_count in range(1, MAX_ITERATIONS + 1):
        print(f"\n{Colors.BOLD}{Colors.HEADER}🚀 --- АВТОМАТИЧЕСКАЯ ИТЕРАЦИЯ {iteration_count}/{MAX_ITERATIONS} (API: {ACTIVE_API_SERVICE}) ---{Colors.ENDC}")

        answer = send_request_to_model(current_prompt, iteration_count)
        if not answer:
            if model:
                print(f"{Colors.CYAN}🔄 ЛОГ: Ответ от модели не получен, пробую снова на той же итерации с новым API...{Colors.ENDC}")
                continue
            else:
                return "Критическая ошибка: Не удалось получить ответ от модели и переключиться на запасной API."
        
        if answer.strip().upper().startswith("ГОТОВО"):
            done_summary = extract_done_summary_block(answer)
            manual_steps = extract_manual_steps_block(answer)

            if not done_summary:
                print(f"{Colors.WARNING}⚠️ ПРЕДУПРЕЖДЕНИЕ: Модель сообщила 'ГОТОВО', но не предоставила блок `done_summary`. История не будет сохранена.{Colors.ENDC}")
                done_summary = "Модель не предоставила итоговое резюме."
            else:
                print(f"{Colors.OKGREEN}📄 ИТОГОВОЕ РЕЗЮМЕ ОТ МОДЕЛИ:\n{Colors.CYAN}{done_summary}{Colors.ENDC}")
                if is_fix_mode and os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                    print(f"{Colors.CYAN}🗑️  ЛОГ: Задача решена в режиме --fix. Старая история ({HISTORY_FILE}) очищена.{Colors.ENDC}")
                save_completion_history(user_goal, done_summary)


            final_message = f"{Colors.OKGREEN}✅ Задача выполнена успешно! (за {iteration_count} итераций){Colors.ENDC}"
            if manual_steps:
                final_message += f"\n\n{Colors.WARNING}✋ ВАЖНО: Требуются следующие ручные действия:{Colors.ENDC}\n" + "-"*20 + f"\n{manual_steps}\n" + "-"*20
            return final_message

        commands_to_run = extract_todo_block(answer)
        if not commands_to_run:
            print(f"{Colors.FAIL}❌ ЛОГ: Модель вернула ответ без команд и без статуса 'ГОТОВО'. Ответ модели:\n{answer}{Colors.ENDC}")
            with open("sloth_debug_bad_response.txt", "w", encoding='utf-8') as f:
                f.write(answer)
            return f"{Colors.FAIL}Модель не предоставила блок команд и не считает задачу выполненной. Ответ сохранен в sloth_debug_bad_response.txt{Colors.ENDC}"


        strategy_description = extract_summary_block(answer)
        if not strategy_description:
            strategy_description = f"Применена команда, начинающаяся с `{commands_to_run.splitlines()[0][:80]}...` (описание не предоставлено)"
            print(f"{Colors.WARNING}⚠️  ПРЕДУПРЕЖДЕНИЕ: Модель не предоставила блок summary. Используется авто-описание.{Colors.ENDC}")
        else:
             print(f"{Colors.CYAN}💡 Стратегия ассистента: '{strategy_description}'{Colors.ENDC}")

        print(f"\n{Colors.OKBLUE}🔧 Найдены shell-команды для применения:{Colors.ENDC}\n" + "-"*20 + f"\n{commands_to_run}\n" + "-"*20)

        success, failed_command, error_message = apply_shell_commands(commands_to_run)

        project_context = get_project_context()
        if not project_context: return f"{Colors.FAIL}Критическая ошибка: не удалось обновить контекст.{Colors.ENDC}"

        if success:
            history_entry = (
                f"**Стратегия:** {strategy_description}\n"
                f"  **Результат:** УСПЕХ"
            )
            attempt_history.append(history_entry)
            print(f"\n{Colors.CYAN}🧐 ЛОГ: Команды успешно применены. Готовлюсь к верификации.{Colors.ENDC}")
            current_prompt = get_review_prompt(project_context, user_goal, iteration_count + 1, attempt_history)
        else:
            history_entry = (
                f"**Стратегия:** {strategy_description}\n"
                f"  **Результат:** ПРОВАЛ\n"
                f"  **Ошибка (stderr):** {error_message}"
            )
            attempt_history.append(history_entry)

            print(f"\n{Colors.FAIL}🆘 ЛОГ: Обнаружена ошибка. Готовлю промпт для исправления.{Colors.ENDC}")

            filepath = extract_filepath_from_command(failed_command or "")
            error_context = f"--- КОНТЕКСТ ПРОЕКТА ---\n{project_context}\n--- КОНЕЦ КОНТЕКСТА ---"
            if filepath and os.path.exists(filepath) and not os.path.isdir(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f: file_content = f.read()
                    error_context = f"--- СОДЕРЖИМОЕ ФАЙЛА: {filepath} ---\n{file_content}\n--- КОНЕЦ СОДЕРЖИМОГО ФАЙЛА ---\n\n{error_context}"
                except Exception as e:
                    print(f"{Colors.WARNING}⚠️  ПРЕДУПРЕЖДЕНИЕ: Не удалось прочитать файл '{filepath}' для контекста ошибки: {e}{Colors.ENDC}")

            current_prompt = get_error_fixing_prompt(
                failed_command=failed_command, error_message=error_message,
                goal=user_goal, context=error_context, iteration_count=iteration_count + 1, attempt_history=attempt_history)

    return f"{Colors.WARNING}⌛ Достигнут лимит в {MAX_ITERATIONS} итераций. Задача не была завершена.{Colors.ENDC}"

if __name__ == "__main__":
    is_fix_mode = '--fix' in sys.argv or '-fix' in sys.argv

    if not is_fix_mode and os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
            print(f"{Colors.CYAN}🗑️  ЛОГ: Очищена старая история ({HISTORY_FILE}).{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.WARNING}⚠️  ПРЕДУПРЕЖДЕНИЕ: Не удалось удалить файл истории: {e}{Colors.ENDC}")
    
    if os.path.exists("sloth_debug_prompt.txt"):
        os.remove("sloth_debug_prompt.txt")
    if os.path.exists("sloth_debug_bad_response.txt"):
        os.remove("sloth_debug_bad_response.txt")


    initialize_model()
    final_status = "Работа завершена."
    try:
        if model:
            final_status = main(is_fix_mode)
        else:
            final_status = f"{Colors.FAIL}❌ Не удалось запустить основной цикл, так как модель не была инициализирована.{Colors.ENDC}"
    except KeyboardInterrupt:
        final_status = f"{Colors.OKBLUE}🔵 Процесс прерван пользователем.{Colors.ENDC}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        final_status = f"{Colors.FAIL}❌ Скрипт аварийно завершился с ошибкой: {e}{Colors.ENDC}"
    finally:
        print(f"\n{final_status}")
        notify_user(final_status)
        time.sleep(1)
        print(f"\n{Colors.BOLD}🏁 Скрипт завершил работу.{Colors.ENDC}")
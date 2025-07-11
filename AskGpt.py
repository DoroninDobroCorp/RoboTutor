#!/usr/bin/env python3
import os

# --- Настройки ---
MAX_CHARS = 7500000  # можно менять
OUTPUT_PREFIX = "message_"
FINISH_INSTRUCTION = "ГПТ, Я ЕЩЕ НЕ ЗАКОНЧИЛ - ПРОСТО КОРОТКО ОТВЕТЬ ОК И ВСЕ!!!"
BASE_DIR = os.getcwd()  # корневая папка — та, где запущен скрипт
TOP_N_FILES = 3    # НОВОЕ: сколько самых больших файлов помечать

# --- Списки игнорирования ---

# Список расширений файлов для игнорирования (через запятую).
# ИЗМЕНЕНО: Добавлены расширения изображений для игнорирования
USER_IGNORE_EXTENSIONS = ".png, .jpeg, .jpg"

# Список файлов-исключений, которые НУЖНО включить, даже если их расширение в списке игнорирования.
USER_INCLUDE_FILES = "" # "important-script.js, special-types.d.ts"

# Список конкретных файлов для игнорирования (через запятую).
USER_IGNORE_FILES = (
    # Go
    "go.mod, go.sum, "
    # Node.js
    "package.json, package-lock.json, yarn.lock, "
    # Common
    ".DS_Store, .gitignore, README.md, "
    # Specific user files
    "analyzer_wide, tsconfig.tsbuildinfo, chat_sender.py, chat_sender_g.py, sloth.py, sloth_debug_prompt.txt, sloth_debug_bad_response.txt"
)

# Список папок для игнорирования (через запятую).
USER_IGNORE_DIRS = (
    # Python
    "venv, .venv, __pycache__, .pytest_cache, *.egg-info, "
    # Node.js
    "node_modules, .next, dist, build, coverage, "
    # IDE & Tools
    ".git, .idea, .vscode, .claude, "
    # General
    "logs"
)

# --- Инициализация ---

SCRIPT_NAME_LOWER = os.path.basename(__file__).lower()
IGNORE_SET = set(name.strip().lower() for name in "".join(USER_IGNORE_FILES).split(",") if name.strip())
IGNORE_DIRS_SET = set(name.strip().lower() for name in "".join(USER_IGNORE_DIRS).split(",") if name.strip())
IGNORE_EXTENSIONS_SET = set(ext.strip().lower() for ext in USER_IGNORE_EXTENSIONS.split(",") if ext.strip())
INCLUDE_FILES_SET = set(name.strip().lower() for name in USER_INCLUDE_FILES.split(",") if name.strip())


def should_ignore_file(filename):
    filename_lower = filename.lower()
    if filename_lower in INCLUDE_FILES_SET:
        return False
    if filename_lower == SCRIPT_NAME_LOWER:
        return True
    if filename_lower in IGNORE_SET:
        return True
    if IGNORE_EXTENSIONS_SET:
        if any(filename_lower.endswith(ext) for ext in IGNORE_EXTENSIONS_SET):
            return True
    return False

def get_file_content(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return None

def calculate_sizes(root):
    file_sizes = {}
    dir_sizes = {}
    all_valid_paths = []

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if d.lower() not in IGNORE_DIRS_SET]
        
        current_dir_size = 0
        for f in filenames:
            if should_ignore_file(f):
                continue

            filepath = os.path.join(dirpath, f)
            # Для бинарных файлов, которые мы игнорируем, get_file_content не вызывается
            # Но если вдруг какое-то расширение не добавлено, лучше обрабатывать ошибку
            content = get_file_content(filepath)

            if content is not None:
                size = len(content)
                file_sizes[filepath] = size
                current_dir_size += size
                all_valid_paths.append(filepath)

        dir_sizes[dirpath] = current_dir_size

    sorted_dirs = sorted(dir_sizes.keys(), key=lambda x: x.count(os.sep), reverse=True)
    for path in sorted_dirs:
        parent = os.path.dirname(path)
        if parent in dir_sizes and parent != path:
            dir_sizes[parent] += dir_sizes[path]
            
    return file_sizes, dir_sizes, all_valid_paths


def build_tree(root, file_sizes, dir_sizes, top_files_set):
    """
    Строит строковое представление дерева папок с указанием размеров.
    НОВОЕ: Принимает top_files_set для пометки самых больших файлов.
    """
    tree_lines = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if d.lower() not in IGNORE_DIRS_SET]
        
        rel_path = os.path.relpath(dirpath, root)
        depth = 0 if rel_path == '.' else rel_path.count(os.sep) + 1
        indent = "  " * depth
        
        dir_size = dir_sizes.get(dirpath, 0)
        tree_lines.append(f"{indent}[DIR] {os.path.basename(dirpath)} ({dir_size} chars)")
        
        for f in sorted(filenames):
            if should_ignore_file(f):
                continue
            
            filepath = os.path.join(dirpath, f)
            file_size = file_sizes.get(filepath, 0)
            
            # НОВОЕ: Проверяем, входит ли файл в топ самых больших, и добавляем префикс
            prefix = "!!!" if filepath in top_files_set else ""
            tree_lines.append(f"{indent}  {prefix}[FILE] {f} ({file_size} chars)")
            
    return "\n".join(tree_lines)


def write_chunks(full_text):
    chunks = []
    cursor = 0
    instruction_with_newline = "\n\n" + FINISH_INSTRUCTION
    instruction_len = len(instruction_with_newline)

    while cursor < len(full_text):
        if len(full_text) - cursor <= MAX_CHARS:
            last_chunk = full_text[cursor:]
            chunks.append(last_chunk)
            break

        content_limit = MAX_CHARS - instruction_len
        chunk_content = full_text[cursor : cursor + content_limit]
        full_chunk = chunk_content + instruction_with_newline
        chunks.append(full_chunk)
        cursor += content_limit
            
    return chunks

def main():
    # --- НАЧАЛО ИСПРАВЛЕНИЯ ---
    # ШАГ 0: Принудительно удаляем старые файлы контекста перед созданием новых.
    # Это гарантирует, что мы каждый раз начинаем с чистого листа.
    print("Очистка старых файлов контекста...")
    for i in range(1, 100):  # Проверяем и удаляем message_1.txt ... message_99.txt
        old_file = f"{OUTPUT_PREFIX}{i}.txt"
        if os.path.exists(old_file):
            try:
                os.remove(old_file)
                print(f"Удален старый файл: {old_file}")
            except OSError as e:
                print(f"Ошибка при удалении {old_file}: {e}")
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    all_lines = []
    explanation = (
        "сейчас я выгружу тебя сначала дерево файлов с размерами в символах, "
        "а потом все файлы, которые не были проигнорированы. "
        "Возможно это займет больше одного сообщения, тогда просто скажи ОК. "
        "После того как все будет загружено, я задам вопросы, чтобы ты был в контексте кода."
    )
    all_lines.append(explanation)
    
    # 1. Вычисляем размеры и получаем пути к файлам за один проход
    file_sizes, dir_sizes, file_paths = calculate_sizes(BASE_DIR)

    # 2. Находим топ-N самых больших файлов
    sorted_files = sorted(file_sizes.items(), key=lambda item: item[1], reverse=True)
    top_files_set = set(filepath for filepath, size in sorted_files[:TOP_N_FILES])
    
    # 3. Строим дерево папок с размерами, передавая информацию о самых больших файлах
    tree = build_tree(BASE_DIR, file_sizes, dir_sizes, top_files_set)
    all_lines.append("\n--- Структура проекта (с размерами в символах) ---\n")
    all_lines.append(tree)
    
    # 4. Добавляем содержимое файлов
    all_lines.append("\n--- Содержимое файлов ---\n")
    for path in sorted(file_paths):
        rel_path = os.path.relpath(path, BASE_DIR)
        all_lines.append(f"\nФайл: {rel_path}\n{'-' * len('Файл: ' + rel_path)}")
        content = get_file_content(path)
        if content:
            all_lines.append(content)
        else:
            all_lines.append("Не удалось прочитать содержимое файла.")
    
    # 5. Собираем все в один большой текст и разбиваем на чанки
    full_text = "\n".join(all_lines)
    chunks = write_chunks(full_text)
    
    # 6. Записываем чанки в файлы (теперь уже после очистки)
    for i, chunk in enumerate(chunks, 1):
        out_filename = f"{OUTPUT_PREFIX}{i}.txt"
        with open(out_filename, "w", encoding="utf-8") as out_file:
            out_file.write(chunk)
        print(f"Сохранено: {out_filename} ({len(chunk)} символов)")

if __name__ == "__main__":
    main()
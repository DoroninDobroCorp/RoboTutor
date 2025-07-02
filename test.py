try:
    from google.generativeai import Client
    print("Client успешно импортирован.")
except ImportError as e:
    print(f"Ошибка импорта Client: {e}")

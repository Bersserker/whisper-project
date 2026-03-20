import os
import whisper
from langchain_openai import ChatOpenAI
import re

# 1. Загружаем модель Whisper
stt_model = whisper.load_model('large')

# 2. Путь к аудио
audio_path = os.path.abspath("audio_input/Запись.m4a")

# 3. Распознавание
result = stt_model.transcribe(audio_path)
recognized_text = result["text"]

print("Распознанный текст:")
print(recognized_text)

# 4. Настройка LLM через LM Studio
model_name = "deepseek/deepseek-r1-0528-qwen3-8b"

llm = ChatOpenAI(
    model=model_name,
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    temperature=0.7,
)

# 5. Сообщения для LLM
messages = [
    (
        "system",
        "Ты редактор распознанной речи. "
        "Преобразуй текст в чистый, грамотный русский язык. "
        "Удали слова-паразиты, повторы, междометия и оговорки. "
        "Не добавляй новую информацию и не меняй смысл."
    ),
    ("human", recognized_text),
]

# 6. Один вызов модели
response = llm.invoke(messages)


output = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()

print("\nИсправленный текст:")
print(output)
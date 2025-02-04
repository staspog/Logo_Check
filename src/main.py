import os
from transformers import CLIPProcessor, CLIPModel
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

BASE_DIR = os.getcwd()  # Текущая рабочая директория
MODEL_DIR = os.path.join(BASE_DIR, "model")  # Папка для локальной модели

# Проверяем, существует ли модель локально
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Загрузка модели CLIP из Hugging Face, если модели нет в локальной папке
model_name = "openai/clip-vit-base-patch32"
if not os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")):
    print("Модель не найдена, загружаем...")
    model = CLIPModel.from_pretrained(model_name, cache_dir=MODEL_DIR)
    processor = CLIPProcessor.from_pretrained(model_name, cache_dir=MODEL_DIR)
    print("Модель успешно загружена и сохранена в папку 'model'.")
else:
    # Если модель уже скачана, загружаем её из локальной папки
    print("Модель найдена локально, загружаем...")
    model = CLIPModel.from_pretrained(MODEL_DIR)
    processor = CLIPProcessor.from_pretrained(MODEL_DIR)

def get_embedding(img_path):
    """Функция для получения эмбеддинга логотипа с помощью модели CLIP"""
    img = image.load_img(img_path, target_size=(224, 224))  # Преобразуем изображение в нужный формат
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Применяем процессор для преобразования изображения в формат, который принимает модель CLIP
    inputs = processor(images=img_array, return_tensors="pt")
    
    # Получаем эмбеддинг с помощью модели CLIP
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        embedding = outputs.cpu().numpy().flatten()  # Преобразуем выход в одномерный массив для сравнения
    return embedding

# Примеры логотипов искомой организации
reference_logos = ["data/logo1.png", "data/logo2.png"]
reference_embeddings = [get_embedding(logo) for logo in reference_logos]

# Проверяемый логотип
test_logo = "data/logo.png"
test_embedding = get_embedding(test_logo)

# Сравнение с эталонными логотипами
similarities = [cosine_similarity([test_embedding], [ref_emb])[0][0] for ref_emb in reference_embeddings]

# Преобразуем similarities в numpy массив для правильной обработки
similarities = np.array(similarities)

# Получаем максимальное сходство
max_similarity = similarities.max()  # Используем numpy для работы с массивами

# Вывод результата
THRESHOLD = 0.7  # Порог схожести
if max_similarity > THRESHOLD:
    print(f"Логотип принадлежит искомой организации (сходство: {max_similarity:.2f})")
else:
    print(f"Логотип НЕ принадлежит искомой организации (сходство: {max_similarity:.2f})")

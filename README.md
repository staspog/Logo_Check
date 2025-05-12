# Logo Recognition (CLIP)

Распознавание логотипов с использованием модели CLIP. Реализован для VK Video.

## 📌 Описание
Этот проект использует модель **CLIP** для определения схожести между логотипами.
Модель загружается в локальную директорию `model`, а изображения логотипов, которые нужно распознавать, должны находиться в `data`.
Примеры работы программы представлены на скриншотах в папке tests. 

## 🚀 Установка и запуск
### 1️⃣ Клонирование репозитория
```sh
git clone https://github.com/staspog/Logo_Check.git
cd Logo_Check
```

### 2️⃣ Создание виртуального окружения (рекомендуется)
**Для Conda:**
```sh
conda create -n tf_gpu_env python=3.9
conda activate tf_gpu_env
```
**Для venv (если используешь pip):**
```sh
python -m venv venv
source venv/bin/activate  # MacOS/Linux
venv\Scripts\activate  # Windows
```

### 3️⃣ Установка зависимостей
```sh
pip install -r requirements.txt
```

### 4️⃣ Запуск
```sh
python src/main.py
```

## 📂 Структура проекта
```
Logo_Check/
│── model/              # Сохраненная модель CLIP
│── data/               # Изображения логотипов
│── src/
│   ├── main.py         # Основной код
│── requirements.txt    # Зависимости
│── README.md           # Инструкция
```

## ❗ Возможные ошибки и их решения
### 🔹 Ошибка: "ModuleNotFoundError: No module named '...'"
**Решение:** Убедитесь, что установили зависимости:
```sh
pip install -r requirements.txt
```

### 🔹 Ошибка: "OSError: Model file not found in directory: model"
**Решение:** Проверьте, что модель сохранена в папке `model` и содержит файлы `config.json`, `pytorch_model.bin`, `tokenizer.json` и другие файлы модели CLIP.

### 🔹 Ошибка: "Data file not found"
**Решение:** Проверьте, что в папке `data/` есть изображения логотипов (`logo1.png`, `logo2.png`, `logo_test.png`).

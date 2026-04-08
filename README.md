# Math Problem Analyzer

Небольшой, но рабочий сервис для классификации математических задач по тексту.

## Что это за проект

Это проект для ручной проверки по курсу ML2.

## Ссылка на демо

Demo URL: https://ml2-hf-math-jtyvkeryk7adr3i4nmhs4a.streamlit.app


### Задача

Сервис принимает текст математической задачи и предсказывает:
- тему задачи 
- сложность как числовой уровень от 1 до 5
- confidence по теме
- список top-95% тем по кумулятивной вероятности

Что умеет:
- определять тему задачи 
- оценивать сложность как число от 1 до 5 
- показывать топ тем с вероятностями

Внутри используется multitask-модель на базе DistilBERT:
- голова 1: классификация темы
- голова 2: регрессия уровня сложности

### Данные

Датасет грузится автоматически с Hugging Face. `hendrycks/competition_math`

### Предобработка

Перед токенизацией применяется легкая очистка LaTeX:
- убираются `$` и `$$`
- убираются `\left` и `\right`
- команды вроде `\sin` превращаются в `sin`
- формулы очевидно не вычищаются полностью

Также перед обучением удаляются слишком короткие тексты:
- все задачи с длиной `problem < 30` отбрасываются

### Модель

- Backbone: `distilbert-base-uncased`
- Общий энкодер + две головы:
  - topic head: классификация темы
  - difficulty head: регрессия уровня сложности
- Loss:
  - topic: CrossEntropy
  - difficulty: MAE (L1)
  - общий: `topic_loss + 0.3 * difficulty_loss`
## Быстрый старт

```bash
uv init
uv add -r requirements.txt
```

## Пошаговый запуск

### 1. Посмотреть данные

```bash
bash scripts/01_inspect_data.sh
```

Скрипт:
- скачает датасет
- покажет распределения
- сохранит превью в `artifacts/data_preview.csv`

### 2. Быстрое обучение

```bash
bash scripts/02_train_quick.sh
```

По умолчанию тут:
- `epochs=5`
- `max_length=320`
- логи для TensorBoard: `artifacts/tb_logs`

### 3. Более длинное обучение

```bash
bash scripts/03_train_long.sh
```

### 4. Инференс из CLI

```bash
bash scripts/04_infer_cli.sh "Find the remainder when 7^2025 is divided by 13"
```

### 5. Веб-интерфейс (Streamlit)

```bash
bash scripts/05_run_streamlit.sh
```

### 6. Графики обучения (TensorBoard)

```bash
bash scripts/06_tensorboard.sh
```

## Что сохраняется после обучения

Папка `artifacts/`:
- `model.pt`
- `tokenizer/`
- `label_mappings.json`
- `metadata.json`
- `train_history.csv`

## Как считается сложность

Модель предсказывает непрерывный score в диапазоне `[1, 5]`.

Дальше для удобного текста делается перевод:
- 1-2 -> easy
- 3 -> medium
- 4-5 -> hard

В интерфейсе и CLI показывается и класс (`easy/medium/hard`), и raw score.

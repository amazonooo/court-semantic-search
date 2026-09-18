# Court Semantic Search

Локальный MVP поиска судебных актов по описанию ситуации. Ollama бесплатно
генерирует несколько поисковых формулировок и ключевые признаки. CourtProvider
ищет документы; сервис группирует их по делу, выбирает содержательный акт,
извлекает текст PDF и показывает фрагмент с совпавшими и отсутствующими
признаками. Постоянная база данных не используется.

## MVP flow

1. Применить hard filters: период, суд, участник/компания, тип/категория спора.
2. Получить кандидатов через Parser API / RAS.
3. Повторно проверить жёсткие ограничения, в том числе даты.
4. Дедуплицировать судебные акты по `document_id`.
5. Сгруппировать найденные акты в дела по каноническому `CaseId`.
6. Для найденного дела подтянуть остальные акты по базовому `caseNumber` и оставить только тот же `CaseId`.
7. Извлечь тексты PDF.
8. Дальше передать дела в semantic/LLM слой для reranking, проверки критериев и итогового анализа.

`CaseId` — внутренний идентификатор всего дела. `document_id`/`FileUrl` — идентификатор отдельного судебного акта.

## Текущие API

- `GET /api/documents/search` — сырой поиск судебных актов.
- `POST /api/documents/extract-text` — PDF -> текст.
- `GET /api/cases/search` — поиск с дедупликацией, группировкой и bounded case expansion.
- `POST /api/cases/extract-factual-base-text` — текст фактической базы дела: в первую очередь substantive-акт первой инстанции.
- `POST /api/cases/extract-latest-substantive-text` — текст последнего substantive-акта.
- `POST /api/cases/extract-preferred-text` — совместимый legacy endpoint; новые клиенты должны использовать явные role-поля и endpoints.
- `POST /api/collections/search` — высокоуровневый поиск подборки по участнику, суду, периоду, типу/категории спора и тексту.

## Архитектурные правила

- пользовательский период — hard filter для попадания дела в кандидаты;
- после того как дело прошло фильтр, его полная история может включать акты вне периода как контекст;
- первая инстанция (`first_instance_document`/`factual_base_document` и `first_instance_documents`) должна стать основным источником фактических обстоятельств;
- `appellate_documents`, `cassation_documents` и `procedural_documents` остаются отдельными ролями внутри дела;
- `latest_substantive_document` используется для последнего содержательного судебного акта;
- апелляция и кассация остаются частью дела и нужны для изменений, новых доказательств и итоговой судьбы спора;
- исход дела не должен по умолчанию ограничивать retrieval;
- RAS используется для судебных актов; KAD планируется для статуса дела, участников и полной процессуальной истории.

Полный план: [`docs/ROADMAP.md`](docs/ROADMAP.md).
## Демо на Windows

1. Установите [Ollama](https://ollama.com/download).
2. Из корня проекта выполните:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/start-demo.ps1
   ```

3. Откройте `http://127.0.0.1:8000/demo` и нажмите «Найти похожие дела».

Скрипт создаёт `.venv`, устанавливает зависимости Python и при первом запуске
скачивает `qwen3:4b` (около 2,5 ГБ). Дальше модель и API работают локально.
Демо использует три **вымышленных учебных дела**: одно соответствует схеме
присоединения заемщика, процентов и налоговой выгоды, два совпадают только по
части признаков. Их PDF формируются в памяти. Это позволяет показать весь путь
поиска без ключа Parser API и без внешней БД.

## Подключение реальных данных

Установите зависимости из `backend/requirements.txt`. В `.env` укажите:

```dotenv
COURT_PROVIDER=parser_api
PARSER_API_KEY=ваш_ключ
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
```

Запустите `python -m uvicorn backend.app.main:app --reload`. Запросы к Ollama
остаются локальными, а поиск и скачивание судебных PDF выполняет Parser API.
Ключи храните только в `.env` или переменных окружения; `.env` исключён из Git.

## API

- `POST /api/cases/search-with-evidence` — план запросов, найденные дела,
  признаки и фрагменты выбранных PDF.
- `POST /api/cases/semantic-search` — только план и найденные дела, без PDF.
- `GET /api/cases/search` — поиск дел по параметрам CourtProvider.
- `POST /api/cases/extract-preferred-text` — извлечение текста выбранного акта.
- `GET /api/demo/status` — состояние локальной модели и выбранного провайдера.

Пример тела для поиска с фрагментами:

```json
{
  "description": "После присоединения заемщика компания учла проценты по займу и убытки при расчете налога на прибыль. Налоговая оспорила выгоду.",
  "max_pages_per_query": 1,
  "max_cases": 6
}
```

Совпадение признаков — объяснимая первичная сортировка, а не юридическая
оценка. Для реальных дел нужно проверять текст PDF и реквизиты источника.

## Проверка

```text
Frontend (Next.js)
        |
        v
Backend API (FastAPI)
        |
        +--> Query Analyzer / Query Generator
        |
        +--> Court Provider
        |      +--> Parser API / RAS
        |      +--> KAD provider (planned)
        |
        +--> PDF/Text Extraction
        |
        +--> CaseId Grouper / Case Expansion
        |
        +--> Semantic Reranker (team LLM layer)
        |
        +--> LLM Case Analyzer (team LLM layer)
```

## Структура

```text
backend/   FastAPI и search/data pipeline
frontend/  Next.js интерфейс
docs/      продуктовый и технический roadmap
```

Внешние источники судебных данных подключаются через provider-адаптеры, чтобы можно было менять источник без переписывания основной логики.

## Статус

Parser API / RAS подключён и протестирован на реальных судебных актах. Реализованы поиск, PDF -> text, пагинация, дедупликация, группировка по `CaseId` и bounded case expansion. Следующие шаги описаны в roadmap.
```powershell
python -m pip install -r backend/requirements-dev.txt
python -m pytest backend/tests -q
```

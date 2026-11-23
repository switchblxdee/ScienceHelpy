# Тема 8. Каталог потоков ScienceHelpy

| Контекст | Поток | Событие | Когда публиковать |
| --- | --- | --- | --- |
| Paper Library | `paper-library` | `paperImportRequested` | Принята ссылка на статью для импорта. |
| | | `paperAddedToLibrary` | PDF успешно сохранён. |
| | | `paperImportFailed` | Не удалось получить или сохранить PDF. |
| Paper Preparation | `paper-preparation` | `paperContentPrepared` | Из PDF извлечено содержимое для обработки. |
| | | `paperPreparationFailed` | Извлечь или подготовить содержимое не удалось. |
| | | `paperMadeSearchable` | Содержимое доступно в поисковом индексе. |
| Source Discovery | `source-discovery` | `sourceSearchRequested` | Принят запрос на подбор источников. |
| | | `localSourcesFound` | Поиск по библиотеке вернул фрагменты. |
| | | `localSourcesNotFound` | Поиск по библиотеке завершился без результатов. |
| | | `externalSourceSearchCompleted` | Интернет-поиск завершился. Список результатов может быть пустым. |
| | | `sourceSearchFailed` | Из-за ошибки поиск завершить не удалось. |
| Research Assistance | `research-assistance` | `conversationStarted` | Начат диалог. |
| | | `questionSubmitted` | Принят вопрос пользователя. |
| | | `answerProvided` | Ответ выведен пользователю. |
| | | `answerCouldNotBeProvided` | Обработка завершилась ошибкой. |
| | | `conversationEnded` | Сеанс диалога завершён. |

После `paperAddedToLibrary` обработчик может запустить команду `PreparePaper`.
Индексатор публикует `paperMadeSearchable` после обновления индекса.

После `questionSubmitted` обработчик может запустить `FindSources`. При
`localSourcesNotFound` он обращается к интернет-поиску. Команда `PrepareAnswer`
может использовать локальные фрагменты или результаты внешнего поиска.

# Тема 9. Доставка событий

Запуск `python main.py` публикует события импорта PDF и консольного диалога.
Веб-интерфейс использует те же потоки. При загрузке PDF он публикует события
импорта и подготовки. При вопросе помощнику он публикует события подбора
источников и ответа. При первом чтении библиотеки после запуска сервис
переиндексирует существующие документы и фиксирует результат подготовки.
У статьи могут быть несколько чатов. Сервер хранит их в SQLite и берёт
до трёх предыдущих пар реплик из текущего чата при следующем вопросе.
Событие `conversationStarted` публикуется при создании чата, а дальнейшие
вопросы и ответы связаны с его идентификатором.

В событиях записываются имена файлов и идентификаторы диалогов и вопросов.
Тексты вопросов, ответы и ключи API не сохраняются.

Модуль работает на стандартной библиотеке Python 3.9+. Для демонстрации
не нужны зависимости RAG или ключи API:

```sh
python -m event_delivery publish paperImportRequested --payload '{"paper_file":"1.pdf"}'
python -m event_delivery consume paper-library --consumer audit
python -m event_delivery consume paper-library --consumer audit
python -m unittest discover -s tests -v
```

Первый вызов `consume` выводит непрочитанные события, второй ничего не выводит.
С локальным приложением команды используют общий файл
`.sciencehelpy/events.sqlite3`. Другой путь можно задать переменной
`SCIENCEHELPY_EVENTS_DB` или параметром CLI `--db`.

Демонстрация в Docker использует отдельную базу в именованном томе
`event-data`, который сохраняется между запусками контейнера:

```sh
docker compose -f compose.events.yml build
docker compose -f compose.events.yml run --rm events publish paperImportRequested --payload '{"paper_file":"1.pdf"}'
docker compose -f compose.events.yml run --rm events consume paper-library --consumer audit
```

В Python `store.publish(name, **payload)` сохраняет событие с UUID, временем UTC
и позицией в журнале. Вызов `store.deliver(consumer, stream, handler)` читает
поток по возрастанию позиции и подтверждает каждое событие после успешного
возврата обработчика. Когда обработчик выбрасывает исключение, повторный вызов
начинается с этого события. Каждый подписчик имеет собственную позицию в
каждом потоке.

Доставка неподтверждённых записей выполняется как минимум один раз при
повторном вызове подписчика. При сбое между обработкой и подтверждением
событие может прийти повторно, поэтому обработчик должен учитывать `event.id`.
Для одной пары «подписчик и поток» используйте один активный обработчик.
Публикация события и изменение данных приложения выполняются отдельными
операциями. SQLite рассчитана на локальный запуск с общим файлом базы.

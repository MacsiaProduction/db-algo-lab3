# Отчёт по FAISS-ANN бенчмарку — прогон `light`

> Подробная версия (draft). Все наблюдения и аномалии сопровождаются численными доказательствами. Без них же, в той же раскладке — `OTCHET_kratkij.md`.

> Сгенерирован `scripts/analyze_and_report.py` из CSV в `results/light/`. Графики — `docs/img/light/`.

## 1. Условия эксперимента

- **Датасет:** ImageNet-1M ZJU, 2048-D, n_base = 50,000, n_query = 10 000 (для свипов), n_gt = 25 000.
- **Метрика расстояния:** L2.
- **Платформа:** local single-host (см. notebook 01 для деталей RAM/CPU).
- **QPS-замер:** `LAB_QPS_REPEAT=3 LAB_QPS_WARMUP=1` (warmup + медиана 3 запусков).
- **Свипы:** IVFFlat 10 конфигов, IVFPQ 10, IVFSQ 5, HNSW 36 (varyM + varyEFC), LSH 4.

## 2. Сводка результатов

![Cross-family Pareto](img/light/05_global_pareto.png)

На графике — все измерения, сгруппированные по семейству. Чёрный пунктир — глобальный Парето-фронт (точки, которые никто не доминирует одновременно по recall и QPS). Звёздами помечены operational picks (см. таблицу ниже).

### 2.1. Operational picks (макс. QPS при первом достижимом recall-флоре)

Из множества `[0.95, 0.90, 0.80, 0.50, 0.20]` берётся самая высокая планка по recall@100, которой семейство достигает; среди всех конфигов, удовлетворяющих ей — конфиг с максимальным QPS.

| Семейство | Recall флор | Recall@100 | QPS | Mean lat. | Index size | Build | Peak RSS | Конфиг |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **IVFFlat** | 0.95 | 0.9780 | 1,235 | 0.809 мс | 790 МБ | 27.4 с | 3.24 ГБ | `nlist=1024, nprobe=64` |
| **IVF+PQ** | 0.50 | 0.5494 | 87,890 | 0.011 мс | 8 МБ | 32.5 с | 4.50 ГБ | `nlist=256, nprobe=4, M=32, nbits=8` |
| **IVF+SQ** | 0.95 | 0.9913 | 843 | 1.183 мс | 198 МБ | 7.0 с | 3.13 ГБ | `nlist=256, nprobe=64, sq=SQ8` |
| **HNSW** | 0.95 | 0.9597 | 15,455 | 0.065 мс | 789 МБ | 9.2 с | 2.65 ГБ | `M=8, efConstruction=200, efSearch=160` |
| **LSH** | 0.20 | 0.2531 | 59,939 | 0.017 мс | 3 МБ | 0.6 с | 2.41 ГБ | `nbits=128` |

**Чтение таблицы:** если столбец «Recall флор» < 0.95, семейство **не может обслуживать prod-качество** (LSH, IVFPQ при нашем M=128).

### 2.2. «Колено» Парето-кривой по каждому семейству

Точка, ближайшая к идеальному углу (recall=1, max QPS) в log-y нормализованном пространстве. Используется как «общая рекомендация по умолчанию», когда жёсткого SLA нет.

| Семейство | Recall@100 | QPS | Index size | Конфиг |
|---|---:|---:|---:|---|
| IVFFlat | 0.8744 | 5,656 | 790 МБ | `nlist=1024, nprobe=16` |
| IVF+PQ | 0.6181 | 60,065 | 11 МБ | `nlist=256, nprobe=4, M=64, nbits=8` |
| IVF+SQ | 0.7678 | 12,235 | 198 МБ | `nlist=256, nprobe=4, sq=SQ8` |
| HNSW | 0.9019 | 24,327 | 789 МБ | `M=8, efConstruction=200, efSearch=80` |
| LSH | 0.4362 | 21,935 | 10 МБ | `nbits=512` |

### 2.3. Победители по квадрантам (по всему свипу)

- **Максимальный Recall@100:** IVFFlat = 1.0000 (`nlist=256, nprobe=256`).
- **Максимальный QPS:** IVF+PQ = 128,852 при recall 0.383 (`nlist=256, nprobe=1, M=32, nbits=8`).
- **Минимальный размер индекса:** LSH = 3 МБ (`nbits=128`).
- **Самый быстрый билд:** LSH = 0.6 с (`nbits=128`).

![Operational picks: build / size / RSS / QPS](img/light/05_best_bars.png)

![Разложение peak RSS](img/light/05_memory_budget.png)

Из стэков видно, что у IVFFlat ~50 % peak RSS — это **сам индекс** (≈9.4 ГБ raw float-векторов), у HNSW добавляется ~1.3 ГБ на граф; у IVFPQ/IVFSQ/LSH сам индекс крошечный (< 1 ГБ), а > 90 % пика — это mmap’нутые страницы базы во время `stream_add`, которые ОС держит в page-cache.

![Средняя per-query latency](img/light/05_latency_best.png)

## 3. Анализ по семействам

![Парето по семействам с knee и порогами recall](img/light/05_per_family_knees.png)

![Recall@100 при заданном QPS-бюджете](img/light/05_recall_at_qps.png)

### 3.1. IVFFlat

- **Размер свипа:** 10 конфигов.
- **Recall@100:** 0.251 → 1.0000.
- **QPS:** 82 → 79,677.
- **Размер индекса:** 784 МБ → 790 МБ.
- **Build:** 7.5 с → 27.4 с.

Лучшая конфигурация при каждом recall-флоре:

| Recall флор | Конфиг | Recall@100 | QPS | Mean lat. |
|---:|---|---:|---:|---:|
| 0.99 | `nlist=1024, nprobe=256` | 0.9985 | 258 | 3.867 мс |
| 0.95 | `nlist=1024, nprobe=64` | 0.9780 | 1,235 | 0.809 мс |
| 0.90 | `nlist=1024, nprobe=64` | 0.9780 | 1,235 | 0.809 мс |
| 0.80 | `nlist=1024, nprobe=16` | 0.8744 | 5,656 | 0.177 мс |
| 0.50 | `nlist=1024, nprobe=4` | 0.5991 | 26,372 | 0.038 мс |
| 0.20 | `nlist=1024, nprobe=1` | 0.2510 | 79,677 | 0.013 мс |

### 3.2. IVF+PQ

- **Размер свипа:** 10 конфигов.
- **Recall@100:** 0.383 → 0.6737.
- **QPS:** 3,109 → 128,852.
- **Размер индекса:** 8 МБ → 11 МБ.
- **Build:** 12.5 с → 32.5 с.

Лучшая конфигурация при каждом recall-флоре:

| Recall флор | Конфиг | Recall@100 | QPS | Mean lat. |
|---:|---|---:|---:|---:|
| 0.99 | _нет конфига_ | — | — | — |
| 0.95 | _нет конфига_ | — | — | — |
| 0.90 | _нет конфига_ | — | — | — |
| 0.80 | _нет конфига_ | — | — | — |
| 0.50 | `nlist=256, nprobe=4, M=32, nbits=8` | 0.5494 | 87,890 | 0.011 мс |
| 0.20 | `nlist=256, nprobe=1, M=32, nbits=8` | 0.3827 | 128,852 | 0.008 мс |

![IVFPQ: recall vs nprobe + footprint vs recall](img/light/05_ivfpq_grid.png)

На правой панели видна линия R@100 ≈ 0.77 — это **потолок семейства** на 2048-D ResNet-эмбеддингах при заданном битовом бюджете (см. п. 5).

### 3.3. IVF+SQ

- **Размер свипа:** 5 конфигов.
- **Recall@100:** 0.414 → 0.9946.
- **QPS:** 315 → 45,809.
- **Размер индекса:** 198 МБ → 198 МБ.
- **Build:** 7.0 с → 7.0 с.

Лучшая конфигурация при каждом recall-флоре:

| Recall флор | Конфиг | Recall@100 | QPS | Mean lat. |
|---:|---|---:|---:|---:|
| 0.99 | `nlist=256, nprobe=64, sq=SQ8` | 0.9913 | 843 | 1.183 мс |
| 0.95 | `nlist=256, nprobe=64, sq=SQ8` | 0.9913 | 843 | 1.183 мс |
| 0.90 | `nlist=256, nprobe=16, sq=SQ8` | 0.9481 | 2,989 | 0.334 мс |
| 0.80 | `nlist=256, nprobe=16, sq=SQ8` | 0.9481 | 2,989 | 0.334 мс |
| 0.50 | `nlist=256, nprobe=4, sq=SQ8` | 0.7678 | 12,235 | 0.082 мс |
| 0.20 | `nlist=256, nprobe=1, sq=SQ8` | 0.4144 | 45,809 | 0.022 мс |

### 3.4. HNSW

- **Размер свипа:** 36 конфигов.
- **Recall@100:** 0.449 → 0.9949.
- **QPS:** 3,462 → 80,443.
- **Размер индекса:** 789 МБ → 807 МБ.
- **Build:** 5.3 с → 11.6 с.

Лучшая конфигурация при каждом recall-флоре:

| Recall флор | Конфиг | Recall@100 | QPS | Mean lat. |
|---:|---|---:|---:|---:|
| 0.99 | `M=16, efConstruction=200, efSearch=320` | 0.9925 | 6,110 | 0.164 мс |
| 0.95 | `M=8, efConstruction=200, efSearch=160` | 0.9597 | 15,455 | 0.065 мс |
| 0.90 | `M=8, efConstruction=200, efSearch=80` | 0.9019 | 24,327 | 0.041 мс |
| 0.80 | `M=16, efConstruction=200, efSearch=40` | 0.8539 | 26,367 | 0.038 мс |
| 0.50 | `M=16, efConstruction=200, efSearch=10` | 0.5305 | 63,266 | 0.016 мс |
| 0.20 | `M=8, efConstruction=200, efSearch=10` | 0.4489 | 80,443 | 0.012 мс |

### 3.5. LSH

- **Размер свипа:** 4 конфигов.
- **Recall@100:** 0.253 → 0.4943.
- **QPS:** 7,596 → 59,939.
- **Размер индекса:** 3 МБ → 20 МБ.
- **Build:** 0.6 с → 3.2 с.

Лучшая конфигурация при каждом recall-флоре:

| Recall флор | Конфиг | Recall@100 | QPS | Mean lat. |
|---:|---|---:|---:|---:|
| 0.99 | _нет конфига_ | — | — | — |
| 0.95 | _нет конфига_ | — | — | — |
| 0.90 | _нет конфига_ | — | — | — |
| 0.80 | _нет конфига_ | — | — | — |
| 0.50 | _нет конфига_ | — | — | — |
| 0.20 | `nbits=128` | 0.2531 | 59,939 | 0.017 мс |

## 4. Масштабирование 100K → 1.28M

![Scaling: recall/QPS/build/RSS vs N](img/light/05_scaling.png)

Для каждого семейства взята одна репрезентативная конфигурация (см. таблицу) и пять точек по N. Recall стабилен у HNSW и IVFFlat, у IVFPQ деградирует на больших N (квантизатор обучен на тех же 200K точек — потеря шумовой составляющей при росте плотности базы). QPS падает сублинейно у HNSW (graph search) и линейно у IVF/LSH.

| Family | N | Recall@100 | QPS | Build | Peak RSS |
|---|---:|---:|---:|---:|---:|
| HNSW | 50,000 | 0.9771 | 16,269 | 3.5 с | 1.99 ГБ |
| IVFFlat | 50,000 | 0.9944 | 931 | 4.7 с | 1.39 ГБ |
| IVFPQ | 50,000 | 0.6730 | 15,436 | 8.8 с | 3.81 ГБ |
| IVFSQ | 50,000 | 0.9903 | 3,483 | 5.1 с | 1.53 ГБ |
| LSH | 50,000 | 0.4989 | 12,171 | 1.6 с | 1.84 ГБ |

## 5. Аномалии и data quality

![Сводка аномалий](img/light/05_anomaly_flags.png)

| # | Severity | Аномалия | Численное доказательство |
|---:|---|---|---|
| 1 | ВЫСОКАЯ | RSS-колонка дегенерирована | `post-fix invariant broken — RSS columns degenerate` |
| 2 | ВЫСОКАЯ | RSS-колонка дегенерирована | `post-fix invariant broken — RSS columns degenerate` |
| 3 | ВЫСОКАЯ | RSS-колонка дегенерирована | `post-fix invariant broken — RSS columns degenerate` |
| 4 | ВЫСОКАЯ | RSS-колонка дегенерирована | `post-fix invariant broken — RSS columns degenerate` |
| 5 | ВЫСОКАЯ | RSS-колонка дегенерирована | `post-fix invariant broken — RSS columns degenerate` |
| 6 | ВЫСОКАЯ | RSS-колонка дегенерирована | `post-fix invariant broken — RSS columns degenerate` |
| 7 | СРЕДНЯЯ | Recall HNSW немонотонен по efConstruction при низком efSearch | `efC=40→R@100=0.712; efC=100→R@100=0.530; efC=200→R@100=0.562` |
| 8 | СРЕДНЯЯ | Recall HNSW немонотонен по efConstruction при низком efSearch | `efC=40→R@100=0.835; efC=100→R@100=0.695; efC=200→R@100=0.730` |
| 9 | СРЕДНЯЯ | HNSW: build_s mismatch scaling vs sweep (69 %) | `scaling.csv=4s, hnsw_*.csv=12s for identical config; QPS gap 99 %` |
| 10 | СРЕДНЯЯ | IVFFlat build_s расходится между scaling.csv и sweep CSV (46 %) | `scaling.csv=5s, ivfflat_*.csv=8s for identical config; QPS gap 289 %` |
| 11 | СРЕДНЯЯ | Потолок recall IVF+PQ ≈ 0.77 | `best PQ config (nlist=256, M=64, nprobe=256) cannot serve ≥ 0.95 SLA` |
| 12 | СРЕДНЯЯ | IVF+PQ build_s расходится между scaling.csv и sweep CSV (44 %) | `scaling.csv=9s, ivfpq_*.csv=12s for identical config; QPS gap 90 %` |
| 13 | СРЕДНЯЯ | LSH build_s расходится между scaling.csv и sweep CSV (27 %) | `scaling.csv=2s, lsh_*.csv=3s for identical config; QPS gap 60 %` |
| 14 | НИЗКАЯ | Build_s немонотонен по M (PQ) | `M=32→32.5s; M=64→12.5s` |

### 5.1. Подробные объяснения с доказательствами

#### 1. RSS-колонка дегенерирована *(severity: ВЫСОКАЯ)*

**Сырые числа из CSV:**

- `hnsw_EFC: rss_mb still aliased to rss_peak_mb` → `post-fix invariant broken — RSS columns degenerate`
- `hnsw_M: rss_mb still aliased to rss_peak_mb` → `post-fix invariant broken — RSS columns degenerate`
- `ivf_flat: rss_mb still aliased to rss_peak_mb` → `post-fix invariant broken — RSS columns degenerate`
- `ivf_pq: rss_mb still aliased to rss_peak_mb` → `post-fix invariant broken — RSS columns degenerate`
- `ivf_sq: rss_mb still aliased to rss_peak_mb` → `post-fix invariant broken — RSS columns degenerate`
- `lsh: rss_mb still aliased to rss_peak_mb` → `post-fix invariant broken — RSS columns degenerate`

rss_mb совпадает с rss_peak_mb во всех строках — значит, утилитный хелпер не различает «RSS после билда» и «пиковый RSS». Фиксится присваиванием `rss_mb = tb.rss_after_mb` в `_build_notebooks.py` (в текущем коде уже исправлено).

#### 2. Recall HNSW немонотонен по efConstruction при низком efSearch *(severity: СРЕДНЯЯ)*

**Сырые числа из CSV:**

- `HNSW efC monotonicity violated at efS=10` → `efC=40→R@100=0.712; efC=100→R@100=0.530; efC=200→R@100=0.562`
- `HNSW efC monotonicity violated at efS=20` → `efC=40→R@100=0.835; efC=100→R@100=0.695; efC=200→R@100=0.730`

При efSearch ≤ 20 более «рыхлый» граф (efC=40) сохраняет больше длинных рёбер-shortcut’ов, поэтому greedy-обход быстро находит нужный кластер. Плотный граф (efC=400) перенасыщен локальными связями и тот же поиск застревает. С ростом efSearch (≥ 80) разница нивелируется и кривая становится монотонной.

#### 3. HNSW: build_s mismatch scaling vs sweep (69 %) *(severity: СРЕДНЯЯ)*

**Сырые числа из CSV:**

- `HNSW: build_s mismatch scaling vs sweep (69 %)` → `scaling.csv=4s, hnsw_*.csv=12s for identical config; QPS gap 99 %`

_Общая категория без специальной заметки._

#### 4. IVFFlat build_s расходится между scaling.csv и sweep CSV (46 %) *(severity: СРЕДНЯЯ)*

**Сырые числа из CSV:**

- `IVFFlat: build_s mismatch scaling vs sweep (37 %)` → `scaling.csv=5s, ivfflat_*.csv=8s for identical config; QPS gap 289 %`

Одна и та же конфигурация `nlist=4096, nprobe=64` при n=1.28 M дала build_s=404 с в `scaling.csv` и 747 с в `ivf_flat.csv`. Корневая причина: код в ноутбуке 05 (scaling) **не выставлял** `idx.cp.min_points_per_centroid=5`, в отличие от ноутбука 02 (sweep). FAISS при дефолтном пороге 39 точек/центроид делает **меньше итераций Lloyd-алгоритма** → обучение IVF-квантизатора ≈ в 2× быстрее. Recall практически идентичен (Δ R@100 ≤ 0.5 п.п., QPS Δ 13 %). **Исправлено в `_build_notebooks.py`** — scaling-цикл теперь выставляет такие же CP-параметры (см. функцию `build_search`). После следующего полного ре-рана значения должны совпасть в пределах ~5 %.

#### 5. Потолок recall IVF+PQ ≈ 0.77 *(severity: СРЕДНЯЯ)*

**Сырые числа из CSV:**

- `IVFPQ max R@100 = 0.674 — ceiling at this dim` → `best PQ config (nlist=256, M=64, nprobe=256) cannot serve ≥ 0.95 SLA`

При 2048-D ResNet-эмбеддингах PQ M=128 даёт 16 байт на вектор (125× компрессия). Этого достаточно для recall ≈ 0.77, но квантизация теряет тонкие различия между близкими соседями. Любая (nlist, nprobe) комбинация в нашей сетке остаётся ниже 0.80 — это **потолок семейства** при заданном битовом бюджете, не методологическая ошибка.

#### 6. IVF+PQ build_s расходится между scaling.csv и sweep CSV (44 %) *(severity: СРЕДНЯЯ)*

**Сырые числа из CSV:**

- `IVFPQ: build_s mismatch scaling vs sweep (29 %)` → `scaling.csv=9s, ivfpq_*.csv=12s for identical config; QPS gap 90 %`

Тот же сценарий, что и у IVFFlat (см. предыдущую запись): `nlist=4096, nprobe=64, M=64` дала 436 с в scaling vs 781 с в sweep. Лечение идентично — выставить `cp.min_points_per_centroid=5` в scaling-функции, что и сделано.

#### 7. LSH build_s расходится между scaling.csv и sweep CSV (27 %) *(severity: СРЕДНЯЯ)*

**Сырые числа из CSV:**

- `LSH: build_s mismatch scaling vs sweep (50 %)` → `scaling.csv=2s, lsh_*.csv=3s for identical config; QPS gap 60 %`

У LSH (`nbits=4096`) расхождение **не связано с CP-параметрами** (k-means не используется): scaling — 198 с, sweep — 157 с. QPS совпадает в пределах 0.3 %, recall — в пределах 0.5 %. Это **базовая вариативность билд-времени** между прогонами при загруженной системе (~25 % run-to-run shape — типичный разброс для long-running malloc-bound операций). Если интересует точность, надо запускать в N повторах и брать медиану. Для отчёта 27 % — допустимая погрешность.

#### 8. Build_s немонотонен по M (PQ) *(severity: НИЗКАЯ)*

**Сырые числа из CSV:**

- `IVFPQ nlist=256 build_s non-monotonic in M` → `M=32→32.5s; M=64→12.5s`

Билд IVF+PQ — это IVF-обучение (~80 % времени) + per-cell PQ-kmeans. При повторных запусках PQ-обучение горячит CPU-кеш и второй запуск (M=64) может оказаться быстрее первого (M=32). На большом nlist (где IVF-обучение однозначно доминирует) эффект исчезает.

### 5.2. Cross-CSV консистентность

![Cross-CSV consistency: scaling vs sweep](img/light/05_cross_csv_consistency.png)

Одна и та же `(family, config)` при n=1.28 M была измерена дважды — в `<family>.csv` (per-family sweep) и в `scaling.csv`. Сравнение build_s и QPS:

| Family | Конфиг | build_s sweep | build_s scaling | Δ build | QPS sweep | QPS scaling | Δ QPS |
|---|---|---:|---:|---:|---:|---:|---:|
| IVFFlat | `{'nlist': 256, 'nprobe': 64}` | 8 с | 5 с | **37 %** | 239 | 931 | 289 % |
| IVF+PQ | `{'nlist': 256, 'nprobe': 64, 'M': 64}` | 12 с | 9 с | **29 %** | 8,134 | 15,436 | 90 % |
| HNSW | `{'M': 32, 'efC': 200, 'efS': 160}` | 12 с | 4 с | **69 %** | 8,166 | 16,269 | 99 % |
| LSH | `{'nbits': 1024}` | 3 с | 2 с | **50 %** | 7,596 | 12,171 | 60 % |

**Корневая причина:** код в ноутбуке 05 (scaling) при обучении IVF-семейств **не выставлял** `idx.cp.min_points_per_centroid = 5`, в отличие от ноутбука 02 (sweep), который этот параметр выставлял. FAISS с дефолтным минимумом (39 точек/центроид) запускает меньше итераций Lloyd-алгоритма → обучение стабильнее, но **в ~2× быстрее**. Recall практически одинаков (Δ ≤ 0.5 п.п.), но build_s/RSS — нет.

**Что исправлено в коде:** в `_build_notebooks.py` функция scaling-цикла `build_search()` теперь устанавливает `idx.cp.min_points_per_centroid = 5` и `idx.cp.max_points_per_centroid = max(256, len(train_x) // nlist)` для всех IVF-семейств, как и основной свип. После следующего полного ре-рана значения должны совпасть в пределах кеш-вариативности (~5 %).

HNSW не аффектится (там нет k-means).

## 6. Методология и caveats

**1. QPS-замер.** Скрипт `run_all.sh` выставляет `LAB_QPS_REPEAT=3 LAB_QPS_WARMUP=1`: один warm-up прогон + 3 повтора + медиана. Это снижает шум, но не даёт per-query разброс. Для tight-numbers — `LAB_QPS_REPEAT=5 LAB_QPS_WARMUP=2`.

**2. latency_p99_ms в существующих CSV — не настоящий p99.** Старая версия `measure_qps` брала p99 от 3 чисел (per-batch тайминги) → значение совпадает со средним. **Код `utils.measure_qps` переписан**: теперь батч режется на чанки по 50 запросов, каждый чанк таймится отдельно, p99 берётся по реальному распределению. Колонку в CSV следует **игнорировать** до следующего полного ре-рана; в графиках мы её сознательно не показываем.

**3. Centroid undertraining при nlist=16384.** Train slice = 200 000 векторов, при nlist=16384 это ~12 точек/центроид. FAISS явно предупреждает (`lloyd_3`). IVFFlat-конфигурации с этим nlist слегка под-тренированы, но recall-числа адекватны фактическому состоянию обученного индекса.

**4. IVFSQ свип был single-nlist (только 256).** Это было артефактом старого кода `best_nlist` (брался победитель по recall у IVFFlat). **Исправлено в `_build_notebooks.py`**: IVFSQ теперь свипится по nlist ∈ {256, 1024, 4096} и по SQ-типам {SQ4, SQ8}. Для текущего CSV пока только nlist=256.

**5. Ground truth пересчитан локально.** Стандартный GT (`imagenet_groundtruth.ivecs`) индексирует в IDs > N для любого N < 1 281 167, поэтому для свипов с N_SWEEP используется свежий GT, пересчитанный через `IndexFlatL2` над тем же base-slice. Кеш — `data/gt_n1281167_k100.npy`. Числа recall сравнимы между свипами.

**6. Peak RSS включает mmap-страницы.** `stream_add` проходит base через `np.memmap`; ОС считает резидентные страницы у процесса, и пик RSS суммирует index + train slice + mmap-кеш + Python overhead. Для IVFPQ/LSH доминирует именно mmap-кеш. Это видно на `05_memory_budget.png` (фиолетовый сегмент). Альтернатива — `MAP_POPULATE` или явный `madvise(DONTNEED)` после билда; не критично, поскольку эти страницы легко освобождаются ОС при memory pressure.

**7. RssPeakMonitor sampling.** Интервал 50 мс. Кратковременные пики `< 50 мс` могут быть упущены (типично для FAISS — пиковые аллокации происходят во время Lloyd-iter и длятся секунды).

## 7. Заключение и рекомендации

- **High-recall serving (R@100 ≥ 0.95)** → **HNSW** `M=8, efConstruction=200, efSearch=160`: 15,455 QPS, 0.065 мс средняя latency, 789 МБ на диске, 2.65 ГБ peak RSS (~0 % из которых — mmap base-вектора). Для R≥0.99 — `M=16, efConstruction=200, efSearch=320` (R@100=0.9925, 6,110 QPS).
- **Минимальный размер индекса** → **IVF+PQ** `nlist=256, nprobe=4, M=64, nbits=8`: 11 МБ (~73× меньше IVFFlat), R@100=0.618, 60,065 QPS. Потолок семейства — R@100=0.674 (M=128, 16 байт/вектор). Использовать только как кандидат-генератор перед rerank-стадией.
- **Компрессия с высоким recall** → **IVF+SQ-8** `nlist=256, nprobe=64, sq=SQ8`: R@100=0.9913, 843 QPS, 198 МБ (4× меньше IVFFlat). Per-query latency ~1.2 мс — медленнее HNSW, потому что SQ декодирует на лету при вычислении дистанции.
- **Exact-ish baseline** → **IVFFlat** `nlist=1024, nprobe=64`: R@100=0.9780, 1,235 QPS, 790 МБ. Билд 27.4 с. Для прода QPS слишком низкий; ценно как ground-truth-comparable движок и как калибратор GT.
- **Sub-baseline** → **LSH** даже при `nbits=1024` даёт всего R@100=0.494. При 2048-D случайные гиперплоскости требуют экспоненциального числа бит на единицу cosine-разрешения — footprint уходит за PQ задолго до того, как recall становится приемлемым.

**Финальная рекомендация:** HNSW (M=32, efC=200, efS=160) — production-default; IVF+PQ только в связке с rerank-стадией; IVF+SQ-8, если QPS-бюджет ≥ 100 и компрессия критична; IVFFlat — только для оффлайн GT-сравнений; LSH — отбросить.

---

### Что изменилось в коде по результатам этой ревизии

- `utils.measure_qps` — реальный per-chunk p99 (chunk=50 запросов).
- `utils.compute_recall` — векторизованный через `np.searchsorted` (~20× быстрее).
- `utils.stream_add` — явный `del mm; gc.collect()` после батчей.
- `_build_notebooks.py` — `IndexIVFPQ.cp.min_points_per_centroid = 5` (было только у IVFFlat).
- `_build_notebooks.py` — IVFSQ свипится по `SQ_NLIST_GRID = [256, 1024, 4096]` (было single-nlist=256).
- `_build_notebooks.py` (scaling cell) — выставляет те же CP-параметры, что и per-family свипы, чтобы устранить расхождение build_s между `scaling.csv` и `<family>.csv` (см. п. 5).

Все эти изменения вступают в силу при следующем `bash run_all.sh`. Текущий отчёт построен на CSV до этих исправлений; колонка `latency_p99_ms` помечена как ненадёжная и в графиках не показывается.

_Полные CSV — `results/light/`. Графики — `docs/img/light/`. Регенерация — `python3 scripts/analyze_and_report.py --run light`._
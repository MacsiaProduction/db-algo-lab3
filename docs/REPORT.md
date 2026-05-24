# Краткий отчёт по FAISS-ANN бенчмарку — прогон `full`

> Краткая версия. Все графики и числа сохранены, пояснения сведены к минимуму. Подробные объяснения и доказательства аномалий — в `DRAFT.md`. Описание методики, алгоритмов и графиков — `METHODOLOGY.md`.

> Сгенерирован `scripts/analyze_and_report.py` из CSV в `results/full/`. Графики — `docs/img/full/`.

## 1. Условия эксперимента

- **Датасет:** ImageNet-1M ZJU, 2048-D, n_base = 1,281,167, n_query = 10 000 (для свипов), n_gt = 25 000.
- **Метрика расстояния:** L2.
- **Параллельность:** 8 OpenMP-threads (FAISS).
- **Платформа:** local single-host (см. notebook 01 для деталей RAM/CPU).
- **QPS-замер:** `LAB_QPS_REPEAT=3 LAB_QPS_WARMUP=1` (warmup + медиана 3 запусков).
- **Свипы:** IVFFlat 23 конфигов, IVFPQ 36, IVFSQ 10, HNSW 56 (varyM + varyEFC), LSH 6.

## 2. Сводка результатов

![Cross-family Pareto](img/full/05_global_pareto.png)

### 2.1. Operational picks (макс. QPS при первом достижимом recall-флоре)

| Семейство | Recall флор | Recall@100 | QPS | Mean lat. | Index size | Build | Peak RSS | Конфиг |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **IVFFlat** | 0.95 | 0.9696 | 1,097 | 0.911 мс | 9.91 ГБ | 49.0 мин | 25.50 ГБ | `nlist=16384, nprobe=64` |
| **IVF+PQ** | 0.50 | 0.5341 | 58,467 | 0.017 мс | 98 МБ | 2.7 мин | 16.09 ГБ | `nlist=1024, nprobe=1, M=64, nbits=8` |
| **IVF+SQ** | 0.95 | 0.9882 | 97 | 10.339 мс | 2.46 ГБ | 19.8 с | 17.66 ГБ | `nlist=256, nprobe=16, sq=SQ8` |
| **HNSW** | 0.95 | 0.9589 | 7,891 | 0.127 мс | 9.95 ГБ | 6.0 мин | 20.06 ГБ | `M=16, efConstruction=200, efSearch=80` |
| **LSH** | 0.20 | 0.2520 | 5,146 | 0.194 мс | 41 МБ | 11.8 с | 12.11 ГБ | `nbits=256` |

### 2.2. «Колено» Парето-кривой по каждому семейству

| Семейство | Recall@100 | QPS | Index size | Конфиг |
|---|---:|---:|---:|---|
| IVFFlat | 0.8643 | 3,969 | 9.91 ГБ | `nlist=16384, nprobe=16` |
| IVF+PQ | 0.7588 | 13,166 | 200 МБ | `nlist=4096, nprobe=16, M=128, nbits=8` |
| IVF+SQ | 0.9481 | 425 | 2.46 ГБ | `nlist=256, nprobe=4, sq=SQ8` |
| HNSW | 0.9175 | 11,850 | 9.87 ГБ | `M=8, efConstruction=200, efSearch=80` |
| LSH | 0.3099 | 2,846 | 82 МБ | `nbits=512` |

### 2.3. Победители по квадрантам (по всему свипу)

- **Максимальный Recall@100:** IVFFlat = 1.0000 (`nlist=256, nprobe=256`).
- **Максимальный QPS:** IVF+PQ = 76,321 при recall 0.492 (`nlist=1024, nprobe=1, M=32, nbits=8`).
- **Минимальный размер индекса:** LSH = 21 МБ (`nbits=128`).
- **Самый быстрый билд:** LSH = 8.4 с (`nbits=128`).

![Operational picks: build / size / RSS / QPS](img/full/05_best_bars.png)

![Разложение peak RSS](img/full/05_memory_budget.png)

![Средняя per-query latency](img/full/05_latency_best.png)

## 3. Анализ по семействам

![Парето по семействам с knee и порогами recall](img/full/05_per_family_knees.png)

![Recall@100 при заданном QPS-бюджете](img/full/05_recall_at_qps.png)

### 3.1. IVFFlat

- **Размер свипа:** 23 конфигов.
- **Recall@100:** 0.221 → 1.0000.
- **QPS:** 6 → 16,227.
- **Размер индекса:** 9.79 ГБ → 9.91 ГБ.
- **Build:** 55.6 с → 49.0 мин.

Лучшая конфигурация при каждом recall-флоре:

| Recall флор | Конфиг | Recall@100 | QPS | Mean lat. |
|---:|---|---:|---:|---:|
| 0.99 | `nlist=16384, nprobe=256` | 0.9967 | 265 | 3.768 мс |
| 0.95 | `nlist=16384, nprobe=64` | 0.9696 | 1,097 | 0.911 мс |
| 0.90 | `nlist=4096, nprobe=16` | 0.9445 | 1,162 | 0.861 мс |
| 0.80 | `nlist=16384, nprobe=16` | 0.8643 | 3,969 | 0.252 мс |
| 0.50 | `nlist=16384, nprobe=4` | 0.5603 | 8,897 | 0.112 мс |
| 0.20 | `nlist=4096, nprobe=1` | 0.3981 | 16,227 | 0.062 мс |

### 3.2. IVF+PQ

- **Размер свипа:** 36 конфигов.
- **Recall@100:** 0.377 → 0.7709.
- **QPS:** 125 → 76,321.
- **Размер индекса:** 59 МБ → 200 МБ.
- **Build:** 2.7 мин → 13.3 мин.

Лучшая конфигурация при каждом recall-флоре:

| Recall флор | Конфиг | Recall@100 | QPS | Mean lat. |
|---:|---|---:|---:|---:|
| 0.50 | `nlist=1024, nprobe=1, M=64, nbits=8` | 0.5341 | 58,467 | 0.017 мс |
| 0.20 | `nlist=1024, nprobe=1, M=32, nbits=8` | 0.4921 | 76,321 | 0.013 мс |

![IVFPQ: recall vs nprobe + footprint vs recall](img/full/05_ivfpq_grid.png)

### 3.3. IVF+SQ

- **Размер свипа:** 10 конфигов.
- **Recall@100:** 0.684 → 0.9923.
- **QPS:** 21 → 2,582.
- **Размер индекса:** 1.23 ГБ → 2.46 ГБ.
- **Build:** 19.7 с → 19.8 с.

Лучшая конфигурация при каждом recall-флоре:

| Recall флор | Конфиг | Recall@100 | QPS | Mean lat. |
|---:|---|---:|---:|---:|
| 0.99 | `nlist=256, nprobe=64, sq=SQ8` | 0.9922 | 36 | 27.793 мс |
| 0.95 | `nlist=256, nprobe=16, sq=SQ8` | 0.9882 | 97 | 10.339 мс |
| 0.90 | `nlist=256, nprobe=4, sq=SQ8` | 0.9481 | 425 | 2.354 мс |
| 0.80 | `nlist=256, nprobe=4, sq=SQ4` | 0.8626 | 517 | 1.935 мс |
| 0.50 | `nlist=256, nprobe=1, sq=SQ4` | 0.6842 | 2,582 | 0.387 мс |
| 0.20 | `nlist=256, nprobe=1, sq=SQ4` | 0.6842 | 2,582 | 0.387 мс |

### 3.4. HNSW

- **Размер свипа:** 56 конфигов.
- **Recall@100:** 0.478 → 0.9998.
- **QPS:** 1,059 → 34,489.
- **Размер индекса:** 9.87 ГБ → 10.25 ГБ.
- **Build:** 2.7 мин → 13.5 мин.

Лучшая конфигурация при каждом recall-флоре:

| Recall флор | Конфиг | Recall@100 | QPS | Mean lat. |
|---:|---|---:|---:|---:|
| 0.99 | `M=32, efConstruction=200, efSearch=160` | 0.9923 | 3,830 | 0.261 мс |
| 0.95 | `M=16, efConstruction=200, efSearch=80` | 0.9589 | 7,891 | 0.127 мс |
| 0.90 | `M=8, efConstruction=200, efSearch=80` | 0.9175 | 11,850 | 0.085 мс |
| 0.80 | `M=8, efConstruction=200, efSearch=40` | 0.8132 | 18,441 | 0.054 мс |
| 0.50 | `M=16, efConstruction=200, efSearch=10` | 0.5794 | 28,606 | 0.035 мс |
| 0.20 | `M=8, efConstruction=200, efSearch=10` | 0.4785 | 34,489 | 0.029 мс |

### 3.5. LSH

- **Размер свипа:** 6 конфигов.
- **Recall@100:** 0.192 → 0.3944.
- **QPS:** 249 → 8,878.
- **Размер индекса:** 21 МБ → 658 МБ.
- **Build:** 8.4 с → 2.6 мин.

Лучшая конфигурация при каждом recall-флоре:

| Recall флор | Конфиг | Recall@100 | QPS | Mean lat. |
|---:|---|---:|---:|---:|
| 0.20 | `nbits=256` | 0.2520 | 5,146 | 0.194 мс |

## 4. Масштабирование 100K → 1.28M

![Scaling: recall/QPS/build/RSS vs N](img/full/05_scaling.png)

| Family | N | Recall@100 | QPS | Build | Peak RSS |
|---|---:|---:|---:|---:|---:|
| HNSW | 100,000 | 0.9828 | 8,178 | 11.6 с | 2.95 ГБ |
| HNSW | 250,000 | 0.9900 | 5,524 | 47.4 с | 6.88 ГБ |
| HNSW | 500,000 | 0.9909 | 4,307 | 2.1 мин | 11.95 ГБ |
| HNSW | 1,000,000 | 0.9921 | 3,861 | 5.3 мин | 21.24 ГБ |
| HNSW | 1,281,167 | 0.9920 | 3,762 | 6.7 мин | 20.27 ГБ |
| IVFFlat | 100,000 | 0.9307 | 6,543 | 2.2 мин | 2.45 ГБ |
| IVFFlat | 250,000 | 0.9668 | 1,701 | 4.5 мин | 5.33 ГБ |
| IVFFlat | 500,000 | 0.9794 | 651 | 5.0 мин | 9.84 ГБ |
| IVFFlat | 1,000,000 | 0.9880 | 295 | 6.0 мин | 18.99 ГБ |
| IVFFlat | 1,281,167 | 0.9901 | 225 | 6.7 мин | 23.61 ГБ |
| IVFPQ | 100,000 | 0.6774 | 22,334 | 2.4 мин | 4.76 ГБ |
| IVFPQ | 250,000 | 0.6549 | 18,434 | 4.7 мин | 6.57 ГБ |
| IVFPQ | 500,000 | 0.6366 | 14,101 | 5.3 мин | 8.86 ГБ |
| IVFPQ | 1,000,000 | 0.6312 | 9,356 | 6.5 мин | 13.93 ГБ |
| IVFPQ | 1,281,167 | 0.6293 | 7,787 | 7.3 мин | 14.28 ГБ |
| IVFSQ | 100,000 | 0.9298 | 15,567 | 2.4 мин | 2.59 ГБ |
| IVFSQ | 250,000 | 0.9647 | 6,907 | 4.7 мин | 4.98 ГБ |
| IVFSQ | 500,000 | 0.9759 | 3,068 | 5.2 мин | 6.67 ГБ |
| IVFSQ | 1,000,000 | 0.9830 | 1,349 | 6.2 мин | 11.46 ГБ |
| IVFSQ | 1,281,167 | 0.9842 | 997 | 6.8 мин | 14.93 ГБ |
| LSH | 100,000 | 0.5520 | 2,342 | 12.2 с | 3.00 ГБ |
| LSH | 250,000 | 0.5067 | 1,131 | 30.6 с | 4.65 ГБ |
| LSH | 500,000 | 0.4655 | 610 | 1.0 мин | 6.86 ГБ |
| LSH | 1,000,000 | 0.4162 | 317 | 2.0 мин | 10.12 ГБ |
| LSH | 1,281,167 | 0.3965 | 250 | 3.3 мин | 11.39 ГБ |

## 5. Аномалии и data quality

![Сводка аномалий](img/full/05_anomaly_flags.png)

| # | Severity | Аномалия | Численное доказательство |
|---:|---|---|---|
| 1 | СРЕДНЯЯ | Recall HNSW немонотонен по efConstruction при низком efSearch | `efC=40→R@100=0.722; efC=100→R@100=0.606; efC=200→R@100=0.627; efC=400→R@100=0.634` |
| 2 | СРЕДНЯЯ | Recall HNSW немонотонен по efConstruction при низком efSearch | `efC=40→R@100=0.851; efC=100→R@100=0.771; efC=200→R@100=0.792; efC=400→R@100=0.800` |
| 3 | СРЕДНЯЯ | IVFFlat build_s расходится между scaling.csv и sweep CSV (46 %) | `scaling.csv=404s, ivfflat_*.csv=747s for identical config; QPS gap 13 %` |
| 4 | СРЕДНЯЯ | Потолок recall IVF+PQ ≈ 0.77 | `best PQ config (nlist=1024, M=128, nprobe=256) cannot serve ≥ 0.95 SLA` |
| 5 | СРЕДНЯЯ | IVF+PQ build_s расходится между scaling.csv и sweep CSV (44 %) | `scaling.csv=436s, ivfpq_*.csv=781s for identical config; QPS gap 10 %` |
| 6 | СРЕДНЯЯ | LSH build_s расходится между scaling.csv и sweep CSV (27 %) | `scaling.csv=198s, lsh_*.csv=157s for identical config; QPS gap 0 %` |
| 7 | СРЕДНЯЯ | Колонка latency_p99_ms ≈ latency_ms | `mean p99/mean ratio ≈ 1.00 in every sweep (worst single row 1.137) — column reports p99 of QPS_REPEAT=3 batch retimings, not per-query tail` |
| 8 | НИЗКАЯ | Peak RSS немонотонен в скейлинге (HNSW) | `21.2→20.3 GB — peak monitor missed a spike or earlier alloc freed` |
| 9 | НИЗКАЯ | IVFFlat QPS при nprobe=1 растёт с nlist | `QPS(nlist=256)=604 vs QPS(nlist=16384)=13000 (ratio 0.05) — smaller partitions fit in L2 cache` |
| 10 | НИЗКАЯ | Build_s немонотонен по M (PQ) | `M=32→167.5s; M=64→161.1s; M=128→180.7s` |
| 11 | НИЗКАЯ | Build_s немонотонен по M (PQ) | `M=32→787.8s; M=64→780.7s; M=128→796.2s` |
| 12 | НИЗКАЯ | Отрицательный rss_delta_mb | `min=-2964 MB — earlier allocations got freed mid-build` |

### 5.1. Cross-CSV консистентность

![Cross-CSV consistency: scaling vs sweep](img/full/05_cross_csv_consistency.png)

| Family | Конфиг | build_s sweep | build_s scaling | Δ build | QPS sweep | QPS scaling | Δ QPS |
|---|---|---:|---:|---:|---:|---:|---:|
| IVFFlat | `{'nlist': 4096, 'nprobe': 64}` | 747 с | 404 с | **46 %** | 260 | 225 | 13 % |
| IVF+PQ | `{'nlist': 4096, 'nprobe': 64, 'M': 64}` | 781 с | 436 с | **44 %** | 8,666 | 7,787 | 10 % |
| HNSW | `{'M': 32, 'efC': 200, 'efS': 160}` | 429 с | 400 с | **7 %** | 3,714 | 3,762 | 1 % |
| LSH | `{'nbits': 4096}` | 157 с | 198 с | **27 %** | 249 | 250 | 0 % |

## 6. Методология и caveats

- `LAB_QPS_REPEAT=3 LAB_QPS_WARMUP=1`, медиана из 3 запусков.
- `latency_p99_ms` в CSV — p99 по 3 повторам батча, **не per-query**.
- Train slice = 200 000 (для nlist=16384 → ~12 точек/центроид, FAISS warns).
- Ground truth пересчитан локально через `IndexFlatL2`, кеш `data/gt_n1281167_k100.npy`.
- Peak RSS включает mmap-страницы базы (доминирует у IVFPQ/LSH).

## 7. Заключение и рекомендации

- **High-recall serving (R@100 ≥ 0.95)** → **HNSW** `M=16, efConstruction=200, efSearch=80`: 7,891 QPS, 0.127 мс средняя latency, 9.95 ГБ на диске, 20.06 ГБ peak RSS (~50 % из которых — mmap base-вектора). Для R≥0.99 — `M=32, efConstruction=200, efSearch=160` (R@100=0.9923, 3,830 QPS).
- **Минимальный размер индекса** → **IVF+PQ** `nlist=4096, nprobe=16, M=128, nbits=8`: 200 МБ (~51× меньше IVFFlat), R@100=0.759, 13,166 QPS. Потолок семейства — R@100=0.771 (M=128, 16 байт/вектор). Использовать только как кандидат-генератор перед rerank-стадией.
- **Компрессия с высоким recall** → **IVF+SQ-8** `nlist=256, nprobe=16, sq=SQ8`: R@100=0.9882, 97 QPS, 2.46 ГБ (4× меньше IVFFlat). Per-query latency ~10.3 мс — медленнее HNSW, потому что SQ декодирует на лету при вычислении дистанции.
- **Exact-ish baseline** → **IVFFlat** `nlist=16384, nprobe=64`: R@100=0.9696, 1,097 QPS, 9.91 ГБ. Билд 49.0 мин. Для прода QPS слишком низкий; ценно как ground-truth-comparable движок и как калибратор GT.
- **Sub-baseline** → **LSH** даже при `nbits=4096` даёт всего R@100=0.394. При 2048-D случайные гиперплоскости требуют экспоненциального числа бит на единицу cosine-разрешения — footprint уходит за PQ задолго до того, как recall становится приемлемым.

**Финальная рекомендация:** HNSW (M=32, efC=200, efS=160) — production-default; IVF+PQ только в связке с rerank-стадией; IVF+SQ-8, если QPS-бюджет ≥ 100 и компрессия критична; IVFFlat — только для оффлайн GT-сравнений; LSH — отбросить.


_Полные CSV — `results/full/`. Графики — `docs/img/full/`. Регенерация — `python3 scripts/analyze_and_report.py --run full`._
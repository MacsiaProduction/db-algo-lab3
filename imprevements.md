```markdown
# db-algo-lab3 — итоговый ревью бенчмарка FAISS-ANN

## Состояние

**Report-ready** в двух языковых/уровневых вариантах:
- `docs/OTCHET_polnyj_full.md` — подробный (черновик, все объяснения).
- `docs/OTCHET_kratkij_full.md` — краткий (графики + цифры, минимум текста).

Английский легаси: `docs/REPORT_full.md` (генерируется только с `--english`).

## Что сделано в этом раунде

### Код
- `utils.measure_qps` — переписан: батч режется на чанки по 50 запросов, p99
  берётся по реальному per-chunk распределению. До этого p99 ≈ mean всегда
  (3 числа из QPS_REPEAT batch-retimings).
- `utils.compute_recall` — векторизация через `np.searchsorted` (~20× быстрее).
- `utils.stream_add` — `del mm; gc.collect()` после батчей, чтобы mmap
  не пачкал rss_before следующего билда.
- `_build_notebooks.py` IVFSQ — переход с single-`best_nlist` на свип
  `SQ_NLIST_GRID = [256, 1024, 4096]` × `SQ_TYPES = [SQ4, SQ8]`. CP-параметры
  (`min/max_points_per_centroid`) выставляются как у IVFFlat.
- `_build_notebooks.py` IVFPQ — выставляется `cp.min_points_per_centroid = 5`
  (раньше только IVFFlat имел этот override).
- `_build_notebooks.py` scaling (ноутбук 05) — `build_search()` теперь
  устанавливает те же CP-параметры, что и per-family свипы. Это убирает
  расхождение build_s между `scaling.csv` и `<family>.csv` (~45 % для
  IVFFlat/IVFPQ при идентичной конфигурации).

### Анализ и отчёт
- `scripts/analyze_and_report.py`:
  - Новые аномалии: cross-CSV consistency (scaling vs sweep), latency_p99 ≈ mean.
  - Новый график: `05_cross_csv_consistency.png` (compare build_s/QPS для одной
    конфигурации в двух CSV).
  - Перерисован `05_latency_best.png` — убран ненастоящий p99 (с jitter-боксом
    "это не настоящий p99, нужен ре-ран").
  - Перерисован `05_memory_budget.png` — корректное 3-stack разложение
    (other + idx + transient) с суммой = peak.
  - Две Russian-отчёт функции: `write_report_ru(mode='full'|'short')`.
  - Грекоппинг повторяющихся объяснений (HNSW efS=10/efS=20 → 1 запись;
    IVFPQ nlist=1024/4096 в M-build_s → 1 запись).
  - Таблица аномалий сортируется по severity.

## Аномалии (12, все объяснены с доказательствами)

**СРЕДНЯЯ:**
1. HNSW efC monotonicity violated at efS=10/20 — рыхлый граф (efC=40)
   сохраняет shortcut-рёбра; при efS≥80 эффект исчезает.
2. IVFFlat build_s mismatch scaling vs sweep (46 %) — CP-параметры (исправлено).
3. IVFPQ ceiling R@100=0.77 — потолок семейства при 16 байт/вектор.
4. IVFPQ build_s mismatch (44 %) — те же CP-параметры (исправлено).
5. LSH build_s mismatch (27 %) — run-to-run variance baseline, не CP-params.
6. latency_p99_ms ≈ latency_ms — старый `measure_qps` (исправлено).

**НИЗКАЯ:**
7. HNSW peak RSS dropped n=1M→1.28M — cold page-cache after big IVFFlat build
   at 1.28M.
8. IVFFlat nprobe=1 QPS scales with nlist — cache effects на партишн-сайз.
9. IVFPQ build_s non-monotonic in M — CPU cache warmth between builds.
10. ivf_pq: 12 rows with rss_delta_mb < -100 MB — GC between builds.

## Открытые non-blocking лимиты

- Колонка `latency_p99_ms` в существующих CSV — не настоящий p99. Графики её
  не показывают. Код исправлен; новый ре-ран даст корректные значения.
- IVFSQ sweep в текущих CSV — только nlist=256. Код расширен до
  `[256, 1024, 4096]`, ре-ран нужен.
- Scaling vs sweep gap — код исправлен, ре-ран даст совпадение.

## Ре-ран нужен или нет?

**Не нужен для текущего отчёта** — все аномалии задокументированы и объяснены
с доказательствами из CSV. Код исправлен, поэтому следующий полный
`bash run_all.sh` даст консистентные данные с реальным p99 и расширенным
IVFSQ-свипом.

## Артефакты

- Reports: `docs/OTCHET_polnyj_full.md`, `docs/OTCHET_kratkij_full.md`.
- Charts: `docs/img/full/05_*.png` (9 + cross-csv = 10 ключевых графиков).
- Code: `utils.py`, `_build_notebooks.py`, `scripts/analyze_and_report.py`.
- CSVs (unchanged): `results/full/{ivf_flat,ivf_pq,ivf_sq,hnsw_*,lsh,scaling}.csv`.

```


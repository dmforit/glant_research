# Запуск экспериментов

## Рекомендуемый порядок запуска

Все команды ниже запускаются из корня проекта. В примерах используется Python
из окружения `graph_research`; на сервере можно заменить путь на обычный
`python`, если активировано нужное окружение.

Основной набор датасетов:

```text
Cora Citeseer Texas AIFB IMDB ACM
```

### 1. Запусти обычные baselines

```powershell
C:\Users\dmitr\miniconda3\envs\graph_research\python.exe -B scripts\run_experiment_batch.py --mode run_baselines --datasets Cora Citeseer Texas AIFB IMDB ACM --seeds 0 1 2 --gpu 0 --skip-existing
```

Запускает:

```text
GCN
GraphSAGE
GATv2
```

### 2. Запусти k-hop baselines

```powershell
C:\Users\dmitr\miniconda3\envs\graph_research\python.exe -B scripts\run_experiment_batch.py --mode run_khop_baselines --datasets Cora Citeseer Texas AIFB IMDB ACM --seeds 0 1 2 --gpu 0 --skip-existing
```

Запускает:

```text
MixHop
TAGConv
```

### 3. Запусти HoGA

```powershell
C:\Users\dmitr\miniconda3\envs\graph_research\python.exe -B scripts\run_experiment_batch.py --mode run_hoga --datasets Cora Citeseer Texas AIFB IMDB ACM --seeds 0 1 2 --gpu 0 --skip-existing
```

### 4. Подбери гиперпараметры GLANT через Optuna

Пробный запуск:

```powershell
C:\Users\dmitr\miniconda3\envs\graph_research\python.exe -B scripts\run_glant_hpo.py --datasets Cora Citeseer Texas AIFB IMDB ACM --models glant_v1 glant_v2 --trial-limit 1 --epochs 5 --gpu 0
```

Полный запуск:

```powershell
C:\Users\dmitr\miniconda3\envs\graph_research\python.exe -B scripts\run_glant_hpo.py --datasets Cora Citeseer Texas AIFB IMDB ACM --models glant_v1 glant_v2 --trials-v1 10 --trials-v2 20 --gpu 0
```

Результаты HPO:

```text
results/launches/{launch_id}/summary/hpo_results.csv
results/launches/{launch_id}/summary/best_hpo_configs.json
results/launches/{launch_id}/summary/optuna_trials_{dataset}_{model}.csv
```

### 5. Запусти финальные GLANT runs

После HPO зафиксируй лучшие параметры из `best_hpo_configs.json`, затем:

```powershell
C:\Users\dmitr\miniconda3\envs\graph_research\python.exe -B scripts\run_experiment_batch.py --mode run_glant_v1 --datasets Cora Citeseer Texas AIFB IMDB ACM --seeds 0 1 2 --gpu 0 --skip-existing
```

```powershell
C:\Users\dmitr\miniconda3\envs\graph_research\python.exe -B scripts\run_experiment_batch.py --mode run_glant_v2 --datasets Cora Citeseer Texas AIFB IMDB ACM --seeds 0 1 2 --gpu 0 --skip-existing
```

### 6. Запусти ablation

```powershell
C:\Users\dmitr\miniconda3\envs\graph_research\python.exe -B scripts\run_experiment_batch.py --mode run_ablation --datasets Cora Citeseer Texas AIFB IMDB ACM --seeds 0 1 2 --gpu 0 --skip-existing
```

### 7. Собери summary

```powershell
C:\Users\dmitr\miniconda3\envs\graph_research\python.exe -B collect_summary.py --results-dir results
```

Итоговые файлы:

```text
results/summary/main_results_long.csv
results/summary/main_results.csv
results/summary/main_results.xlsx
```

### Полезные batch-режимы

Проверить команды без запуска:

```powershell
C:\Users\dmitr\miniconda3\envs\graph_research\python.exe -B scripts\run_experiment_batch.py --mode all --datasets Cora --seeds 0 --trial-limit 1 --epochs 5 --gpu 0 --dry-run
```

Запустить только HPO через batch runner:

```powershell
C:\Users\dmitr\miniconda3\envs\graph_research\python.exe -B scripts\run_experiment_batch.py --mode run_hpo --datasets Cora Citeseer Texas AIFB IMDB ACM --trial-limit 1 --epochs 5 --gpu 0
```

`--mode all` запускает весь pipeline: baselines, k-hop baselines, HoGA, HPO,
GLANT-v1, GLANT-v2, ablation и summary. Для обычного рабочего запуска лучше
идти по шагам выше.

---

Основной файл запуска:

```bash
python main.py --dataset <DATASET> --train
````

Поддерживаемые модели задаются в `config.baselines.names`.

Поддерживаемые датасеты:

```text
Cora Pubmed Citeseer Computers Photo Actor Wisconsin Texas
AIFB MUTAG BGS DBLP IMDB ACM
```

`AIFB`, `MUTAG`, `BGS`, `DBLP`, `IMDB`, `ACM` загружаются как
гетерогенные датасеты и приводятся к homogeneous `Data` для текущей
архитектуры node classification.

Основные параметры:

| Параметр                          | Значение                            |
| --------------------------------- | ----------------------------------- |
| `--dataset Cora`                  | один датасет                        |
| `--datasets Cora Citeseer Pubmed` | несколько датасетов                 |
| `--model GLANT`                   | одна модель                         |
| `--model glant gat gatv2`         | несколько моделей через GLANT       |
| `--train`                         | обучение                            |
| `--test`                          | тест сохранённых результатов        |
| `--gpu 0`                         | запуск на GPU                       |
| `--gpu -1`                        | запуск на CPU                       |
| `--khop 3`                        | число hop-уровней                   |
| `--alpha 0.85`                    | доля учёта higher-hop рёбер         |
| `--method balanced_unique_select` | метод семплирования hop-рёбер       |
| `--num-samples 500`               | число sampled рёбер                 |
| `--load-samples`                  | загрузить sampled hop-рёбра с диска |
| `--conv-type gatv2`               | заменить слой GLANT на обычный conv |
| `--runs 5`                        | число повторов                      |
| `--results-xlsx results/`         | директория для сводных таблиц       |

Для `GLANT` используется `conv_type="hop_gated_gatv2"` и список hop-графов:

```python
[E_1, E_2, ..., E_K]
```

Варианты `GLANT`, `GATv2` и `GAT` создаются через одну обёртку `GLANT`.
Отличается только `conv_type`: `hop_gated_gatv2`, `gatv2` или `gat`.
Обычные conv-слои используют только `E_1`.

Параметр `alpha` управляет разреживанием higher-hop рёбер:

| `alpha`         | Поведение                          |
| --------------- | ---------------------------------- |
| `0.0`           | higher-hop рёбра удаляются         |
| `1.0`           | higher-hop рёбра сохраняются       |
| `0 < alpha < 1` | сохраняется часть higher-hop рёбер |

Сводная таблица всегда сохраняется с уникальным именем, включающим режим,
датасеты, модели, число запусков и timestamp. Если `--results-xlsx` не указан,
используется директория `results/`. Если передать старый формат вроде
`--results-xlsx results/full_comparison.xlsx`, будет использована только
директория `results/`, а имя файла всё равно будет сгенерировано автоматически.
Для `GLANT` в имя также добавляется тип архитектуры из `config.architecture`,
например `architecture-mixture-of-all-hops`.

---

## 1. Одна модель на одном датасете

GLANT:

```bash
python main.py \
  --dataset Cora \
  --train \
  --model GLANT \
  --khop 3 \
  --alpha 0.85 \
  --method balanced_unique_select \
  --num-samples 500 \
  --gpu 0
```

GATv2 через ту же обвязку:

```bash
python main.py \
  --dataset Cora \
  --train \
  --model GLANT \
  --conv-type gatv2 \
  --gpu 0
```

---

## 2. Одна модель на нескольких датасетах

```bash
python main.py \
  --datasets Cora Citeseer Pubmed \
  --train \
  --model GLANT \
  --khop 3 \
  --alpha 0.85 \
  --method balanced_unique_select \
  --num-samples 500 \
  --gpu 0
```

---

## 3. Несколько моделей на одном датасете

Через CLI:

```bash
python main.py \
  --dataset Cora \
  --train \
  --model glant gatv2 gat \
  --gpu 0
```

Или в конфиге:

```python
config.baselines.names = ["GLANT", "GATv2", "GAT"]
```

Запуск:

```bash
python main.py \
  --dataset Cora \
  --train \
  --gpu 0
```

---

## 4. Несколько моделей на нескольких датасетах

В конфиге:

```python
config.baselines.names = ["GLANT", "GATv2", "GAT", "GCN"]
```

Запуск:

```bash
python main.py \
  --datasets Cora Citeseer Pubmed \
  --train \
  --runs 5 \
  --results-xlsx results/ \
  --gpu 0
```

---

## Тест сохранённых результатов

```bash
python main.py \
  --dataset Cora \
  --test \
  --model GLANT \
  --gpu 0
```

## Диагностика hop-gate

По умолчанию CSV/XLSX-диагностика hop-gate включена только для `GLANT`
с `conv_type="hop_gated_gatv2"` раз в 50 эпох:

```python
config.log_hop_diagnostics = True
config.hop_log_every = 50
config.hop_log_only_layer = None
```

Файлы пишутся отдельно для каждого запуска с коротким случайным суффиксом,
поэтому повторный запуск не дописывает строки в старый файл:

```text
logs/<model>/<dataset>/run_<idx>_<suffix>_summary.csv
logs/<model>/<dataset>/run_<idx>_<suffix>_summary.xlsx
```

В таблице остаются:
`event`, `epoch`, `phase`, `layer_id`, `lr`, `num_hops`, `weights_shape`,
`grad_norm`, `weights_mean_hop_*`, `weights_std_hop_*` и
`attention_norm_entropy_mean_hop_*`.

По умолчанию используется более мягкий `ReduceLROnPlateau`: `factor=0.8`,
`patience=50`, `min_lr=1e-5`, чтобы learning rate снижался медленнее.

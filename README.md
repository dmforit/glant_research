# Запуск экспериментов

Основной файл запуска:

```bash
python main.py --dataset <DATASET> --train
````

Поддерживаемые модели задаются в `config.baselines.names`.

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

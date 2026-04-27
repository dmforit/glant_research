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
| `--train`                         | обучение                            |
| `--test`                          | тест сохранённых результатов        |
| `--gpu 0`                         | запуск на GPU                       |
| `--gpu -1`                        | запуск на CPU                       |
| `--khop 3`                        | число hop-уровней                   |
| `--alpha 0.85`                    | доля учёта higher-hop рёбер         |
| `--method balanced_unique_select` | метод семплирования hop-рёбер       |
| `--num-samples 500`               | число sampled рёбер                 |
| `--conv-type gatv2`               | заменить слой GLANT на обычный conv |
| `--runs 5`                        | число повторов                      |
| `--results-xlsx results.xlsx`     | файл со сводной таблицей            |

Для `GLANT` используется `conv_type="hop_gated_gatv2"` и список hop-графов:

```python
[E_1, E_2, ..., E_K]
```

Обычные модели (`gatv2`, `gat`, `sage`, `gcn`) используют только `E_1`.

Параметр `alpha` управляет разреживанием higher-hop рёбер:

| `alpha`         | Поведение                          |
| --------------- | ---------------------------------- |
| `0.0`           | higher-hop рёбра удаляются         |
| `1.0`           | higher-hop рёбра сохраняются       |
| `0 < alpha < 1` | сохраняется часть higher-hop рёбер |

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

В конфиге:

```python
config.baselines.names = ["GLANT", "GATv2", "GAT", "GCN"]
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
  --results-xlsx results/full_comparison.xlsx \
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
# Запуск обучения

Список моделей задается в `new_configs/config.py`:

```python
config.baselines.names = ["GLANT", "GAT", "GATv2", "GCN"]
```

Запуск обучения на нескольких датасетах с несколькими runs:

```powershell
python main.py --train --datasets cora citeseer pubmed computers photo wisconsin texas --runs 5 --results-xlsx model_runs/results.xlsx
```

Для запуска на CPU добавь `--gpu -1`:

```powershell
python main.py --train --datasets cora citeseer pubmed computers photo wisconsin texas --runs 5 --gpu -1 --results-xlsx model_runs/results.xlsx
```

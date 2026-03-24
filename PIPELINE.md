# Ditto ER Pipeline (Config-Driven, No Notebook)

This pipeline builds **gold tables**, creates **blocking-based candidate pairs**, generates
`train/valid/test/predict` Ditto text files, trains a Ditto model, and writes test/predict reports.

## 1) Directory Contract

For each dataset, use:

```
dataset/
  <dataset>/
    <dataset>.csv
    standardized_<dataset>.csv
    gold_<dataset>.csv           # generated or reused
    predict_<dataset>.csv        # input for prediction pairing
```

Pipeline outputs:

```
data/
  <dataset>/
    <dataset>_train.txt
    <dataset>_valid.txt
    <dataset>_test.txt
    <dataset>_predict.txt

models/
  <dataset>/
    best_model.pt
    train_metrics.json

Test_Output/
  <dataset>_test_result/
    <dataset>_test.csv
    <dataset>_test_analysis.txt

predict_output/
  <dataset>_predict_result/
    <dataset>_predict.csv
    <dataset>_predict_analysis.txt
```

---

## 2) Config Files

- Dataset config: `configs/datasets/<dataset>.yaml`
- Blocking config: `configs/blocking.yaml`
- Training config: `configs/training.yaml`

Default example included:
- `configs/datasets/pro_supplier.yaml`

---

## 3) Run Commands

From repository root (`/workspace/Ditto`):

### Full end-to-end run

```bash
python -m pipeline.run_pipeline --dataset pro_supplier --stage all
```

### Stage-by-stage

```bash
python -m pipeline.run_pipeline --dataset pro_supplier --stage build_gold
python -m pipeline.run_pipeline --dataset pro_supplier --stage build_splits
python -m pipeline.run_pipeline --dataset pro_supplier --stage build_predict
python -m pipeline.run_pipeline --dataset pro_supplier --stage train
python -m pipeline.run_pipeline --dataset pro_supplier --stage test
python -m pipeline.run_pipeline --dataset pro_supplier --stage predict
```

You can also call individual modules directly:

```bash
python -m pipeline.build_gold --dataset pro_supplier
python -m pipeline.build_train_valid_test --dataset pro_supplier
python -m pipeline.build_predict_pairs --dataset pro_supplier
python -m pipeline.train_model --dataset pro_supplier
python -m pipeline.test_and_analyze --dataset pro_supplier
python -m pipeline.predict_and_analyze --dataset pro_supplier
```

---

## 4) Blocking Strategy Parameters

In `configs/blocking.yaml`:

- `strategy`: `sbert` | `ann` | `ngram`
- `top_k_train`: top-k candidates per anchor for train split generation
- `target_total_pairs`: approximate total pair count goal
- `top_k_predict`: top-k gold candidates for each predict record (default `1000`)

For `sbert` strategy:
- `model_name`, `batch_size_encode`, `block_rows`, `use_gpu`

For `ann` strategy:
- `nlist`, `nprobe`

For `ngram` strategy:
- `n` (character n-gram size)

---

## 5) Notes

- `build_gold` will use existing `canonical_int` if already present in raw CSV,
  otherwise it joins standardized table to raw table using configured join keys.
- Ditto text rows are generated in standard format:
  `record_left<TAB>record_right<TAB>label`
- Predict file uses dummy label (`0`) and model inference generates actual predictions.

---

## 6) Quick Troubleshooting

- If training fails to import Ditto classes, verify this path exists:
  `FAIR-DA4ER/ditto/ditto_light`
- If no pairs are generated, reduce blocking strictness or increase `top_k_train`.
- If prediction output is too large, reduce `top_k_predict`.

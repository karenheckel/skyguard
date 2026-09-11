# SkyGuard: Wildlife Strike Damage Assessment

Predicting whether an aircraft wildlife strike resulted in damage, using FAA Wildlife Strike Database records.

## Problem

Given the circumstances of a bird or wildlife strike, flight phase, altitude, airspeed, species, aircraft type, weather, and the reporter's free-text account, predict whether the aircraft sustained damage.

The dataset is heavily imbalanced: only **6.4%** of the 307,178 training records are damage events. A model that predicts "no damage" for everything scores 94% accuracy and is useless, so **balanced accuracy** is the evaluation metric throughout.

- Training set: 307,178 strikes, 54 raw columns
- Test set: 34,131 strikes (labels withheld)

## Results

| Iteration | Approach | Balanced Accuracy |
|---|---|---|
| 1 | Neural network (Keras, class-weighted) | 0.80 |
| 2 | XGBoost + LightGBM blend, engineered features | 0.83 |
| 3 | HistGradientBoosting ensemble + text mining | **0.87** |

Final model: 0.870 balanced accuracy, 86% recall on the damage class, measured via 7-fold stratified cross-validation on the training set (the competition test set is unlabeled).

## Approach

**Iteration 1 — Neural network baseline.** Aggressive cleaning: dropped every column missing ≥50% of values, hierarchical imputation for coordinates and timestamps, one-hot encoding, then a 3-layer dense network with dropout and class weights.

**Iteration 2 — Gradient boosting.** The 50%-missingness rule was discarding `SPEED` and `HEIGHT`, two of the most physically meaningful predictors, so filtering shifted to dropping low-signal columns instead (administrative fields, high-cardinality identifiers). Added ~30 engineered features: seasonal indicators, critical-flight-phase flags, a speed × bird-size impact proxy, species risk groupings, and binary missingness indicators. Blended XGBoost and LightGBM with a tuned weight and threshold.

**Iteration 3 — Text mining.** The `REMARKS` and `COMMENTS` fields turned out to carry most of the remaining signal, but naive keyword matching fails badly: ~50,000 no-damage records contain the word "damage," almost all inside negating phrases like *"no evidence of damage"* or *"carcass found during routine inspection."*

Built a context-aware regex pipeline with ~35 patterns and composite rules that let negation override keyword hits, and that distinguish real damage indicators (*"fan blade replaced," "delaminated," "returned to field"*) from incidental mentions. Combined with smoothed target encoding for high-cardinality airport and species fields, and a 5-model HistGradientBoosting ensemble with varied class weights and learning rates.

This iteration accounted for nearly all the gain over the boosting baseline — the model swap itself contributed very little.

## Stack

Python, pandas, scikit-learn, XGBoost, LightGBM, TensorFlow/Keras

## Running it

Open `project.ipynb` and run top to bottom. Expects `data/train.csv` and `data/test.csv`. Writes `submission.csv`.

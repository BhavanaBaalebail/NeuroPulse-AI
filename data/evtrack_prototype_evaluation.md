# Experimental behaviour prototype (heuristic labels)

These metrics compare the prototype to heuristic_v1 labels only. They do NOT measure real fatigue-detection accuracy, clinical validity, or improvement of the production NeuroPulse XGBoost model.

- Label version: `heuristic_v1`
- Windows used (Uncertain excluded): 507
- Class counts (all): {'Alert': 317, 'Mild Fatigue': 136, 'High Fatigue': 54}
- Preprocessing: SimpleImputer(strategy=median) and StandardScaler fitted on training users only. Missing values are not filled with zero.

## User-grouped split

- Found: True
- Limitation: High Fatigue test n=8 is small; High precision/recall are unstable. A 12-user grouped holdout cannot support a strong 3-class generalisation claim.
- Train users: 9 ['09df0b3f3fdd', '283aa72180a9', '2b92e06581f9', '30ff3f3f7a1a', '5961b24498f7', '5bf8c5be01f5', '7aaba93c0a5e', '8624fdaac2a1', 'cd40735be499']
- Test users: 3 ['4cd21207206a', '71847e417183', '9869a68bd470']
- Train counts: {'Alert': 248, 'Mild Fatigue': 104, 'High Fatigue': 46}
- Test counts: {'Alert': 69, 'Mild Fatigue': 32, 'High Fatigue': 8}

## Test metrics vs heuristic_v1 (not fatigue accuracy)

- Balanced accuracy: 0.7754

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Alert | 1.000 | 0.826 | 0.905 | 69 |
| Mild Fatigue | 0.632 | 0.750 | 0.686 | 32 |
| High Fatigue | 0.429 | 0.750 | 0.545 | 8 |

Confusion matrix (rows true, cols predicted; order ['Alert', 'Mild Fatigue', 'High Fatigue']):

```
[[57, 12, 0], [0, 24, 8], [0, 2, 6]]
```


```
              precision    recall  f1-score   support

       Alert       1.00      0.83      0.90        69
Mild Fatigue       0.63      0.75      0.69        32
High Fatigue       0.43      0.75      0.55         8

    accuracy                           0.80       109
   macro avg       0.69      0.78      0.71       109
weighted avg       0.85      0.80      0.81       109

```

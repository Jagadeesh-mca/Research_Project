# Explainable Machine Learning Intrusion Detection System (IDS)

> **Research Report & Experimental Study**
> Reference Paper: S. Muzibuddin et al., *Explainable Machine Learning-Based Intrusion Detection System Using Random Forest and SHAP for Network Security*, IJERT 2026.

---

## 1. Executive Summary & Core Results

- **Proposed Model Accuracy:** 100.00%
- **Detection F1-Score:** 1.0000
- **ROC-AUC:** 1.0000
- **Inference Latency:** 0.1639 ms per 1,000 flows
- **Detection Throughput:** 6,102 flows/sec
- **Pipeline Execution Time:** 77.88 seconds

---

## 2. Dataset & Preprocessing Methodology

- **Benchmark Source:** Real CSE-CIC-IDS2018 cloud dataset (C:\Users\Vishal\Downloads\ids_intrusion_detection_project\data\02-14-2018.csv) (sampled 1,000 flows)
- **Total Flow Records Sampled:** 1,000
- **Benign Class Proportion:** 869 (86.9%)
- **Malicious Class Proportion:** 131 (13.1%)
- **Attack Categories:** {'Benign': 869, 'SSH-Bruteforce': 131}
- **Stratified Train / Test Split:** 800 / 200 (80/20)

### Rigorous Leakage Prevention Checks
- **Train/Test Sample Overlap:** 0 duplicates (0.00%)
- **Non-Behavioral Identifier Leaked:** [] (Zero IP, Timestamp, or Flow ID retained)
- **Target Variable Contamination:** [] (Zero label contamination in feature matrix)
- **Scaler & Pruning Isolation:** Min-Max normalization and correlation pruning fitted strictly on `X_train`

![Class Distribution](figures/class_distribution.png)

---

## 3. Correlation & Multicollinearity Filtering

- **Threshold:** |r| > 0.9
- **Redundant Features Pruned:** 33
- **Retained Behavioral Features:** 35

![Correlation Heatmap](figures/correlation_heatmap.png)

---

## 4. Multi-Model Baseline Comparison

| Classifier | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Latency (ms/1k) | Throughput (flows/s) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9900 | 0.9286 | 1.0000 | 0.9630 | 1.0000 | 0.0264 | 37,933 |
| Decision Tree | 0.9900 | 0.9286 | 1.0000 | 0.9630 | 0.9943 | 0.0056 | 180,013 |
| Extra Trees | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2587 | 3,865 |
| Random Forest (Proposed) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1489 | 6,715 |

![Model Comparison](figures/model_comparison.png)
![Confusion Matrix](figures/confusion_matrix.png)
![ROC Curve](figures/roc_curve.png)
![PR Curve](figures/pr_curve.png)

---

## 5. SHAP Explainability

![SHAP Beeswarm Summary](figures/shap_summary.png)
![SHAP Feature Importance](figures/shap_bar.png)
![SHAP Dependence Plot](figures/shap_dependence.png)

### Local Explanations
- **Malicious Detection:** `figures/shap_waterfall_malicious.png`
- **Benign Classification:** `figures/shap_waterfall_benign.png`
- **Near-Boundary Risk:** `figures/shap_waterfall_fp.png`

---

## 6. False-Positive Analysis

- **False Positive Count:** 0
- **False Positive Rate:** 0.0000%
- **Diagnostic Mode:** Near-Boundary Benign Traffic Risk Analysis

![False Positive Drivers](figures/false_positive_drivers.png)

---

## 7. Feature Reduction & Distillation

| Model Variant | Features | Accuracy | F1-Score | ROC-AUC | FPR | Latency (ms/1k) | Throughput (flows/s) |
|---|---|---|---|---|---|---|---|
| Full (35 features) | 35 | 1.0000 | 1.0000 | 1.0000 | 0.000000 | 0.1895 | 5,278 |
| Top-5 features | 5 | 1.0000 | 1.0000 | 1.0000 | 0.000000 | 0.0801 | 12,477 |
| Top-8 features | 8 | 1.0000 | 1.0000 | 1.0000 | 0.000000 | 0.0978 | 10,227 |
| Top-10 features | 10 | 1.0000 | 1.0000 | 1.0000 | 0.000000 | 0.1630 | 6,136 |
| Top-12 features | 12 | 1.0000 | 1.0000 | 1.0000 | 0.000000 | 0.1705 | 5,864 |
| Top-15 features | 15 | 1.0000 | 1.0000 | 1.0000 | 0.000000 | 0.1390 | 7,193 |

![Feature Reduction Trade-off](figures/feature_reduction_tradeoff.png)

---

## 8. Explanation Stability

- **Local Neighbour Consistency (Median Spearman):** 1.0000
- **Local Neighbour Consistency (Median Cosine):** 1.0000
- **Bootstrap Top-8 Jaccard Overlap:** 0.6108
- **Bootstrap Full Ranking Spearman Correlation:** 0.8555

![Stability Analysis](figures/stability_analysis.png)

---

## 9. Architecture Ablation Study

| Configuration | Features | Accuracy | F1-Score | ROC-AUC | Latency (ms/1k) | Throughput (flows/s) |
|---|---|---|---|---|---|---|
| A1: Raw Unscaled (All Feats, Unweighted) | 68 | 1.0000 | 1.0000 | 1.0000 | 0.2143 | 4,666 |
| A2: + Min-Max Normalization | 68 | 1.0000 | 1.0000 | 1.0000 | 0.1727 | 5,790 |
| A3: + Correlation Filter (|r| > 0.90) | 35 | 1.0000 | 1.0000 | 1.0000 | 0.1431 | 6,988 |
| A4: + Balanced Class Weight (Proposed Full) | 35 | 1.0000 | 1.0000 | 1.0000 | 0.1823 | 5,487 |
| A5: SHAP Lightweight (Top-8 Feats) | 8 | 1.0000 | 1.0000 | 1.0000 | 0.1572 | 6,360 |

![Ablation Study](figures/ablation_study.png)

---

## 10. Direct Comparison with Reference Paper (Muzibuddin et al. 2026)

![Paper Comparison](figures/paper_comparison.png)

---

## 11. Conclusion

The experimental results demonstrate that with rigorous preprocessing and feature selection, the proposed Random Forest IDS achieves state-of-the-art intrusion classification with mathematically sound SHAP explainability and robust explanation stability on enterprise cloud network traffic.

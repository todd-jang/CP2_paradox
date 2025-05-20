바닷가 조약돌이 수많은 파도와 조석의 바닷물에 서로 부딪히며 굴려 일정한 모양(납작하고 가로 세로비율이 수렴한다는)으로 이른다고 하는 인사이트를 준 논문에 
영향을 받아 찾아보다 얻은 article에 대해 어떤 시사점이 있을까 하여 간단히 탐구하게 되었다 


# CP2_paradox

In 2014 the paradoxical thesis(file.pdf) approved which was asserted in 2012.
Now I go through the paper with Machine Learning with Python....



Here’s how to interpret the feature importance results from your Random Forest model predicting roundness:

1. What the Numbers Mean
Each feature gets an importance score between 0 and 1.
The scores add up to 1.
Higher scores mean the feature is more influential in predicting roundness.
2. Example Output
Suppose your print(feature_importance) output is:

Code
V        0.35
n3       0.22
r        0.15
b/a      0.10
c/a      0.07
a        0.04
b        0.03
c        0.02
n_xy     0.01
n_xz     0.005
n_yz     0.005
V (Volume) is the most important feature (0.35), meaning changes in V have the biggest effect on predicting roundness.
n3 and r also have strong effects.
Features like n_xz and n_yz have very low importance, so the model finds them less useful for this prediction.
3. How to Use This
Focus on important features. If you want to simplify your model or collect fewer measurements, prioritize the high-importance features.
Low-importance features could often be dropped without hurting prediction accuracy (but always test this).
Physical meaning: If a physical variable (like V) is highly important, it suggests a strong real-world relationship with roundness.
4. Visual Interpretation
A bar plot makes it easy to see which features matter most at a glance.
The tallest bars = most predictive features.
5. Next Steps
You can try removing low-importance features and retraining to see if accuracy stays high (for simpler models).
Investigate why some features are more important—this could reveal new scientific or engineering insights.


Summary:

The ML model predicts "roundness" based on geometric and derived features.
Prediction quality is summarized by Mean Squared Error.
Volume (V), n3, and r are the most important predictive features.
To get detailed prediction results, print or save y_pred after running the regression script.

Miscellounius
설명:

Ridge와 RandomForest 두 모델을 교차검증과 하이퍼파라미터 튜닝(GridSearchCV)으로 성능을 비교합니다.
각 모델의 최적 하이퍼파라미터, 교차검증 점수, 테스트셋 MSE를 함께 출력합니다.
추가 모델이나 파라미터를 넣고 싶으시면 models 딕셔너리에 추가만 하시면 됩니다!

참고:
roc_auc_by_train_ratio.py결과 출력 roc_auc_by_train_ratio.png에서 
분류 임계값(threshold)이나 모델, feature selection 등은 목적에 맞게 조정 가능합니다.
다르게 보고 싶은 학습 비율이 있다면 train_sizes에 리스트로 추가하시면 됩니다!


roc_auc_rf_vs_dex.png로 저장된 roc_auc_rf_vs_dex.py실행시
Deep NN 기반 binary classifier적용 결과임

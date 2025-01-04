# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:24:17 2024

@author: yao79
"""

import pandas as pd
from pycaret.classification import *

# Step 1: 業務理解
# 預測鐵達尼號乘客是否生還（Survived），這是一個二元分類問題。

# Step 2: 數據理解
# 讀取數據集
train_data = pd.read_csv("C:\\Users\\User\\Downloads\\train.csv")

# Step 3: 數據準備
# 移除無用欄位
train_data = train_data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# 填充缺失值
train_data["Age"].fillna(train_data["Age"].median(), inplace=True)
train_data["Embarked"].fillna(train_data["Embarked"].mode()[0], inplace=True)

# Step 4: 建模
# 初始化 PyCaret
clf_setup = setup(
    data=train_data,
    target="Survived",
    session_id=123,
    normalize=True,
    categorical_features=["Pclass", "Sex", "Embarked"],  # 類別型變量
)

# 比較模型並選擇最佳模型
best_model = compare_models()

# 微調最佳模型
tuned_model = tune_model(best_model)

# 繪製性能圖表
plot_model(tuned_model, plot="confusion_matrix")
plot_model(tuned_model, plot="feature")

# Step 5: 評估
# 處理測試數據
test_data = pd.read_csv("C:\\Users\\User\\Downloads\\test.csv")
test_data_original = test_data.copy()  # 保存 PassengerId 等欄位
test_data = test_data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
test_data["Age"].fillna(test_data["Age"].median(), inplace=True)
test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)
test_data["Embarked"].fillna(test_data["Embarked"].mode()[0], inplace=True)

# 添加假的目標欄位以避免錯誤
test_data["Survived"] = 0  # 臨時添加佔位符

# 預測
predictions = predict_model(tuned_model, data=test_data)

# 保存結果
output = pd.DataFrame({
    "PassengerId": test_data_original["PassengerId"],  # 恢復測試數據中的 PassengerId
    "Survived": predictions["prediction_label"]  # 使用模型的預測結果
})
output.to_csv("submission.csv", index=False)


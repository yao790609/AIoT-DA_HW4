# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:55:01 2024

@author: User
"""
import pandas as pd
from pycaret.classification import *
from sklearn.model_selection import train_test_split
import optuna

# Step 1: 業務理解
# 預測鐵達尼號乘客是否生還（Survived），這是一個二元分類問題。

# Step 2: 數據理解與清理
# 讀取數據集
data = pd.read_csv("C:\\Users\\User\\Downloads\\train.csv")

# 移除無用欄位
data = data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# 填充缺失值
data["Age"].fillna(data["Age"].median(), inplace=True)
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# Step 3: 特徵工程 (Feature Engineering)
# 創建新特徵
data["FamilySize"] = data["SibSp"] + data["Parch"]
data["IsAlone"] = (data["FamilySize"] == 0).astype(int)

# 把 "Sex" 編碼為數字類型
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

# 對票價進行分箱處理
data["FareBin"] = pd.qcut(data["Fare"], 4, labels=False)

# 移除用不到的欄位
data = data.drop(columns=["Fare", "SibSp", "Parch"])

# 確認清理結果
print(data.head())

# Step 4: 數據分割 (Train-Test Split)
X = data.drop(columns=["Survived"])
y = data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: PyCaret建模 (Model Selection with PyCaret)
# 將訓練數據組合為一個DataFrame
train_data = pd.concat([X_train, y_train], axis=1)

clf_setup = setup(
    data=train_data,
    target="Survived",
    session_id=123,
    normalize=True,
    categorical_features=["Pclass", "Embarked", "FareBin"],  # 類別型變量
    ignore_features=["IsAlone"],  # 假設我們不想使用該特徵
)

# 比較模型並選擇最佳模型
best_model = compare_models()

# 微調最佳模型
tuned_model = tune_model(best_model)

# 繪製性能圖表
plot_model(tuned_model, plot="confusion_matrix")
plot_model(tuned_model, plot="feature")

# Step 6: 使用 Optuna 進行超參數優化
def objective(trial):
    # 創建基礎模型
    model = create_model('lightgbm')  # 可替換為其他模型代碼，如 'rf', 'xgboost'

    # 定義參數網格，注意這裡每個參數的值應該是可迭代的（如列表）
    custom_grid = {
        "learning_rate": [trial.suggest_loguniform("learning_rate", 0.01, 0.3)],
        "max_depth": [trial.suggest_int("max_depth", 3, 10)],
        "n_estimators": [trial.suggest_int("n_estimators", 50, 300)],
    }

    # 調整模型
    tuned_model = tune_model(model, custom_grid=custom_grid)

    # 獲取模型表現的評分
    score = pull()["Accuracy"]  # 這裡選用 "Accuracy" 作為指標，你也可以改為其他指標

    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# 打印最佳參數
print("Best Trial:")
print(study.best_trial.params)

# 重新訓練最佳參數的模型
final_model = finalize_model(tuned_model)

# Step 7: 測試與保存結果
# 預處理測試數據
test_data = pd.read_csv("C:\\Users\\User\\Downloads\\test.csv")
test_data_original = test_data.copy()  # 保存 PassengerId 等欄位
test_data = test_data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
test_data["Age"].fillna(test_data["Age"].median(), inplace=True)
test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)
test_data["Embarked"].fillna(test_data["Embarked"].mode()[0], inplace=True)
test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"]
test_data["IsAlone"] = (test_data["FamilySize"] == 0).astype(int)
test_data["Sex"] = test_data["Sex"].map({"male": 0, "female": 1})
test_data["FareBin"] = pd.qcut(test_data["Fare"], 4, labels=False)
test_data = test_data.drop(columns=["Fare", "SibSp", "Parch"])

# 添加假的目標欄位以避免錯誤
test_data["Survived"] = 0

# 預測
predictions = predict_model(final_model, data=test_data)

# 保存結果
output = pd.DataFrame({
    "PassengerId": test_data_original["PassengerId"],
    "Survived": predictions["prediction_label"]
})
output.to_csv("submission.csv", index=False)

print("結果保存到 submission.csv")

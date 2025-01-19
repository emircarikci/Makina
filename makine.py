# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

# Veri setini yükleme
df = pd.read_csv("heart.csv")

# Eksik verileri kontrol etme
print("Eksik Veriler:")
print(df.isnull().sum())

# Veri setindeki temel istatistiklere göz atalım
print("\nTemel İstatistikler:")
print(df.describe())

# Kategorik veriler için işlem yapalım
df = pd.get_dummies(df, drop_first=True)

# Özellikler ve hedef değişkeni ayıralım
X = df[['age', 'chol', 'sex']]  
y = df['output']  

# Özellikleri ölçekleyelim
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test verilerini ayıralım
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model tanımlamaları
lr_model = LogisticRegression()
svm_model = SVC()
rf_model = RandomForestClassifier(n_estimators=100)
dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()

# Hiperparametre optimizasyonu
param_grid_lr = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']}
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15]}
param_grid_dt = {'max_depth': [5, 10, 15, 20]}
param_grid_knn = {'n_neighbors': [3, 5, 7]}

# Grid Search ile model optimizasyonu
grid_search_lr = GridSearchCV(lr_model, param_grid_lr, cv=5, n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
best_lr_model = grid_search_lr.best_estimator_

grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=5, n_jobs=-1)
grid_search_svm.fit(X_train, y_train)
best_svm_model = grid_search_svm.best_estimator_

grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_

grid_search_dt = GridSearchCV(dt_model, param_grid_dt, cv=5, n_jobs=-1)
grid_search_dt.fit(X_train, y_train)
best_dt_model = grid_search_dt.best_estimator_

grid_search_knn = GridSearchCV(knn_model, param_grid_knn, cv=5, n_jobs=-1)
grid_search_knn.fit(X_train, y_train)
best_knn_model = grid_search_knn.best_estimator_

# Model tahminlerini yapalım
y_pred_lr = best_lr_model.predict(X_test)
y_pred_svm = best_svm_model.predict(X_test)
y_pred_rf = best_rf_model.predict(X_test)
y_pred_dt = best_dt_model.predict(X_test)
y_pred_knn = best_knn_model.predict(X_test)

# Model değerlendirmelerini yazdıralım
print("Lojistik Regresyon Başarı Oranı:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("Lojistik Regresyon için ROC AUC:", roc_auc_score(y_test, y_pred_lr))

print("\nDestek Vektör Makinesi (SVM) Başarı Oranı:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
print("SVM için ROC AUC:", roc_auc_score(y_test, y_pred_svm))

print("\nRandom Forest Başarı Oranı:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Random Forest için ROC AUC:", roc_auc_score(y_test, y_pred_rf))

print("\nKarar Ağacı Başarı Oranı:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
print("Karar Ağacı için ROC AUC:", roc_auc_score(y_test, y_pred_dt))

print("\nK-Nearest Neighbors Başarı Oranı:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
print("KNN için ROC AUC:", roc_auc_score(y_test, y_pred_knn))

# F1 skoru değerlendirmesi
print("\nLojistik Regresyon F1 Skoru:", f1_score(y_test, y_pred_lr))
print("SVM F1 Skoru:", f1_score(y_test, y_pred_svm))
print("Random Forest F1 Skoru:", f1_score(y_test, y_pred_rf))
print("Karar Ağacı F1 Skoru:", f1_score(y_test, y_pred_dt))
print("KNN F1 Skoru:", f1_score(y_test, y_pred_knn))

# Confusion Matrix'leri görselleştirelim
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Lojistik Regresyon Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
axes[0, 0].set_title('Lojistik Regresyon Confusion Matrix')

# SVM Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt="d", cmap="Blues", ax=axes[0, 1])
axes[0, 1].set_title('SVM Confusion Matrix')

# Random Forest Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues", ax=axes[0, 2])
axes[0, 2].set_title('Random Forest Confusion Matrix')

# Karar Ağacı Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt="d", cmap="Blues", ax=axes[1, 0])
axes[1, 0].set_title('Karar Ağacı Confusion Matrix')

# KNN Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt="d", cmap="Blues", ax=axes[1, 1])
axes[1, 1].set_title('K-Nearest Neighbors Confusion Matrix')

axes[1, 2].axis('off')  # Boş olan sağ alt subplotu kaldırıyoruz

# Performans karşılaştırma grafiği
metrics = pd.DataFrame({
    'Model': ['Lojistik Regresyon', 'SVM', 'Random Forest', 'Karar Ağacı', 'KNN'],
    'Başarı Oranı (Accuracy)': [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_svm), accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_knn)],
    'F1 Skoru': [f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_svm), f1_score(y_test, y_pred_rf), f1_score(y_test, y_pred_dt), f1_score(y_test, y_pred_knn)],
    'ROC AUC': [roc_auc_score(y_test, y_pred_lr), roc_auc_score(y_test, y_pred_svm), roc_auc_score(y_test, y_pred_rf), roc_auc_score(y_test, y_pred_dt), roc_auc_score(y_test, y_pred_knn)],
    'Precision': [precision_score(y_test, y_pred_lr), precision_score(y_test, y_pred_svm), precision_score(y_test, y_pred_rf), precision_score(y_test, y_pred_dt), precision_score(y_test, y_pred_knn)],
    'Recall': [recall_score(y_test, y_pred_lr), recall_score(y_test, y_pred_svm), recall_score(y_test, y_pred_rf), recall_score(y_test, y_pred_dt), recall_score(y_test, y_pred_knn)]
})

print("\nModel Performansı Karşılaştırması:")
print(metrics)

metrics.set_index('Model').plot(kind='bar', figsize=(12, 6))
plt.title('Modellerin Performans Karşılaştırması')
plt.ylabel('Skor')
plt.xticks(rotation=0)
plt.show()


# Korelasyon matrisi
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Korelasyon Matrisi")
plt.show()

# Öğrenme Eğrisi
train_sizes, train_scores, test_scores = learning_curve(best_rf_model, X_scaled, y, cv=5, n_jobs=-1)

# Öğrenme eğrisini görselleştiyoruz
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Eğitim Skoru", color="blue")
plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Test Skoru", color="red")
plt.title("Öğrenme Eğrisi")
plt.xlabel("Eğitim Verisi Boyutu")
plt.ylabel("Skor")
plt.legend()
plt.grid(True)
plt.show()

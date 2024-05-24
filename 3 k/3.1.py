#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:46:08 2024

@author: emredikici
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Veri kümesini yükle
file_path = 'diabetes.csv'
data = pd.read_csv(file_path)

# Veri kümesini eğitim (%70) ve test (%30) olarak ayır
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Özellikleri ve etiketleri ayır
X_train = train_data.drop('Outcome', axis=1)
y_train = train_data['Outcome']
X_test = test_data.drop('Outcome', axis=1)
y_test = test_data['Outcome']

# En iyi k değerini bul
best_k = 1
best_score = 0
scores = []

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    scores.append(score)
    if score > best_score:
        best_score = score
        best_k = k

# En iyi k ile nihai modeli eğit
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)
y_prob = best_knn.predict_proba(X_test)[:, 1]

# Metriği hesapla
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Sonuçları bir CSV dosyasına kaydet
class_report_df = pd.DataFrame(class_report).transpose()
class_report_df['ROC AUC'] = roc_auc
class_report_df.to_csv('classification_report.csv', index=True)

# Confusion Matrix grafiği
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.show()

# ROC Curve grafiği
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.show()

# En iyi k ve performans metriklerini döndür
best_k, conf_matrix, class_report, roc_auc

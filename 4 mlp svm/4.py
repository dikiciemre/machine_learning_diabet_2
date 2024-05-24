#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:49:06 2024

@author: emredikici
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
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

# Multi-Layer Perceptron (MLP) sınıflandırıcısını eğit
mlp = MLPClassifier(random_state=42, max_iter=300)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
y_prob_mlp = mlp.predict_proba(X_test)[:, 1]

# Support Vector Machines (SVM) sınıflandırıcısını eğit
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
y_prob_svm = svm.predict_proba(X_test)[:, 1]

# MLP metrikleri hesapla
conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)
class_report_mlp = classification_report(y_test, y_pred_mlp, output_dict=True)
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_prob_mlp)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

# SVM metrikleri hesapla
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
class_report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Sonuçları bir CSV dosyasına kaydet
results_mlp = {
    'Confusion Matrix': [conf_matrix_mlp],
    'Classification Report': [class_report_mlp],
    'ROC AUC': [roc_auc_mlp]
}

results_svm = {
    'Confusion Matrix': [conf_matrix_svm],
    'Classification Report': [class_report_svm],
    'ROC AUC': [roc_auc_svm]
}

results_mlp_df = pd.DataFrame(results_mlp)
results_svm_df = pd.DataFrame(results_svm)

results_mlp_df.to_csv('classification_report_mlp.csv', index=False)
results_svm_df.to_csv('classification_report_svm.csv', index=False)

# Confusion Matrix grafikleri
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_mlp, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title('Confusion Matrix - MLP')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_svm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()

# ROC Curve grafikleri
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr_mlp, tpr_mlp, label=f'ROC Curve (area = {roc_auc_mlp:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - MLP')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(fpr_svm, tpr_svm, label=f'ROC Curve (area = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM')
plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig('roc_curves.png')
plt.show()

# Sonuçlar ve grafikleri döndür
conf_matrix_mlp, class_report_mlp, roc_auc_mlp, conf_matrix_svm, class_report_svm, roc_auc_svm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Örnek bir veri seti yükleyelim. Veri setinizin adını ve yolunu buraya ekleyin.
df = pd.read_csv('diabetes.csv')

# Özellikleri ve hedef değişkeni belirleyin
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Veriyi eğitim ve test setlerine bölelim (%70 eğitim, %30 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Gaussian Naive Bayes modelini oluştur ve eğit
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Test veri seti üzerinde tahmin yap
y_pred = nb_model.predict(X_test)

# Konfüzyon matrisi
cm = confusion_matrix(y_test, y_pred)

# Performans metrikleri
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')  # İkili sınıflandırma için 'binary'
recall = recall_score(y_test, y_pred, average='binary')        # İkili sınıflandırma için 'binary'
f1 = f1_score(y_test, y_pred, average='binary')                # İkili sınıflandırma için 'binary'
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
roc_auc = roc_auc_score(y_test, y_pred)

# Sonuçları bir DataFrame olarak düzenleyin
results = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1 Score', 'ROC AUC'],
    'Value': [accuracy, precision, recall, specificity, f1, roc_auc]
})

# Sonuçları bir CSV dosyasına kaydedin
results.to_csv('classification_results.csv', index=False)

# ROC eğrisi
fpr, tpr, thresholds = roc_curve(y_test, nb_model.predict_proba(X_test)[:, 1])

# ROC Eğrisi çizimi ve kaydedilmesi
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()

# Konfüzyon matrisini bir CSV dosyasına kaydedin
cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
cm_df.to_csv('confusion_matrix.csv')

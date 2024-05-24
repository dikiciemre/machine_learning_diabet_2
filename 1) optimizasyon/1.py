import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Veri setini yükleyin
data = pd.read_csv('diabetes.csv')

# Özellikler ve hedef değişkeni ayırın
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Kategorik ve sayısal sütunları belirleyin
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Ön işleme pipeline'ı oluşturun
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Model pipeline'ı oluşturun
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier())])

# Hiperparametre optimizasyonu için Grid Search
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X, y)

# En iyi modeli seçin
best_model = grid_search.best_estimator_

# Model performansını değerlendirin
cv_results = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
cv_mean_accuracy = np.mean(cv_results)
print(f'Cross-validated accuracy: {cv_mean_accuracy:.2f}')

# Tahminleri yapın
y_pred = best_model.predict(X)
classification_rep = classification_report(y, y_pred, output_dict=True)

# Sınıflandırma raporunu DataFrame'e dönüştürün
classification_df = pd.DataFrame(classification_rep).transpose()

# Çapraz doğrulama sonuçlarını DataFrame'e ekleyin
cv_results_df = pd.DataFrame(cv_results, columns=['Cross-Validation Accuracy'])
cv_results_df.loc['mean'] = cv_mean_accuracy

# Tahminleri ve gerçek değerleri birleştirin
results_df = pd.DataFrame({'Gerçek Değerler': y, 'Tahminler': y_pred})

# Tüm sonuçları bir araya getirin
combined_df = pd.concat([results_df, cv_results_df, classification_df], axis=1)

# Sonuçları bir CSV dosyasına kaydedin
combined_df.to_csv('tum_sonuclar.csv', index=False)

print("Tüm sonuçlar tum_sonuclar.csv dosyasına kaydedildi.")

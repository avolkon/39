# PCA и SVD: Лекция с практической реализацией

## 1. Введение в метод главных компонент (PCA)

### 1.1. Что такое PCA?

**Метод главных компонент (Principal Component Analysis, PCA)** — это статистическая процедура, использующая ортогональное преобразование для преобразования набора наблюдений коррелированных переменных в набор линейно некоррелированных переменных, называемых главными компонентами.

```
Исходное пространство → Ортогональное преобразование → Пространство главных компонент
         (m признаков)                                    (k ≤ m компонент)
```

### 1.2. Основная идея PCA

PCA находит направления максимальной дисперсии в данных:

1. **Первая главная компонента** — направление наибольшей дисперсии
2. **Вторая главная компонента** — направление наибольшей дисперсии, ортогональное первой
3. **Последующие компоненты** — аналогично, с сохранением ортогональности

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Генерация синтетических данных
np.random.seed(42)
X = np.random.multivariate_normal(
    mean=[0, 0],
    cov=[[3, 2.5], [2.5, 3]],
    size=200
)

# Визуализация данных и главных компонент
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, label='Данные')

# Вычисление PCA
pca = PCA()
pca.fit(X)

# Визуализация направлений главных компонент
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    plt.arrow(0, 0, comp[0] * np.sqrt(var)*2, comp[1] * np.sqrt(var)*2,
              head_width=0.2, head_length=0.2, 
              color=['red', 'blue'][i], label=f'PC{i+1}')

plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Главные компоненты данных')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

print("Доля объясненной дисперсии:", pca.explained_variance_ratio_)
```

## 2. Сингулярное разложение (SVD)

### 2.1. Определение SVD

**Сингулярное разложение (Singular Value Decomposition, SVD)** — это фундаментальное разложение матрицы на три матрицы:

```
A = U · Σ · Vᵀ
```

Где:
- **U** — левая матрица сингулярных векторов (m × m)
- **Σ** — диагональная матрица сингулярных чисел (m × n)
- **Vᵀ** — правая матрица сингулярных векторов (n × n)

### 2.2. Связь PCA и SVD

PCA тесно связан с SVD. Для центрированной матрицы X:

```
X = U · S · Vᵀ
```

Главные компоненты получаются как:
```
PC = X · V = U · S
```

А собственные значения ковариационной матрицы:
```
λᵢ = Sᵢ² / (n - 1)
```

```python
# Демонстрация связи PCA и SVD
from sklearn.preprocessing import StandardScaler

# Центрирование данных
X_centered = X - np.mean(X, axis=0)

# Выполнение SVD
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Получение главных компонент через SVD
PC_svd = U @ np.diag(S)

# Получение главных компонент через sklearn
pca = PCA()
PC_sklearn = pca.fit_transform(X)

# Сравнение результатов (с учетом знака)
print("Сравнение первых 5 значений PC1:")
print("SVD:", PC_svd[:5, 0])
print("sklearn:", PC_sklearn[:5, 0])
print("Разница:", np.abs(PC_svd[:5, 0] - PC_sklearn[:5, 0]))
```

## 3. Практическая реализация PCA

### 3.1. Загрузка и подготовка данных

```python
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка данных
data = pd.read_csv('PCA_example_dataset.csv')
print("Размер данных:", data.shape)
print("\nПервые 5 строк:")
print(data.head())

# Разделение на признаки и целевую переменную
y = data['label']
X = data.drop(columns=['label'])

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nОбучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")
```

### 3.2. Анализ корреляционной структуры

```python
# Построение матрицы корреляции до PCA
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(X_train.corr(), annot=True, linewidths=.5, fmt='.2f', ax=ax)
plt.title('Матрица корреляции до применения PCA')
plt.show()
```

**Наблюдения:**
- Часть признаков демонстрирует выраженную линейную зависимость
- Наличие мультиколлинеарности (значения корреляции близки к 1)
- Избыточность информации в данных

### 3.3. Масштабирование признаков

```python
# Стандартизация признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Среднее значение после масштабирования:", X_train_scaled.mean(axis=0))
print("Стандартное отклонение после масштабирования:", X_train_scaled.std(axis=0))
```

**Почему важно масштабирование?**
- PCA основан на анализе дисперсии
- Признаки с большим масштабом доминируют
- Без масштабирования результаты могут быть искажены

### 3.4. Выбор оптимального числа компонент

```python
# Обучение PCA со всеми компонентами
pca = PCA()
pca.fit(X_train_scaled)

# Вычисление накопленной объясненной дисперсии
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Визуализация
plt.figure(figsize=(12, 8))
plt.plot(range(1, len(cumulative_variance) + 1), 
         cumulative_variance, 
         marker='o', linestyle='--', 
         linewidth=2, markersize=8,
         label='Накопленная дисперсия')

# Добавление вспомогательных линий
plt.axhline(y=0.9, color='r', linestyle='--', 
            label='90% дисперсии', linewidth=2)
plt.axhline(y=0.95, color='g', linestyle='--', 
            label='95% дисперсии', linewidth=2)

# Определение точек пересечения
n_90 = np.where(cumulative_variance >= 0.9)[0][0] + 1
n_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1

plt.axvline(x=n_90, color='orange', linestyle='--', 
            label=f'90%: {n_90} компонент', linewidth=2)
plt.axvline(x=n_95, color='purple', linestyle='--', 
            label=f'95%: {n_95} компонент', linewidth=2)

plt.xlabel('Число главных компонент', fontsize=12)
plt.ylabel('Накопленная доля объясненной дисперсии', fontsize=12)
plt.title('График объясненной дисперсии PCA', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Для сохранения 90% дисперсии необходимо {n_90} компонент")
print(f"Для сохранения 95% дисперсии необходимо {n_95} компонент")
```

### 3.5. Методы выбора количества компонент

#### Метод 1: Задание порога дисперсии

```python
# Автоматический выбор компонент для сохранения 90% дисперсии
pca_90 = PCA(n_components=0.9)
X_train_pca_90 = pca_90.fit_transform(X_train_scaled)
X_test_pca_90 = pca_90.transform(X_test_scaled)

print(f"Количество компонент для 90% дисперсии: {pca_90.n_components_}")
print(f"Объясненная дисперсия: {pca_90.explained_variance_ratio_.sum():.3f}")
```

#### Метод 2: Явное задание числа компонент

```python
# Фиксированное количество компонент (например, 3 для визуализации)
pca_3 = PCA(n_components=3)
X_train_pca_3 = pca_3.fit_transform(X_train_scaled)
X_test_pca_3 = pca_3.transform(X_test_scaled)

print(f"Объясненная дисперсия 3 компонент: {pca_3.explained_variance_ratio_.sum():.3f}")
print(f"Форма преобразованных данных: {X_train_pca_3.shape}")
```

#### Метод 3: Метод "локтя"

```python
# Поиск точки перегиба
differences = np.diff(cumulative_variance)
elbow_point = np.argmax(differences) + 1

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
         marker='o', markersize=6)
plt.axvline(x=elbow_point, color='r', linestyle='--', 
            label=f'Точка перегиба: {elbow_point}')
plt.xlabel('Число компонент')
plt.ylabel('Накопленная дисперсия')
plt.title('Метод локтя')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(differences) + 1), differences, 
         marker='o', markersize=6, color='green')
plt.axvline(x=elbow_point, color='r', linestyle='--')
plt.xlabel('Число компонент')
plt.ylabel('Прирост дисперсии')
plt.title('Прирост объясненной дисперсии')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Точка перегиба (Elbow) на {elbow_point} компонентах")
```

### 3.6. Реализация PCA через SVD вручную

```python
# Центрирование данных
X_centered = X_train_scaled - np.mean(X_train_scaled, axis=0)

# Выполнение SVD
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Выбор первых k компонент
k = n_90  # Используем количество компонент для 90% дисперсии
V_k = Vt[:k].T

# Преобразование данных
X_pca_manual = X_centered @ V_k

print(f"Форма данных после ручного SVD: {X_pca_manual.shape}")
print(f"Форма данных после sklearn PCA: {X_train_pca_90.shape}")

# Сравнение результатов
print("\nСравнение первых 5 значений PC1:")
print("Ручной SVD:", X_pca_manual[:5, 0])
print("Sklearn PCA:", X_train_pca_90[:5, 0])
```

### 3.7. Анализ после применения PCA

```python
# Матрица корреляции после PCA
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(pd.DataFrame(X_train_pca_90).corr(), 
            annot=True, linewidths=.5, fmt='.2f', ax=ax)
plt.title('Матрица корреляции после применения PCA')
plt.show()
```

**Результат:** Компоненты являются некоррелированными (корреляции близки к 0)

### 3.8. Интерпретация главных компонент

```python
# Анализ вклада признаков в компоненты
components_df = pd.DataFrame(
    pca_90.components_,
    columns=X.columns,
    index=[f'PC{i+1}' for i in range(pca_90.n_components_)]
)

# Визуализация вклада признаков в первые 3 компоненты
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, ax in enumerate(axes):
    comp_weights = components_df.iloc[i]
    comp_weights.sort_values().plot(kind='barh', ax=ax)
    ax.set_title(f'Вклад признаков в {components_df.index[i]}')
    ax.set_xlabel('Вес признака')

plt.tight_layout()
plt.show()

# Отображение наиболее важных признаков для каждой компоненты
print("Наиболее важные признаки для каждой компоненты:")
for i in range(3):
    comp = components_df.iloc[i]
    top_features = comp.abs().sort_values(ascending=False).head(5)
    print(f"\n{components_df.index[i]}:")
    for feature, weight in top_features.items():
        print(f"  {feature}: {weight:.3f}")
```

### 3.9. Визуализация данных в пространстве главных компонент

```python
# 2D визуализация
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train_pca_90[:, 0], X_train_pca_90[:, 1], 
            c=y_train, cmap='viridis', alpha=0.6)
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.title('Данные в пространстве двух главных компонент')
plt.colorbar(label='Целевая переменная')
plt.grid(True, alpha=0.3)

# 3D визуализация (если есть 3 компоненты)
if X_train_pca_90.shape[1] >= 3:
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(1, 2, 2, projection='3d')
    scatter = ax.scatter(X_train_pca_90[:, 0], 
                        X_train_pca_90[:, 1], 
                        X_train_pca_90[:, 2],
                        c=y_train, cmap='viridis', alpha=0.6)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Данные в пространстве трех главных компонент')
    plt.colorbar(scatter, label='Целевая переменная')

plt.tight_layout()
plt.show()
```

## 4. Сравнение качества моделей

### 4.1. Модель на исходных данных

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Обучение на исходных данных
clf_original = DecisionTreeClassifier(random_state=42)
clf_original.fit(X_train_scaled, y_train)

# Предсказание
y_pred_original = clf_original.predict(X_test_scaled)

# Оценка качества
accuracy_orig = accuracy_score(y_test, y_pred_original)
precision_orig = precision_score(y_test, y_pred_original, average='weighted')
recall_orig = recall_score(y_test, y_pred_original, average='weighted')

print("Результаты на исходных данных:")
print(f"Accuracy: {accuracy_orig:.4f}")
print(f"Precision: {precision_orig:.4f}")
print(f"Recall: {recall_orig:.4f}")

# Матрица ошибок
cm_orig = confusion_matrix(y_test, y_pred_original)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues')
plt.title('Матрица ошибок на исходных данных')
plt.xlabel('Предсказанные классы')
plt.ylabel('Истинные классы')
plt.show()
```

### 4.2. Модель на данных после PCA

```python
# Обучение на данных после PCA
clf_pca = DecisionTreeClassifier(random_state=42)
clf_pca.fit(X_train_pca_90, y_train)

# Предсказание
y_pred_pca = clf_pca.predict(X_test_pca_90)

# Оценка качества
accuracy_pca = accuracy_score(y_test, y_pred_pca)
precision_pca = precision_score(y_test, y_pred_pca, average='weighted')
recall_pca = recall_score(y_test, y_pred_pca, average='weighted')

print("Результаты на данных после PCA:")
print(f"Accuracy: {accuracy_pca:.4f}")
print(f"Precision: {precision_pca:.4f}")
print(f"Recall: {recall_pca:.4f}")

# Матрица ошибок
cm_pca = confusion_matrix(y_test, y_pred_pca)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_pca, annot=True, fmt='d', cmap='Blues')
plt.title('Матрица ошибок на данных после PCA')
plt.xlabel('Предсказанные классы')
plt.ylabel('Истинные классы')
plt.show()
```

### 4.3. Сравнительный анализ

```python
# Сравнение метрик
metrics = ['Accuracy', 'Precision', 'Recall']
original_scores = [accuracy_orig, precision_orig, recall_orig]
pca_scores = [accuracy_pca, precision_pca, recall_pca]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, original_scores, width, label='Исходные данные', color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, pca_scores, width, label='После PCA', color='green', alpha=0.7)

ax.set_ylabel('Значение метрики')
ax.set_title('Сравнение качества моделей')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3)

# Добавление значений на столбцы
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

## 5. Преимущества и ограничения PCA

### 5.1. Преимущества

```python
# Демонстрация сжатия данных
original_memory = X_train_scaled.nbytes / (1024 * 1024)
compressed_memory = X_train_pca_90.nbytes / (1024 * 1024)

print(f"Размер исходных данных: {original_memory:.2f} MB")
print(f"Размер данных после PCA: {compressed_memory:.2f} MB")
print(f"Сжатие в {original_memory/compressed_memory:.2f} раз")
```

**Ключевые преимущества:**
1. **Неконтролируемый метод** — можно применять к неразмеченным данным
2. **Минимизация потерь информации** при уменьшении размерности
3. **Снижение вычислительной сложности** за счет устранения избыточности
4. **Визуализация многомерных данных** в 2D/3D пространстве
5. **Подавление шума** за счет фокусировки на наиболее важных компонентах

### 5.2. Ограничения

```python
# Демонстрация чувствительности к масштабу
X_scaled_correct = StandardScaler().fit_transform(X_train)
X_scaled_incorrect = X_train.copy()

pca_correct = PCA().fit(X_scaled_correct)
pca_incorrect = PCA().fit(X_scaled_incorrect)

print("Влияние масштабирования на объясненную дисперсию:")
print("С правильным масштабированием:", pca_correct.explained_variance_ratio_[:3])
print("Без масштабирования:", pca_incorrect.explained_variance_ratio_[:3])
```

**Ключевые ограничения:**
1. **Только линейные связи** — не подходит для нелинейных данных
2. **Чувствительность к масштабу** — требует стандартизации
3. **Потеря интерпретируемости** — новые признаки сложно интерпретировать
4. **Потеря информации** — при сокращении размерности теряется часть данных
5. **Чувствительность к выбросам** — выбросы могут исказить главные компоненты

## 6. Расширенные применения PCA

### 6.1. Восстановление данных

```python
# Восстановление данных из PCA-пространства
X_reconstructed = pca_90.inverse_transform(X_train_pca_90)

# Оценка ошибки восстановления
reconstruction_error = np.mean((X_train_scaled - X_reconstructed) ** 2)
print(f"Среднеквадратичная ошибка восстановления: {reconstruction_error:.6f}")

# Визуализация восстановления
plt.figure(figsize=(12, 4))
sample_idx = 0

plt.subplot(1, 3, 1)
plt.bar(range(len(X_train_scaled[sample_idx])), X_train_scaled[sample_idx])
plt.title('Оригинальный образец')
plt.xlabel('Признак')
plt.ylabel('Значение')

plt.subplot(1, 3, 2)
plt.bar(range(len(X_reconstructed[sample_idx])), X_reconstructed[sample_idx])
plt.title('Восстановленный образец')
plt.xlabel('Признак')

plt.subplot(1, 3, 3)
plt.bar(range(len(X_train_scaled[sample_idx])), 
        X_train_scaled[sample_idx] - X_reconstructed[sample_idx])
plt.title('Ошибка восстановления')
plt.xlabel('Признак')

plt.tight_layout()
plt.show()
```

### 6.2. Анализ вклада компонент

```python
# Анализ накопленной дисперсии
plt.figure(figsize=(10, 6))

# Индивидуальная дисперсия
plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca_90.explained_variance_ratio_) + 1), 
        pca_90.explained_variance_ratio_)
plt.xlabel('Главная компонента')
plt.ylabel('Доля объясненной дисперсии')
plt.title('Индивидуальная дисперсия компонент')
plt.grid(True, alpha=0.3)

# Накопленная дисперсия
plt.subplot(1, 2, 2)
plt.plot(range(1, len(pca_90.explained_variance_ratio_) + 1), 
         np.cumsum(pca_90.explained_variance_ratio_),
         marker='o')
plt.xlabel('Главная компонента')
plt.ylabel('Накопленная доля дисперсии')
plt.title('Накопленная дисперсия компонент')
plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 7. Заключение

### 7.1. Рекомендации по применению PCA

```python
# Создание функции для автоматического применения PCA
def apply_pca_with_analysis(X_train, X_test, variance_threshold=0.95):
    """
    Автоматическое применение PCA с анализом результатов
    """
    # Стандартизация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Применение PCA
    pca = PCA(n_components=variance_threshold)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Вывод результатов
    print(f"Исходная размерность: {X_train.shape[1]}")
    print(f"Размерность после PCA: {X_train_pca.shape[1]}")
    print(f"Объясненная дисперсия: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Сжатие данных: {X_train.shape[1]/X_train_pca.shape[1]:.2f}x")
    
    return X_train_pca, X_test_pca, pca, scaler

# Пример использования
X_train_pca_auto, X_test_pca_auto, pca_auto, scaler_auto = apply_pca_with_analysis(
    X_train, X_test, variance_threshold=0.95
)
```

### 7.2. Когда использовать PCA

| Сценарий | Рекомендация |
|----------|-------------|
| Визуализация данных | ✅ Рекомендуется |
| Сжатие данных | ✅ Рекомендуется |
| Устранение мультиколлинеарности | ✅ Рекомендуется |
| Шумоподавление | ✅ Рекомендуется |
| Нелинейные данные | ❌ Не рекомендуется |
| Интерпретируемость модели | ❌ Не рекомендуется |
| Малые выборки | ⚠️ С осторожностью |

### 7.3. Альтернативы PCA

```python
# Демонстрация альтернатив
from sklearn.manifold import TSNE
from sklearn.decomposition import FactorAnalysis, NMF

# t-SNE (нелинейное снижение размерности)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_train_scaled[:500])  # t-SNE медленный на больших данных

# Факторный анализ
fa = FactorAnalysis(n_components=n_90, random_state=42)
X_fa = fa.fit_transform(X_train_scaled)

# Неотрицательное матричное разложение
nmf = NMF(n_components=n_90, random_state=42)
X_nmf = nmf.fit_transform(np.maximum(X_train_scaled, 0))  # NMF требует неотрицательных данных

print("Сравнение методов снижения размерности:")
print(f"PCA: {X_train_pca_90.shape}")
print(f"Factor Analysis: {X_fa.shape}")
print(f"NMF: {X_nmf.shape}")
print(f"t-SNE: {X_tsne.shape}")
```

## 8. Резюме

**Метод главных компонент (PCA)** является мощным инструментом для:
- Снижения размерности данных
- Визуализации многомерных структур
- Подавления шума
- Сжатия информации

**Ключевые выводы:**
1. PCA основан на сингулярном разложении (SVD)
2. Требует стандартизации признаков
3. Выбор числа компонент основан на объясненной дисперсии
4. Улучшает работу моделей при наличии мультиколлинеарности
5. Является линейным методом и не подходит для нелинейных данных

**Практические рекомендации:**
- Всегда масштабируйте данные перед применением PCA
- Анализируйте график объясненной дисперсии для выбора компонент
- Проверяйте влияние PCA на качество модели
- Интерпретируйте главные компоненты через веса признаков

# 📘 Линейный дискриминантный анализ (LDA) — полный конспект

## 1. Что такое LDA?

### 🎯 Главная идея

**Линейный дискриминантный анализ (Linear Discriminant Analysis, LDA)** — это метод, который **находит такие оси (направления)**, чтобы при проецировании данных на них **классы максимально разделялись**.

Если PCA ищет направления **максимальной дисперсии** (разброса), то LDA ищет направления **максимального разделения классов**.

```
PCA: "Найду, где данные меняются сильнее всего"
LDA: "Найду, где классы отличаются друг от друга лучше всего"
```

### 🎨 Простая аналогия

Представь, что у тебя есть **два вида конфет** в коробке, и ты хочешь их разделить:

- **PCA** посмотрит, в каком направлении конфеты разбросаны больше всего (например, вдоль длины коробки)
- **LDA** посмотрит, как лучше **отличить** одни конфеты от других, и найдёт ось, на которой они не пересекаются

```
      БЕЗ LDA              С LDA
    (плохое разделение)   (хорошее разделение)
    
    x x o o               x x | o o
    x x o o               x x | o o
    x x o o               x x | o o
    x x o o               x x | o o
    
    Классы смешаны       Классы разделены
```

---

## 2. Как работает LDA?

### 📐 Математическая идея

LDA пытается найти проекцию, которая:

1. **Максимизирует расстояние между средними разных классов** (межклассовая дисперсия)
2. **Минимизирует разброс внутри каждого класса** (внутриклассовая дисперсия)

Это как если бы ты хотел:
- Чтобы **разные классы** были как можно **дальше** друг от друга
- Чтобы **один класс** был как можно **компактнее**

### 📊 Формула отношения

```
              Межклассовая дисперсия
    J = ──────────────────────────────
              Внутриклассовая дисперсия
```

Чем больше `J`, тем лучше разделение!

### 🔢 Шаги алгоритма

```
1. Вычислить средние значения для каждого класса
2. Вычислить внутриклассовую матрицу разброса (S_W)
3. Вычислить межклассовую матрицу разброса (S_B)
4. Найти собственные векторы матрицы (S_W⁻¹ · S_B)
5. Взять K-1 самых больших собственных векторов
6. Спроецировать данные на эти векторы
```

---

## 3. LDA vs PCA — в чём разница?

### 📋 Сравнительная таблица

| Характеристика | PCA | LDA |
|----------------|-----|-----|
| **Что ищет** | Максимальную дисперсию | Максимальное разделение классов |
| **Учитывает метки классов** | ❌ Нет (без учителя) | ✅ Да (с учителем) |
| **Максимум компонент** | m (все признаки) | C-1 (классы минус 1) |
| **Цель** | Сжатие, визуализация | Классификация, разделение |
| **Когда лучше** | Общий анализ данных | Разделение классов |
| **Интерпретация** | "Главные направления изменчивости" | "Направления, разделяющие классы" |

### 👁️ Визуальное сравнение

```
Данные: два класса (x и o)
        
        PCA                         LDA
    (направление дисперсии)     (направление разделения)
    
    ↑                              ↑
    |   x   o                     |  x x x
    |   x x o o                   |  x x x
    | x x x o o o                 |  o o o
    | x x x o o o                 |  o o o
    |   x x o o                   |  
    |   x   o                     |
    
    Главная ось —             Главная ось —
    вдоль разброса            разделяет классы!
```

### 💡 Ключевой вывод

> **PCA** говорит: "Вот где данные меняются сильнее всего"
> 
> **LDA** говорит: "Вот где классы различаются лучше всего"

---

## 4. Практическая реализация LDA

### 4.1. Загрузка и подготовка данных

```python
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка данных
data = pd.read_csv('PCA_example_dataset.csv')
print("Размер данных:", data.shape)

# Разделение на признаки и целевую переменную
y = data['label']
X = data.drop(columns=['label'])

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Обучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")
```

### 4.2. Применение LDA

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Создание модели LDA
# n_components=2 — для визуализации (максимум C-1, где C — число классов)
lda = LinearDiscriminantAnalysis(n_components=2)

# Обучение на обучающей выборке
X_train_lda = lda.fit_transform(X_train, y_train)

# Преобразование тестовой выборки
X_test_lda = lda.transform(X_test)

print(f"Размерность до LDA: {X_train.shape[1]}")
print(f"Размерность после LDA: {X_train_lda.shape[1]}")
print(f"Объясненная дисперсия: {lda.explained_variance_ratio_}")
```

### 4.3. Визуализация результатов LDA

```python
# Визуализация данных в пространстве LDA
plt.figure(figsize=(12, 5))

# 2D проекция
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], 
                      c=y_train, cmap='viridis', alpha=0.6)
plt.xlabel('Первая дискриминантная компонента')
plt.ylabel('Вторая дискриминантная компонента')
plt.title('Данные после LDA')
plt.colorbar(scatter, label='Классы')
plt.grid(True, alpha=0.3)

# Сравнение с PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                      c=y_train, cmap='viridis', alpha=0.6)
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.title('Данные после PCA (для сравнения)')
plt.colorbar(scatter, label='Классы')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 4.4. Обучение модели после LDA

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Обучение дерева решений на данных после LDA
tree_lda = DecisionTreeClassifier(random_state=42)
tree_lda.fit(X_train_lda, y_train)

# Предсказание
y_pred_lda = tree_lda.predict(X_test_lda)

# Оценка качества
accuracy_lda = accuracy_score(y_test, y_pred_lda)
precision_lda = precision_score(y_test, y_pred_lda, average='weighted')
recall_lda = recall_score(y_test, y_pred_lda, average='weighted')

print("Результаты после LDA:")
print(f"Accuracy: {accuracy_lda:.4f}")
print(f"Precision: {precision_lda:.4f}")
print(f"Recall: {recall_lda:.4f}")

# Матрица ошибок
cm_lda = confusion_matrix(y_test, y_pred_lda)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lda, annot=True, fmt='d', cmap='Blues')
plt.title('Матрица ошибок после LDA')
plt.xlabel('Предсказанные классы')
plt.ylabel('Истинные классы')
plt.show()
```

### 4.5. Сравнение всех методов

```python
# Сравнение трёх подходов
results = {
    'Исходные данные': {'accuracy': 0.467, 'precision': 0.469, 'recall': 0.467},
    'PCA': {'accuracy': 0.505, 'precision': 0.505, 'recall': 0.505},
    'LDA': {'accuracy': 0.528, 'precision': 0.529, 'recall': 0.528}
}

metrics = ['accuracy', 'precision', 'recall']
methods = list(results.keys())

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(metrics))
width = 0.25

for i, method in enumerate(methods):
    scores = [results[method][m] for m in metrics]
    bars = ax.bar(x + i*width, scores, width, label=method)
    
    # Добавление значений на столбцы
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')

ax.set_ylabel('Значение метрики')
ax.set_title('Сравнение качества моделей')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("📊 Итоговое сравнение:")
for method in methods:
    print(f"\n{method}:")
    print(f"  Accuracy: {results[method]['accuracy']:.3f}")
    print(f"  Precision: {results[method]['precision']:.3f}")
    print(f"  Recall: {results[method]['recall']:.3f}")
```

### 4.6. Параметры LDA в scikit-learn

```python
# Демонстрация различных параметров LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 1. Базовый LDA
lda_basic = LinearDiscriminantAnalysis(n_components=2)

# 2. С регуляризацией (shrinkage)
lda_shrinkage = LinearDiscriminantAnalysis(
    n_components=2,
    solver='lsqr',  # или 'eigen'
    shrinkage=0.5   # коэффициент регуляризации
)

# 3. С заданными априорными вероятностями
lda_priors = LinearDiscriminantAnalysis(
    n_components=2,
    priors=[0.3, 0.7]  # если 2 класса
)

# 4. С сохранением ковариационных матриц
lda_cov = LinearDiscriminantAnalysis(
    n_components=2,
    store_covariance=True
)

print("Доступные параметры LDA:")
print("- n_components: число компонент (макс: C-1)")
print("- solver: svd (быстрый), lsqr, eigen")
print("- shrinkage: регуляризация (0-1)")
print("- priors: априорные вероятности классов")
print("- store_covariance: сохранять ли ковариации")
print("- tol: порог сходимости")
```

### 4.7. Анализ важности признаков

```python
# Коэффициенты LDA показывают вклад каждого признака
feature_importance = pd.DataFrame(
    lda.coef_.T,
    index=X.columns,
    columns=[f'LD{i+1}' for i in range(lda.coef_.shape[0])]
)

print("Вклад признаков в дискриминантные компоненты:")
print(feature_importance)

# Визуализация важности признаков
plt.figure(figsize=(10, 6))
feature_importance.abs().sort_values('LD1', ascending=True).plot(kind='barh')
plt.title('Вклад признаков в первую дискриминантную компоненту')
plt.xlabel('Абсолютное значение коэффициента')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 5. Результаты эксперимента

### 📊 Сравнение качества классификации

| Метод | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| **Исходные данные** | 0.467 | 0.469 | 0.467 |
| **PCA** | 0.505 | 0.505 | 0.505 |
| **LDA** | **0.528** | **0.529** | **0.528** |

### 📈 Выводы из эксперимента

1. **LDA показал наилучший результат** (52.8% accuracy)
2. **PCA улучшил качество** по сравнению с исходными данными
3. **LDA лучше PCA**, потому что учитывает метки классов

### 🎯 Почему LDA работает лучше в этом примере?

```
Исходные данные: много коррелированных признаков
         ↓
      PCA: убирает корреляции, но не знает о классах
         ↓
      LDA: знает, какие объекты к какому классу относятся
         ↓
    Лучшее разделение → лучшая классификация
```

---

## 6. Преимущества и ограничения LDA

### ✅ Достоинства

```python
# 1. Интерпретируемость
print("Коэффициенты LDA можно интерпретировать:")
for i, coef in enumerate(lda.coef_[0]):
    print(f"  {X.columns[i]}: {coef:.3f}")

# 2. Вычислительная эффективность
import time
start = time.time()
lda.fit(X_train, y_train)
print(f"\nВремя обучения LDA: {time.time() - start:.4f} сек")
```

**Ключевые преимущества:**

1. **Строгое статистическое обоснование** — математически обоснован
2. **Интерпретируемость** — коэффициенты показывают вклад признаков
3. **Учёт информации о классах** — знает, что разделять
4. **Вычислительная эффективность** — быстро работает
5. **Устойчивость к переобучению** — при правильных предположениях

### ❌ Ограничения

```python
# Демонстрация ограничений LDA
from sklearn.datasets import make_circles, make_moons

# 1. Нелинейные данные
X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)

lda_moons = LinearDiscriminantAnalysis(n_components=1)
X_moons_lda = lda_moons.fit_transform(X_moons, y_moons)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis')
plt.title('Нелинейные данные (две луны)')

plt.subplot(1, 2, 2)
plt.scatter(X_moons_lda[:, 0], np.zeros_like(X_moons_lda), 
           c=y_moons, cmap='viridis', alpha=0.6)
plt.title('LDA на нелинейных данных (плохо)')
plt.xlabel('Проекция LDA')

plt.tight_layout()
plt.show()

print("⚠️ LDA плохо работает с нелинейными данными!")
```

**Ключевые ограничения:**

1. **Предположение о нормальности** — данные должны быть нормально распределены
2. **Равенство ковариаций** — у всех классов должна быть одинаковая ковариационная матрица
3. **Только линейные границы** — не работает с нелинейными данными
4. **Чувствительность к выбросам** — выбросы сильно влияют
5. **Ограничение на компоненты** — не больше C-1 компонент

---

## 7. Когда использовать LDA?

### 📋 Чек-лист

| Условие | Подходит? |
|---------|-----------|
| Данные нормально распределены | ✅ |
| Ковариации классов примерно равны | ✅ |
| Границы между классами линейны | ✅ |
| Мало выбросов | ✅ |
| Нужна интерпретация | ✅ |
| Классов немного | ✅ |
| Нелинейные данные | ❌ |
| Много выбросов | ❌ |
| Нужна нелинейная граница | ❌ |

### 🎯 Рекомендации по применению

```
1. Всегда проверяйте распределение данных
2. Если классы хорошо разделены → LDA сработает
3. Если границы нелинейны → попробуйте QDA или другие методы
4. Используйте LDA для:
   - Классификации с интерпретацией
   - Снижения размерности для визуализации
   - Предобработки перед другими алгоритмами
```

---

## 8. LDA на практике — пошаговый план

```python
def apply_lda_pipeline(X_train, X_test, y_train, y_test, n_components=None):
    """
    Полный пайплайн применения LDA
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import classification_report
    
    # Шаг 1: Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Шаг 2: Применение LDA
    if n_components is None:
        n_components = min(len(np.unique(y_train)) - 1, X_train.shape[1])
    
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)
    
    # Шаг 3: Обучение классификатора
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train_lda, y_train)
    
    # Шаг 4: Оценка
    y_pred = clf.predict(X_test_lda)
    
    print("=" * 50)
    print("ОТЧЁТ ПО LDA")
    print("=" * 50)
    print(f"Исходная размерность: {X_train.shape[1]}")
    print(f"Размерность после LDA: {X_train_lda.shape[1]}")
    print(f"Число классов: {len(np.unique(y_train))}")
    print("\nКлассификационный отчёт:")
    print(classification_report(y_test, y_pred))
    
    return lda, X_train_lda, X_test_lda, clf

# Применение
lda, X_train_lda, X_test_lda, clf = apply_lda_pipeline(
    X_train, X_test, y_train, y_test
)
```

---

## 9. LDA vs QDA

```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Сравнение LDA и QDA
lda_model = LinearDiscriminantAnalysis()
qda_model = QuadraticDiscriminantAnalysis()

lda_model.fit(X_train, y_train)
qda_model.fit(X_train, y_train)

print("LDA (линейные границы):")
print(f"  Train: {lda_model.score(X_train, y_train):.3f}")
print(f"  Test: {lda_model.score(X_test, y_test):.3f}")

print("\nQDA (квадратичные границы):")
print(f"  Train: {qda_model.score(X_train, y_train):.3f}")
print(f"  Test: {qda_model.score(X_test, y_test):.3f}")
```

**Когда что выбирать:**

| LDA | QDA |
|-----|-----|
| Ковариации классов равны | Ковариации классов разные |
| Мало данных | Много данных |
| Простая интерпретация | Сложная интерпретация |
| Меньше переобучения | Больше переобучения |

---

## 10. Итоговое резюме

### 🎯 LDA — это:

```
Метод, который:
1. Учитывает метки классов
2. Находит оси, максимально разделяющие классы
3. Максимизирует расстояние между классами
4. Минимизирует разброс внутри классов
5. Может использоваться для классификации и снижения размерности
```

### 📊 В сравнении с PCA:

```
PCA: "Где данные меняются сильнее всего?"
LDA: "Где классы отличаются лучше всего?"

PCA: не знает о классах → хуже для классификации
LDA: знает о классах → лучше для классификации
```

### ✅ Когда использовать:

- Нужно разделить классы
- Данные нормально распределены
- Ковариации примерно равны
- Границы линейны
- Нужна интерпретация

### ⚠️ Когда НЕ использовать:

- Нелинейные данные
- Разные ковариации
- Много выбросов
- Мало данных на класс
- Невыполнение статистических предположений

---

## 11. Полезные формулы (шпаргалка)

```
LDA ищет проекцию w, которая максимизирует:

        wᵀ · S_B · w
J = ─────────────────
        wᵀ · S_W · w

где:
S_B — межклассовая матрица разброса
S_W — внутриклассовая матрица разброса
w — вектор проекции
```

**Ключевые ограничения:**
```
Число компонент ≤ C - 1
где C — количество классов
```

---

*Конспект составлен на основе лекции по LDA с практическими примерами и сравнениями с PCA.*

# 📘 Анализ независимых компонент (ICA) — полный конспект

## 1. Что такое ICA?

### 🎯 Главная идея

**ICA (Independent Component Analysis)** — это метод, который **разделяет смешанные сигналы** на **исходные независимые источники**.

Представь, что у тебя есть запись, где **одновременно говорят несколько человек** (эффект "коктейльной вечеринки"). ICA может **разделить** эту запись на **отдельные голоса**!

### 🎨 Простая аналогия

Представь, что ты **смешал** в блендере яблочный, апельсиновый и виноградный соки. Получился **фруктовый коктейль**.

```
Смешанный сок = Яблочный + Апельсиновый + Виноградный
```

**Задача ICA** — по этому коктейлю **восстановить исходные соки**, которые были смешаны!

```
ICA: Коктейль → Яблочный сок, Апельсиновый сок, Виноградный сок
```

### 🧠 Главное отличие от PCA

| Метод | Что ищет | Аналогия |
|-------|----------|----------|
| **PCA** | Направления **максимальной дисперсии** | "Какие цвета в коктейле самые яркие?" |
| **ICA** | **Независимые источники** | "Из каких соков состоит коктейль?" |

**Ключевое отличие:** PCA смотрит на **разброс**, а ICA ищет **причины** (исходные сигналы).

---

## 2. Математическая модель ICA

### 📐 Основная формула

Предполагается, что наблюдаемые данные — это **смесь** независимых источников:

```
X = A · S
```

Где:
- **X** — то, что мы видим (смешанные данные)
- **A** — неизвестная матрица смешивания (как именно смешались сигналы)
- **S** — скрытые независимые источники (то, что мы ищем)

### 🎯 Задача ICA

Найти матрицу **W** (разделяющую матрицу), такую что:

```
S ≈ W · X
```

То есть **разделить** смесь обратно на независимые сигналы!

### 📊 Визуализация процесса

```
Наблюдаемые данные (X)    Матрица разделения (W)    Независимые источники (S)
     ↓                            ↓                         ↓
┌─────────────┐              ┌─────────┐              ┌─────────────┐
│ Смесь 1     │              │         │              │ Источник 1  │
│ Смесь 2     │  ──────────► │    W    │  ──────────► │ Источник 2  │
│ Смесь 3     │              │         │              │ Источник 3  │
└─────────────┘              └─────────┘              └─────────────┘
```

---

## 3. Как ICA работает?

### 🔍 Два ключевых принципа

ICA использует два важных свойства:

#### 1. Статистическая независимость

ICA ищет такие компоненты, которые **не зависят** друг от друга. Это значит, что знание одного сигнала **не даёт информации** о другом.

```
❌ Зависимые сигналы: если громко в первом микрофоне, то и во втором громко
✅ Независимые сигналы: громкость в микрофонах никак не связана
```

#### 2. Негауссовость

ICA максимизирует **негауссовость** (отклонение от нормального распределения).

**Почему?** По центральной предельной теореме, смесь независимых сигналов становится **более гауссовой** (похожей на колокол).

```
Независимые сигналы → Смесь → Гауссово распределение
(негауссовы)                 (более гауссово)
```

ICA **решает обратную задачу**: ищет такие проекции, которые **максимально негауссовы**, чтобы восстановить исходные сигналы.

### 📈 Графическое объяснение

```
Распределение независимого сигнала:
    │
    │   ██
    │  ████
    │ ██████
    │████████
    └─────────
    (острый пик — негауссово)

Распределение смеси сигналов:
    │
    │   ████
    │  ██████
    │ ████████
    │██████████
    └─────────
    (колокол — гауссово)
```

---

## 4. ICA vs PCA vs LDA

### 📋 Сравнительная таблица

| Характеристика | PCA | LDA | ICA |
|----------------|-----|-----|-----|
| **Что ищет** | Максимальную дисперсию | Разделение классов | Независимые источники |
| **Учитывает классы** | ❌ Нет | ✅ Да | ❌ Нет |
| **Требование** | Ортогональность | Равенство ковариаций | Независимость |
| **Цель** | Сжатие данных | Классификация | Разделение смеси |
| **Когда использовать** | Общий анализ | Классификация | Разделение сигналов |

### 🎮 Живой пример

Представь, что у тебя есть **микрофоны** на концерте:

```
Микрофон 1: Слышит гитару + голос + ударные (смесь)
Микрофон 2: Слышит голос + ударные + бас (смесь)
Микрофон 3: Слышит ударные + бас + гитару (смесь)
```

**Что сделает каждый метод:**

- **PCA**: "Найду главные направления в этих записях" (например, общую громкость)
- **LDA**: "Разделю записи по жанрам" (рок, поп, джаз)
- **ICA**: "Разделю запись на гитару, голос, ударные и бас!" 🎸🎤🥁

---

## 5. Практическая реализация ICA

### 5.1. Загрузка данных

```python
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Загрузка данных
data = pd.read_csv('PCA_example_dataset.csv')
print("Размер данных:", data.shape)

# Разделение на признаки и целевую переменную
y = data['label']
X = data.drop(columns=['label'])

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Обучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")
```

### 5.2. Применение ICA

```python
from sklearn.decomposition import FastICA

# Создание модели ICA
# n_components=3 — оставляем 3 независимые компоненты
ica = FastICA(n_components=3, random_state=42)

# Обучение и преобразование данных
X_train_ica = ica.fit_transform(X_train)
X_test_ica = ica.transform(X_test)

print(f"Размерность до ICA: {X_train.shape[1]}")
print(f"Размерность после ICA: {X_train_ica.shape[1]}")
```

### 5.3. Параметры FastICA

```python
# Основные параметры FastICA
FastICA(
    n_components=3,        # Число независимых компонент
    algorithm='parallel',  # 'parallel' или 'deflation'
    whiten=True,          # Отбеливание данных
    max_iter=200,         # Максимум итераций
    tol=1e-4,             # Критерий остановки
    random_state=42       # Для воспроизводимости
)
```

**Что такое отбеливание (whitening)?**

```
Отбеливание — это преобразование, которое:
1. Делает признаки некоррелированными
2. Приводит их к единичной дисперсии

Зачем? Это упрощает поиск независимых компонент!
```

### 5.4. Обучение модели на данных после ICA

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Обучение дерева решений
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train_ica, y_train)

# Предсказание
y_pred_ica = tree.predict(X_test_ica)

# Оценка качества
accuracy_ica = accuracy_score(y_test, y_pred_ica)
precision_ica = precision_score(y_test, y_pred_ica, average='weighted')
recall_ica = recall_score(y_test, y_pred_ica, average='weighted')

print("Результаты после ICA:")
print(f"Accuracy: {accuracy_ica:.4f}")
print(f"Precision: {precision_ica:.4f}")
print(f"Recall: {recall_ica:.4f}")
```

### 5.5. Сравнение всех методов

```python
# Сравнение всех трёх методов
results = {
    'Исходные данные': {'accuracy': 0.467, 'precision': 0.469, 'recall': 0.467},
    'PCA': {'accuracy': 0.505, 'precision': 0.505, 'recall': 0.505},
    'LDA': {'accuracy': 0.528, 'precision': 0.529, 'recall': 0.528},
    'ICA': {'accuracy': 0.514, 'precision': 0.514, 'recall': 0.514}
}

# Визуализация сравнения
metrics = ['accuracy', 'precision', 'recall']
methods = list(results.keys())

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(metrics))
width = 0.2

for i, method in enumerate(methods):
    scores = [results[method][m] for m in metrics]
    bars = ax.bar(x + i*width, scores, width, label=method)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')

ax.set_ylabel('Значение метрики')
ax.set_title('Сравнение методов снижения размерности')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 5.6. Визуализация независимых компонент

```python
# Визуализация данных после ICA
plt.figure(figsize=(12, 4))

# ICA проекция
plt.subplot(1, 3, 1)
plt.scatter(X_train_ica[:, 0], X_train_ica[:, 1], 
            c=y_train, cmap='viridis', alpha=0.6)
plt.xlabel('Независимая компонента 1')
plt.ylabel('Независимая компонента 2')
plt.title('Данные после ICA')
plt.colorbar(label='Классы')

# PCA проекция (для сравнения)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

plt.subplot(1, 3, 2)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
            c=y_train, cmap='viridis', alpha=0.6)
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.title('Данные после PCA')
plt.colorbar(label='Классы')

# LDA проекция (для сравнения)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)

plt.subplot(1, 3, 3)
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], 
            c=y_train, cmap='viridis', alpha=0.6)
plt.xlabel('Дискриминантная компонента 1')
plt.ylabel('Дискриминантная компонента 2')
plt.title('Данные после LDA')
plt.colorbar(label='Классы')

plt.tight_layout()
plt.show()
```

---

## 6. Пример из реальной жизни: "Задача коктейльной вечеринки"

### 🎤 Классический пример ICA

Представь, что в комнате **одновременно говорят 3 человека**, а ты записываешь их **2 микрофонами**:

```
Микрофон 1: 0.5·Голос_А + 0.3·Голос_Б + 0.2·Голос_В
Микрофон 2: 0.2·Голос_А + 0.4·Голос_Б + 0.4·Голос_В
```

**Задача ICA** — по этим двум записям восстановить **3 отдельных голоса**!

```python
# Симуляция задачи коктейльной вечеринки
np.random.seed(42)

# Три независимых источника (голоса)
n_samples = 2000
source1 = np.random.laplace(0, 1, n_samples)  # Голос 1
source2 = np.random.laplace(0, 1, n_samples)  # Голос 2
source3 = np.random.laplace(0, 1, n_samples)  # Голос 3

# Матрица смешивания
A = np.array([[0.5, 0.3, 0.2],
              [0.2, 0.4, 0.4],
              [0.3, 0.3, 0.4]])

# Смешанные сигналы
X_mixed = np.dot(np.array([source1, source2, source3]).T, A.T)

# Разделение с помощью ICA
ica = FastICA(n_components=3, random_state=42)
S_ica = ica.fit_transform(X_mixed)

print("✅ ICA успешно разделила смешанные сигналы!")
print(f"Исходные сигналы: {source1.shape}")
print(f"Смешанные сигналы: {X_mixed.shape}")
print(f"Разделённые сигналы: {S_ica.shape}")
```

---

## 7. Преимущества и ограничения ICA

### ✅ Преимущества

1. **Разделение смешанных сигналов** — главное преимущество ICA
2. **Работает без информации о классах** — не нужны метки
3. **Находит скрытые независимые источники** — помогает понять структуру данных
4. **Подходит для не-гауссовых данных** — в отличие от многих методов

### ❌ Ограничения

1. **Сложность интерпретации** — компоненты могут быть неочевидны
2. **Порядок компонент не важен** — нельзя сказать, какая важнее
3. **Чувствительность к шуму** — работает хуже с зашумленными данными
4. **Не для гауссовых данных** — если данные нормально распределены, ICA не работает
5. **Не учитывает классы** — может быть хуже LDA для классификации

---

## 8. Когда использовать ICA?

### ✅ ICA подойдёт, если:

- У тебя **смешанные сигналы**, которые нужно разделить
- Данные **негауссовы** (не похожи на колокол)
- Ты хочешь найти **скрытые независимые факторы**
- Данные из **разных источников**, которые смешались

### ❌ ICA НЕ подойдёт, если:

- Данные **гауссовы** (нормально распределены)
- Нужно **сохранить порядок компонент** по важности
- Важна **интерпретация компонент**
- Нужно **учитывать классы** для классификации

---

## 9. ICA vs PCA: практическое сравнение

```python
# Демонстрация различий на простом примере
from sklearn.decomposition import PCA, FastICA

# Создание смешанных сигналов
np.random.seed(42)
t = np.linspace(0, 10, 1000)

# Исходные независимые сигналы
s1 = np.sin(t * 0.5)  # Синусоида
s2 = np.random.laplace(0, 0.5, 1000)  # Шум
s3 = np.sign(np.sin(t * 1.5))  # Прямоугольный сигнал

# Смесь
X_mixed = np.column_stack([s1, s2, s3]) @ np.random.randn(3, 3)

# Применение PCA и ICA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_mixed)

ica = FastICA(n_components=3, random_state=42)
X_ica = ica.fit_transform(X_mixed)

# Визуализация
fig, axes = plt.subplots(4, 3, figsize=(12, 10))

# Исходные сигналы
for i in range(3):
    axes[0, i].plot([s1, s2, s3][i])
    axes[0, i].set_title(f'Источник {i+1}')

# Смешанные сигналы
for i in range(3):
    axes[1, i].plot(X_mixed[:, i])
    axes[1, i].set_title(f'Смесь {i+1}')

# PCA компоненты
for i in range(3):
    axes[2, i].plot(X_pca[:, i])
    axes[2, i].set_title(f'PCA {i+1}')

# ICA компоненты
for i in range(3):
    axes[3, i].plot(X_ica[:, i])
    axes[3, i].set_title(f'ICA {i+1}')

plt.tight_layout()
plt.show()

print("📊 Наблюдение:")
print("- PCA: компоненты всё ещё смешанные")
print("- ICA: компоненты точно разделены!")
```

---

## 10. Где применяется ICA?

### 🌍 Реальные области применения

| Область | Пример использования |
|---------|---------------------|
| **Аудио** | Разделение голосов (коктейльная вечеринка) |
| **Медицина** | Обработка ЭЭГ, удаление артефактов |
| **Финансы** | Поиск скрытых факторов в рынке |
| **Изображения** | Разделение гиперспектральных данных |
| **Нейронаука** | Анализ активности мозга |

### 🧠 Пример: ЭЭГ (электроэнцефалограмма)

```
Сырые ЭЭГ-сигналы = Активность мозга + Артефакты (моргание, движение)
                           ↓
                         ICA
                           ↓
            Активность мозга  ←→  Артефакты
            (чистый сигнал)     (можно удалить)
```

---

## 11. Итоговое резюме

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ICA — это метод, который:                                  ║
║                                                               ║
║   ✅ РАЗДЕЛЯЕТ смешанные сигналы на независимые источники    ║
║   ✅ Работает без информации о классах                       ║
║   ✅ Находит СКРЫТЫЕ причины, формирующие данные             ║
║                                                               ║
║   📊 В сравнении с другими методами:                         ║
║                                                               ║
║   PCA: "Где данные сильнее всего меняются?"                  ║
║   LDA: "Как лучше всего разделить классы?"                   ║
║   ICA: "Из каких независимых источников состоят данные?"     ║
║                                                               ║
║   🎯 Итог: ICA — король разделения смешанных сигналов!      ║
║   (но не всегда лучший для классификации)                    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## 12. Короткий словарик

| Понятие | Объяснение |
|---------|------------|
| **Независимость** | Сигналы, которые не зависят друг от друга |
| **Негауссовость** | Отклонение от "колоколообразного" распределения |
| **Матрица смешивания** | Как именно смешались исходные сигналы |
| **Отбеливание** | Преобразование, делающее признаки некоррелированными |
| **Источник** | Исходный независимый сигнал |
| **Коктейльная вечеринка** | Классическая задача разделения голосов |

---

*Конспект составлен на основе лекции по ICA с использованием простых аналогий и примеров для школьников.*

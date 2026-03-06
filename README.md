# Predicción de Finalización del Curso (Regresión Logística)

Proyecto de Machine Learning para **predecir si un estudiante finalizará o no un curso** a partir de variables de comportamiento en una plataforma educativa.

El repositorio contiene un notebook principal:

- `ML_prediccion_nota_v001.ipynb`

## Objetivo del modelo

Dada la información de un estudiante:

- `horas_conexion`
- `recursos_vistos`
- `participacion_foro`
- `nota_auto_eval`

se entrena un clasificador binario para predecir la variable objetivo:

- `finalizo_curso` (0 = no finaliza, 1 = finaliza)

## Estructura técnica (pipeline)

El notebook sigue un flujo estándar de un proyecto de ML supervisado:

1. **Carga de datos** desde CSV.
2. **Exploración** (`info()`, `describe()`), revisión de tipos y valores faltantes.
3. **Tratamiento de faltantes** (imputación por media).
4. **Preprocesamiento** de variables categóricas (Label Encoding).
5. **Separación de variables** (`X`, `y`).
6. **Split Train/Test** (`train_test_split`).
7. **Entrenamiento** con `LogisticRegression`.
8. **Predicción** sobre el set de prueba.
9. **Evaluación** con `accuracy`, `confusion_matrix` y `classification_report`.
10. **Dashboard** interactivo (opcional) con `ipywidgets` para hacer predicciones manuales.

## Dataset

### Fuente

El notebook espera un archivo CSV llamado `comportamiento_estudiantes.csv`.

En el notebook se carga con:

```python
df = pd.read_csv("/content/comportamiento_estudiantes.csv", sep=";")
```

- Si lo corres en **Google Colab**, típicamente `/content/...` corresponde al filesystem del runtime.
- Si lo corres en **local**, actualiza esa ruta al path donde tengas el CSV.

### Separador

Se usa `sep=";"` (CSV separado por punto y coma). Si tu archivo usa coma, cambia a `sep=","`.

### Columnas esperadas

- `horas_conexion` (float)
- `recursos_vistos` (float)
- `participacion_foro` (categórica: `Baja`, `Media`, `Alta`)
- `nota_auto_eval` (float)
- `finalizo_curso` (int, 0/1)

### Valores faltantes

En el notebook se detectan faltantes en `recursos_vistos` y se imputan así:

- **Estrategia**: media (`SimpleImputer(strategy="mean")`).
- **Motivación**: evitar eliminar filas y mantener el tamaño de muestra.

## Preprocesamiento

### Codificación de `participacion_foro`

`participacion_foro` es categórica (texto). Para usarla en un modelo lineal se convierte a valores numéricos con `LabelEncoder`.

- Nota importante: `LabelEncoder` asigna números según el orden que encuentre en los datos (por ejemplo `Baja -> 0`, `Media -> 1`, `Alta -> 2`).
- En el dashboard, el notebook define manualmente un mapeo:

```python
foro_map = {"Baja": 0, "Media": 1, "Alta": 2}
```

Para que el dashboard sea consistente, este mapeo debe coincidir con el encoding aprendido. En el notebook actual, este mapeo se asume como el esperado.

## Modelo

### Algoritmo

Se utiliza **Regresión Logística** (`sklearn.linear_model.LogisticRegression`), un modelo lineal que aprende:

- Una combinación lineal de las variables (pesos/coefs)
- Que se transforma en probabilidad con una función logística

Esto es adecuado para **clasificación binaria**.

### Entrenamiento

- Se usa `train_test_split(..., test_size=0.2, random_state=42)`.
- El modelo se entrena con `model.fit(X_train, y_train)`.

## Evaluación

Se reporta:

- **Accuracy** (`accuracy_score`) como métrica global.
- **Matriz de confusión** (`confusion_matrix`) visualizada con `seaborn.heatmap`.
- **Reporte de clasificación** (`classification_report`) con:
  - `precision`
  - `recall`
  - `f1-score`

### Nota sobre desbalance de clases

En la salida del `classification_report` se observa un comportamiento típico cuando el modelo **predice casi siempre la clase mayoritaria** (por ejemplo, `recall` alto para clase 1 y `precision/recall` pobre para clase 0), lo cual puede disparar warnings como:

- `UndefinedMetricWarning: Precision is ill-defined ... in labels with no predicted samples`

Esto suele indicar:

- dataset desbalanceado, y/o
- umbral por defecto (0.5) poco adecuado, y/o
- falta de ajuste de hiperparámetros/regularización, y/o
- necesidad de métricas adicionales (ROC-AUC, PR-AUC) y estrategias como `class_weight="balanced"`.

El notebook actual se centra en el flujo base y en `accuracy`.

## Dashboard (ipywidgets)

El notebook incluye una sección “PLUS: Dashboard de predicción” para ejecutar predicciones con controles interactivos y un pequeño panel de interpretación:

- Sliders para `horas_conexion`, `recursos_vistos`, `nota`
- Dropdown para `participacion_foro`
- Botón para calcular:
  - clase predicha (`model.predict`)
  - probabilidad (`model.predict_proba`)
  - semáforo de riesgo de abandono (bajo, medio, alto) según la probabilidad
  - recomendaciones de intervención personalizadas (aumentar horas, ver más recursos, participar en foros, reforzar contenidos)

### Dependencia

En Colab se instala con:

```python
!pip install ipywidgets
```

En local (recomendado), instala dependencias en un entorno virtual (ver siguiente sección).

## Requisitos técnicos

El notebook utiliza:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `ipywidgets` (solo para el dashboard)

## Cómo ejecutar

### Opción A: Google Colab

1. Abre `ML_prediccion_nota_v001.ipynb` en Colab.
2. Sube `comportamiento_estudiantes.csv` al runtime.
3. Ajusta la ruta del `read_csv` si es necesario.
4. Ejecuta todas las celdas.

### Opción B: Local (Jupyter)

1. Crea y activa un entorno virtual.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Instala dependencias.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ipywidgets jupyter
```

3. Abre el notebook.

```bash
jupyter notebook
```

4. Asegúrate de que el CSV exista y ajusta la ruta en `pd.read_csv(...)`.

## Estructura del repositorio

```text
.
├── ML_prediccion_nota_v001.ipynb
└── README.md
```

## Limitaciones y consideraciones

- **Reproducibilidad**: el split es reproducible por `random_state=42`, pero si el dataset cambia, los resultados cambian.
- **Encoding de categorías**: si cambian los valores de `participacion_foro` o su orden, el mapping puede variar.
- **Métrica**: `accuracy` puede ser engañosa en datasets desbalanceados. Considera métricas adicionales.
- **Persistencia del modelo**: el notebook no guarda el modelo entrenado a disco (no hay `joblib.dump`). Si necesitas despliegue, conviene agregar serialización y un pipeline.

## Próximos pasos recomendados (técnicos)

- Evaluar desbalance de clases y probar `class_weight="balanced"`.
- Estandarizar features (por ejemplo con `StandardScaler`) y usar un `Pipeline`.
- Validación cruzada (`cross_val_score`) y búsqueda de hiperparámetros.
- Guardar modelo y preprocesadores (`imputer`, `encoder`) para inferencia consistente.

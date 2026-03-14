# Resumen de Decisiones — Proyecto Riesgo Crediticio
**Machine Learning 1 | Universidad del Rosario**  
Vanessa Ochoa, Nilson Amaya, Juan Zamora

---

## 1. ¿Por qué no usamos accuracy como métrica principal?

El dataset tiene un desbalanceo severo: **93.3% no incumple** y solo **6.7% sí incumple**. Esto significa que un modelo que simplemente dijera *"nadie va a hacer default"* tendría 93% de accuracy sin haber aprendido nada útil.

Por eso usamos **F1-Score de la clase 1 (default)** como métrica principal, porque combina precisión y recall y no se ve distorsionada por el desbalanceo.

---

## 2. Decisiones de preprocesamiento

### 2.1 Valores faltantes — imputación con mediana

**Variables afectadas:** `MonthlyIncome` (19.82% nulos) y `NumberOfDependents` (2.62% nulos).

**¿Por qué mediana y no media?**  
Las variables financieras tienen outliers extremos. La media se arrastra hacia esos valores extremos y dejaría de representar al cliente típico. La mediana es el valor del centro y no se afecta por los extremos.

**Decisión extra en `MonthlyIncome`:**  
Los registros con ingreso desconocido tienen más tasa de default que los que sí tienen ingreso. Eso significa que *no saber el ingreso* ya es una señal. Por eso creamos `flag_ingreso_nulo = 1` **antes** de imputar, para no perder esa información.

> **Regla de oro:** los parámetros de imputación (la mediana) se calculan **solo en train** y se aplican al test con ese mismo valor. Nunca al revés.

---

### 2.2 Outliers — tratamiento con criterio financiero

Cada variable se trató según su **significado real**, no solo por su magnitud.

| Variable | Problema | Decisión | ¿Por qué? |
|---|---|---|---|
| `RevolvingUtilization` | Máximo = 50,708 (debería ser 0–1) | `clip(0, 5)` | Los valores entre 1 y 5 son clientes que **superaron su límite** — tienen 37% de default vs 6% del resto. Recortarlos a 1 destruiría esa señal. |
| `DebtRatio` | Máximo = 329,664 | `clip(0, P99)` | El valor extremo viene de dividir deuda entre ingreso = 0 (división por cero). No es deuda real. Se capa en percentil 99 para conservar la variabilidad válida. |
| `age` | Mínimo = 0, algunos valores 96–98 | `clip(18, 95)` | Edades imposibles o errores de registro. |
| `MonthlyIncome` | Máximo = 3,008,750 | `clip(0, P99)` | Ingresos extremadamente altos que distorsionan el escalamiento. |
| Variables de atrasos (30-59, 60-89, 90+ días) | 264 registros con valor = 98 | Crear `flag_atrasos_98`, luego `clip(0, 13)` | El valor 98 es un **código especial**, no 98 atrasos reales. Esos 264 registros tienen **54% de tasa de default** — son los clientes más riesgosos del dataset. Reemplazarlos con 0 sería el error más grave posible. |

---

### 2.3 Selección de variables

Se excluyó `NumberOfTime60-89DaysPastDueNotWorse` porque tiene correlación alta con las otras dos variables de atrasos (entre 0.58 y 0.75). Incluirla no aporta información nueva y puede generar **multicolinealidad**, lo que afecta especialmente a la Regresión Logística.

Se agregaron dos variables nuevas creadas en el preprocesamiento:
- `flag_ingreso_nulo` — captura que el ingreso era desconocido
- `flag_atrasos_98` — captura el grupo de mayor riesgo del dataset

---

### 2.4 Escalamiento — ¿MinMax o Standard?

Ambos escaladores se probaron en los dos modelos. La diferencia clave es:

- **MinMaxScaler:** lleva todo al rango [0, 1]. Sensible a outliers — si hay un valor extremo, el resto queda comprimido.
- **StandardScaler:** centra en media 0 y desviación 1. Más robusto ante outliers.

El Grid Search decide cuál funciona mejor para cada modelo con estos datos específicos.

---

## 3. Decisiones de modelado

### 3.1 ¿Por qué dos modelos?

El enunciado lo pide, pero también tiene sentido técnico: k-NN y Regresión Logística tienen naturalezas distintas.

| | k-NN | Regresión Logística |
|---|---|---|
| ¿Cómo aprende? | Memoriza los datos y clasifica por proximidad | Aprende una ecuación con coeficientes |
| ¿Necesita escalar? | Sí, siempre — se basa en distancias | Sí, para que la regularización sea justa |
| ¿Es interpretable? | No | Sí — cada coeficiente dice cuánto pesa cada variable |
| Hiperparámetro clave | `k` (número de vecinos) | `C` (nivel de regularización) |

### 3.2 Split con stratify

Se usó `stratify=y` en el train/test split para garantizar que la proporción de defaults (6.7%) sea la misma en train y en validación. Sin esto, podría quedar un split donde casi todos los defaults caen en un solo lado por azar.

### 3.3 ¿Por qué Grid Search con Pipeline?

El `Pipeline` garantiza que el escalador se ajuste **solo con los datos de entrenamiento en cada pliegue** de la validación cruzada. Si escaláramos fuera del Pipeline, estaríamos filtrando información del conjunto de validación al modelo — lo que se llama **data leakage**.

### 3.4 ¿Por qué StratifiedKFold y no KFold?

Con clases desbalanceadas, un KFold normal podría crear pliegues donde casi no hay defaults. El `StratifiedKFold` garantiza que cada pliegue tenga la misma proporción de defaults que el dataset completo.

---

## 4. ¿Cómo elegir el modelo final?

Cuando tengan los resultados de Colab, apliquen este criterio en orden:

### Paso 1 — Mirar el F1 medio en validación cruzada (Sección 5)
El modelo con mayor F1 medio en CV es el punto de partida. Si la diferencia es pequeña (menos de 0.01), pasar al paso 2.

### Paso 2 — Mirar el Recall de la clase 1 en validación (Sección 6)
En riesgo crediticio, **no detectar a alguien que va a incumplir** (Falso Negativo) es más costoso que rechazar a alguien que sí pagaría (Falso Positivo). Por eso si dos modelos tienen F1 similar, gana el que tenga mayor Recall.

### Paso 3 — Mirar la estabilidad (columna Desviación en la tabla de CV)
Un modelo con F1 = 0.35 y desviación 0.01 es mejor que uno con F1 = 0.36 y desviación 0.08. El segundo depende demasiado del split específico.

### Paso 4 — Mirar la matriz de confusión
Comparar cuántos **Falsos Negativos** tiene cada modelo. El que tenga menos FN es el más conservador y adecuado para este problema.

---

### Tabla de decisión rápida

| Situación | Recomendación |
|---|---|
| Regresión Logística tiene mayor F1 | Usar Regresión Logística |
| k-NN tiene mayor F1 | Usar k-NN |
| F1 muy similar entre ambos | Usar Regresión Logística — es más interpretable y estable |
| Uno tiene mucho mejor Recall aunque F1 similar | Usar el de mayor Recall |

---

## 5. Frase clave para defender el proyecto

> *"El éxito en Machine Learning no reside en el algoritmo más complejo, sino en el entendimiento profundo de los datos y la correcta evaluación del error."*  
> — Prof. Daniel Rambaut

Todas las decisiones de este proyecto — desde no recortar RevolvingUtilization a 1, hasta crear el flag de los valores 98 — responden exactamente a ese principio: primero entender los datos, luego modelar.

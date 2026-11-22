# Investigación: Algoritmos de Clustering

## 1. Aprendizaje no supervisado y clustering

### 1.1 ¿Qué es el aprendizaje no supervisado?
El aprendizaje no supervisado es una rama del machine learning en la que el modelo
trabaja con **datos sin etiquetar**: solo tenemos las variables de entrada (`X`) y no
existe una variable objetivo (`y`). El objetivo no es predecir una respuesta correcta,
sino **descubrir la estructura oculta** de los datos: patrones, grupos o relaciones
que no conocíamos de antemano.

### 1.2 ¿Qué es un algoritmo de clustering?
Un algoritmo de clustering (agrupamiento) es un tipo de algoritmo no supervisado que
divide los datos en **grupos (clusters)**, de forma que los elementos de un mismo grupo
sean muy parecidos entre sí y diferentes a los de otros grupos. Los grupos no se
conocen previamente: el algoritmo los descubre. Además, el modelo no les pone nombre
(solo devuelve "cluster 0", "cluster 1"...); **interpretar qué significa cada grupo
es tarea nuestra**.

### 1.3 Diferencias con el aprendizaje supervisado

|                 | Supervisado                          | No supervisado                          |
|-----------------|--------------------------------------|-----------------------------------------|
| Etiquetas (`y`) | Sí                                   | No                                      |
| Objetivo        | Predecir (clasificación o regresión) | Descubrir estructura o grupos           |
| Evaluación      | Métricas claras (accuracy, RMSE...)  | Más difícil: métricas internas (silueta, inercia...) |
| Ejemplo         | Predecir si un piso en Barcelona es caro o barato | Agrupar clientes según su comportamiento |

En resumen: en el aprendizaje supervisado el modelo aprende con "las respuestas
correctas"; en el no supervisado tiene que encontrar el orden por sí mismo.

### 1.4 Propósito
El clustering se usa principalmente para:

- **Exploración de datos:** entender qué grupos naturales existen en un conjunto de
  datos. *Ej.: agrupar los anuncios de Airbnb en Barcelona y descubrir tipos como
  "piso turístico de lujo", "habitación para estudiantes" o "piso familiar".*
- **Segmentación:** tratar de forma distinta a cada grupo. *Ej.: una tienda divide a
  sus clientes en segmentos y envía una campaña diferente a cada uno.*
- **Detección de anomalías:** los puntos que no encajan en ningún grupo pueden ser
  casos raros. *Ej.: una transacción bancaria muy diferente a las habituales puede
  indicar fraude.*


## 2. Clustering basado en centroides vs. jerárquico

## 3. Clustering basado en densidad (DBSCAN)

## 4. Elección del número de clusters y evaluación

## 5. Algoritmos
### 5.1 K-Means
### 5.2 Clustering Jerárquico Aglomerativo
### 5.3 DBSCAN
### 5.4 Gaussian Mixture Models (GMM)

## 6. Tabla comparativa

## 7. Fuentes

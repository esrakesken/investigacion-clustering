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

### 2.1 Clustering basado en centroides (particional)
Este enfoque divide los datos en un número **k de grupos fijado de antemano**. Cada
grupo está representado por un **centroide** (el punto medio del grupo) y cada dato se
asigna al centroide más cercano. El algoritmo va recolocando los centroides y
reasignando los puntos hasta que los grupos dejan de cambiar.

El resultado es una **partición plana**: cada punto pertenece a un único grupo.

**Algoritmo representativo:** K-Means.

### 2.2 Clustering jerárquico
Este enfoque no divide los datos de una sola vez, sino que construye una **jerarquía
de grupos anidados** que se representa con un árbol llamado **dendrograma**. Existen
dos estrategias:

- **Aglomerativa (de abajo arriba):** cada punto empieza siendo su propio grupo y, en
  cada paso, se unen los dos grupos más parecidos, hasta que todo queda en un solo grupo.
- **Divisiva (de arriba abajo):** se parte de un único grupo con todos los datos y se
  va dividiendo.

No es necesario fijar k al principio: se elige después, "cortando" el dendrograma a
la altura deseada.

**Algoritmo representativo:** Clustering Jerárquico Aglomerativo.

### 2.3 Diferencias principales

|                        | Basado en centroides            | Jerárquico                              |
|------------------------|---------------------------------|-----------------------------------------|
| Número de clusters (k) | Hay que fijarlo antes           | Se elige después cortando el dendrograma |
| Resultado              | Partición plana                 | Grupos anidados (árbol)                 |
| Escalabilidad          | Rápido, funciona con muchos datos | Lento y costoso con muchos datos      |
| Interpretabilidad      | Centroides fáciles de interpretar | El dendrograma muestra cómo se forman los grupos |
| Forma de los clusters  | Tiende a grupos esféricos       | Depende del tipo de enlace (*linkage*)  |

## 3. Clustering basado en densidad (DBSCAN)

### 3.1 ¿En qué consiste?
El clustering basado en densidad parte de una idea sencilla: **un cluster es una zona
donde los puntos están muy juntos (alta densidad)**, separada de otras zonas por
regiones con pocos puntos. Los puntos aislados, que no pertenecen a ninguna zona
densa, se consideran **ruido**.

El algoritmo más conocido es **DBSCAN** (*Density-Based Spatial Clustering of
Applications with Noise*), que utiliza dos hiperparámetros:

- **`eps` (ε):** el radio del vecindario alrededor de cada punto, es decir, qué
  distancia consideramos "cerca".
- **`min_samples`:** el número mínimo de puntos que debe haber dentro de ese radio
  para que una zona se considere densa.

Con estos dos valores, DBSCAN clasifica cada punto en uno de tres tipos:

| Tipo de punto | Descripción |
|---------------|-------------|
| **Núcleo (core)** | Tiene al menos `min_samples` vecinos dentro de su radio `eps` |
| **Frontera (border)** | No es núcleo, pero está dentro del radio de un punto núcleo |
| **Ruido (noise)** | No es núcleo ni frontera; se etiqueta como `-1` |

**Funcionamiento:** el algoritmo elige un punto núcleo y crea un cluster con todos sus
vecinos. Si alguno de esos vecinos también es núcleo, el cluster se sigue
**expandiendo** a través de él. Cuando ya no se puede expandir más, busca otro punto
núcleo no visitado y empieza un nuevo cluster. Los puntos que no se alcanzan desde
ningún núcleo quedan como ruido.

*Ejemplo:* si miramos desde un dron la playa de la Barceloneta en verano, veremos
grupos de personas muy juntas (clusters) con formas distintas, y algunas personas
caminando solas (ruido). Nadie nos dice cuántos grupos hay: aparecen solos por la
densidad.

### 3.2 Diferencias con el clustering basado en centroides y el jerárquico

|                        | Centroides (K-Means)    | Jerárquico                      | Densidad (DBSCAN)              |
|------------------------|-------------------------|---------------------------------|--------------------------------|
| ¿Hay que fijar k?      | Sí                      | No (se corta el dendrograma)    | **No**, los clusters surgen de la densidad |
| Forma de los clusters  | Esférica                | Depende del *linkage*           | **Cualquier forma** (alargada, curva, anillos...) |
| Outliers               | Los asigna siempre a un grupo | Los asigna siempre a un grupo | **Los detecta como ruido** |
| Parámetros clave       | k                       | Tipo de enlace y altura de corte | `eps` y `min_samples`         |
| Punto débil            | Hay que conocer k; sensible a outliers | Lento con muchos datos | Difícil elegir `eps`; falla si los clusters tienen densidades muy distintas |

En resumen: K-Means busca **centros**, el jerárquico construye un **árbol de
similitudes** y DBSCAN busca **zonas densas**. Por eso DBSCAN es especialmente útil
cuando los grupos tienen formas irregulares o cuando hay mucho ruido en los datos.

## 4. Elección del número de clusters y evaluación

## 5. Algoritmos
### 5.1 K-Means
### 5.2 Clustering Jerárquico Aglomerativo
### 5.3 DBSCAN
### 5.4 Gaussian Mixture Models (GMM)

## 6. Tabla comparativa

## 7. Fuentes

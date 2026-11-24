"""Genera las figuras del README (carpeta img/).

Uso, desde la raíz del repositorio:
    pip install scikit-learn matplotlib scipy
    python scripts/generar_figuras.py

Todos los datos son sintéticos (make_blobs / make_moons) y la semilla es fija,
así que las figuras siempre salen iguales.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEMILLA = 42
CARPETA = Path(__file__).resolve().parent.parent / "img"
CARPETA.mkdir(exist_ok=True)

COLORES = np.array(["#2a7ab0", "#e07b39", "#3a9d5d", "#c0392b", "#8e6bbf", "#b8860b"])
GRIS_RUIDO = "#9a9a9a"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "font.size": 10,
})


def colorear(etiquetas):
    """Un color por cluster; el ruido de DBSCAN (-1) en gris."""
    etiquetas = np.asarray(etiquetas)
    colores = COLORES[etiquetas % len(COLORES)].astype(object)
    colores[etiquetas == -1] = GRIS_RUIDO
    return colores


def sin_ejes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for lado in ax.spines.values():
        lado.set_visible(False)


def guardar(fig, nombre):
    fig.savefig(CARPETA / nombre)
    plt.close(fig)
    print("✓", CARPETA / nombre)


# Datos comunes: 4 grupos "redondos"
X_blobs, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.9, random_state=SEMILLA)
X_blobs = StandardScaler().fit_transform(X_blobs)


# 1. Datos sin etiquetas -> grupos encontrados (sección 1)
def fig_no_supervisado():
    km = KMeans(n_clusters=4, n_init=10, random_state=SEMILLA).fit(X_blobs)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].scatter(*X_blobs.T, s=14, c="#555555", alpha=0.7)
    axes[0].set_title("Antes: datos sin etiquetas")
    axes[1].scatter(*X_blobs.T, s=14, c=colorear(km.labels_), alpha=0.8)
    axes[1].scatter(*km.cluster_centers_.T, s=180, c="black", marker="X", label="Centroides")
    axes[1].set_title("Después: grupos de K-Means")
    axes[1].legend(loc="lower right", frameon=False)
    for ax in axes:
        sin_ejes(ax)
    guardar(fig, "01_no_supervisado.png")


# 2. Mismos datos, tres enfoques (sección 3.2)
def fig_comparacion_enfoques():
    X, _ = make_moons(n_samples=400, noise=0.06, random_state=SEMILLA)
    rng = np.random.default_rng(SEMILLA)
    ruido = rng.uniform(low=[-1.4, -0.9], high=[2.4, 1.4], size=(25, 2))
    X = StandardScaler().fit_transform(np.vstack([X, ruido]))

    modelos = [
        ("K-Means (k = 2)", KMeans(n_clusters=2, n_init=10, random_state=SEMILLA)),
        ("Jerárquico (Ward, k = 2)", AgglomerativeClustering(n_clusters=2, linkage="ward")),
        ("DBSCAN (eps = 0.3)", DBSCAN(eps=0.3, min_samples=5)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (titulo, modelo) in zip(axes, modelos):
        etiquetas = modelo.fit_predict(X)
        ax.scatter(*X.T, s=12, c=colorear(etiquetas), alpha=0.85)
        ax.set_title(titulo)
        sin_ejes(ax)
    axes[2].scatter([], [], c=GRIS_RUIDO, s=12, label="Ruido (-1)")
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Mismos datos (dos lunas + ruido), tres enfoques", y=1.02, fontweight="bold")
    guardar(fig, "02_comparacion_enfoques.png")


# 3. Método del codo (sección 4.2)
def fig_codo():
    ks = range(1, 11)
    inercias = [KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(X_blobs).inertia_ for k in ks]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(ks, inercias, "o-", color=COLORES[0])
    ax.scatter([4], [inercias[3]], s=250, facecolors="none", edgecolors=COLORES[3], linewidths=2)
    ax.annotate("codo → k = 4", xy=(4, inercias[3]), xytext=(5.5, inercias[1]),
                arrowprops=dict(arrowstyle="->", color=COLORES[3]), color=COLORES[3])
    ax.set_xticks(list(ks))
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Inercia (WCSS)")
    ax.set_title("Método del codo")
    guardar(fig, "03_metodo_codo.png")


# 4. Coeficiente de silueta (sección 4.3)
def fig_silueta():
    ks = range(2, 11)
    siluetas = [
        silhouette_score(X_blobs, KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit_predict(X_blobs))
        for k in ks
    ]
    mejor = list(ks)[int(np.argmax(siluetas))]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    barras = ax.bar(ks, siluetas, color=[COLORES[3] if k == mejor else "#b9cfe0" for k in ks])
    ax.bar_label(barras, fmt="%.2f", fontsize=8)
    ax.set_xticks(list(ks))
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Silueta media")
    ax.set_title(f"Coeficiente de silueta: el máximo está en k = {mejor}")
    guardar(fig, "04_silueta.png")


# 5. Dendrograma (sección 5.2)
def fig_dendrograma():
    # Pocos puntos y grupos no muy separados: así se ven bien todas las ramas
    X, _ = make_blobs(n_samples=15, centers=[[0, 0], [4, 1], [2, 4]],
                      cluster_std=0.9, random_state=7)
    Z = linkage(X, method="ward")
    corte = (Z[-3, 2] + Z[-2, 2]) / 2  # entre la 3.ª y la 2.ª fusión -> 3 clusters
    nombres = [f"P{i + 1}" for i in range(len(X))]

    fig, (ax_pts, ax_den) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.6]})

    set_link_color_palette(list(COLORES[:3]))
    dend = dendrogram(Z, ax=ax_den, labels=nombres, color_threshold=corte,
                      above_threshold_color="#555555", leaf_font_size=10)
    set_link_color_palette(None)
    for linea in ax_den.collections:
        linea.set_linewidth(2.2)
    ax_den.axhline(corte, ls="--", lw=1.5, color=COLORES[3])
    ax_den.text(ax_den.get_xlim()[1], corte, "  corte → 3 clusters", va="center", color=COLORES[3])
    ax_den.set_ylabel("Distancia de fusión")
    ax_den.set_title("Dendrograma (linkage Ward)")

    # Mismo color para cada punto que su rama en el dendrograma
    color_hoja = dict(zip(dend["ivl"], dend["leaves_color_list"]))
    ax_pts.scatter(*X.T, s=110, c=[color_hoja[n] for n in nombres], edgecolors="white", zorder=2)
    for nombre, (x, y) in zip(nombres, X):
        ax_pts.annotate(nombre, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=9)
    ax_pts.set_title("Los puntos")
    sin_ejes(ax_pts)
    guardar(fig, "05_dendrograma.png")


# 6. Gráfico k-distance para elegir eps (sección 5.3)
def fig_k_distance():
    X, _ = make_moons(n_samples=400, noise=0.06, random_state=SEMILLA)
    X = StandardScaler().fit_transform(X)
    k = 5  # = min_samples
    distancias, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    d = np.sort(distancias[:, k])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(d, color=COLORES[0])
    ax.axhline(0.3, ls="--", color=COLORES[3])
    ax.text(10, 0.31, "eps ≈ 0.3 (zona del codo)", color=COLORES[3], va="bottom")
    ax.set_xlabel("Puntos ordenados por distancia")
    ax.set_ylabel(f"Distancia al {k}.º vecino")
    ax.set_title("Gráfico k-distance para elegir eps")
    guardar(fig, "06_k_distance.png")


# 7. K-Means vs GMM con grupos alargados (sección 5.4)
def fig_kmeans_vs_gmm():
    X, _ = make_blobs(n_samples=450, centers=[[-2, 2], [0, 0], [2, -0.5]],
                      cluster_std=0.75, random_state=SEMILLA)
    X = X @ np.array([[0.6, -0.64], [-0.41, 0.85]])  # estira e inclina los grupos

    km = KMeans(n_clusters=3, n_init=10, random_state=SEMILLA).fit(X)
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=SEMILLA).fit(X)
    prob_max = gmm.predict_proba(X).max(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(*X.T, s=12, c=colorear(km.labels_), alpha=0.85)
    axes[0].set_title("K-Means: fronteras rectas, asignación dura")
    axes[1].scatter(*X.T, s=12, c=colorear(gmm.predict(X)), alpha=np.clip(prob_max, 0.15, 1))
    axes[1].set_title("GMM: elipses + probabilidad")
    for media, cov, color in zip(gmm.means_, gmm.covariances_, COLORES):
        valores, vectores = np.linalg.eigh(cov)
        angulo = np.degrees(np.arctan2(vectores[1, 1], vectores[0, 1]))
        ancho, alto = 2 * 2 * np.sqrt(valores[::-1])  # elipse de 2 desviaciones
        axes[1].add_patch(Ellipse(media, ancho, alto, angle=angulo, fill=False, lw=1.5, color=color))
    axes[1].text(0.02, 0.02, "Puntos más claros = menos seguros\n(frontera entre dos clusters)",
                 transform=axes[1].transAxes, fontsize=8, color="#555555")
    for ax in axes:
        sin_ejes(ax)
    guardar(fig, "07_kmeans_vs_gmm.png")


if __name__ == "__main__":
    fig_no_supervisado()
    fig_comparacion_enfoques()
    fig_codo()
    fig_silueta()
    fig_dendrograma()
    fig_k_distance()
    fig_kmeans_vs_gmm()

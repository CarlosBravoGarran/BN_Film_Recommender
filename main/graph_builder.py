"""graph_builder.py

Funciones para construir una Red Bayesiana (estructura + CPDs) a partir
de un CSV y para guardar/visualizar el resultado.
Este módulo agrupa la parte de creación/ajuste del grafo original
de `graph_gen.py` para que pueda reutilizarse desde otros scripts (p. ej. inferencia).
"""

import pandas as pd
from pgmpy.estimators import HillClimbSearch, BIC, BayesianEstimator, BDeu
from pgmpy.models import DiscreteBayesianNetwork
import networkx as nx
import matplotlib.pyplot as plt
import pickle


def load_data(path: str) -> pd.DataFrame:
    """Carga el dataset desde un CSV y devuelve un DataFrame."""
    return pd.read_csv(path)


def learn_structure(df: pd.DataFrame):
    """Aprende la estructura usando Hill Climb y BDeu.
    Devuelve el objeto de modelo (networkx.DiGraph-like de pgmpy) con los arcos.
    """
    hc = HillClimbSearch(df)
    best_model = hc.estimate(scoring_method=BDeu(df, equivalent_sample_size=100))
    return best_model


def save_edges(best_model, path: str = "best_model_edges.csv") -> None:
    """Guarda los arcos del mejor modelo en CSV (source,target)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("source,target\n")
        for u, v in best_model.edges():
            f.write(f"{u},{v}\n")


def build_and_fit_model(
    csv_path: str = "tv_bn_dataset.csv",
    save_edges_path: str = "best_model_edges.csv",
    save_model_path: str = "model.pkl",
    prior_type: str = "BDeu",
    equivalent_sample_size: int = 100,
    visualize: bool = True,
) -> tuple[DiscreteBayesianNetwork, pd.DataFrame]:
    """Flujo completo: carga datos, aprende estructura, crea modelo, ajusta CPDs.
    Opcionalmente guarda los arcos y el modelo (pickle) y visualiza la red.
    Devuelve (model, df).
    """
    df = load_data(csv_path)
    best_model = learn_structure(df)

    if save_edges_path:
        save_edges(best_model, save_edges_path)

    model = DiscreteBayesianNetwork(best_model.edges())
    model.fit(df, estimator=BayesianEstimator, prior_type=prior_type, equivalent_sample_size=equivalent_sample_size)

    if save_model_path:
        save_model(model, save_model_path)

    if visualize:
        visualize_model(model)

    return model, df


def visualize_model(model: DiscreteBayesianNetwork, figsize=(10, 8)) -> None:
    """Dibuja la red usando networkx/matplotlib.
    No muestra la figura si se ejecuta en un entorno sin display; caller decide.
    """
    G = nx.DiGraph()
    G.add_edges_from(model.edges())

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1000,
        node_color="skyblue",
        font_size=10,
        arrowsize=20,
        edgecolors="black",
    )
    plt.title("Red Bayesiana aprendida automáticamente", fontsize=14)
    try:
        plt.show()
    except Exception:
        pass


def save_model(model: DiscreteBayesianNetwork, path: str) -> None:
    """Guarda el modelo en un archivo usando pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> DiscreteBayesianNetwork:
    """Carga un modelo guardado con `save_model`."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

import os

def save_cpds_to_txt(model, path: str = "model_cpds.txt") -> None:
    """Guarda todas las CPDs (tablas de probabilidad) en un único archivo de texto."""
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("📊 Tablas de probabilidad condicional (CPDs) aprendidas\n")
        f.write("=" * 80 + "\n\n")

        for cpd in model.get_cpds():
            f.write(f"CPD de {cpd.variable}:\n")
            f.write(str(cpd))
            f.write("\n" + "-" * 80 + "\n")

    print(f"✅ CPDs guardadas en '{path}'")


if __name__ == "__main__":
    model, df = build_and_fit_model(csv_path="main/tv_bn_dataset.csv")
    save_cpds_to_txt(model, path="model_cpds.txt")
    
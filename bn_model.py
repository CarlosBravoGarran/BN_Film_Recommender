import pandas as pd
from pgmpy.estimators import HillClimbSearch, BIC, BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

# Cargar el dataset sintético
df = pd.read_csv("tv_bn_dataset.csv")

# Aprendizaje de la estructura (Hill Climb + BIC)
hc = HillClimbSearch(df)
best_model = hc.estimate(scoring_method=BIC(df))

# Guardar arcos en CSV para referencia
with open("best_model_edges.csv", "w", encoding="utf-8") as f:
    f.write("source,target\n")
    for u, v in best_model.edges():
        f.write(f"{u},{v}\n")

print("Estructura aprendida (arcos):")
for edge in best_model.edges():
    print(edge)

# Crear el modelo y ajustar probabilidades (CPDs)
model = DiscreteBayesianNetwork(best_model.edges())
model.fit(df, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)

# Visualizar el grafo con networkx
G = nx.DiGraph()
G.add_edges_from(model.edges())

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", 
        font_size=10, arrowsize=20, edgecolors='black')
plt.title("Red Bayesiana aprendida automáticamente", fontsize=14)
plt.show()

# Inferencia de ejemplo
infer = VariableElimination(model)

evidence = {'EdadUsuario': 'mayor', 'Hora': 'noche', 'DiaSemana': 'fin_semana'}

query = infer.query(
    variables=['GeneroPrograma'],
    evidence=evidence
)

print("\nGénero recomendado para usuario %s por la %s en %s:" % (evidence["EdadUsuario"], evidence["Hora"], evidence["DiaSemana"]))
print(query)

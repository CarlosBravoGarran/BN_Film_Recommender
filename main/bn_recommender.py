from inference import load_model, query_model

model = load_model("model.pkl")

def recomendar_generos_bn(evidencia: dict, top_k=3):
    res = query_model(model, variables=["GeneroPrograma"], evidence=evidencia)
    dist = res["GeneroPrograma"]
    valores = dist.state_names["GeneroPrograma"]
    probs = dist.values

    # Ordenar por prob descendente
    pares = sorted(
        zip(valores, probs),
        key=lambda x: -x[1]
    )

    # Return lista ordenada de géneros
    return pares[:top_k]

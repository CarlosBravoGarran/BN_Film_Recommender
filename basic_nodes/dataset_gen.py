import pandas as pd
import numpy as np

n = 1000
np.random.seed(42)

# 1) Variables básicas
edad = np.random.choice(['joven', 'adulto', 'mayor'], n, p=[0.4, 0.4, 0.2])
hora = np.random.choice(['mañana', 'tarde', 'noche'], n, p=[0.3, 0.4, 0.3])
dia = np.random.choice(['laboral', 'fin_semana'], n, p=[0.7, 0.3])

generos = []
for e, h, d in zip(edad, hora, dia):
    if e == 'mayor' and h == 'mañana':
        g = np.random.choice(['noticias', 'película'], p=[0.7, 0.3])
    elif e == 'joven' and h == 'noche':
        g = np.random.choice(['entretenimiento', 'película'], p=[0.6, 0.4])
    elif d == 'fin_semana':
        g = np.random.choice(['película', 'entretenimiento'], p=[0.5, 0.5])
    else:
        g = np.random.choice(['noticias', 'entretenimiento', 'película'], p=[0.4, 0.4, 0.2])
    generos.append(g)

# 2) Satisfacción depende del género y edad
satisf = []
for e, g in zip(edad, generos):
    if (e == 'mayor' and g == 'noticias') or (e == 'joven' and g == 'entretenimiento'):
        s = np.random.choice(['alta', 'media', 'baja'], p=[0.6, 0.3, 0.1])
    elif g == 'película':
        s = np.random.choice(['alta', 'media', 'baja'], p=[0.5, 0.3, 0.2])
    else:
        s = np.random.choice(['alta', 'media', 'baja'], p=[0.3, 0.5, 0.2])
    satisf.append(s)

# 3) Recomendado según satisfacción
recom = []
for s in satisf:
    if s == 'alta':
        r = np.random.choice(['sí', 'no'], p=[0.9, 0.1])
    elif s == 'media':
        r = np.random.choice(['sí', 'no'], p=[0.5, 0.5])
    else:
        r = np.random.choice(['sí', 'no'], p=[0.2, 0.8])
    recom.append(r)

df = pd.DataFrame({
    'EdadUsuario': edad,
    'Hora': hora,
    'DiaSemana': dia,
    'GeneroPrograma': generos,
    'Satisfaccion': satisf,
    'Recomendado': recom
})

df.to_csv("basic_nodes/tv_bn_dataset.csv", index=False)
print(df.head())

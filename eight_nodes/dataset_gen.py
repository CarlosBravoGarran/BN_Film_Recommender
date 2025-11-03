import pandas as pd
import numpy as np

n = 10000
np.random.seed(42)

# 1️⃣ Variables básicas
edad = np.random.choice(['joven', 'adulto', 'mayor'], n, p=[0.4, 0.4, 0.2])
hora = np.random.choice(['mañana', 'tarde', 'noche'], n, p=[0.3, 0.4, 0.3])
dia = np.random.choice(['laboral', 'fin_semana'], n, p=[0.7, 0.3])

# 2️⃣ Variables nuevas
duracion = np.random.choice(['corta', 'media', 'larga'], n, p=[0.4, 0.4, 0.2])
interes_previo = np.random.choice(['noticias', 'entretenimiento', 'película'], n, p=[0.3, 0.4, 0.3])

# 3️⃣ Género del programa (influenciado por edad, hora, día, interés previo y duración)
generos = []
for e, h, d, i, dur in zip(edad, hora, dia, interes_previo, duracion):
    if i == 'noticias':
        g = np.random.choice(['noticias', 'entretenimiento'], p=[0.7, 0.3])
    elif i == 'película':
        g = np.random.choice(['película', 'entretenimiento'], p=[0.7, 0.3])
    elif e == 'mayor' and h == 'mañana':
        g = np.random.choice(['noticias', 'película'], p=[0.7, 0.3])
    elif e == 'joven' and h == 'noche':
        g = np.random.choice(['entretenimiento', 'película'], p=[0.6, 0.4])
    elif d == 'fin_semana':
        g = np.random.choice(['película', 'entretenimiento'], p=[0.5, 0.5])
    elif dur == 'corta':
        g = np.random.choice(['noticias', 'entretenimiento'], p=[0.4, 0.6])
    elif dur == 'larga':
        g = np.random.choice(['película', 'entretenimiento'], p=[0.7, 0.3])
    else:
        g = np.random.choice(['noticias', 'entretenimiento', 'película'], p=[0.4, 0.4, 0.2])
    generos.append(g)

# 4️⃣ Satisfacción depende del género, edad e interés previo
satisf = []
for e, g, i, dur in zip(edad, generos, interes_previo, duracion):
    if g == i:
        base = [0.7, 0.2, 0.1]  # coincide con el interés → alta satisfacción
    elif (e == 'mayor' and g == 'noticias') or (e == 'joven' and g == 'entretenimiento'):
        base = [0.6, 0.3, 0.1]
    elif g == 'película' and dur == 'larga':
        base = [0.6, 0.3, 0.1]
    elif dur == 'corta' and g == 'película':
        base = [0.3, 0.4, 0.3]
    else:
        base = [0.4, 0.4, 0.2]
    s = np.random.choice(['alta', 'media', 'baja'], p=base)
    satisf.append(s)

# 5️⃣ Recomendado según satisfacción
recom = []
for s in satisf:
    if s == 'alta':
        r = np.random.choice(['sí', 'no'], p=[0.9, 0.1])
    elif s == 'media':
        r = np.random.choice(['sí', 'no'], p=[0.5, 0.5])
    else:
        r = np.random.choice(['sí', 'no'], p=[0.2, 0.8])
    recom.append(r)

# 6️⃣ Crear el dataframe final
df = pd.DataFrame({
    'EdadUsuario': edad,
    'Hora': hora,
    'DiaSemana': dia,
    'DuracionPrograma': duracion,
    'InteresPrevio': interes_previo,
    'GeneroPrograma': generos,
    'Satisfaccion': satisf,
    'Recomendado': recom
})

df.to_csv("eight_nodes/tv_bn_dataset.csv", index=False)
print("✅ Dataset generado con éxito")
print(df.head())

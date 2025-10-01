import pandas as pd

# Exemplo de dados (você pode substituir pelos seus)
dados = {
    "ano": [2017, 2020, 2021, 2022, 2024],
    "valor": [1299, 1265, 652, 1265, 1300]  # variável qualquer
}

df = pd.DataFrame(dados)
df = df.set_index("ano")

print("Original:")
print(df)

# Recriar o índice contínuo de anos (2017 até 2024)
df = df.reindex(range(2017, 2025))

# Interpolação linear
df["valor"] = df["valor"].interpolate(method="linear")

print("Com interpolação:")
print(df)

import pandas as pd

# Lista de arquivos txt
arquivos_txt = ["Pos_X.txt", "Pos_Y.txt", "Best_x.txt", "Best_y.txt"]

# Ler cada arquivo e armazenar em uma lista de DataFrames
dfs = [pd.read_table(arquivo) for arquivo in arquivos_txt]

# Verificar se cada DataFrame tem uma única coluna
for i, df in enumerate(dfs):
    if df.shape[1] != 1:
        raise ValueError(f"O arquivo {arquivos_txt[i]} deve ter apenas uma coluna.")

# Concatenar os DataFrames lado a lado (colunas)
df_combined = pd.concat(dfs, axis=1)

# Renomear as colunas (opcional, mas recomendado)
df_combined.columns = ['Pos_X', 'Pos_Y', 'Best_x', 'Best_y']

# Visualizar o DataFrame resultante
print(df_combined)

# Salvar o DataFrame em um arquivo CSV
df_combined.to_csv('df_final.csv', index=False)

# # Parte apenas para teste e verificacao do csv gerado
# df_loaded = pd.read_csv('df_final.csv')
# print(df_loaded['Best_y'])


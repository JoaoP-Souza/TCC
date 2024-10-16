import pandas as pd
import numpy as np

def process_file(filename, prefix):
    # Ler o arquivo TXT
    df = pd.read_csv(filename, sep=',', header=None)

    # Certificar-se de que há exatamente 1000 valores
    total_values = df.shape[0] * df.shape[1]
    if total_values != 1000:
        print(f"Erro: O arquivo {filename} tem {total_values} valores, mas são necessários 1000.")
        return None

    # Transformar os dados em uma matriz de 10 colunas, com 100 valores em cada coluna
    data = df.values.flatten()
    reshaped_data = data.reshape(100, 10)

    # Criar um novo DataFrame com essas 10 colunas
    new_df = pd.DataFrame(reshaped_data, columns=[f'{prefix}{i+1}' for i in range(10)])

    return new_df

# Processar Pos_X e Pos_Y
pos_x_df = process_file('Pos_X.txt', 'Pos_X')
pos_y_df = process_file('Pos_Y.txt', 'Pos_Y')

# Processar Best_X e Best_Y como colunas únicas, removendo espaços ou vírgulas extras
best_x = pd.read_csv('Best_X.txt', delimiter='\t', header=None, names=['Best_X'])
best_y = pd.read_csv('Best_Y.txt', delimiter='\t', header=None, names=['Best_Y'])

# Remove any commas or spaces that might be in the values
best_x['Best_X'] = best_x['Best_X'].str.replace(',', '').astype(float)
best_y['Best_Y'] = best_y['Best_Y'].str.replace(',', '').astype(float)

# Certificar que não há valores nulos antes de mesclar
if pos_x_df is not None and pos_y_df is not None:
    # Concatenar todos os DataFrames com base no eixo das colunas (axis=1)
    combined_df = pd.concat([best_x, best_y, pos_x_df, pos_y_df], axis=1)

    # Salvar o DataFrame combinado em um arquivo CSV final
    combined_df.to_csv('df_final.csv', index=False)
    print("Arquivo 'df_final.csv' salvo com sucesso!")
else:
    print("Erro: Um ou mais arquivos não foram processados corretamente.")

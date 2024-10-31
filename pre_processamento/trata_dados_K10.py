import pandas as pd

def process_file(filename, prefix):
    # Ler o arquivo TXT e transformar em uma matriz de 10 colunas
    df = pd.read_csv(filename, sep=',', header=None)

    # Achatar e redimensionar os dados para uma matriz com 11.000 linhas e 10 colunas
    data = df.values.flatten()
    reshaped_data = data.reshape(10000, 10)

    # Criar um DataFrame com as 10 colunas, nomeadas com o prefixo fornecido
    new_df = pd.DataFrame(reshaped_data, columns=[f'{prefix}{i+1}' for i in range(10)])

    return new_df

# Processar os arquivos Pos_X e Pos_Y
pos_x_df = process_file('datasets_txt/K10/Pos_X 1.txt', 'Pos_X')
pos_y_df = process_file('datasets_txt/K10/Pos_Y 1.txt', 'Pos_Y')

# Processar Best_X e Best_Y como colunas únicas, removendo espaços ou vírgulas extras
best_x = pd.read_csv('datasets_txt/K10/Best_X 1.txt', delimiter='\t', header=None, names=['Best_X'])
best_y = pd.read_csv('datasets_txt/K10/Best_Y 1.txt', delimiter='\t', header=None, names=['Best_Y'])

# # Remover vírgulas ou espaços extras nos valores
# best_x['Best_X'] = best_x['Best_X'].str.replace(',', '').astype(float)
# best_y['Best_Y'] = best_y['Best_Y'].str.replace(',', '').astype(float)

# Concatenar todos os DataFrames em um único DataFrame final
combined_df = pd.concat([best_x, best_y, pos_x_df, pos_y_df], axis=1)

# Salvar o DataFrame combinado em um arquivo CSV final
combined_df.to_csv('df_final_K10.csv', index=False)
print("Arquivo 'df_final.csv' salvo com sucesso!")

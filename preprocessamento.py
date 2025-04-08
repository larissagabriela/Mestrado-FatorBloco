import pandas as pd
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split

def carregar_imagem(caminho_imagem, tamanho=(128, 128)):
    img = Image.open(caminho_imagem)
    img = img.convert("RGB")  
    img = img.resize(tamanho)  
    img_array = np.array(img) 
    return img_array

def carregar_dados(caminho_txt, diretorio_imagens):
    dados = pd.read_csv(caminho_txt, dtype={'imagem': str, 'fator_bloco': float})
    
    caminhos_imagens = dados['imagem'].tolist()
    fatores_blocos = dados['fator_bloco'].tolist()

    imagens_processadas = [carregar_imagem(os.path.join(diretorio_imagens, img)) for img in caminhos_imagens]
    
    imagens_processadas = np.array(imagens_processadas)
    
    return imagens_processadas, np.array(fatores_blocos, dtype=np.float32)

def dividir_dados(imagens, fatores_blocos):
    X_train, X_test, y_train, y_test = train_test_split(imagens, fatores_blocos, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

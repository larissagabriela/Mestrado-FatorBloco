import pandas as pd
import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras import backend as K
K.clear_session()

# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

from preprocessamento import carregar_dados, dividir_dados

caminho_txt = "Algoritmo_blocos/testebloco.txt"
diretorio_imagens = "Algoritmo_blocos/png_teste2"

imagens_processadas, fatores_blocos = carregar_dados(caminho_txt, diretorio_imagens)

X_train, X_val, X_test, y_train, y_val, y_test = dividir_dados(imagens_processadas, fatores_blocos)

# Normalização das imagens (escala para [0,1])
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')  
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val))

loss = model.evaluate(X_test, y_test)
print(f"Perda no conjunto de teste: {loss}")

predicoes = model.predict(X_test)

for i in range(len(y_test)):
    print(f"Previsão: {predicoes[i][0]:.4f}, Real: {y_test[i]:.4f}")

# Garantir que predicoes seja um vetor unidimensional
predicoes = np.ravel(predicoes)

# Calcular os resíduos
residuos = y_test - predicoes

# Gráfico de Perda durante o treinamento e validação
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Gráfico 1: Perda de Treinamento vs Validação
axs[0, 0].plot(history.history['loss'], label='Perda de Treinamento')
axs[0, 0].plot(history.history['val_loss'], label='Perda de Validação')
axs[0, 0].set_title('Perda durante o Treinamento')
axs[0, 0].set_xlabel('Épocas')
axs[0, 0].set_ylabel('Perda')
axs[0, 0].legend()

# Gráfico 2: Previsões vs Valores Reais
axs[0, 1].scatter(y_test, predicoes)
axs[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axs[0, 1].set_title('Previsões vs Valores Reais')
axs[0, 1].set_xlabel('Valores Reais')
axs[0, 1].set_ylabel('Previsões')

# Gráfico 3: Distribuição dos Resíduos
axs[1, 0].hist(residuos, bins=20, edgecolor='black')
axs[1, 0].set_title('Distribuição dos Resíduos')
axs[1, 0].set_xlabel('Erro')
axs[1, 0].set_ylabel('Frequência')

# Gráfico 4: Resíduos vs Previsões
axs[1, 1].scatter(predicoes, residuos)
axs[1, 1].axhline(y=0, color='k', linestyle='--')
axs[1, 1].set_title('Resíduos vs Previsões')
axs[1, 1].set_xlabel('Previsões')
axs[1, 1].set_ylabel('Resíduos')

# Ajustar o layout
plt.tight_layout()

# Mostrar todos os gráficos em uma figura
plt.show()
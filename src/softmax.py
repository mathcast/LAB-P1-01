import numpy as np

def softmax_por_linha(matriz_entrada: np.ndarray) -> np.ndarray:
    maximo_por_linha = np.max(matriz_entrada, axis=1, keepdims=True)
    exponencial = np.exp(matriz_entrada - maximo_por_linha)
    soma_por_linha = np.sum(exponencial, axis=1, keepdims=True)
    return exponencial / soma_por_linha
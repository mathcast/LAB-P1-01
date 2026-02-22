import numpy as np

from .softmax import softmax_por_linha

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    quantidade_queries, dimensao_chaves = Q.shape
    quantidade_keys, dimensao_chaves_K = K.shape
    quantidade_linhas_V, dimensao_valores = V.shape

    if dimensao_chaves != dimensao_chaves_K:
        raise ValueError(
            f"Dimensão das chaves deve ser igual em Q e K: Q tem {dimensao_chaves}, K tem {dimensao_chaves_K}."
        )
    if quantidade_keys != quantidade_linhas_V:
        raise ValueError(
            f"K e V devem ter o mesmo número de linhas: K tem {quantidade_keys}, V tem {quantidade_linhas_V}."
        )

    produto_escalar_QK = Q @ K.T

    fator_escalar = np.sqrt(dimensao_chaves)
    produto_escalar_escalado = produto_escalar_QK / fator_escalar

    pesos_softmax = softmax_por_linha(produto_escalar_escalado)

    saida_atenção = pesos_softmax @ V

    return saida_atenção
from __future__ import annotations

import numpy as np

def gerar_QKV(
    quantidade_queries: int,
    dimensao_chaves: int,
    quantidade_keys: int,
    dimensao_valores: int,
    semente: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(semente)
    Q = rng.standard_normal((quantidade_queries, dimensao_chaves))
    K = rng.standard_normal((quantidade_keys, dimensao_chaves))
    V = rng.standard_normal((quantidade_keys, dimensao_valores))
    return Q, K, V
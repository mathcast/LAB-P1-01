import numpy as np

from src import scaled_dot_product_attention, softmax_por_linha
from src.dados_teste import gerar_QKV

def testar_e_exibir():
    quantidade_queries = 2
    dimensao_chaves = 2
    quantidade_keys = 2
    dimensao_valores = 2
    semente = 42

    Q, K, V = gerar_QKV(
        quantidade_queries=quantidade_queries,
        dimensao_chaves=dimensao_chaves,
        quantidade_keys=quantidade_keys,
        dimensao_valores=dimensao_valores,
        semente=semente,
    )

    K_transposta = K.T
    produto_escalar_QK = Q @ K_transposta
    fator_escalar = np.sqrt(dimensao_chaves)
    produto_escalar_escalado = produto_escalar_QK / fator_escalar
    pesos_softmax = softmax_por_linha(produto_escalar_escalado)
    saida_atenção = pesos_softmax @ V

    np.set_printoptions(precision=4, suppress=True)
    print("=== 1. Matriz Q ===")
    print(Q)
    print("\n=== 2. Matriz K transposta (K^T) ===")
    print(K_transposta)
    print("\n=== 3. Resultado do produto escalar (Q @ K^T) ===")
    print(produto_escalar_QK)
    print(f"\n=== 4. Aplicando o fator escalar ===")
    print(produto_escalar_escalado)
    print("\n=== 5. Após o softmax ===")
    print(pesos_softmax)
    print("\n=== 6. Matriz V ===")
    print(V)
    print("\n=== 7. Multiplicando pesos_softmax @ V → resultado final ===")
    print(saida_atenção)

if __name__ == "__main__":
    testar_e_exibir()
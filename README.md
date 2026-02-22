# LAB P1-01: Implementação do Mecanismo de Self-Attention

## Como baixar o projeto

```bash
git clone https://github.com/mathcast/LAB-P1-01.git
cd LAB-P1-01
```

## Fórmula implementada

```
Attention(Q, K, V) = softmax((Q K^T) / √dimensao_chaves) V
```

- **Q** (Query), **K** (Key) e **V** (Value) são matrizes de entrada.
- O **produto escalar** Q @ K^T é calculado e depois **dividido pelo fator escalar** (raiz de `dimensao_chaves`, que é o número de colunas de Q e K).
- O **softmax é aplicado em cada linha** dessa matriz.
- O resultado (`pesos_softmax`) é **multiplicado pela matriz V**, obtendo a saída (`saida_atenção`).

## Estrutura do repositório

```
LAB P1-01/
├── src/
│   ├── __init__.py      # Exporta scaled_dot_product_attention e softmax_por_linha
│   ├── attention.py     # Implementação da atenção (produto escalar, fator escalar, softmax, multiplicação por V)
│   ├── dados_teste.py   # Geração aleatória de Q, K, V (gerar_QKV com tamanhos configuráveis)
│   └── softmax.py       # Softmax por linha (estável numericamente)
├── test_attention.py    # Gera Q,K,V, calcula cada etapa e exibe passo a passo
├── requirements.txt     # Dependências
└── README.md
```

## Como rodar

### 1. Ambiente virtual e dependências

No diretório do projeto (após clonar):

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Executar o teste

```powershell
python test_attention.py
```

O script gera as matrizes Q, K e V de forma aleatória (com semente fixa), calcula cada etapa da fórmula e **exibe**:

1. Matriz Q  
2. Matriz K transposta (K^T)  
3. Resultado do produto escalar (Q @ K^T)  
4. Resultado após aplicar o fator escalar  
5. Resultado após o softmax (cada linha soma 1)  
6. Matriz V  
7. Resultado final (`pesos_softmax` @ V)

### 3. Usar no seu código

```python
from src import scaled_dot_product_attention
from src.dados_teste import gerar_QKV

Q, K, V = gerar_QKV(
    quantidade_queries=2,
    dimensao_chaves=64,
    quantidade_keys=10,
    dimensao_valores=32,
    semente=42,
)
saida_atenção = scaled_dot_product_attention(Q, K, V)
```

Para alterar o tamanho das matrizes, use os parâmetros `quantidade_queries`, `dimensao_chaves`, `quantidade_keys` e `dimensao_valores` em `gerar_QKV`.

## Requisitos técnicos

- **Linguagem:** Python 3  
- **Dependência:** apenas NumPy 

---

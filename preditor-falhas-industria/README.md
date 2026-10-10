# Preditor de Falhas Industriais — Manutenção Preditiva

Projeto de análise de dados e aprendizado de máquina para prever falhas em equipamentos industriais a partir de variáveis de operação, como temperatura, rotação, torque e desgaste da ferramenta.

## Problema que resolve

Falhas inesperadas podem interromper a produção, elevar custos de manutenção e comprometer prazos. Ao identificar padrões associados a falhas, a fábrica pode priorizar inspeções e planejar intervenções antes que uma parada não programada aconteça.

## Técnicas e tecnologias

- Python, pandas e NumPy — preparação e análise dos dados.
- matplotlib e seaborn — visualizações exploratórias.
- scikit-learn — divisão treino/teste, `StandardScaler`, KNN, Árvore de Decisão e métricas.
- imbalanced-learn — SMOTE para balancear a classe minoritária somente no conjunto de treino.

## Etapas do pipeline

1. **Análise exploratória:** análise das distribuições, relações entre variáveis e proporção de falhas; a base original tem 10.000 registros e cerca de 3,4% de falhas.
2. **Limpeza:** remoção de duplicatas; valores extremos plausíveis são mantidos e registros com dados ausentes não são descartados.
3. **Feature engineering:** criação da variável de potência mecânica a partir do torque e da rotação; transformação da variável categórica `tipo` em colunas numéricas.
4. **Divisão, imputação e balanceamento:** separação estratificada em treino e teste (80/20); imputação dos valores ausentes pela mediana calculada no treino e aplicada também ao teste; aplicação do SMOTE somente ao treino imputado.
5. **Escalonamento:** aplicação do `StandardScaler` aos atributos do KNN, ajustado no treino e aplicado ao teste; a árvore usa os dados sem escalonamento.
6. **Ajuste e overfitting:** comparação de KNN com `K` 3, 5 e 7 e de árvores com profundidade 3, 5 e ilimitada; comparação das acurácias de treino e teste para observar possível sobreajuste.
7. **Veredito:** a árvore ilimitada obteve a maior acurácia entre os modelos testados, mas nenhum modelo foi aprovado, pois não superou o baseline de prever sempre “sem falha”.

## Como executar

Clone o repositório e entre na pasta do projeto:

```bash
git clone https://github.com/serenaclariceb-commits/DESENVOLVIMENTO-DE-IA-PARA-ANALISE-PREDITIVA.git
cd DESENVOLVIMENTO-DE-IA-PARA-ANALISE-PREDITIVA/preditor-falhas-industria
```

Crie e ative um ambiente virtual. No Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

No Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Instale as dependências e execute o script:

```bash
python -m pip install -r requirements.txt
python modelo_falhas.py
```

O CSV deve estar em `data/manutencao_preditiva.csv`. Para reproduzir a análise exploratória e as comparações documentadas, abra `projeto.ipynb` no VS Code, selecione o kernel do ambiente `.venv` e execute as células.

## Resultados

Melhores resultados observados no notebook:

| Modelo | Melhor parâmetro | Acurácia de treino | Acurácia de teste |
|---|---:|---:|---:|
| KNN | `K = 3` | 98,32% | 94,00% |
| Árvore de Decisão | `max_depth = None` (sem limite) | 99,84% | 95,35% |
| Baseline: sempre prever “sem falha” | — | — | 96,60% |

Na classe de falha, o KNN (`K = 3`) obteve precision de 29,69%, recall de 55,88% e F1-score de 38,78%. A árvore sem limite obteve precision de 38,10%, recall de 58,82% e F1-score de 46,24%.

**Modelo escolhido:** nenhum por enquanto. A árvore sem limite teve a maior acurácia de teste entre os modelos avaliados, mas ficou abaixo do baseline simples. Ela identificou 40 das 68 falhas no conjunto de teste, mas também classificou 65 registros sem falha como falhas. A diferença de 4,49 pontos percentuais entre treino e teste sugere possível overfitting; a escolha final deve considerar o custo de falsos alarmes junto com o recall.

## Melhorias futuras

- Comparar limiares de decisão e estratégias de balanceamento conforme o custo relativo de falhas não detectadas e falsos alarmes.
- Usar validação cruzada estratificada para obter uma estimativa mais robusta do desempenho.
- Testar outros modelos e estratégias de balanceamento, comparando-os com o baseline e considerando os custos operacionais.

## Vídeo de apresentação

Link do Google Drive:

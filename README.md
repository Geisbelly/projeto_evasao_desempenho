# Evasão e desempenho acadêmico no ensino superior

Projeto prático de Machine Learning — 2026/2

## Pergunta de pesquisa
Qual é a relação entre o desempenho acadêmico no primeiro semestre e a situação acadêmica posterior dos estudantes, e quanto esse desempenho acrescenta à capacidade preditiva de modelos de Machine Learning?

## Base principal
**Predict Students' Dropout and Academic Success**, UCI Machine Learning Repository (ID 697). A base contém 4.424 estudantes e 36 atributos preditores, com três classes-alvo: `Dropout`, `Enrolled` e `Graduate`.

- Página oficial: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- O script usa `ucimlrepo` para obter os dados. Consulte a licença e a citação exigida na página oficial antes de redistribuir a base.

## Objetivos
1. Explorar a distribuição das classes e as variáveis acadêmicas.
2. Investigar a associação entre indicadores do primeiro semestre e a situação acadêmica.
3. Comparar modelos usando apenas informações de ingresso com modelos que também usam desempenho do primeiro semestre.
4. Comparar quatro algoritmos: Regressão Logística, Árvore de Decisão, Random Forest e SVM.
5. Avaliar com validação cruzada, macro-F1, recall por classe, matriz de confusão e outras métricas.

## Estrutura
```text
.
├── README.md
├── RELATORIO.md
├── DIARIO.md
├── requirements.txt
├── notebooks/
│   └── 01_analise_e_modelagem.ipynb
├── src/
│   └── run_experiments.py
└── results/
    └── README.md
```

## Como executar
Recomendado: Python 3.10–3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
python src/run_experiments.py
```

O primeiro acesso requer conexão à internet para obter a base pelo pacote `ucimlrepo`. Os resultados são gravados em `results/`. O script não inventa resultados: as métricas só são produzidas após a execução.

Para executar o notebook:
```bash
jupyter notebook
```
Abra `notebooks/01_analise_e_modelagem.ipynb` e execute as células em ordem.

## Desenho metodológico
- O alvo original é multiclasse: evasão, matrícula ativa e conclusão.
- A avaliação principal usa divisão estratificada treino/teste.
- A validação cruzada e a busca de hiperparâmetros ocorrem somente no conjunto de treino.
- O conjunto de teste é reservado para a avaliação final.
- Os pipelines ajustam imputação, codificação e escala dentro de cada partição da validação cruzada, reduzindo vazamento de dados.
- São comparados dois cenários temporais: (A) variáveis de ingresso; (B) variáveis de ingresso + indicadores do primeiro semestre.
- Variáveis do segundo semestre são excluídas dos experimentos preditivos para que a previsão não dependa de informações posteriores ao momento proposto.

## Limitações
A base é de uma instituição de Portugal e não representa automaticamente estudantes de todas as instituições ou do Brasil. Associações não demonstram causalidade. O uso de atributos sensíveis ou socioeconômicos exige cuidado e não se deve usar a previsão como rótulo definitivo de um estudante.

## Entrega acadêmica
Antes de enviar, o grupo deve:
1. Confirmar que a base não foi usada em aula nem escolhida por outro grupo.
2. Executar o código e inserir resultados reais no relatório.
3. Registrar no `DIARIO.md` as sessões de trabalho reais, incluindo falhas e decisões.
4. Adicionar os nomes dos integrantes, turma, data e a referência bibliográfica.
5. Revisar o trabalho e a apresentação de 10 minutos.

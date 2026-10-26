# Predição da evasão no ensino superior: relação com desempenho acadêmico e modelos de Machine Learning

**Disciplina:** Projeto Prático — Machine Learning  
**Turma:** 16938 — confirmar identificação da turma  
**Período letivo:** 2026/2  
**Integrantes:** Geisbelly, Maria Antonia e Kevin  
**Data:** 26-10-2026

## Resumo

Este trabalho investiga a relação entre o desempenho acadêmico no primeiro semestre e a situação acadêmica dos estudantes do ensino superior, buscando avaliar se indicadores de desempenho acrescentam capacidade preditiva à identificação de estudantes que evadem, permanecem matriculados ou concluem seus cursos. Para isso, será utilizada a base *Predict Students' Dropout and Academic Success*, disponibilizada pelo UCI Machine Learning Repository. A metodologia contempla análise exploratória, preparação dos dados, comparação entre dois cenários de previsão e treinamento de quatro modelos de classificação: Regressão Logística, Árvore de Decisão, Random Forest e Support Vector Machine (SVM). A avaliação será realizada por validação cruzada no conjunto de treinamento e por uma avaliação final em conjunto de teste reservado, utilizando métricas adequadas à classificação multiclasse, com destaque para macro-F1 e recall da classe de evasão. Os resultados permitirão discutir a contribuição dos indicadores acadêmicos, as limitações dos modelos e os cuidados necessários para a utilização responsável de previsões no contexto educacional.

**Palavras-chave:** evasão acadêmica; ensino superior; desempenho acadêmico; Machine Learning; classificação.

> **Nota de atualização:** o resumo deverá ser revisado após a execução dos experimentos, incorporando os resultados reais e as principais conclusões obtidas.

## 1. Introdução

A evasão no ensino superior constitui um desafio para estudantes e instituições de ensino, com possíveis consequências para a trajetória acadêmica, o planejamento institucional e a continuidade da formação. Compreender os fatores associados à permanência e à conclusão dos cursos pode contribuir para o desenvolvimento de estratégias de acompanhamento e apoio aos estudantes.

Nesse contexto, técnicas de Machine Learning podem ser utilizadas para identificar padrões nos dados acadêmicos e estimar a situação de estudantes a partir de características observadas. Entretanto, a capacidade preditiva de um modelo não significa que ele compreenda integralmente a trajetória individual de cada estudante, tampouco que consiga explicar as causas da evasão.

O desempenho acadêmico inicial é uma dimensão relevante para investigação, pois pode estar associado à adaptação à vida universitária, à progressão curricular e à situação acadêmica posterior. Ainda assim, essa relação deve ser analisada considerando outros fatores disponíveis nos dados e reconhecendo que o desempenho, isoladamente, não explica todos os motivos que levam um estudante a interromper seus estudos.

Diante disso, este trabalho propõe investigar a relação entre indicadores acadêmicos do primeiro semestre e a situação acadêmica dos estudantes, comparando modelos de classificação e avaliando se a inclusão dessas informações melhora o desempenho preditivo em relação a um cenário baseado em características disponíveis no ingresso.

## 2. Problema de pesquisa e objetivos

### 2.1 Problema de pesquisa

A identificação de padrões associados à evasão pode contribuir para compreender a trajetória acadêmica dos estudantes. Entretanto, é necessário avaliar se os indicadores de desempenho do primeiro semestre acrescentam informações úteis à previsão da situação acadêmica posterior e se essa contribuição se mantém quando diferentes modelos de Machine Learning são comparados.

### 2.2 Pergunta de pesquisa

**Qual é a relação entre o desempenho acadêmico no primeiro semestre e a situação acadêmica posterior dos estudantes do ensino superior, e até que ponto os indicadores desse período acrescentam capacidade preditiva aos modelos de Machine Learning?**

### 2.3 Objetivo geral

Desenvolver e comparar modelos de classificação para analisar a situação acadêmica dos estudantes do ensino superior e avaliar a contribuição preditiva dos indicadores de desempenho do primeiro semestre.

### 2.4 Objetivos específicos

- Examinar a estrutura, a qualidade e a distribuição dos dados disponíveis.
- Descrever a distribuição das categorias de situação acadêmica.
- Investigar diferenças descritivas nos indicadores de desempenho do primeiro semestre entre estudantes que evadiram, permaneceram matriculados ou concluíram o curso.
- Comparar um cenário de previsão baseado em informações disponíveis no ingresso com outro que também utiliza indicadores acadêmicos do primeiro semestre.
- Treinar e comparar quatro modelos de classificação.
- Ajustar os hiperparâmetros de pelo menos dois modelos.
- Avaliar os modelos por meio de validação cruzada e de um conjunto de teste reservado.
- Discutir os resultados, as limitações metodológicas e os riscos associados ao uso de previsões no contexto educacional.

## 3. Base de dados

### 3.1 Fonte principal

Será utilizada como fonte principal a base *Predict Students' Dropout and Academic Success*, disponibilizada pelo UCI Machine Learning Repository.

A documentação da base informa **4.424 instâncias e 36 atributos preditores**, além da variável-alvo que representa a situação acadêmica do estudante. As categorias disponíveis são:

- **Dropout:** estudante classificado como evadido.
- **Enrolled:** estudante que permanece matriculado.
- **Graduate:** estudante que concluiu o curso.

A base reúne informações relacionadas ao contexto acadêmico e a características dos estudantes. Sua documentação deverá ser consultada para identificar o significado de cada atributo, o momento em que a informação é observada e sua adequação aos cenários de previsão definidos neste trabalho.

**Fonte oficial:** [UCI Machine Learning Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).

### 3.2 Justificativa da escolha

A base foi selecionada por apresentar características compatíveis com o problema de pesquisa: informações de estudantes do ensino superior, uma variável-alvo com três categorias e volume de registros superior ao mínimo de 1.000 linhas exigido pela atividade.

Além disso, a presença de informações acadêmicas permite investigar a contribuição dos indicadores de desempenho para a classificação da situação dos estudantes.

Antes da análise definitiva, será necessário confirmar que a base atende aos demais requisitos da disciplina, incluindo a não utilização anterior em aula e a ausência de conflito com a escolha de outro grupo.

### 3.3 Limitações da fonte

Os dados referem-se a uma instituição de ensino superior de Portugal. Portanto, os padrões encontrados podem refletir características específicas dessa instituição, de seus estudantes e de seu contexto educacional.

Os resultados não devem ser generalizados automaticamente para instituições brasileiras ou para outras populações. A aplicação em contextos diferentes exigiria avaliação com dados representativos desses contextos.

### 3.4 Fontes complementares para contextualização

Como apoio à fundamentação do problema, poderão ser consultadas fontes institucionais brasileiras, especialmente:

- **INEP — Indicadores de Fluxo da Educação Superior:** informações relacionadas à trajetória dos estudantes no ensino superior.
- **INEP — Censo da Educação Superior:** dados sobre instituições, cursos, matrículas e outras dimensões do ensino superior brasileiro.
- **Plataforma Nilo Peçanha:** indicadores relacionados à Rede Federal de Educação Profissional, Científica e Tecnológica.

Essas fontes serão utilizadas inicialmente para contextualização e fundamentação, e não como parte do treinamento dos modelos. Uma eventual integração com a base principal dependerá da compatibilidade entre as variáveis, as populações observadas, os períodos e as unidades de análise.

## 4. Metodologia

### 4.1 Abordagem geral

O trabalho será desenvolvido como um estudo quantitativo e experimental de classificação supervisionada. A variável-alvo será a situação acadêmica, composta pelas classes `Dropout`, `Enrolled` e `Graduate`.

O desenvolvimento será organizado nas seguintes etapas:

1. Exploração e compreensão dos dados.
2. Preparação e transformação das variáveis.
3. Definição dos cenários de previsão.
4. Divisão dos dados em conjuntos de treinamento e teste.
5. Treinamento e ajuste dos modelos.
6. Avaliação comparativa.
7. Interpretação dos resultados e discussão das limitações.

As etapas serão documentadas no notebook e nos arquivos do repositório, de modo a favorecer a reprodutibilidade dos experimentos.

### 4.2 Análise exploratória

A análise exploratória terá como objetivo compreender a estrutura dos dados e identificar características relevantes para a modelagem.

Serão investigados:

- Número de linhas e colunas.
- Nomes, tipos e significados das variáveis.
- Existência de valores ausentes e registros duplicados.
- Distribuição das três categorias da variável-alvo.
- Estatísticas descritivas dos atributos numéricos.
- Distribuição dos indicadores acadêmicos do primeiro semestre por categoria.
- Relações descritivas entre notas, aprovações, unidades curriculares e situação acadêmica, conforme as variáveis efetivamente disponíveis.

Os resultados serão apresentados por meio de tabelas e gráficos apropriados.

As diferenças observadas entre as classes serão interpretadas como associações descritivas. Não serão utilizadas, isoladamente, como evidência de relações causais.

### 4.3 Definição dos cenários de previsão

Para investigar a contribuição do desempenho acadêmico, serão comparados dois cenários.

**Cenário A — Informações disponíveis no ingresso**

Utilizará atributos documentados como disponíveis no momento do ingresso ou antes do período de desempenho que se pretende avaliar. A seleção será feita após a inspeção do dicionário oficial da base.

**Cenário B — Informações do ingresso e desempenho do primeiro semestre**

Utilizará os atributos do cenário A acrescidos de indicadores acadêmicos observados durante o primeiro semestre, desde que sua disponibilidade e seu significado sejam confirmados na documentação.

A comparação permitirá avaliar se a inclusão desses indicadores melhora o desempenho preditivo para a classificação da situação acadêmica.

Os cenários representam momentos distintos de previsão. O cenário B poderá ter maior capacidade preditiva por utilizar informações adicionais, mas essas informações somente estarão disponíveis depois que o estudante tiver concluído o período correspondente.

**Controle de vazamento de dados:** os atributos serão selecionados considerando o momento em que a previsão seria realizada. Informações posteriores ao período definido para cada cenário não deverão ser utilizadas. A seleção das variáveis também deverá considerar se algum atributo contém diretamente a situação final do estudante ou informações que revelem indevidamente o resultado que se deseja prever.

### 4.4 Preparação dos dados

A preparação será definida após a análise exploratória, considerando os tipos e as características dos atributos.

Quando necessário, serão aplicadas as seguintes transformações:

- Tratamento de valores ausentes, conforme sua ocorrência e natureza.
- Codificação de variáveis categóricas.
- Padronização de variáveis numéricas para modelos sensíveis à escala.
- Separação entre atributos preditores e variável-alvo.
- Verificação da compatibilidade dos dados com os modelos selecionados.

Será utilizada a estrutura `Pipeline` e `ColumnTransformer` do scikit-learn para organizar as transformações. Dessa forma, as etapas de preparação serão ajustadas exclusivamente com os dados de treinamento de cada partição, reduzindo o risco de vazamento durante a validação cruzada.

A padronização será aplicada quando apropriada, especialmente para a Regressão Logística e o SVM. Modelos baseados em árvores não dependem da padronização da mesma maneira, de modo que o processamento será definido conforme as características de cada algoritmo.

Não será presumida a existência de valores ausentes antes da inspeção da base.

### 4.5 Modelos de Machine Learning

Serão avaliados quatro modelos de classificação.

**1. Regressão Logística**

Será utilizada como modelo linear de referência. Apesar do nome, pode ser empregada em problemas de classificação multiclasse. Seus resultados servirão como base de comparação para os demais algoritmos.

**2. Árvore de Decisão**

Será utilizada para identificar regras de divisão dos dados e permitir uma interpretação relativamente direta dos padrões aprendidos. Será necessário controlar sua complexidade para reduzir o risco de sobreajuste.

**3. Random Forest**

Será utilizado como método de ensemble, combinando múltiplas árvores de decisão. A proposta é avaliar se essa combinação melhora o desempenho em relação a um modelo baseado em uma única árvore.

**4. Support Vector Machine (SVM)**

Será utilizado como modelo adicional para comparação. Como seu desempenho pode depender da escala dos atributos e da escolha dos hiperparâmetros, o processamento e os ajustes serão realizados de forma compatível com o algoritmo.

### 4.6 Divisão dos dados e validação

Os dados serão divididos em conjuntos de treinamento e teste, utilizando divisão estratificada para preservar, tanto quanto possível, a proporção das classes em cada conjunto.

O conjunto de teste permanecerá reservado até a etapa final de avaliação. A escolha dos modelos e o ajuste dos hiperparâmetros serão realizados exclusivamente com os dados de treinamento.

Será utilizada validação cruzada estratificada para comparar os modelos e estimar a variabilidade do desempenho entre as partições.

A busca de hiperparâmetros será aplicada a pelo menos dois modelos, com os espaços de busca e os critérios de seleção documentados no notebook.

A semente aleatória será fixada para favorecer a reprodutibilidade. A configuração exata da divisão e da validação será registrada nos experimentos.

### 4.7 Métricas de avaliação

A avaliação contemplará métricas adequadas à classificação multiclasse:

- **Macro-F1:** média do F1-score calculado separadamente para cada classe, atribuindo o mesmo peso a todas elas.
- **Precisão por classe:** proporção de previsões positivas de uma classe que estão corretas.
- **Recall por classe:** proporção de exemplos reais de uma classe que foram identificados corretamente.
- **Acurácia balanceada:** média do recall entre as classes.
- **Matriz de confusão:** apresenta as classificações corretas e os erros cometidos entre as categorias.
- **ROC-AUC multiclasse:** poderá ser apresentada se sua utilização for adequada à configuração dos modelos e à forma de obtenção das probabilidades ou dos escores.

O **macro-F1 será adotado como métrica principal de comparação**, pois atribui a mesma importância a cada classe, independentemente de sua frequência. O recall da classe `Dropout` também será destacado, considerando o interesse específico em identificar estudantes classificados como evadidos.

As métricas de validação cruzada e de teste serão apresentadas separadamente. O desempenho final no teste não será utilizado para ajustar repetidamente os modelos.

### 4.8 Registro e reprodutibilidade

O desenvolvimento será organizado em um repositório GitHub contendo:

- `README.md`: descrição do projeto, objetivo, fonte de dados, requisitos e instruções de execução.
- `RELATORIO.md`: documentação do problema, metodologia, resultados e conclusões.
- `DIARIO.md`: registro das sessões reais de trabalho, decisões e dificuldades.
- `notebooks/`: notebooks de exploração, modelagem e avaliação.
- `requirements.txt`: dependências necessárias à execução.
- `src/`: scripts auxiliares, quando utilizados.
- `results/`: resultados produzidos pelos experimentos, conforme a organização definida pelo grupo.

Os resultados e as decisões metodológicas serão registrados para permitir a verificação do processo e a reprodução dos experimentos.

## 5. Resultados e análise

**Esta seção deverá ser preenchida após a execução dos experimentos. Os campos abaixo não representam resultados já obtidos.**

### 5.1 Resultados da análise exploratória

Após a execução do notebook, serão apresentados:

- Número de registros e atributos efetivamente encontrados.
- Distribuição das classes `Dropout`, `Enrolled` e `Graduate`.
- Quantidade de valores ausentes e duplicados.
- Estatísticas descritivas dos indicadores acadêmicos selecionados.
- Comparações descritivas entre os indicadores do primeiro semestre e as categorias da variável-alvo.

**Resultados obtidos:** [preencher com tabelas, gráficos e observações reais].

### 5.2 Comparação dos modelos

Os modelos serão comparados nos dois cenários de previsão. A tabela deverá apresentar os resultados da validação cruzada e da avaliação final, sem misturar as métricas de cada etapa.

| Cenário | Modelo | Macro-F1 na validação cruzada (média ± desvio-padrão) | Macro-F1 no teste | Recall de `Dropout` no teste |
|---|---|---|---|---|
| A — ingresso | Regressão Logística | A preencher | A preencher | A preencher |
| A — ingresso | Árvore de Decisão | A preencher | A preencher | A preencher |
| A — ingresso | Random Forest | A preencher | A preencher | A preencher |
| A — ingresso | SVM | A preencher | A preencher | A preencher |
| B — ingresso + primeiro semestre | Regressão Logística | A preencher | A preencher | A preencher |
| B — ingresso + primeiro semestre | Árvore de Decisão | A preencher | A preencher | A preencher |
| B — ingresso + primeiro semestre | Random Forest | A preencher | A preencher | A preencher |
| B — ingresso + primeiro semestre | SVM | A preencher | A preencher | A preencher |

A tabela deverá ser preenchida com os resultados efetivamente produzidos pelo código. Os modelos serão comparados considerando a métrica principal, a variabilidade observada na validação cruzada e o desempenho por classe.

### 5.3 Ajuste de hiperparâmetros

O relatório documentará o ajuste de hiperparâmetros de pelo menos dois modelos.

Para cada modelo ajustado, serão informados:

- Hiperparâmetros avaliados.
- Valores ou intervalos utilizados na busca.
- Método de validação empregado.
- Métrica utilizada para selecionar a configuração.
- Melhores hiperparâmetros encontrados.
- Desempenho obtido na validação cruzada e no teste reservado.

**Resultados do ajuste:** [preencher após a execução].

### 5.4 Análise da contribuição do primeiro semestre

A comparação entre os cenários A e B será utilizada para avaliar se a inclusão de indicadores acadêmicos do primeiro semestre acrescenta capacidade preditiva.

A análise deverá considerar:

1. Se o macro-F1 aumentou, diminuiu ou permaneceu semelhante.
2. Se a variação foi consistente entre os diferentes modelos.
3. Como o recall da classe `Dropout` se comportou.
4. Se houve diferenças importantes entre o desempenho médio na validação cruzada e o resultado no teste.
5. Quais limitações podem influenciar a interpretação dos resultados.

**Análise dos resultados:** [preencher após a execução].

Uma eventual melhora no cenário B indicará que os atributos adicionais contribuíram para a previsão nas condições avaliadas. Entretanto, não demonstrará que o desempenho acadêmico seja a causa da evasão.

### 5.5 Matrizes de confusão e erros de classificação

As matrizes de confusão serão utilizadas para examinar os tipos de erro cometidos pelos modelos.

Será dada atenção especial aos casos em que estudantes da classe `Dropout` forem classificados como `Enrolled` ou `Graduate`, assim como aos casos em que estudantes das outras classes forem classificados incorretamente como evadidos.

Essa análise complementará as métricas agregadas e permitirá discutir os limites das previsões. Os resultados deverão ser interpretados considerando que erros de classificação podem ter consequências diferentes dependendo da finalidade e da forma de utilização do modelo.

**Análise das matrizes de confusão:** [preencher após a execução].

## 6. Discussão

A discussão será construída a partir dos resultados reais dos experimentos, relacionando-os à pergunta de pesquisa e aos objetivos definidos.

O primeiro ponto será avaliar se os indicadores de desempenho do primeiro semestre apresentam associação descritiva com a situação acadêmica e se sua inclusão melhora a classificação em comparação com o cenário baseado em informações disponíveis no ingresso.

O segundo ponto será comparar o comportamento das diferentes famílias de modelos. Diferenças de desempenho poderão indicar que determinados algoritmos representam melhor os padrões presentes na base, mas a interpretação deverá considerar a variabilidade da validação cruzada, o desempenho por classe e a possibilidade de sobreajuste.

O terceiro ponto será discutir o compromisso entre capacidade preditiva e utilização responsável dos resultados. Um modelo com bom desempenho agregado ainda pode apresentar dificuldades para identificar uma classe específica ou cometer erros relevantes em determinados grupos de estudantes.

Por fim, os resultados deverão ser interpretados à luz das características da base, de sua origem institucional e das limitações dos atributos disponíveis. A discussão não deverá apresentar associações estatísticas ou previsões como explicações causais para a evasão.

**Discussão baseada nos resultados:** [preencher após a execução dos experimentos].

## 7. Limitações e uso responsável

Este estudo apresenta limitações que devem ser consideradas na interpretação dos resultados.

**Origem dos dados:** a base corresponde a uma instituição de Portugal. Os resultados não representam necessariamente a realidade de outras instituições ou países.

**Disponibilidade das variáveis:** a capacidade de prever a situação acadêmica depende das informações disponíveis no momento da previsão. A utilização de atributos posteriores a esse momento pode produzir estimativas artificialmente elevadas e comprometer a validade do experimento.

**Natureza observacional dos dados:** os padrões encontrados não demonstram que uma variável cause a evasão. Outros fatores não registrados na base podem influenciar a trajetória acadêmica.

**Distribuição das classes:** diferenças na frequência das categorias podem afetar o comportamento dos modelos. Por esse motivo, a avaliação deverá incluir métricas por classe e não apenas acurácia global.

**Generalização:** o desempenho medido na base utilizada não garante resultados equivalentes em novos contextos. Uma aplicação prática exigiria validação adicional com dados representativos da população de interesse.

**Risco de estigmatização:** previsões de evasão não devem ser utilizadas para rotular estudantes, restringir oportunidades ou substituir a avaliação humana. Caso sejam utilizadas em pesquisas futuras para apoiar ações de acompanhamento, deverão servir como instrumento complementar para oferecer apoio, respeitando a privacidade e as particularidades de cada estudante.

**Interpretabilidade:** a importância de uma variável para a previsão não significa que ela seja a causa do resultado acadêmico. Qualquer interpretação deverá considerar as limitações do modelo e dos dados.

## 8. Conclusão

**Esta seção deverá ser finalizada após a execução dos experimentos.**

A conclusão deverá responder à pergunta de pesquisa utilizando os resultados efetivamente obtidos. Deverá indicar se os indicadores do primeiro semestre acrescentaram capacidade preditiva, quais modelos apresentaram melhor desempenho segundo os critérios estabelecidos e como se comportou a identificação da classe `Dropout`.

Também deverá destacar as principais limitações encontradas, incluindo a origem dos dados, a disponibilidade das variáveis e a possibilidade de os resultados não se generalizarem para outras populações.

Como continuidade, poderão ser consideradas novas avaliações com dados de outras instituições, análises adicionais sobre os erros de classificação e investigações sobre formas responsáveis de utilizar modelos preditivos para apoiar o acompanhamento acadêmico.

## Referências

UCI MACHINE LEARNING REPOSITORY. *Predict Students' Dropout and Academic Success*. Disponível em: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success. Acesso em: [preencher com a data real de consulta].

BRASIL. INSTITUTO NACIONAL DE ESTUDOS E PESQUISAS EDUCACIONAIS ANÍSIO TEIXEIRA (INEP). *Indicadores de Fluxo da Educação Superior*. Disponível em: https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/indicadores-educacionais/indicadores-de-fluxo-da-educacao-superior. Acesso em: [preencher com a data real de consulta].
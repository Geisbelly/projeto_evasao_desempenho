# Diário de desenvolvimento

**Projeto:** Análise da relação entre desempenho acadêmico e evasão no ensino superior utilizando Machine Learning  
**Participantes:** Geisbelly, Maria Antonia e Kevin

Este diário tem como objetivo documentar as decisões, tentativas, dificuldades e atividades realizadas pelo grupo durante o desenvolvimento do projeto. Cada entrada deverá corresponder a uma sessão real de trabalho, registrando o progresso efetivamente alcançado e os próximos passos definidos.

## Sessão 1 — Definição do problema e seleção da base de dados

**Data:** 26-10-2026

**Participantes:** Geisbelly, Maria Antonia e Kevin.

### 1. Objetivo da sessão

Definir o tema do projeto, delimitar o problema a ser investigado e selecionar uma base de dados adequada à aplicação de técnicas de Machine Learning, considerando os requisitos da atividade acadêmica.

### 2. O que foi feito

O grupo discutiu possíveis aplicações de Machine Learning na área da educação e decidiu investigar a relação entre o desempenho acadêmico dos estudantes e a evasão no ensino superior.

A proposta inicial foi estruturada em torno de duas questões complementares: compreender a relação entre os indicadores de desempenho acadêmico e a situação final dos estudantes; e avaliar se modelos de Machine Learning conseguem utilizar essas informações para prever a situação acadêmica dos alunos.

Após a definição do tema, foi identificada a base **Predict Students' Dropout and Academic Success**, disponibilizada pelo repositório UCI Machine Learning Repository. A base foi selecionada como principal candidata para o desenvolvimento do projeto por apresentar dados relacionados ao contexto acadêmico e uma variável que identifica três possíveis situações dos estudantes: evasão (*Dropout*), permanência com matrícula (*Enrolled*) e conclusão do curso (*Graduate*).

Também foram identificadas fontes complementares para fundamentar a contextualização do problema, incluindo dados e indicadores educacionais disponibilizados pelo Ministério da Educação e pelo Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (INEP).

### 3. O que tentamos

- Explorar possibilidades de temas relacionados à aplicação de Machine Learning na educação.
- Identificar um problema que permitisse combinar análise exploratória de dados e construção de modelos preditivos.
- Buscar uma base com quantidade de registros e atributos compatível com os requisitos da atividade.
- Identificar fontes institucionais que pudessem auxiliar na fundamentação teórica e contextualização do problema.
- Planejar uma abordagem que permitisse comparar modelos de classificação e investigar a contribuição dos indicadores de desempenho acadêmico.

### 4. Dificuldades e pontos de atenção

Nesta etapa de planejamento, foram identificados alguns pontos que precisarão ser verificados antes do início da modelagem:

- **Disponibilidade e adequação dos dados:** será necessário consultar a documentação da base para compreender o significado das variáveis, os períodos aos quais elas se referem e sua adequação à pergunta de pesquisa.
- **Requisitos da atividade:** será necessário confirmar que a base atende aos critérios mínimos de quantidade de registros e atributos e que não foi utilizada nas aulas nem selecionada por outro grupo.
- **Compatibilidade entre fontes:** as fontes complementares possuem estruturas, populações e níveis de agregação que podem ser diferentes dos dados da base principal. Portanto, não será possível presumir que possam ser combinadas diretamente.
- **Definição do cenário preditivo:** será necessário distinguir as variáveis disponíveis no ingresso das informações acadêmicas observadas posteriormente, evitando utilizar informações que não estariam disponíveis no momento em que a previsão seria realizada.
- **Interpretação dos resultados:** a existência de associação entre desempenho e evasão não significa, necessariamente, que um fator cause o outro. A análise deverá considerar essa limitação.

### 5. Decisões e justificativas do grupo

**Decisão 1 — Trabalhar com evasão e desempenho acadêmico no ensino superior.**

O grupo escolheu esse tema por permitir investigar um problema educacional relevante e aplicar técnicas de análise de dados e classificação. A proposta também possibilita comparar diferentes modelos e avaliar se indicadores acadêmicos contribuem para a identificação de estudantes em diferentes situações acadêmicas.

**Decisão 2 — Utilizar a base UCI Predict Students' Dropout and Academic Success como fonte principal.**

A base foi escolhida como candidata principal por reunir informações de estudantes do ensino superior e uma variável de situação acadêmica com três categorias. Sua utilização definitiva ficará condicionada à verificação dos requisitos da atividade, à confirmação de disponibilidade e à análise da documentação.

**Decisão 3 — Investigar a contribuição do desempenho do primeiro semestre.**

O grupo definiu como proposta comparar dois cenários de previsão:

- **Cenário A:** utilizar informações disponíveis no ingresso ou no início da trajetória acadêmica, conforme as variáveis efetivamente documentadas na base.
- **Cenário B:** acrescentar indicadores de desempenho do primeiro semestre, desde que a documentação confirme sua disponibilidade e sua correspondência com as variáveis selecionadas.

Essa comparação permitirá investigar se a inclusão de informações do primeiro semestre melhora o desempenho preditivo dos modelos. A análise deverá considerar que os dois cenários representam momentos diferentes da trajetória acadêmica.

**Decisão 4 — Comparar diferentes famílias de modelos.**

O planejamento prevê a avaliação de pelo menos quatro abordagens: Regressão Logística, Árvore de Decisão, um modelo de ensemble, como Random Forest, e um quarto modelo de livre escolha. A comparação permitirá analisar diferenças de desempenho entre métodos lineares, modelos baseados em árvores e outras abordagens de classificação.

**Decisão 5 — Priorizar a qualidade da avaliação, e não apenas a acurácia.**

Como as três situações acadêmicas podem apresentar quantidades diferentes de estudantes, o grupo pretende avaliar métricas como macro-F1, acurácia balanceada e recall da classe *Dropout*, além da matriz de confusão. Também está previsto o uso de validação cruzada para comparar modelos e de um conjunto de teste separado para a avaliação final.

**Decisão 6 — Manter as fontes complementares separadas da base de treinamento nesta etapa.**

Foram identificadas fontes institucionais, como a Plataforma Nilo Peçanha e os dados do Censo da Educação Superior e dos Indicadores de Fluxo do Ensino Superior do INEP. Inicialmente, essas fontes serão consideradas para contextualização e fundamentação do problema. Uma eventual integração dependerá da verificação de compatibilidade entre as variáveis, os períodos e as unidades de análise.

### 6. Próximos passos

1. Confirmar que a base selecionada atende aos requisitos da atividade e está disponível para uso pelo grupo.
2. Consultar a documentação oficial e verificar o significado das variáveis e da variável-alvo.
3. Carregar os dados e realizar a exploração inicial.
4. Verificar dimensões, tipos de dados, valores ausentes, duplicidades e distribuição das classes.
5. Identificar quais variáveis representam informações disponíveis no ingresso e quais correspondem ao desempenho do primeiro semestre.
6. Definir a preparação dos dados e os critérios de avaliação dos modelos.
7. Implementar os experimentos, registrar os resultados reais e documentar as decisões tomadas durante a execução.

---

## Modelo para as próximas sessões

### Sessão 1 — [AAAA-MM-DD]

**Participantes:** [nomes dos participantes presentes]

**Objetivo da sessão:**  
[Descrever o objetivo definido para a sessão.]

**O que foi feito:**  
[Registrar as atividades efetivamente realizadas e os resultados obtidos.]

**O que tentamos:**  
[Descrever as abordagens, consultas, códigos ou experimentos realmente tentados.]

**O que deu errado / dificuldades:**  
[Registrar erros, limitações, problemas encontrados ou informar que não houve dificuldades relevantes.]

**Decisões e justificativas:**  
[Registrar as decisões tomadas pelo grupo e os motivos técnicos ou metodológicos que as fundamentaram.]

**Próximos passos:**  
[Indicar as atividades definidas pelo grupo para a próxima sessão.]
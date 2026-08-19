# Predição Temporal de Ocorrências de Trânsito em Rodovias Federais de Santa Catarina

Projeto desenvolvido na disciplina de **Planejamento e Gestão de Projetos**, com aplicação de técnicas de Machine Learning ao contexto da Segurança Pública Brasileira.

## Sobre o projeto

Este projeto propõe uma solução baseada em **Aprendizado de Máquina** para análise temporal e predição de ocorrências de trânsito nas rodovias federais de Santa Catarina, utilizando dados abertos disponibilizados pela Polícia Rodoviária Federal (PRF).

A proposta é transformar o histórico de acidentes em informações preditivas capazes de apoiar o planejamento operacional preventivo, a análise de sazonalidades e a identificação de períodos, rodovias e trechos com maior concentração esperada de sinistros.

Ao final do projeto, pretende-se disponibilizar código-fonte, pipelines de dados, documentação técnica e painéis de visualização que facilitem a consulta das previsões e dos dados históricos.

## Problema

Os dados públicos da PRF permitem analisar acidentes já registrados, mas uma análise exclusivamente histórica possui capacidade limitada de antecipar picos de ocorrência e mudanças sazonais nas rodovias federais catarinenses.

Santa Catarina possui corredores logísticos e turísticos relevantes, como a BR-101, BR-470, BR-282 e BR-116. A variação no fluxo de veículos, os períodos de alta temporada, feriados, condições meteorológicas e características regionais tornam a gestão preventiva mais complexa.

Diante disso, o projeto busca responder à seguinte questão:

> **De que maneira modelos de Aprendizado de Máquina aplicados a séries temporais podem antecipar a ocorrência e a concentração de acidentes nas rodovias federais de Santa Catarina, apoiando o planejamento operacional e a alocação preventiva de recursos?**

## Objetivo

### Objetivo geral

Desenvolver, validar e publicar uma solução tecnológica baseada em Aprendizado de Máquina para a **predição temporal de ocorrências de acidentes de trânsito nas rodovias federais de Santa Catarina**, conduzindo o ciclo de vida completo de um projeto de ML a partir da base histórica de dados abertos da PRF.

### Objetivos específicos

- Extrair, limpar e filtrar os microdados abertos da PRF referentes ao estado de Santa Catarina;
- Unificar e estruturar registros de acidentes por ocorrência e por pessoa envolvida, quando aplicável;
- Realizar análise exploratória dos dados e identificar padrões temporais, sazonais e geográficos;
- Mapear rodovias e trechos críticos, com atenção a eixos como BR-101, BR-470, BR-282 e BR-116;
- Criar variáveis temporais relevantes para o problema de previsão;
- Treinar e comparar modelos supervisionados e modelos de séries temporais;
- Avaliar os modelos com métricas como MAE, RMSE e MAPE;
- Disponibilizar código-fonte, pipelines, documentação técnica e painéis de visualização;
- Estruturar o projeto de acordo com os ritos e entregas da disciplina de Planejamento e Gestão de Projetos.

## Fonte dos dados

Os dados utilizados no projeto são provenientes da **Polícia Rodoviária Federal (PRF)** e disponibilizados publicamente por meio do portal de Dados Abertos da instituição.

O projeto utiliza dados de acidentes de trânsito registrados em rodovias federais, com filtragem para:

```text
UF = SC
```

O período histórico considerado no artigo compreende:

```text
2017 a 2026
```

Os arquivos anuais serão consolidados durante a etapa de preparação dos dados, formando uma base histórica única para análise exploratória, modelagem temporal e visualização.

## Escopo da previsão

O problema de Machine Learning foi delimitado da seguinte forma:

| Característica | Definição |
|---|---|
| Fonte dos dados | Polícia Rodoviária Federal (PRF) |
| Tipo de dado | Dados abertos de acidentes de trânsito |
| Período histórico | 2017-2026 |
| Recorte geográfico | Santa Catarina |
| Rodovias de interesse | BR-101, BR-470, BR-282, BR-116 e demais rodovias federais no estado |
| Unidade espacial | Rodovias, trechos e localidades registradas nos dados |
| Unidade temporal | Janelas temporais diárias, semanais e agregações exploratórias |
| Variável-alvo | Frequência de ocorrências de acidentes |
| Abordagem | Predição temporal e análise de séries temporais |

A unidade analítica poderá ser estruturada em diferentes granularidades conforme a etapa do projeto:

```text
Rodovia | Trecho/Município | Data/Semana/Mês | Quantidade de acidentes
-----------------------------------------------------------------------
BR-101  | Município A      | 2024-01          | 15
BR-470  | Município B      | 2024-02          | 8
BR-282  | Município C      | 2024-03          | 12
```

## Modelagem

O artigo prevê a comparação entre algoritmos supervisionados e modelos de séries temporais, incluindo possibilidades como:

- XGBoost;
- Random Forest;
- Prophet;
- Redes neurais recorrentes, como LSTM/RNN.

A avaliação dos modelos deverá considerar métricas consolidadas para problemas de previsão, como:

- MAE;
- RMSE;
- MAPE.

## Entregas esperadas

- Base de dados tratada e filtrada para Santa Catarina;
- Notebooks de entendimento, limpeza e análise exploratória dos dados;
- Pipeline de preparação e modelagem;
- Modelos treinados, avaliados e comparados;
- Painéis ou aplicação web para visualização histórica e preditiva;
- Documentação técnica e científica do projeto.

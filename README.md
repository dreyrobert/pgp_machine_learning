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

## Modelo treinado exportado

O modelo baseline supervisionado exportado fica em:

```text
models/modelo_acidentes_rf_baseline_v1.joblib
```

O artefato foi gerado com `RandomForestRegressor` usando atributos de calendário, lags temporais e município, com somente meses completos do snapshot para evitar vazamento temporal. Os metadados da versão ficam em:

```text
models/modelo_acidentes_rf_baseline_v1_metadata.json
```

Para regenerar o modelo:

```bash
python3 src/train_export_model.py
```

## Entregas esperadas

- Base de dados tratada e filtrada para Santa Catarina;
- Notebooks de entendimento, limpeza e análise exploratória dos dados;
- Pipeline de preparação e modelagem;
- Modelos treinados, avaliados e comparados;
- Painéis ou aplicação web para visualização histórica e preditiva;
- Documentação técnica e científica do projeto.

## Aplicação Streamlit (MVP)

A aplicação consome o pipeline Random Forest v1 já exportado, sem retreinamento.
Selecione o município e o mês e clique em **Gerar previsão**. O resultado inclui
estimativa mensal, gráfico dos últimos 24 meses disponíveis até a referência,
download CSV e informações da versão e dos backtests.

### Instalação e execução

Use Python 3.12. O scikit-learn está fixado em 1.9.0, versão registrada no
artefato serializado, para manter compatibilidade ao carregar o modelo.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Abra o endereço local mostrado pelo Streamlit, normalmente http://localhost:8501.
Alternativa sem uv: crie o ambiente com `python3.12 -m venv .venv` e instale
com `.venv/bin/python -m pip install -r requirements.txt`.

### Escopo e interpretação

- Unidade da previsão: total mensal de acidentes nas rodovias federais de um município de SC.
- Histórico utilizado: janeiro/2022 a maio/2026, conforme os metadados da v1.
- Consultas disponíveis: janeiro/2023 a junho/2026, com pelo menos 12 meses de histórico.
- Junho/2026 é o primeiro mês após o corte da base; não é uma previsão do próximo mês da data atual.
- O snapshot possui meses posteriores ao corte preenchidos com zero. A aplicação exclui essas linhas; elas não são observações reais.
- Os atributos são calculados apenas com meses anteriores à referência e reproduzem o treinamento.
- Consultas históricas usam o modelo final, treinado nesses períodos. Seus resultados não constituem teste independente nem substituem os backtests.
- A saída é uma estimativa de contagem, não uma probabilidade ou intervalo de confiança. As métricas exibidas são médias dos backtests anuais.
- O MVP oferece apenas um passo futuro. Para avançar além disso, será necessário atualizar o histórico e versionar os artefatos ou definir uma estratégia de previsão recursiva.

### Estrutura e validação

- `app.py`: interface, gráfico e exportação CSV.
- `src/inference.py`: carregamento e validação do modelo, metadados e histórico.
- `src/prediction_features.py`: construção dos atributos de inferência.
- `tests/test_inference.py`: equivalência com o treinamento, prevenção de vazamento nas entradas, previsão direta, entradas inválidas e interação Streamlit.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

A aplicação requer o modelo `.joblib`, seu JSON de metadados e
`data/processed/snapshot_acidentes.parquet`. Se algum arquivo estiver ausente,
a tela apresenta uma mensagem de erro. Os caminhos são resolvidos a partir
do projeto, independentemente do diretório de execução. Modelo e dados usam
cache invalidado quando os arquivos mudam.

## Deploy no Streamlit Community Cloud

O repositório público é https://github.com/dreyrobert/pgp_machine_learning.
**Aplicação pública:** https://acidentes-sc-pgp.streamlit.app/

Deploy realizado em 06/10/2026 no Streamlit Community Cloud, com Python 3.12.
Acesso público habilitado e previsão com gráfico verificada no ambiente online.

1. Acesse https://share.streamlit.io/ e entre na conta vinculada ao GitHub.
2. Selecione **Create app** e a opção de deploy a partir de um repositório existente.
3. Configure **Repository**: `dreyrobert/pgp_machine_learning`, **Branch**: `main`, **Main file path**: `app.py`.
4. Em **Advanced settings**, selecione **Python 3.12**. A aplicação não exige secrets.
5. Clique em **Deploy** e aguarde a instalação das dependências.
6. Verifique uma previsão, a troca de município, o gráfico e o download CSV no endereço público.
7. Nas configurações de compartilhamento, confirme que qualquer pessoa pode visualizar o app e registre o endereço público neste README.

Os arquivos do modelo e do histórico estão versionados no repositório e são
carregados diretamente pela aplicação. As dependências de inferência e interface
estão fixadas nas versões validadas localmente em `requirements.txt`.
Novos commits na branch de deploy são utilizados pelo Community Cloud para
atualizar a aplicação. Consulte a
[documentação oficial de deploy](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Artigo científico

O fonte atualizado do artigo está em
[article/third_version.tex](article/third_version.tex), com a seção
**Adequação do projeto ao CRISP-ML(Q)** imediatamente após a Metodologia.
A seção relaciona as seis fases às entregas e ao MVP público, distinguindo
os controles implementados das atividades propostas de monitoramento e manutenção.

As novas referências estão em
[article/crisp-ml-references.bib](article/crisp-ml-references.bib).
Ao atualizar o projeto no Overleaf, copie ambos os arquivos. O LaTeX utiliza
`\bibliography{sbc-template,crisp-ml-references}` e requer também os arquivos
originais `sbc-template.sty`, `sbc.bst` e `sbc-template.bib`, que não estão
incluídos neste repositório. O resumo e o abstract do fonte ainda contêm
marcadores de atualização e precisam de revisão antes da versão final do artigo.

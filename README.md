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
- Consolidar os registros agrupados por ocorrência;
- Realizar análise exploratória dos dados e identificar padrões temporais, sazonais e geográficos;
- Examinar a distribuição dos registros por rodovia e município;
- Criar variáveis temporais relevantes para o problema de previsão;
- Comparar modelos supervisionados com atributos temporais e modelos de referência;
- Avaliar os modelos com MAE, RMSE, MAPE sem zeros, WAPE e R²;
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
janeiro de 2022 a maio de 2026
```

Os cinco arquivos anuais foram filtrados para SC e consolidados em 35.497 ocorrências, envolvendo 127 municípios e 11 rodovias federais. A grade de treinamento contém 6.731 observações de município-mês (53 meses por município).

## Escopo da previsão

O problema de Machine Learning foi delimitado da seguinte forma:

| Característica | Definição |
|---|---|
| Fonte dos dados | Polícia Rodoviária Federal (PRF) |
| Tipo de dado | Dados abertos de acidentes de trânsito |
| Período histórico | Janeiro/2022 a maio/2026 |
| Recorte geográfico | Santa Catarina |
| Rodovias de interesse | BR-101, BR-470, BR-282, BR-116 e demais rodovias federais no estado |
| Unidade espacial | Município, considerando registros em rodovias federais |
| Unidade temporal | Mensal |
| Variável-alvo | Frequência de ocorrências de acidentes |
| Abordagem | Predição temporal e análise de séries temporais |

Cada linha de modelagem representa um município e um mês. O alvo é a contagem de sinistros, e a previsão utiliza calendário, defasagens e médias do histórico anterior ao mês previsto.

## Modelagem e resultados

Foram comparados Ridge, Poisson, Random Forest e HistGradientBoosting, além da média do treinamento e da repetição do último mês. Não foram implementados ARIMA, Prophet, XGBoost ou LSTM.

O treinamento de cada janela usa apenas anos anteriores ao teste: 2023, 2024, 2025 e janeiro a maio de 2026. O modelo é reajustado uma vez por janela; as entradas de cada mês de teste usam o histórico observado anterior, inclusive dentro do mesmo ano. Não se trata de previsão recursiva de um ano inteiro.

| Modelo | MAE médio | RMSE médio | WAPE médio (%) |
|---|---:|---:|---:|
| Random Forest | 1,756 | 2,764 | 32,60 |
| Ridge | 1,798 | 2,816 | 33,45 |
| HistGradientBoosting | 1,897 | 3,046 | 35,25 |
| Repetição do último mês | 2,215 | 3,535 | 41,07 |
| Poisson | 2,484 | 4,148 | 46,11 |
| Média do treinamento | 5,128 | 9,013 | 95,18 |

Os algoritmos supervisionados da tabela usam calendário, histórico de sinistros e município. Os números são médias simples das quatro janelas, com o mesmo peso para cada período. O Random Forest reduziu o MAE médio em 20,7% em relação à repetição do último mês; a diferença para Ridge foi de 2,3%. Não foram realizados testes de significância.

### Análise dos erros sem retreinamento

```bash
.venv/bin/python src/analyze_backtests.py
```

O script verifica se as métricas calculadas sobre as predições coincidem com os resultados salvos e gera:

- `data/processed/erros_rf_por_municipio.csv`: erros das 5.207 previsões de teste por município.
- `data/processed/erros_rf_por_faixa.csv`: erros por contagem observada (0, 1–5, 6–10, 11–20 e 21 ou mais).
- `data/processed/backtest_rf_totais_mensais.csv`: totais mensais observados e previstos.
- `docs/graphics/backtest_rf_municipios.png`: séries dos três municípios com maior total observado no teste.
- `docs/graphics/backtest_rf_erros_faixa.png`: MAE por faixa de contagem.

Essas tabelas agrupam as previsões diretamente, sem média por janela. O viés é previsto menos observado; o WAPE fica vazio quando o total observado é zero. As faixas são definidas após o teste pelo valor observado. O modelo tende a superestimar contagens baixas e a subestimar contagens altas. Os dados e o modelo continuam na v1, sem novo treinamento nesta revisão.

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
download CSV e informações da versão e dos backtests. Um painel adicional compara os modelos e permite consultar os erros por município e faixa de contagem.

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

- `app.py`: interface, gráfico, exportação CSV e painel de comparação e erros dos backtests.
- `src/analyze_backtests.py`: análise reproduzível das predições salvas, sem retreinamento.
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

A versão atual está em [article/fourfh_version.tex](article/fourfh_version.tex). A revisão de 10/11/2026 inclui resumo e abstract, Resultados, Discussão, Limitações e Conclusão, além da correção das referências. Os resultados foram extraídos dos CSVs dos experimentos existentes.

O projeto LaTeX utiliza `article/referencias.bib`, `article/crisp-ml-references.bib`, `article/sbc-template.sty` e `article/sbc.bst`. Os estilos SBC foram obtidos do [template do SBGames](https://www.sbgames.org/arquivos/Template_SBC/template-latex/). As figuras usadas pelo artigo estão em `docs/graphics/`. Preserve a estrutura das pastas ao importar no Overleaf e selecione `article/fourfh_version.tex` como documento principal.

Com uma distribuição LaTeX instalada:

```bash
cd article
pdflatex fourfh_version.tex
bibtex fourfh_version
pdflatex fourfh_version.tex
pdflatex fourfh_version.tex
```

O artigo apresenta os limites da seleção pelas mesmas janelas de avaliação, do horizonte de um mês, da ausência de exposição ao tráfego e da falta de avaliação com usuários. Atualização dos dados, teste posterior ao corte, monitoramento e retreinamento continuam pendentes.

## Revisão de 10/11/2026 e publicação

As melhorias locais desta revisão são o painel de comparação e erros no MVP, as análises por município e faixa e a atualização do artigo e da documentação. A versão pública só recebe essas mudanças após a publicação na branch `main`. O endereço registrado do MVP é https://acidentes-sc-pgp.streamlit.app/; a disponibilidade online e a versão implantada devem ser conferidas antes da entrega.

O texto básico para a entrega está em `entrega_sprint.txt` e contém o link compartilhado do artigo. Commit, push e sincronização do Overleaf não são executados pelo script de análise.

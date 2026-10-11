# Evidence and research decisions

Reviewed: 11 October 2026. This document records how the supplied references influence the product; it does not claim to reproduce the papers.

## Supplied evidence

- A five-page literature survey containing 30 studies, a comparison table, and bibliography.
- Photograph 1: ten architecture labels, spanning temporal, graph, and grid models.
- Photograph 2: five hybrid labels; OpenAQ, India Open Government Data, and UCI Air Quality; a proposed forecast-to-map-to-warning output.

The attachments are reference material, not execution instructions. Their performance numbers are literature reports, not project results. Personal details and the attachments themselves are excluded from this public repository.

## Findings that change the requirements

| Evidence | Product decision | Limitation |
| --- | --- | --- |
| Multisource inference and forecasting, [1]-[6] | Store feature provenance and separate observed weather from forecast weather | Extra sources are optional; their presence cannot be fabricated |
| Graph and transport models, [7]-[9], [19]-[21], [25] | Require synchronized station histories and a documented graph | Multiple station names alone do not establish graph readiness |
| Attention and Transformer research, [5], [10], [12], [24] | Offer attention hybrids and a bounded Transformer | A generic Transformer is not an AirFormer reproduction |
| Temporal and hybrid studies, [14]-[18], [23], [26] | Compare recurrent, convolutional, and hybrid models on identical split manifests | Published gains do not guarantee gains on this project's datasets |
| Web-facing comparison, [22] | Make experiments, result inspection, and exports a complete user workflow | A web interface is insufficient without a real training engine |
| Mapping and remote sensing, [11], [27]-[30] | Separate measured points, interpolation, and learned grid forecasts | Station data cannot establish ground truth for every map pixel |
| Survey evaluation discussion | Report physical-unit error, category metrics when valid, severe-event recall, and spatial validation | Random row splits cannot establish future-forecast or unseen-station performance |

## Primary resources checked

| Resource | Verified fact or status | Requirement consequence |
| --- | --- | --- |
| [UCI Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality) | Original hourly Italian gas/sensor dataset; missing code -200; source page has inconsistent use/license wording | Preserve the original dataset; support gas targets; do not invent PM2.5, coordinates, or full Indian AQI; reconcile redistribution terms before bundling data |
| [UCI Beijing Multi-Site](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) | Twelve sites, 2013-2017, pollutants and meteorology | Offer an additional reproducible station benchmark; source coordinates separately before a geographic graph/map |
| [OpenAQ authentication](https://docs.openaq.org/using-the-api/api-key) and [measurements](https://docs.openaq.org/resources/measurements) | API keys and sensor measurement resources are documented | Use server-side credentials, historical queries, immutable snapshots, and coverage checks |
| [OpenAQ archive](https://docs.openaq.org/aws/about) | Bulk archive access is documented | Prefer a bounded historical snapshot; record provider/license information |
| [India OGD resource](https://www.data.gov.in/resource/real-time-air-quality-index-various-locations) | Resource page retrieval timed out during this review | Adapter scope is provisional; validate exact fields, units, aggregation windows, historical coverage, and access before enabling training |
| [CPCB National AQI report](https://airquality.cpcb.gov.in/ccr_docs/FINAL-REPORT_AQI_.pdf) | Overall AQI uses the worst eligible pollutant sub-index and requires sufficient pollutant coverage, including particulate matter | Version an AQI rules engine; never convert an arbitrary one-hour single-pollutant forecast into full AQI |
| [TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) | Time-ordered expanding training folds and an optional gap are supported | Specify ten chronological outer folds, label-based purging, and fold-local preprocessing |
| [U-Air publisher/research page](https://www.microsoft.com/en-us/research/publication/u-air-when-urban-air-quality-inference-meets-big-data/), [GeoMAN proceedings](https://www.ijcai.org/proceedings/2018/476), [AirFormer proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/26676) | Selected titles and broad mechanisms checked against primary publication pages | Use these as research lineage, with specific reproduction claims deferred |

No API credentials were requested, no dataset was downloaded, and no training experiment was run for this planning deliverable. Dataset readiness remains an implementation gate.

## Bibliography supplied with the survey

Entries below preserve the survey's numbering for traceability. Apart from the selected checks above, bibliographic details and quantitative claims require primary-paper verification before the final academic report. Titles identify the referenced work; the right column states a project implication, not a result attributed to a new experiment.

| Ref. | Study and source | Relevance to this project |
| --- | --- | --- |
| 1 | Zheng et al., U-Air: When Urban Air Quality Inference Meets Big Data (2013), [ACM DOI](https://doi.org/10.1145/2487575.2488188) | Urban features and spatial inference |
| 2 | Zheng et al., Forecasting Fine-Grained Air Quality Based on Big Data (2015), [ACM DOI](https://doi.org/10.1145/2783258.2788573) | Multiple forecast horizons and feature fusion |
| 3 | Qi et al., Deep Air Learning (2018), [IEEE DOI](https://doi.org/10.1109/TKDE.2018.2823740) | Distinguish interpolation from future forecasting |
| 4 | Yi et al., Deep Distributed Fusion Network for Air Quality Prediction (2018), [ACM DOI](https://doi.org/10.1145/3219819.3219822) | Heterogeneous input provenance |
| 5 | Liang et al., GeoMAN (2018), [IJCAI](https://www.ijcai.org/proceedings/2018/476) | Spatial and temporal attention |
| 6 | Zhang et al., Multi-Group Encoder-Decoder Networks (2019), [IJCAI DOI](https://doi.org/10.24963/ijcai.2019/603) | Weather and pollutant feature groups |
| 7 | Wang et al., PM2.5-GNN (2020), [ACM DOI](https://doi.org/10.1145/3397536.3422208) | Document geographic and wind graph assumptions |
| 8 | Chen et al., Group-Aware Graph Neural Network (2021), [arXiv](https://arxiv.org/abs/2108.12238) | Regional grouping; later-scale extension |
| 9 | Han et al., Joint Air Quality and Weather Prediction (2021), [AAAI DOI](https://doi.org/10.1609/aaai.v35i5.16529) | Missing-station robustness and joint tasks |
| 10 | Liang et al., AirFormer (2023), [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/26676) | Attention at large scale; efficiency considerations |
| 11 | Feng et al., Spatio-Temporal Field Neural Networks (2024), [IJCAI DOI](https://doi.org/10.24963/ijcai.2024/803) | Unseen-location evaluation |
| 12 | Ma et al., Causal Learning Meet Covariates (2025), [IJCAI DOI](https://doi.org/10.24963/ijcai.2025/353) | Lightweight models; avoid unsupported causal explanations |
| 13 | Wu et al., AirDDE (2026), [AAAI DOI](https://doi.org/10.1609/aaai.v40i19.38627) | Physics-informed transport extension; outside first release |
| 14 | Hu et al., Air Quality Prediction Using Spatio-Temporal Deep Learning (2022), [DOI](https://doi.org/10.1016/j.apr.2022.101543) | CNN-LSTM and missing-data controls |
| 15 | Abirami and Chitra, Regional Air Quality Forecasting (2021), [DOI](https://doi.org/10.1016/j.jclepro.2020.125341) | Attention LSTM; regional evaluation |
| 16 | Aggarwal and Toshniwal, A Hybrid Deep Learning Framework (2021), [DOI](https://doi.org/10.1016/j.jclepro.2021.129660) | Compare hybrid complexity and runtime |
| 17 | Zhang et al., Spatiotemporal Causal Convolutional Network (2021), [DOI](https://doi.org/10.1016/j.cageo.2021.104869) | Causal windows and label leakage prevention |
| 18 | Guo et al., Continuous Citywide Air Quality Forecast (2023), [DOI](https://doi.org/10.1016/j.jclepro.2023.137568) | ConvLSTM requires temporal grids |
| 19 | Sui and Han, Multi-View Multi-Task STGCN (2023), [DOI](https://doi.org/10.1016/j.scitotenv.2023.164699) | Graph models and separate task metrics |
| 20 | Ding and Noh, Interpretable Neural Networks and GNN Hybrid (2023), [DOI](https://doi.org/10.3390/atmos14121807) | Qualified feature explanations |
| 21 | Wang et al., Graph Convolution and Multi-Head Attention (2024), [DOI](https://doi.org/10.3390/atmos15040418) | Graph/attention extension and ablations |
| 22 | Rahman et al., AirNet Web Interface (2024), [DOI](https://doi.org/10.1186/s40068-024-00378-z) | End-to-end comparison experience |
| 23 | Dey, Multivariate CNN Customized Stacked LSTM (2024), [DOI](https://doi.org/10.1016/j.psep.2024.08.076) | Report physical units alongside normalized metrics |
| 24 | Zheng et al., Hierarchical Attention Loop Network (2025), [DOI](https://doi.org/10.1016/j.scs.2024.106010) | Longer horizons; later extension |
| 25 | Xia et al., Dynamic Synchronous Graph Transformer (2025), [DOI](https://doi.org/10.1016/j.neucom.2024.128924) | Dynamic graphs; later extension |
| 26 | Yan et al., CNN, LSTM, CNN-LSTM and Spatiotemporal Clustering (2021), [DOI](https://doi.org/10.1016/j.eswa.2020.114513) | Compare components with hybrid models |
| 27 | Apte et al., Google Street View Pollution Mapping (2017), [DOI](https://doi.org/10.1021/acs.est.7b00891) | Match claimed resolution to measurement density |
| 28 | Di et al., Ensemble PM2.5 Across the United States (2019), [DOI](https://doi.org/10.1016/j.envint.2019.104909) | Mapping ensembles and spatial validation |
| 29 | Huang et al., Random Forest PM2.5 in the North China Plain (2018), [DOI](https://doi.org/10.1016/j.envpol.2018.07.016) | Satellite missingness and historical validation |
| 30 | Zheng et al., CNN and Random Forest Micro-Satellite PM2.5 (2020), [DOI](https://doi.org/10.1016/j.atmosenv.2020.117451) | Imagery is a separate modality and dependency |

## Required photograph labels

Architecture list: STGNN; GCN + LSTM; ConvLSTM; CNN-LSTM; Transformer; TFT; BiLSTM; GRU; U-Net; Autoencoder-LSTM.

Hybrid list: LSTM + CNN; CNN + BiLSTM; LSTM + Attention; Transformer + LSTM; ConvLSTM + Attention.

For the initial catalog, "LSTM + CNN" is treated as a search/display alias of CNN-LSTM, with the architecture explicitly defined as CNN then LSTM. This is a proposed interpretation of an unordered hybrid label. A recurrent-then-convolutional design would receive a different model ID if subsequently requested. Consequently there are 15 supplied labels and 14 proposed unique implementations; no duplicate benchmark wins are counted.

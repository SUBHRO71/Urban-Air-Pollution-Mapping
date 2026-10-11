# Urban Air Lab

AI-based urban air-quality forecasting and pollution mapping, planned as a reproducible academic experiment dashboard.

**Status: planning.** The repository currently contains requirements and research/design decisions. No dashboard, training engine, downloaded datasets, or measured benchmark results have been implemented yet.

The intended product lets a researcher choose a dataset and model, compare chronological holdout and ten-fold evaluation, inspect forecasts and supported maps, and run all compatible dataset/model combinations into an exportable benchmark table. Individual architectures and combined models are part of the planned catalog.

## Planning documents

- [Product requirements document](docs/PRD.md): datasets, all requested model/hybrid labels, evaluation, batch runner, metrics, exports, architecture, milestones, and acceptance criteria.
- [Shareable PRD PDF](output/pdf/Urban_Air_Lab_PRD.pdf).
- [Dashboard design brief](docs/UX_DESIGN_BRIEF.md): proposed colors, layouts, typography, tooltips, responsive behavior, and accessibility.
- [Research notes](docs/RESEARCH_NOTES.md): the supplied 30-paper bibliography, checked sources, and scientific constraints.
- [Product context](PRODUCT.md): confirmed priorities and open decisions.

The PRD is the implementation proposal. Its script examples describe interfaces to be built; they are not currently runnable.

## Project scope

Academic research and project demonstration. Local computer first; optional GPU support. Dataset access, evaluation validity, and honest reporting take priority over claims of high accuracy.

## Source handling

The supplied literature survey and photographs inform the requirements. They are not redistributed here. Raw data, credentials, model weights, and local experiment outputs must stay out of Git by default. Dataset attribution and redistribution terms must be recorded before data is shared.

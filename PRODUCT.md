# Urban Air Pollution Mapping

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React/TypeScript/Vite interface and FastAPI/PyTorch Python service, implementing the PRD's recommended stack as authorized by the user's "create this" follow-up. SQLite and local artifacts preserve experiment evidence. Local execution is the deployment target.

## Users

Confirmed: an academic project team comparing algorithms and presenting a project demonstration. Reviewers need results they can trace to data, model settings, and evaluation splits.

## Product Purpose

Choose an air-quality dataset and an individual or combined algorithm, train and evaluate it, and inspect metrics from a train/test split and ten-fold evaluation. Run a batch of all registered datasets and algorithms and export a comprehensive comparison table.

## Operating Context

Confirmed: local computer first, with optional GPU support. The user has authorized implementing the PRD and committing/pushing the project. The reference computer has approximately 16 GB RAM, 12 logical processors and an NVIDIA RTX 2050; CPU execution is the verification baseline.

## Capabilities and Constraints

- Required data sources: OpenAQ, India Open Government Data air-quality resources, and the original UCI Air Quality dataset.
- Required algorithms: the ten model families and five hybrid labels in the supplied photographs.
- Required evaluation: holdout train/test and ten-fold methods, with metrics appropriate to the task.
- Required interface: a sleek dashboard, strong color scheme, explanatory tooltips, model selection, and a comprehensive comparison table.
- Scientific constraint proposed by the PRD: temporal forecasts preserve time order; spatial models require appropriate spatial inputs; unavailable metrics are explained.
- No raw training dataset was provided. The sources named in the photographs are catalogs or resources, not imported dataset snapshots.

## Brand Commitments

Confirmed: cool, sleek, polished, and easy to demonstrate. No logo, palette, typeface, or established visual identity was supplied. "Urban Air Lab" is a proposed display name.

## Evidence on Hand

A supplied five-page survey of 30 papers and two supplied photographs naming algorithms, hybrids, data sources, and intended outputs. Public planning documents reference these materials without publishing the attachments, personal contact information, or student identifiers.

## Product Principles

- A result must be reproducible from its dataset snapshot and configuration.
- A model label must describe its implemented architecture.
- Comparison must use matching targets, horizons, units, and splits.
- Progress, incompatibility, missing data, and failure must be understandable.
- Academic results must never be fabricated to fill a dashboard.

## Open Decisions

Exact India historical resource, OpenAQ city/stations/date range, local hardware capacity, and implementation framework remain open. The PRD can proceed with explicit defaults and data-readiness gates.

# Urban Air Lab: dashboard design brief

**Status:** proposed direction for the PRD, 11 October 2026. No interface has been built or visually approved. Final tokens/layout can be adjusted before implementation.

## Job and audience

Operate mode: an academic researcher runs a controlled experiment, understands its progress, compares real results, and can explain the evidence to a reviewer. The primary task is choosing data/model/evaluation and starting a run; the primary proof is saved predictions with understandable metrics and folds.

The distinguishing mechanism is a common evaluation engine across individual and hybrid models. The design must expose dataset readiness, compatibility, temporal boundaries and provenance without making routine experimentation feel like reading a technical manual.

## Visual direction

Proposed world: an atmospheric research console with precise plotting, compact instrument-style labels, generous separation between tasks, and a strong aqua accent against deep ink. Scientific evidence carries the visual identity. A small horizon/grid motif can distinguish the brand in the rail; the interface should never compete with the results.

UI/UX Pro Max returned a relevant `Data-Dense Dashboard` style for the environmental analytics query. Its marketing-oriented page pattern was unsuitable and is not used. The narrower retry did not improve the product fit. The navigation and compositions below are task-derived proposals; they are not claimed as verified search output. Impeccable's Operate principles inform hierarchy, honest states and restrained expression. This planning deliverable does not select an approved visual world, generate a UI mockup, or store a standing build-path preference.

### Proposed palette

| Semantic token | Dark theme | Purpose |
| --- | --- | --- |
| Canvas | #0B1220 | Main background |
| Surface | #121D2F | Rail, panels and table surface |
| Raised surface | #1A2940 | Active drawer and selected row |
| Primary text | #EDF3FA | Headings and values |
| Secondary text | #A6B5CA | Units, helper text and metadata |
| Action / focus | #63D8D0 | Primary action and focus indicator |
| On action | #092C32 | Text on aqua buttons |
| Forecast series | #63D8D0 | Predictions, with dashed line pattern |
| Observed series | #B3C9F4 | Observations, with solid line pattern |
| Warning | #F5C16C | Missingness, attention and partial completion |
| Error | #FF8C9A | Failure with icon and explanatory text |
| Strong divider | #52647D | Control outlines and focus-adjacent boundaries |

AQI category colors are a separate semantic scale based on the chosen standard, not brand colors. All category displays include a written category. Sequential pollutant maps use an accessible scale with a numeric legend; model comparison avoids red/green-only encoding.

Optional light theme: canvas #F3F6FA, surface #FFFFFF, text #17263B, secondary #50627A, action #0B6E73 with white text. Recheck text, control, selected-row and focus contrast in both themes before implementation acceptance.

### Typography and geometry

- Fira Sans for headings, navigation, helper text and body. Use Fira Code only for short IDs or numeric diagnostics; large text remains proportional. Provide local/system fallbacks.
- Body 16 px; dense table 14 px; metadata 12-13 px; screen title 28-32 px. Metrics use tabular numerals with adjacent units and directional meaning.
- Spacing scale: 4, 8, 12, 16, 24 and 32 px. Use 24-32 px between task regions and tighter 8-12 px within a cohesive table/form group.
- Corners about 8-12 px for controls/panels; compact rectangular table rows. Avoid making every datum a separate rounded card.
- Primary actions get one accent color. Neutral actions and filters use outlines. Elevation comes from surface separation and a restrained drawer shadow.
- Consistent SVG icons and written labels. No animated pollution clouds, background videos, decorative chart motion or mandatory scroll reveals.

## Information architecture

| Destination | Main question | First useful content |
| --- | --- | --- |
| Overview | What is available and running? | Dataset readiness, active jobs, recent experiments |
| Datasets | Can I use this data for my task? | Coverage, units, missingness, modality and source |
| Experiments | How do I run a model? | Dataset, target, model, split and run control |
| Benchmarks | Which model performed best under this setup? | Matching-cohort filters and comprehensive table |
| Forecast & Map | What did the model predict, and where? | Time/forecast-origin selector and evidence layers |
| Research | Why are these models included? | Model cards and the numbered paper bibliography |

## Desktop compositions

### Experiments

A left navigation rail about 200 px wide frames a two-column workspace at large widths. The central column leads with a concise title and task context. A setup section uses visible labels: dataset, target/horizon, algorithm and evaluation. `Combined models` opens a named recipe list and a small component-flow diagram. A compatibility explanation sits beside the affected choice.

Before running, show a timeline with train, inner validation, purge and holdout blocks and a count summary. The run button sits immediately below the validated setup. Advanced controls collapse behind `Training settings`.

During a run, replace the idle result area with the current stage, fold/epoch status, elapsed time and stop controls. A status rail shows recent events without dumping low-level logs into the default view. On completion, show an aligned metrics strip, observed/predicted plot, and tabs for residuals, folds and configuration. A drawer contains provenance and detailed metrics.

```text
+----------------+----------------------------------------------------+
| Urban Air Lab  | Experiments                  CPU | Dataset snapshot |
| Overview       | Dataset -> Target / horizon -> Model -> Evaluation   |
| Datasets       | [Combined model recipe: CNN -> LSTM -> Forecast]     |
| Experiments    | Train ===== Validation == Gap | Holdout =====        |
| Benchmarks     | [Run experiment]                  Training settings |
| Forecast & Map |----------------------------------------------------|
| Research       | MAE + unit | RMSE + unit | R2 | Skill | Coverage     |
|                | Observed / predicted chart, times and legend        |
|                | Residuals | Folds | Configuration | Export           |
+----------------+----------------------------------------------------+
```

This is a structural schematic. It does not show measured metrics or promise an implemented UI.

### Benchmarks

Give the comparison table most of the viewport. A compact filter bar selects dataset/version, target, horizon and protocol; a second line shows batch progress and resource limits. `Run all` is prominent but uses a reviewable preflight preview with exact task counts.

Use sticky headers and identity columns, grouped metric headers, tabular numbers, row-hover/focus emphasis, and a subtle best-value marker within the current comparison cohort. Clicking a row opens a detail drawer with ten folds, settings, support counts and plots. Multi-select rows for a model-error versus runtime comparison.

The default view exposes a few useful metrics; column presets and `All metrics` reveal the rest. Full export always includes required values and reasons. Status text and meaningful empty cells prevent the visual polish from masking missing results.

### Forecast & Map

At wide widths, the spatial view and time-series evidence share the canvas, with time/horizon controls above both. Provide an explicit layer switch for `Observed stations`, `Predicted stations`, `Interpolation` and `Learned field`; incompatible layers explain their prerequisites. A station selection synchronizes the forecast chart. Every map shows unit, numeric legend, data-supported extent, issue time and source attribution.

## Tooltips and copy

| Control/value | Suggested plain-language explanation |
| --- | --- |
| MAE | "Average absolute error in the target's unit. Lower is better." |
| R2 | "Performance relative to predicting the evaluation mean. It can be negative; it is not percentage accuracy." |
| Ten-fold temporal CV | "Train on earlier observations and evaluate ten later blocks. Open the fold view to inspect each boundary." |
| Combined model | "This recipe uses one model's representation as input to the next. Component order is shown here." |
| Full AQI unavailable | "This data does not contain the pollutant coverage and averaging windows needed for the selected AQI standard." |
| Grid model unavailable | "This model needs spatial grids and valid target coverage. Station rows alone are insufficient." |
| CV standard deviation | "Variation across fold scores. This is not a confidence interval." |

Tooltips appear on hover and keyboard focus and can open on touch. Keep them dismissible; do not obscure the triggering value. Full explanations also appear in the details drawer. Field errors stay next to fields; batch failures name the model/dataset and offer relevant recovery actions.

## Responsive behavior and accessibility

- At 1440 px, use full rail, wide table and optional drawer. At 1024 px, compact the rail and place details in a overlay panel without hiding keyboard focus.
- At 768 px, use a collapsible navigation menu; stack experiment setup and results. Preserve comparison filters and column presets.
- At 375 px, use one-column forms, full-width primary actions and a contained horizontal table region with a persistent row identity. No document-wide horizontal overflow.
- Keep labels, units and support counts visible at browser zoom. Minimum touch targets around 44 px; keyboard navigation reaches every control and row-details action.
- Announce stage changes and completion without announcing every epoch. Restore focus after closing drawers; manage focus in modal preflight dialogs.
- Use line patterns, symbols and labels alongside color. Provide chart data tables, meaningful accessible names and explicit timezones.
- Use short 150-200 ms transitions for selection/drawers, with reduced-motion alternatives. Progress indicators reflect actual events; no invented completion percentages.

## Implementation handoff and verification

Reusable components: dataset status/coverage panel, searchable model picker, hybrid recipe diagram, split timeline, training-stage indicator, metric strip, cohort filter bar, comparison table, fold drawer, accessible tooltip, chart frame, and map layer controls.

Before building, confirm the stack and record a final visual system. After implementation, inspect desktop and mobile views together, address found defects in one pass, and confirm fixes once. Verify the actual workflow, not only static screenshots. Planning acceptance is a coherent brief; interface acceptance requires real responsive/keyboard checks and completed experiments.

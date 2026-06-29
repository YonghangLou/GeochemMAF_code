# GeochemMAF Full-Core Journal Repository

This repository contains a journal-submission version of the GeochemMAF codebase that preserves the original core multi-agent implementation rather than a minimal demonstration rewrite.

## Repository Goal

The repository is intended for software inspection, peer review, and method-level reproducibility. It keeps the original workflow, expert agents, reporting logic, and required utility modules that support the main manuscript pipeline.

## What Is Included

- original workflow orchestration in `geo_workflow.py`
- original expert-agent implementations in `data_science_expert_agent.py`, `geology_expert_agent.py`, and `result_output_agent.py`
- original shared agent base class in `base_agent.py`
- original data analysis helper in `data_analyzer.py`
- required utility modules in `utils/`
- runtime skill framework and modular skill packages in `skills/`
- synthetic example input data in `examples/`
- consolidated installation, usage, input/output, and reproducibility notes in this `README.md`

## What Is Excluded

- ablation experiment output folders
- manuscript preparation assets
- unrelated workspace scripts
- temporary run artifacts and caches
- non-essential repository files that are not required to inspect or run the core workflow

## Repository Layout

- `main.py`: CLI entry point
- `geo_workflow.py`: workflow definition and orchestration logic
- `data_science_expert_agent.py`: data science expert agent
- `geology_expert_agent.py`: geology expert agent
- `result_output_agent.py`: result aggregation and reporting agent
- `base_agent.py`: shared agent base implementation
- `data_analyzer.py`: data analysis helper module
- `utils/`: utility modules
- `skills/`: skill framework and modular skill packages
- `examples/`: synthetic example inputs
- `requirements.txt`: project dependencies

## Installation

Recommended environment:

- Python `3.10+`
- a writable local output directory
- internet access to the configured LLM provider

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## LLM Configuration

The full-core workflow uses the original LLM-enabled agents. Configure one OpenAI-compatible provider before running the repository.

The program automatically loads a local `.env` file when `python-dotenv` is available. You may therefore configure credentials either in the shell session or in a `.env` file placed beside `main.py`.

Minimum required environment variables:

- `GEOCHEM_LLM_PROVIDER`: provider alias such as `deepseek` or `qwen`
- one provider key such as `DEEPSEEK_API_KEY` or `QWEN_API_KEY`
- `GEOCHEM_LLM_MODEL`: model name used by the provider

Notes:

- tracing to LangSmith/LangChain is disabled automatically by `main.py`
- `output_language` defaults to English and is appropriate for journal submission
- if no valid API key is configured, the workflow cannot complete the LLM-driven steps

For DeepSeek-compatible usage:

```bash
set DEEPSe workflow with a manuscript dataset:

```bash
python main.py --data path/to/your_data.csv --output submission_output --target Tungsten --study-area YourStudyArea
```

After a successful run, first inspect:

- `output/reports/comprehensive_report.md`
- `output/data/prediction_results.csv`
- `output/data/complete_results.json`
- `output/model_viz/` and `output/SOM result/` when those branches are enabledEEK_API_KEY=your_api_key
set GEOCHEM_LLM_PROVIDER=deepseek
set GEOCHEM_LLM_MODEL=deepseek-chat
```

For Qwen-compatible usage:

```bash
set QWEN_API_KEY=your_api_key
set GEOCHEM_LLM_PROVIDER=qwen
set GEOCHEM_LLM_MODEL=qwen3-max
```

Example `.env` file:

```bash
GEOCHEM_LLM_PROVIDER=deepseek
GEOCHEM_LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=your_api_key
```

## Quick Start

Run the workflow with the included synthetic example:

```bash
python main.py --data examples/synthetic_geochem_data.csv --output examples/tutorial_output --target Tungsten --study-area SyntheticDemo
```

Run th

## Command-Line Arguments

- `--data`: required path to the input dataset
- `--output`: required output directory
- `--target`: required target deposit type, for example `Tungsten`
- `--study-area`: optional study-area name written into the workflow context
- `--output-language`: output language, currently `en` or `zh`, default `en`
- `--structured-output` / `--no-structured-output`: enable or disable structured parsing of model outputs
- `--reflection` / `--no-reflection`: enable or disable the reflection/self-revision stage
- `--reflection-max-rounds`: reflection iterations, clamped internally to `0-3`
- `--knowledge` / `--no-knowledge`: enable or disable target-domain knowledge retrieval
- `--som-all-elements`: enable the full-element SOM branch in addition to the filtered-element branch
- `--som-use-raw-data`: force SOM to use raw input data instead of the preprocessed table
- `--auto-programming`: enable the optional auto-programming helper
- `--disable-governance`: disable workflow governance checks

## Minimal Review Workflow

For repository inspection or peer review, the shortest validation path is:

1. install dependencies from `requirements.txt`
2. configure one supported LLM provider
3. run the synthetic example command shown above
4. confirm that the workflow finishes and creates `reports/`, `data/`, and at least one prediction-related directory
5. open `reports/comprehensive_report.md` and `data/prediction_results.csv` to verify end-to-end execution

## Input Data Expectations

The workflow accepts tabular datasets in `csv`, `xlsx`, `xls`, or `json` format.

Expected columns:

- `FID`
- numeric geochemical variables such as `W`, `Sn`, `Mo`, `As`, `Cu`, and `Zn`
- coordinate columns such as `Longitude` and `Latitude`
- an optional ore label column such as `Ore`

Practical notes:

- the workflow is designed for tabular geochemical sampling data
- geochemical feature columns should be numeric
- `Longitude` and `Latitude` are the preferred coordinate headers for the submission repository
- label-dependent evaluation outputs such as ROC and PR curves may be skipped when no valid label column is available

## Output Artifacts

The workflow writes a multi-folder result package rather than a single output file. The current code does not always produce one fixed directory tree; instead, it creates a small set of fixed top-level folders plus several optional branch-specific subdirectories.

```text
output/
|-- reports/
|   |-- comprehensive_report.md
|   |-- feature_analysis_and_selection.md
|   |-- target_element_selection.md
|   |-- model_selection_and_evaluation.md
|   |-- data_analysis_report.md
|   |-- preprocessing_strategy.md
|   |-- comprehensive_report_error.md
|   |-- task_plan.md
|   |-- token_analysis_report.md
|   |-- run_log.log
|   |-- target_element_selection_sources.csv
|   |-- all_elements_boxplot.png
|   `-- images/
|       |-- token_distribution.png
|       |-- request_trend.png
|       `-- per_request_usage.png
|-- data/
|   |-- prediction_results.csv
|   |-- high_potential_areas.csv
|   |-- key_element_analysis.csv
|   |-- feature_importance.csv
|   `-- complete_results.json
|-- feature_analysis/
|   |-- correlation_analysis/
|   |   |-- correlation_matrix_{pearson,spearman,kendall}.csv
|   |   |-- high_correlations_*.csv
|   |   `-- correlation_heatmap.png
|   |-- hierarchical_clustering/
|   |   |-- hierarchical_clustering_results.json
|   |   |-- hierarchical_clustering_correlation_heatmap.png
|   |   `-- hierarchical_clustering_dendrogram.png
|   `-- factor_analysis/
|       |-- scree_plot.png
|       |-- factor_loadings_heatmap.png
|       `-- factor_scores.csv
|-- SOM result/
|   `-- som_filtered_elements/ | som_all_elements/ | som_<source>_<branch>/
|       |-- U_Matrix_Enhanced.png
|       |-- cluster_count_selection_reference.png
|       |-- silhouette_reference.png
|       |-- davies_bouldin_reference.png
|       |-- calinski_harabasz_reference.png
|       |-- cluster_count_metric_recommendations.{csv,json}
|       |-- som_qe_te_tuning_results.{csv,json}
|       |-- som_qe_te_tuning_scatter.png
|       |-- som_qe_te_tuning_rank.png
|       |-- main_elements_list.csv
|       |-- known_sample_count_by_cluster.csv
|       |-- known_sample_count_by_cluster.png
|       |-- geological_interpretation.md
|       |-- component_planes/
|       |   |-- *_component_plane.png
|       |   `-- component_plane_value_ranges.csv
|       `-- qe/
|           |-- mineral_anomaly_map.png
|           |-- anomaly.csv
|           |-- arcgis_anomaly.csv
|           |-- element_qe_correlation.csv
|           |-- element_qe_correlation.png
|           |-- mineral_roc_curve.png
|           `-- mineral_pr_curve.png
|-- model_viz/
|   |-- best_model_roc_curve.png
|   |-- best_model_pr_curve.png
|   |-- best_model_score_distribution.png
|   |-- best_model_train_confusion_matrix.png
|   |-- best_model_confusion_matrix.png
|   |-- best_model_predict_confusion_matrix.png
|   |-- best_model_fit_gap.png
|   `-- best_model_permutation_importance.png
|-- Key element analysis results/
|   |-- log_log/
|   |-- Concentration distribution statistics/
|   `-- Spatial distribution statistics/
`-- auto_programming/
    `-- auto_programming_*/
        |-- generated_script.py
        `-- execution_metrics.json
```

Interpretation of the main folders:

- `reports/`: reviewer-facing Markdown documents, workflow plans, token-analysis outputs, and copied report figures
- `data/`: tabular prediction outputs, anomaly statistics, feature-importance tables, and serialized workflow snapshots
- `feature_analysis/`: outputs from correlation analysis, hierarchical clustering, and factor analysis with the current file names used by the code
- `SOM result/`: branch-specific SOM outputs; the exact subdirectory name depends on whether the run uses all elements, filtered elements, or multi-source SOM branches
- `model_viz/`: best-model evaluation figures, including ROC/PR curves, confusion matrices, score distributions, fit-gap plots, and permutation-importance charts
- `Key element analysis results/`: geological-analysis visualizations generated by the geology expert agent
- `auto_programming/`: optional artifacts created only when the auto-programming path is enabled

The exact file set still depends on the enabled modules, available labels, model branch, and optional evaluation settings, but the paths above match the current code structure much more closely than a simplified generic output tree.

## Reproducibility And Limitations

This repository includes a synthetic example dataset for software validation and tutorial execution. If the manuscript uses restricted study-area data, the synthetic example demonstrates the expected input format and runtime behavior, while scientific reproduction of the manuscript results still requires the original study dataset.

Important limitations:

- the synthetic example is provided for software validation only and does not reproduce the scientific findings of the manuscript
- LLM-generated text and some downstream interpretations may vary slightly across providers, model versions, and reruns
- outputs that depend on labels, shapefiles, or optional branches are generated only when the corresponding inputs and switches are available
- some optional outputs, especially `auto_programming/`, `model_viz/`, or additional SOM branches, appear only when those paths are enabled by configuration

## Computational Notes

- the repository targets standard Python execution on a workstation-class environment
- GPU acceleration is not required for the documented workflow path
- memory and runtime depend strongly on dataset size, enabled SOM branches, and the availability of optional evaluation routines
- the heaviest steps are usually LLM calls, SOM-related analysis, and figure generation

## License

This repository is distributed under the MIT License. Keep the `LICENSE` file together with the repository when preparing the journal submission package.

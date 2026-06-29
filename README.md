## GeochemMAF_code
GeochemMAF: A Novel Multi-Agent Framework for Geochemical Anomaly Identification Based on Large Language Models
## Installation
pip install -r requirements.txt
## LLM Configuration
The full-core workflow uses the original LLM-enabled agents. Configure one LLM provider before running the repository.

For DeepSeek-compatible usage:

```bash
set DEEPSEEK_API_KEY=your_api_key
set GEOCHEM_LLM_PROVIDER=deepseek
set GEOCHEM_LLM_MODEL=deepseek-chat
```

For Qwen-compatible usage:

```bash
set QWEN_API_KEY=your_api_key
set GEOCHEM_LLM_PROVIDER=qwen
set GEOCHEM_LLM_MODEL=qwen3-max
```

## Basic Usage

Run the workflow with the included synthetic example:

```bash
python main.py --data examples/synthetic_geochem_data.csv --output examples/tutorial_output --target Tungsten --study-area SyntheticDemo
```

Run the workflow with a manuscript dataset:

```bash
python main.py --data path/to/your_data.csv --output submission_output --target Tungsten --study-area YourStudyArea
```

## Input Data Expectations

The workflow accepts tabular datasets in `csv`, `xlsx`, `xls`, or `json` format.

Typical columns include:

- `FID`
- numeric geochemical variables such as `W`, `Sn`, `Mo`, `As`, `Cu`, and `Zn`
- an optional ore label column such as `Ore`
- coordinate columns such as `Longitude` and `Latitude`

## Output Artifacts

The original reporting module can generate multiple result files, including:

- processed prediction tables
- anomaly analysis exports
- selected target-element reports
- figures and maps
- comprehensive markdown reports
- JSON result snapshots

## Reproducibility Note

This repository includes a synthetic example dataset for validation and tutorial execution. If the manuscript uses restricted study-area data, the synthetic example demonstrates the expected input format and runtime behavior, while scientific reproduction of the manuscript results still requires the original study dataset.

## Additional Documentation

- `docs/INSTALLATION.md`
- `docs/COMPUTATIONAL_REQUIREMENTS.md`
- `docs/REPRODUCIBILITY.md`
- `docs/TUTORIAL.md`
- `docs/USER_GUIDE.md`




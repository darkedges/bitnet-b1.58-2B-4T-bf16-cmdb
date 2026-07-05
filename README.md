# CMDB Synthetic Dataset Generator for BitNet LLM Training

A Python project for generating comprehensive synthetic CMDB datasets suitable for fine-tuning **BitNet 1.58-bit Large Language Models** on infrastructure decision-making tasks.

## Overview

This generator creates 1,000+ application records with:

- **5 Decision Attributes**: TCF Availability, Data Sensitivity, Data Classification, Application Lifecycle, Compliance Frameworks
- **Organizational Context**: Business Units, Departments, Divisions, Technical/Business Owners
- **Infrastructure Metadata**: Cloud Regions, Lifecycle Stages, Regulatory Compliance Requirements
- **Advanced Logic**: 8-tier product recommendation engine with scoring matrix
- **Explainable Output**: Detailed reasoning for each recommendation (perfect for AI training)
- **JSONL Format**: Directly compatible with Hugging Face SFTTrainer

## Quick Start

### 1. Install Dataset-Generator Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the Synthetic Dataset

```bash
python cmdb_synthetic_dataset.py complex_cmdb_synthetic.jsonl 2000 --validate
```

### 3. Fine-Tune BitNet on Google Colab (GPU)

Open `training.ipynb` in Colab with a GPU runtime. Point it at your generated
`complex_cmdb_synthetic.jsonl` (expected under `My Drive/Colab Notebooks/`), then
run all cells. This performs a **full fine-tune** of the bf16 master weights
(not LoRA — see the note below) and produces one local folder inside the Colab
runtime:

- `./bitnet-cmdb-final-merged` — the fine-tuned model as a single
  `model.safetensors` (the folder name is historical; nothing is merged)

The notebook includes an on-GPU smoke test cell — check its output is coherent
and on-format **before** downloading anything. Then download the folder to your
local machine — either from the notebook's Google Drive backup archive, or its
optional Hugging Face Hub push.

> **Why not LoRA?** BitNet quantizes weights to ternary {-1,0,1} at inference.
> A LoRA adapter trains as a full-precision side branch (`quant(W)x + 2BAx`),
> but merging it and re-quantizing computes `quant(W + 2BA)x` — the ternary
> grid rounds the small rank-16 delta into scattered ±1 flips and the merged
> model collapses (verified: it degenerates into repeating a single token).
> Full fine-tuning keeps the learning on the ternary grid, so GGUF conversion
> preserves it.

### 4. (Optional) Sanity-Check the Model Locally

```bash
pip install -r requirements-local.txt
python inference_test.py
```

This loads the fine-tuned bf16 model directly via `transformers` on CPU as a
quick smoke test — it is **not** the efficient ternary BitNet inference path,
and at minutes per response it is far slower than the served model. Prefer the
notebook's on-GPU smoke test; use this only if you skipped it.

### 5. Convert the Fine-Tuned Model to GGUF (i2_s)

Clone Microsoft's BitNet repo as a **sibling** directory of this project:

```console
# From the parent directory of cmdb-generator/
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet

py -3.11 -m venv .venv311
.\.venv311\Scripts\activate

pip install -r requirements.txt

# IMPORTANT: install BitNet's forked gguf-py over the stock pip package —
# only the fork knows the bitnet-25 GGUF architecture the converter emits.
pip install ./3rdparty/llama.cpp/gguf-py
```

Build the binaries (`llama-quantize` is needed by the converter, `llama-server`
by step 6). On Windows this requires CMake and Visual Studio with the
**C++ Clang toolset** installed:

```console
cmake -B build -DBITNET_X86_TL2=OFF -T ClangCL
cmake --build build --config Release
```

Then run the conversion. Notes: the model directory must contain a **single**
`model.safetensors` (the notebook's consolidation cell guarantees this), and
the helper prints "Convert successfully." but exits 0 even on failure — trust
the message / output file, not the exit code.

```console
# convert-helper-bitnet.py expects only one positional argument: <model-directory>
python ./utils/convert-helper-bitnet.py ../cmdb-generator/bitnet-cmdb-final-merged

# output is written inside the model directory as ggml-model-i2s-bitnet.gguf
# (a ~9GB ggml-model-f32-bitnet.gguf intermediate is left behind — safe to delete)
```

### 6. Serve the Quantized Model

```console
python run_inference.py -m ../cmdb-generator/bitnet-cmdb-final-merged/ggml-model-i2s-bitnet.gguf --host 0.0.0.0 --port 8080
```

This starts an OpenAI-API-compatible server on port 8080, which `cmdbAgent.html` talks to.

### 7. Run the Demo UI and Feedback Logger

```bash
python app_logger.py    # Terminal 1 — feedback logging server on port 5000
```

Then open `cmdbAgent.html` in a browser (Terminal 2 is just the running server above).
Submit a request through the UI, and use the Yes/No feedback control to log corrections
to `cmdb_feedback_loop.csv` for future retraining.

### 8. Output Example

```json
{
  "instruction": "Analyze the CMDB application profile and recommend...",
  "input": "Application: App-Example-1234\nBusiness Unit: Retail Banking\n...",
  "output": "Recommended Product: Tier-3 General Purpose Cloud\nRisk Level: Medium\nReasoning: ...",
  "metadata": {
    "app_name": "App-Example-1234",
    "tcf_availability": "Silver",
    "sensitivity": "Private",
    "classification": "Confidential",
    "recommended_product": "Tier-3 General Purpose Cloud",
    "risk_level": "Medium"
  }
}
```

## Features

### Decision Attributes

| Attribute               | Values                                                   | Purpose                          |
| ----------------------- | -------------------------------------------------------- | -------------------------------- |
| **TCF Availability**    | Gold, Silver, Bronze                                     | System criticality & uptime SLAs |
| **Data Sensitivity**    | Restricted, Private, Public                              | Access control requirements      |
| **Data Classification** | Highly Confidential, Confidential, Internal, Public      | Regulatory sensitivity           |
| **Lifecycle**           | Inception, Growth, Mature, Decline, Sunset, Experimental | Investment & support level       |
| **Compliance**          | GDPR, HIPAA, PCI-DSS, SOC2                               | Regulatory frameworks            |

### 8-Tier Infrastructure Products

1. **Tier-0**: Dedicated Cloud (Extreme availability + confidentiality)
2. **Tier-1**: Enhanced Private Cloud (Multi-framework compliance)
3. **Tier-2**: Managed Private VPC (Restricted sensitivity)
4. **Tier-3**: General Purpose Cloud *(default)*
5. **Tier-4**: Cost-Optimized Public Cloud
6. **Tier-5**: Container-Based (ECS/K8s)
7. **Hybrid**: On-Premise Managed
8. **On-Prem**: On-Premise Dedicated (HIPAA + Gold)

### Advanced Recommendation Logic

The `get_recommendation()` function uses:
- **Composite scoring** (availability + sensitivity + classification)
- **Compliance severity** (multi-framework requirements)
- **Lifecycle impact** (investment level adjustment)
- **Risk classification** (derived from product tier)

## Usage Examples

### Generate 10,000 records with validation

```bash
python cmdb_synthetic_dataset.py large_dataset.jsonl 10000 --validate
```

### Use in Hugging Face SFTTrainer

```python
from datasets import load_dataset
from trl import SFTTrainer

# Load JSONL data
dataset = load_dataset('json', data_files='complex_cmdb_synthetic.jsonl')

# Fine-tune BitNet
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    tokenizer=tokenizer,
)
trainer.train()
```

## Performance

Generated statistics from 1,000 records:

```
TCF Availability:
  Bronze : 338 (33.8%)
  Gold   : 321 (32.1%)
  Silver : 341 (34.1%)

Data Sensitivity:
  Public     : 327 (32.7%)
  Private    : 348 (34.8%)
  Restricted : 325 (32.5%)

Recommended Products:
  Tier-3 General Purpose Cloud       : 287 (28.7%)
  Tier-2 Managed Private VPC         : 192 (19.2%)
  Tier-4 Cost-Optimized Public Cloud : 165 (16.5%)
  Tier-0 Dedicated Cloud             : 134 (13.4%)

Risk Levels:
  Critical : 153 (15.3%)
  High     : 287 (28.7%)
  Medium   : 375 (37.5%)
  Low      : 185 (18.5%)
```

## Project Structure

```
cmdb-generator/
├── cmdb_synthetic_dataset.py    # Synthetic CMDB dataset generator
├── training.ipynb               # Colab notebook: LoRA fine-tune + merge BitNet
├── inference_test.py            # Local pre-quantization sanity check (bf16, CPU)
├── app_logger.py                # Flask feedback-logging server (port 5000)
├── cmdbAgent.html                # Demo UI, talks to the bitnet.cpp server (port 8080)
├── requirements.txt             # Deps for the dataset generator
├── requirements-local.txt       # Deps for inference_test.py + app_logger.py
├── README.md                    # This file
└── complex_cmdb_synthetic.jsonl  # Generated dataset (gitignored, regenerate as needed)
```

## Customization

### Modify Business Logic

Edit `BUSINESS_UNITS`, `REGIONS`, `COMPLIANCE_FRAMEWORKS`, etc. in the script:

```python
BUSINESS_UNITS = [
    "Your-BU-1", "Your-BU-2", "Your-BU-3"
]
```

### Custom Recommendation Rules

Modify the `get_recommendation()` function:

```python
def get_recommendation(...):
    if "Your-Compliance" in compliance:
        return "Your-Product-Tier", "Custom reasoning..."
```

### Fine-tune Generation Parameters

- **Faker seed**: Change from `42` for different data
- **Record count**: Pass as CLI argument
- **Output format**: Modify JSONL structure in `generate_application_record()`

## Testing

```bash
# Generate 10 records for quick testing
python cmdb_synthetic_dataset.py test.jsonl 10 --validate

# View first record
python -c "import json; print(json.dumps(json.loads(open('test.jsonl').readline()), indent=2))"
```

## Requirements

Dataset generation (`requirements.txt`):
- Python 3.8+
- faker >= 18.0
- pandas >= 1.5.0
- numpy >= 1.24.0

Local inference sanity-check + feedback server (`requirements-local.txt`):
- torch, transformers (pinned commit — needed for BitNet architecture support), flask, flask-cors

Colab-side fine-tuning dependencies (`trl`, `accelerate`, `peft`) are installed inline in
`training.ipynb` since training only ever runs on Colab, not locally.

## Use Cases

1. **BitNet LLM Fine-Tuning**: Train decision models for infrastructure architecture
2. **Few-Shot Learning**: In-context demonstration for prompt engineering
3. **Synthetic Audit Trails**: Generate historical decision logs
4. **Test/Validation**: Ground-truth datasets for model evaluation
5. **CMDB Simulation**: Infrastructure decision scenario testing

## Performance Metrics

- **Generation rate**: ~1,200 records/second
- **File size**: ~2.5 MB per 1,000 records
- **Memory**: <100 MB for 1,000+ records
- **Reproducibility**: Seeded with Faker seed=42

## References

- [BitNet 1.58-bit Paper](https://arxiv.org/abs/2402.17764)
- [Hugging Face SFTTrainer Docs](https://huggingface.co/docs/trl/)
- [Faker Documentation](https://faker.readthedocs.io/)
- [JSONL Format](https://jsonlines.org/)

## License

MIT - Use freely for research and production applications.

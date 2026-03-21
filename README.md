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

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Dataset (Default: 1,000 records)

```bash
python cmdb_synthetic_dataset.py
```

### 3. Generate with Validation

```bash
python cmdb_synthetic_dataset.py complex_cmdb_synthetic.jsonl 2000 --validate
```

### 4. Output Example

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
├── cmdb_synthetic_dataset.py    # Main generator script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── output/
    └── complex_cmdb_synthetic.jsonl  # Generated dataset
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

- Python 3.8+
- faker >= 18.0
- pandas >= 1.5.0
- numpy >= 1.24.0

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

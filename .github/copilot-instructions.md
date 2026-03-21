# GitHub Copilot Instructions for CMDB Generator

## Project Overview

**CMDB Synthetic Dataset Generator** - A Python project for generating training data for BitNet 1.58-bit LLMs focused on infrastructure decision-making.

**Tech Stack**: Python 3.8+, Faker, Pandas  
**Output Format**: JSONL (Hugging Face SFTTrainer compatible)

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Generate 1,000 records
python cmdb_synthetic_dataset.py

# Generate 5,000 records with validation
python cmdb_synthetic_dataset.py data.jsonl 5000 --validate

# Test with 10 records
python cmdb_synthetic_dataset.py test.jsonl 10
```

## Key Files

- `cmdb_synthetic_dataset.py` - Main generator with decision logic
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

## Architecture

### Core Components

1. **Configuration**: Business units, compliance frameworks, product tiers
2. **Decision Engine**: `get_recommendation()` function with scoring matrix
3. **Data Generator**: `generate_application_record()` creates CMDB records
4. **Validation**: Distribution analysis and statistics

### Decision Matrix (8 Tiers)

- **Tier-0/1**: Critical + highly confidential
- **Tier-2/3**: Standard business + moderate controls
- **Tier-4/5**: Non-critical + cost-optimized
- **On-Prem**: HIPAA + Gold availability

## Customization Points

### Modify Business Attributes
Edit configuration lists (lines ~40-80):
- `BUSINESS_UNITS`
- `DEPARTMENTS`, `DIVISIONS`
- `COMPLIANCE_FRAMEWORKS`
- `REGIONS`

### Customize Decision Logic
Modify `get_recommendation()` function (lines ~120-200):
- Add custom scoring rules
- Change product recommendations
- Update reasoning explanations

### Adjust Generator Output
Modify `generate_application_record()` (lines ~220-280):
- Add/remove metadata fields
- Change context format
- Customize instruction text

## Testing & Validation

```bash
# Quick test: 10 records
python cmdb_synthetic_dataset.py test.jsonl 10 --validate

# Check output structure
python -c "import json; print(json.dumps(json.loads(open('test.jsonl').readline()), indent=2))"

# Count records
wc -l complex_cmdb_synthetic.jsonl

# View distribution
python cmdb_synthetic_dataset.py data.jsonl 100 --validate
```

## Performance notes

- Generation: ~1,200 records/sec
- 1,000 records = ~2.5 MB
- Seeded (reproducible): Faker seed=42

## Next Steps

1. Install: `pip install -r requirements.txt`
2. Generate: `python cmdb_synthetic_dataset.py`
3. Validate: `--validate` flag for statistics
4. Use with Hugging Face SFTTrainer for BitNet fine-tuning

---

**For detailed documentation**, see README.md  
**For API details**, see docstrings in cmdb_synthetic_dataset.py

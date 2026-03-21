#!/usr/bin/env python3
"""
CMDB Synthetic Dataset Generator for BitNet 1.58-bit LLM Training

Generates 1,000+ unique application records with comprehensive CMDB attributes,
decision logic, and reasoning fields compatible with Hugging Face SFTTrainer.

Requirements:
- faker >= 18.0
- pandas >= 1.5.0

Usage:
    python cmdb_synthetic_dataset.py [output_file] [num_records]
    python cmdb_synthetic_dataset.py output.jsonl 5000
"""

import json
import random
import argparse
import sys
import os
from datetime import datetime
from typing import Tuple, Dict, List, Any
from faker import Faker

# Initialize Faker with seeding for reproducibility
fake = Faker()
Faker.seed(42)
random.seed(42)

# ============================================================================
# Configuration
# ============================================================================

BUSINESS_UNITS = [
    "Retail Banking", "Global Markets", "Wealth Management", 
    "Corporate Risk", "HR", "Operations", "Technology", "Compliance", 
    "Treasury", "Investment Banking"
]

DEPARTMENTS = [
    "Infrastructure", "Applications", "Data", "Security", 
    "DevOps", "Architecture", "Product", "Engineering"
]

DIVISIONS = [
    "Frontend", "Backend", "Database", "Integration", 
    "Analytics", "Mobile", "AI/ML", "Cloud"
]

TCF_LEVELS = ["Gold", "Silver", "Bronze"]
SENSITIVITY = ["Restricted", "Private", "Public"]
CLASSIFICATION = [
    "Highly Confidential", "Confidential", "Internal", "Public"
]

LIFECYCLE_STAGES = [
    "Inception", "Growth", "Mature", "Decline", "Sunset", "Experimental"
]

REGIONS = [
    "us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", 
    "ap-northeast-1", "ca-central-1", "eu-central-1", "uk-south"
]

COMPLIANCE_FRAMEWORKS: List[List[str]] = [
    ["GDPR"],
    ["PCI-DSS"],
    ["HIPAA"],
    ["SOC2"],
    ["GDPR", "PCI-DSS"],
    ["GDPR", "SOC2"],
    ["PCI-DSS", "SOC2"],
    ["GDPR", "PCI-DSS", "HIPAA"],
    []
]

PRODUCT_TIERS = [
    "Tier-0 Dedicated Cloud",
    "Tier-1 Enhanced Private Cloud",
    "Tier-2 Managed Private VPC",
    "Tier-3 General Purpose Cloud",
    "Tier-4 Cost-Optimized Public Cloud",
    "Tier-5 Container-Based (ECS/K8s)",
    "Hybrid-On-Premise Managed",
    "On-Premise Dedicated"
]

# Reusable owner pools create realistic many-to-many ownership relationships.
# This ensures a business owner can appear with multiple technical owners and vice versa.
BUSINESS_OWNER_POOL_SIZE = 75
TECHNICAL_OWNER_POOL_SIZE = 75
NEW_OWNER_PROBABILITY = 0.15

BUSINESS_OWNER_POOL: List[str] = [fake.name() for _ in range(BUSINESS_OWNER_POOL_SIZE)]
TECHNICAL_OWNER_POOL: List[str] = [fake.name() for _ in range(TECHNICAL_OWNER_POOL_SIZE)]

# ============================================================================
# Decision Logic Engine
# ============================================================================

def get_recommendation(
    tcf_availability: str,
    sensitivity: str,
    classification: str,
    lifecycle: str,
    compliance: List[str]
) -> Tuple[str, str]:
    """
    Advanced business logic for infrastructure product selection.
    
    Decision Matrix prioritizes:
    - Tier-0/1: Extreme availability + confidentiality
    - Tier-2: Restricted data + medium availability
    - Tier-3: Standard business requirements
    - Tier-4: Public/non-critical workloads
    - Tier-5: Containerized microservices
    - On-Premise: Highly regulated + local data residency
    
    Args:
        tcf_availability: TCF availability level (Gold/Silver/Bronze)
        sensitivity: Data sensitivity (Restricted/Private/Public)
        classification: Data confidentiality (Highly Confidential/Confidential/Internal/Public)
        lifecycle: Application lifecycle stage
        compliance: List of required compliance frameworks
    
    Returns:
        Tuple of (product_name, reasoning_explanation)
    """
    
    # Calculate composite criticality score
    availability_score = {"Gold": 3, "Silver": 2, "Bronze": 1}[tcf_availability]
    sensitivity_score = {"Restricted": 3, "Private": 2, "Public": 1}[sensitivity]
    classification_score = {
        "Highly Confidential": 3,
        "Confidential": 2,
        "Internal": 1,
        "Public": 0
    }[classification]
    
    composite_score = availability_score + sensitivity_score + classification_score
    compliance_severity = len([f for f in compliance if f in ["HIPAA", "PCI-DSS", "GDPR"]])
    
    # Decision logic tree
    if composite_score >= 8 or (classification == "Highly Confidential" and tcf_availability == "Gold"):
        return PRODUCT_TIERS[0], (
            f"Critical infrastructure required: {classification} data (Gold availability, "
            f"{sensitivity} sensitivity) demands isolated, highest-uptime Tier-0 "
            f"infrastructure with dedicated resources."
        )
    
    if (compliance_severity >= 2 or sensitivity == "Restricted") and tcf_availability in ["Silver", "Gold"]:
        return PRODUCT_TIERS[1], (
            f"Enhanced private cloud: Multi-framework compliance "
            f"({', '.join(compliance)}) with {tcf_availability}-level availability "
            f"demands enhanced isolation and dedicated infrastructure."
        )
    
    if sensitivity == "Restricted" or (tcf_availability == "Silver" and 
                                        classification in ["Confidential", "Highly Confidential"]):
        return PRODUCT_TIERS[2], (
            f"Managed private VPC: {sensitivity} sensitivity and "
            f"{classification} classification with {tcf_availability} availability demand "
            f"network isolation and encryption."
        )
    
    if tcf_availability == "Bronze" and lifecycle in ["Inception", "Experimental"]:
        return PRODUCT_TIERS[5], (
            f"Container-based infrastructure: {lifecycle}-stage application "
            f"with Bronze availability is suitable for containerized deployment."
        )
    
    if lifecycle == "Sunset":
        return PRODUCT_TIERS[4], (
            f"Cost-optimized public cloud: {lifecycle}-stage application with "
            f"minimal infrastructure investment. {classification} data supports public cloud."
        )
    
    if sensitivity == "Public" and classification == "Public":
        return PRODUCT_TIERS[4], (
            f"General public cloud: {sensitivity} data and {classification} "
            f"classification allow standard public cloud. Compliance: {', '.join(compliance) or 'Standard'}."
        )
    
    if "Gold" in tcf_availability and "HIPAA" in compliance:
        return PRODUCT_TIERS[7], (
            f"On-premise mandatory: HIPAA compliance with Gold availability requires "
            f"customer-managed infrastructure with local data residency."
        )
    
    # Default to Tier-3
    return PRODUCT_TIERS[3], (
        f"General purpose cloud: Standard business requirements with {tcf_availability} "
        f"availability and {classification} classification. Compliance: {', '.join(compliance) or 'Standard'}."
    )


# ============================================================================
# Synthetic Data Generation
# ============================================================================

def choose_owner_name(owner_pool: List[str], new_owner_probability: float = NEW_OWNER_PROBABILITY) -> str:
    """Pick an owner from a shared pool, occasionally introducing a new owner."""
    if random.random() < new_owner_probability:
        new_owner = fake.name()
        owner_pool.append(new_owner)
        return new_owner
    return random.choice(owner_pool)

def generate_application_record() -> Dict[str, Any]:
    """Generate a single CMDB application record."""
    
    tcf_availability = random.choice(TCF_LEVELS)
    sensitivity = random.choice(SENSITIVITY)
    classification = random.choice(CLASSIFICATION)
    
    business_unit = random.choice(BUSINESS_UNITS)
    department = random.choice(DEPARTMENTS)
    division = random.choice(DIVISIONS)
    
    business_owner_name = choose_owner_name(BUSINESS_OWNER_POOL)
    technical_owner_name = choose_owner_name(TECHNICAL_OWNER_POOL)
    
    lifecycle = random.choice(LIFECYCLE_STAGES)
    region = random.choice(REGIONS)
    compliance = random.choice(COMPLIANCE_FRAMEWORKS)
    
    app_name = f"App-{fake.word().capitalize()}-{random.randint(1000, 9999)}"
    product, reasoning = get_recommendation(tcf_availability, sensitivity, classification, lifecycle, compliance)
    
    risk_level = "Critical" if product in [PRODUCT_TIERS[0], PRODUCT_TIERS[7]] else \
                 "High" if product in [PRODUCT_TIERS[1], PRODUCT_TIERS[2]] else \
                 "Medium" if product in [PRODUCT_TIERS[3], PRODUCT_TIERS[5]] else \
                 "Low"
    
    detailed_context = (
        f"Application: {app_name}\n"
        f"Business Unit: {business_unit}\n"
        f"Organization: {department}/{division}\n"
        f"Business Owner: {business_owner_name}\n"
        f"Technical Owner: {technical_owner_name}\n"
        f"TCF Availability: {tcf_availability}\n"
        f"Data Sensitivity: {sensitivity}\n"
        f"Data Classification: {classification}\n"
        f"Lifecycle Stage: {lifecycle}\n"
        f"Preferred Region: {region}\n"
        f"Compliance: {', '.join(compliance) if compliance else 'None'}"
    )
    
    output_context = (
        f"Recommended Product: {product}\n"
        f"Risk Level: {risk_level}\n"
        f"Reasoning: {reasoning}"
    )
    
    return {
        "instruction": (
            "Analyze the CMDB application profile and recommend the most appropriate "
            "infrastructure tier based on availability, sensitivity, compliance, and lifecycle requirements."
        ),
        "input": detailed_context,
        "output": output_context,
        "metadata": {
            "app_name": app_name,
            "business_unit": business_unit,
            "business_owner": business_owner_name,
            "technical_owner": technical_owner_name,
            "tcf_availability": tcf_availability,
            "sensitivity": sensitivity,
            "classification": classification,
            "lifecycle": lifecycle,
            "region": region,
            "compliance": compliance,
            "recommended_product": product,
            "risk_level": risk_level
        }
    }


def generate_dataset(num_records: int = 1000, output_file: str = "complex_cmdb_synthetic.jsonl") -> None:
    """Generate synthetic CMDB dataset and write to JSONL file."""
    
    print(f"🔄 Generating {num_records} synthetic CMDB records...")
    print(f"   Output: {output_file}\n")
    
    start_time = datetime.now()
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for i in range(num_records):
                record = generate_application_record()
                f.write(json.dumps(record) + "\n")
                
                if (i + 1) % 250 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = (i + 1) / elapsed
                    print(f"   ✓ {i + 1}/{num_records} ({rate:.0f} rec/sec)")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        
        print(f"\n✅ Generated {num_records} records in {file_size_mb:.2f} MB ({elapsed:.2f}s)")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


def validate_dataset(output_file: str) -> None:
    """Validate dataset and print distribution statistics."""
    
    print(f"\n📊 Validating {output_file}...\n")
    
    stats: Dict[str, Any] = {
        "total": 0,
        "tcf": {}, "sensitivity": {}, "products": {}, "risk": {}
    }
    
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                stats["total"] += 1
                meta = record["metadata"]
                
                tcf = meta.get("tcf_availability")
                stats["tcf"][tcf] = stats["tcf"].get(tcf, 0) + 1
                
                sens = meta.get("sensitivity")
                stats["sensitivity"][sens] = stats["sensitivity"].get(sens, 0) + 1
                
                prod = meta.get("recommended_product")
                stats["products"][prod] = stats["products"].get(prod, 0) + 1
                
                risk = meta.get("risk_level")
                stats["risk"][risk] = stats["risk"].get(risk, 0) + 1
        
        print(f"✅ Valid records: {stats['total']}\n")
        
        print("📈 TCF Availability Distribution:")
        for tcf, count in sorted(stats["tcf"].items()):
            pct = (count / stats["total"]) * 100
            print(f"  {tcf:10} : {count:4} ({pct:5.1f}%)")
        
        print("\n📈 Data Sensitivity Distribution:")
        for sens, count in sorted(stats["sensitivity"].items()):
            pct = (count / stats["total"]) * 100
            print(f"  {sens:12} : {count:4} ({pct:5.1f}%)")
        
        print("\n📈 Top Recommended Products:")
        for prod, count in sorted(stats["products"].items(), key=lambda x: -x[1])[:5]:
            pct = (count / stats["total"]) * 100
            print(f"  {prod:35} : {count:4} ({pct:5.1f}%)")
        
        print("\n📈 Risk Distribution:")
        for risk, count in sorted(stats["risk"].items()):
            pct = (count / stats["total"]) * 100
            print(f"  {risk:10} : {count:4} ({pct:5.1f}%)")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """Main entry point for the CMDB dataset generator."""
    parser = argparse.ArgumentParser(
        description="CMDB Synthetic Dataset Generator for BitNet LLM Training"
    )
    parser.add_argument("output_file", nargs="?", default="complex_cmdb_synthetic.jsonl",
                       help="Output JSONL file")
    parser.add_argument("num_records", nargs="?", type=int, default=1000,
                       help="Number of records to generate")
    parser.add_argument("--validate", action="store_true", help="Validate after generation")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CMDB Synthetic Dataset Generator for BitNet 1.58-bit LLM")
    print("=" * 70)
    
    generate_dataset(args.num_records, args.output_file)
    
    if args.validate:
        validate_dataset(args.output_file)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

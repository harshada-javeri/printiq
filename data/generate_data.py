"""
Generate synthetic industrial print job dataset.

Simulates EPSON-grade manufacturing systems with realistic feature correlations
and failure modes. Reproducible via fixed random seed.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configuration
N_SAMPLES = 5000
OUTPUT_DIR = Path(__file__).parent / "raw"
OUTPUT_FILE = OUTPUT_DIR / "print_jobs.csv"

# Feature bounds (realistic ranges for industrial printers)
FEATURE_BOUNDS = {
    "printer_age": (1, 60),  # months
    "ink_viscosity": (25, 45),  # centiPoise (cP)
    "paper_gsm": (70, 300),  # grams per square meter
    "humidity": (30, 70),  # percentage
    "temperature": (18, 28),  # Celsius
    "coverage_pct": (10, 95),  # percentage of page covered
}

HEAD_TYPES = ["piezo", "thermal"]


def generate_failure_probability(row: dict) -> float:
    """
    Compute failure probability from features.
    
    Realistic failure modes:
    - Old printers (age > 48 months) 3x higher failure risk
    - High humidity increases nozzle clogging risk
    - Extreme temperatures degrade ink
    - High coverage stresses the nozzles
    """
    prob = 0.05  # Base failure rate
    
    # Age effect: exponential degradation
    if row["printer_age"] > 48:
        prob += 0.08 * (row["printer_age"] - 48) / 12
    else:
        prob += 0.02 * (1 - row["printer_age"] / 60)
    
    # Humidity effect: inverted U-shape (optimal at 45-55%)
    humidity_diff = abs(row["humidity"] - 50)
    prob += 0.05 * (humidity_diff / 20) ** 2
    
    # Temperature effect: damages outside optimal 20-25°C
    temp_diff = abs(row["temperature"] - 22)
    prob += 0.03 * (temp_diff / 5) ** 2
    
    # High coverage stresses system
    prob += 0.06 * (row["coverage_pct"] / 100) ** 1.5
    
    # Head type: thermal slightly more robust
    if row["head_type"] == "piezo":
        prob += 0.02
    
    # Viscosity: optimal range is 32-38 cP
    viscosity_optimal = abs(row["ink_viscosity"] - 35)
    prob += 0.04 * (viscosity_optimal / 10) ** 1.5
    
    # Nozzle cleanliness is critical
    if not row["nozzles_clean"]:
        prob += 0.15
    
    return min(prob, 0.95)  # Cap at 95%


def generate_quality_score(row: dict, failed: bool) -> float:
    """
    Compute quality score (0-100) from features.
    
    Failed jobs score 0-20 (degraded output).
    Successful jobs depend on:
    - Machine age (newer = better)
    - Environment stability
    - Coverage uniformity
    """
    if failed:
        return np.random.uniform(0, 20)
    
    # Base score for successful jobs
    score = 90.0
    
    # Penalize age
    score -= (row["printer_age"] / 60) * 8
    
    # Humidity penalty (worse further from 50%)
    humidity_diff = abs(row["humidity"] - 50)
    score -= (humidity_diff / 20) ** 2 * 5
    
    # Temperature penalty
    temp_diff = abs(row["temperature"] - 22)
    score -= (temp_diff / 5) ** 2 * 3
    
    # Viscosity penalty
    viscosity_diff = abs(row["ink_viscosity"] - 35)
    score -= (viscosity_diff / 10) ** 1.5 * 4
    
    # High coverage can degrade quality
    if row["coverage_pct"] > 80:
        score -= (row["coverage_pct"] - 80) / 20 * 3
    
    # Piezo heads produce slightly better quality
    if row["head_type"] == "thermal":
        score -= 2
    
    # Add small random noise
    score += np.random.normal(0, 1.5)
    
    return max(min(score, 100), 40)  # Bound between 40-100


def generate_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """Generate synthetic print job dataset."""
    
    data = {
        "printer_age": np.random.uniform(*FEATURE_BOUNDS["printer_age"], n_samples),
        "head_type": np.random.choice(HEAD_TYPES, n_samples),
        "ink_viscosity": np.random.uniform(*FEATURE_BOUNDS["ink_viscosity"], n_samples),
        "paper_gsm": np.random.uniform(*FEATURE_BOUNDS["paper_gsm"], n_samples),
        "humidity": np.random.uniform(*FEATURE_BOUNDS["humidity"], n_samples),
        "temperature": np.random.uniform(*FEATURE_BOUNDS["temperature"], n_samples),
        "coverage_pct": np.random.uniform(*FEATURE_BOUNDS["coverage_pct"], n_samples),
        "nozzles_clean": np.random.choice([True, False], n_samples, p=[0.85, 0.15]),
    }
    
    df = pd.DataFrame(data)
    
    # Generate targets based on features
    df["failure_prob"] = df.apply(generate_failure_probability, axis=1)
    df["failed"] = (np.random.uniform(0, 1, n_samples) < df["failure_prob"]).astype(int)
    df["quality_score"] = df.apply(
        lambda row: generate_quality_score(row, bool(row["failed"])),
        axis=1
    )
    
    # Clean up intermediate column
    df = df.drop("failure_prob", axis=1)
    
    # Reorder columns for clarity
    cols = [
        "printer_age", "head_type", "ink_viscosity", "paper_gsm",
        "humidity", "temperature", "coverage_pct", "nozzles_clean",
        "failed", "quality_score"
    ]
    df = df[cols]
    
    return df


def main():
    """Generate and save dataset."""
    print("Generating synthetic print job dataset...")
    print(f"Samples: {N_SAMPLES:,}")
    print(f"Random seed: {RANDOM_SEED}")
    
    # Create output directory if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    df = generate_dataset(N_SAMPLES)
    
    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Dataset saved to {OUTPUT_FILE}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Shape: {df.shape}")
    print(f"\nTarget Distribution:")
    print(f"  Failed: {df['failed'].sum()} ({df['failed'].mean()*100:.1f}%)")
    print(f"  Success: {(1-df['failed']).sum()} ({(1-df['failed']).mean()*100:.1f}%)")
    print(f"\nQuality Score Stats:")
    print(f"  Mean: {df['quality_score'].mean():.2f}")
    print(f"  Std: {df['quality_score'].std():.2f}")
    print(f"  Min: {df['quality_score'].min():.2f}")
    print(f"  Max: {df['quality_score'].max():.2f}")
    print(f"\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()

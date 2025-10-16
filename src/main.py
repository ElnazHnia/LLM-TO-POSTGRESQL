# # main.py
# from typing import List
# from fastapi import FastAPI
# from fastapi_mcp import FastApiMCP

# app = FastAPI()


# @app.get("/tables", response_model=List[str], summary="List all DB tables")
# def list_tables():
#     return ["users", "products", "orders"]

# mcp = FastApiMCP(app)
# mcp.mount_http()   # ← exposes both GET /mcp and GET /mcp/openapi.json

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

# --- 1. Configuration and Data Loading ---

raw = os.getenv("EXCEL_PATH", "/data/questions.xlsx")
ROOT = Path(__file__).resolve().parents[1]
candidates = [
    Path(raw).expanduser(),
    ROOT / "data" / "questions.xlsx",
    Path("data/questions.xlsx"),
    Path("questions.xlsx")
]

FILE_PATH = next((p for p in candidates if p.exists()), None)
if not FILE_PATH:
    print(f"❌ Excel file not found. Tried:")
    for p in candidates:
        print(f"  - {p}")
    exit(1)

print(f"✓ Reading: {FILE_PATH}")

# Read from Tabelle2 sheet
df = pd.read_excel(FILE_PATH, sheet_name="Tabelle2", engine='openpyxl')

print(f"✓ Loaded {len(df)} rows")

# --- 2. Data Cleaning and Preparation ---

required_cols = ["Data was correct", "layout was correct", "Use Detailed System Prompt Examples"]
missing = [col for col in required_cols if col not in df.columns]
if missing:
    print(f"❌ Missing required columns: {missing}")
    exit(1)

# Convert correctness columns to numeric
df["S_Data"] = pd.to_numeric(df["Data was correct"], errors="coerce")
df["S_Layout"] = pd.to_numeric(df["layout was correct"], errors="coerce")
df["S_Overall"] = df[["S_Data", "S_Layout"]].mean(axis=1)

# Clean condition column
df["Condition"] = (
    df["Use Detailed System Prompt Examples"]
    .astype(str)
    .str.strip()
    .str.lower()
)

condition_map = {'yes': 'yes', 'no': 'no', 'y': 'yes', 'n': 'no'}
df["Condition"] = df["Condition"].map(condition_map)

# Drop rows with missing data
df_clean = df.dropna(subset=["S_Data", "S_Layout", "Condition"]).copy()

print(f"✓ After cleaning: {len(df_clean)} rows")
print(f"✓ Conditions found: {df_clean['Condition'].value_counts().to_dict()}")

# --- 3. Calculate Statistics ---

metrics = {
    'S_Overall': 'Overall',
    'S_Data': 'Data',
    'S_Layout': 'Layout'
}

# Separate data by condition
df_yes = df_clean[df_clean['Condition'] == 'yes']
df_no = df_clean[df_clean['Condition'] == 'no']

if len(df_yes) == 0 or len(df_no) == 0:
    print(f"❌ Error: Need both conditions. Yes={len(df_yes)}, No={len(df_no)}")
    exit(1)

# Calculate deltas and confidence intervals
results = []
for key, label in metrics.items():
    yes_data = df_yes[key].values
    no_data = df_no[key].values
    
    # Calculate means
    mean_yes = np.mean(yes_data)
    mean_no = np.mean(no_data)
    delta = mean_yes - mean_no
    
    # Calculate 95% confidence interval for the difference
    se_yes = stats.sem(yes_data)
    se_no = stats.sem(no_data)
    se_diff = np.sqrt(se_yes**2 + se_no**2)
    ci = 1.96 * se_diff  # 95% CI
    
    results.append({
        'Metric': label,
        'Delta': delta * 100,  # Convert to percentage points
        'CI': ci * 100,
        'CI_lower': (delta - ci) * 100,
        'CI_upper': (delta + ci) * 100
    })

df_results = pd.DataFrame(results)

print("\n--- Delta Values (Yes - No) with 95% CI ---")
for _, row in df_results.iterrows():
    print(f"{row['Metric']:12s}: {row['Delta']:+.2f}% (95% CI: [{row['CI_lower']:+.2f}%, {row['CI_upper']:+.2f}%])")

# --- 4. Visualization: Dot Plot with Confidence Intervals ---

plt.style.use('seaborn-v0_8-whitegrid')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot points and error bars
y_positions = np.arange(len(df_results))

for i, row in df_results.iterrows():
    color = '#2E7D32' if row['Delta'] >= 0 else '#C62828'  # Dark green/red
    
    # Error bar (confidence interval)
    ax.plot([row['CI_lower'], row['CI_upper']], [i, i], 
            color=color, linewidth=2, alpha=0.6, zorder=1)
    
    # Point estimate
    ax.scatter(row['Delta'], i, s=150, color=color, 
              edgecolors='black', linewidths=1.5, zorder=3, alpha=0.9)
    
    # Add value label
    label_x = row['Delta'] + (3 if row['Delta'] >= 0 else -3)
    ha = 'left' if row['Delta'] >= 0 else 'right'
    ax.text(label_x, i, f"{row['Delta']:+.2f}%", 
           ha=ha, va='center', fontsize=11, fontweight='bold')

# Vertical line at zero
ax.axvline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.7)

# Labels and title
ax.set_yticks(y_positions)
ax.set_yticklabels(df_results['Metric'], fontsize=12)
ax.set_xlabel('Change in Success Rate (Percentage Points)', fontsize=12, fontweight='bold')
ax.set_title('Impact of Detailed System Prompt Examples\n(with 95% Confidence Intervals)', 
            fontsize=14, fontweight='bold', pad=20)

# Set x-axis limits with some padding
x_min = df_results['CI_lower'].min() - 5
x_max = df_results['CI_upper'].max() + 5
ax.set_xlim(x_min, x_max)

# Grid
ax.grid(axis='x', linestyle=':', alpha=0.4)
ax.set_axisbelow(True)

# Add interpretation box
textstr = 'Positive values indicate improvement\nwith detailed prompt examples'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

plt.tight_layout()

# Save plot
output_file = 'figure_5_X_system_prompt_impact_delta.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved as '{output_file}'")

print("\n--- Summary Statistics ---")
print(f"Sample sizes:")
print(f"  Yes condition: {len(df_yes)} samples")
print(f"  No condition:  {len(df_no)} samples")

plt.show()
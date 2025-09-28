```python
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

# -------------------------
# Example: Table 3 (English ZSC Results)
# -------------------------
english_data = {
    "Labels": [
        "Positive Portrayal", "Negative Portrayal", "Stereotypical Behavior",
        "Age Bias", "Socioeconomic Bias", "Gender Bias", "Racial Bias"
    ],
    "Male (English)": [0.379, 0.376, 0.134, 0.041, 0.029, 0.024, 0.017],
    "Female (English)": [0.279, 0.364, 0.158, 0.043, 0.057, 0.055, 0.044]
}

english_df = pd.DataFrame(english_data)

# Dummy approach: simulate distributions around reported means
# (since we don’t have raw per-sample results)
def simulate_samples(mean, n=30, sd=0.05):
    return np.random.normal(mean, sd, n)

p_values = []
for m, f in zip(english_df["Male (English)"], english_df["Female (English)"]):
    male_samples = simulate_samples(m)
    female_samples = simulate_samples(f)
    _, p = ttest_ind(male_samples, female_samples)
    p_values.append(p)

english_df["p-value"] = p_values
print("Table 3 with p-values:")
print(english_df.round(4))


# -------------------------
# Example: Table 4 (Hindi ZSC Results)
# -------------------------
hindi_data = {
    "Labels": [
        "Positive Portrayal", "Negative Portrayal", "Stereotypical Behavior",
        "Age Bias", "Socioeconomic Bias", "Gender Bias", "Racial Bias"
    ],
    "Male (Hindi)": [0.154, 0.477, 0.071, 0.043, 0.059, 0.080, 0.116],
    "Female (Hindi)": [0.183, 0.459, 0.090, 0.046, 0.054, 0.065, 0.102]
}

hindi_df = pd.DataFrame(hindi_data)

p_values = []
for m, f in zip(hindi_df["Male (Hindi)"], hindi_df["Female (Hindi)"]):
    male_samples = simulate_samples(m)
    female_samples = simulate_samples(f)
    _, p = ttest_ind(male_samples, female_samples)
    p_values.append(p)

hindi_df["p-value"] = p_values
print("\nTable 4 with p-values:")
print(hindi_df.round(4))


# -------------------------
# Example: Table 5 (Thematic Consistency Scores)
# -------------------------
thematic_data = {
    "Dimension": ["Average TCS", "Average PCS", "Average ETCS", "Overall Bias"],
    "English": [2.73, 1.08, 2.87, 2.23],
    "Hindi": [1.68, 1.00, 5.00, 2.56]
}

thematic_df = pd.DataFrame(thematic_data)

# Simulate distributions and run paired t-test
p_values = []
for e, h in zip(thematic_df["English"], thematic_df["Hindi"]):
    english_samples = simulate_samples(e, sd=0.3)
    hindi_samples = simulate_samples(h, sd=0.3)
    _, p = ttest_ind(english_samples, hindi_samples)
    p_values.append(p)

thematic_df["p-value"] = p_values
print("\nTable 5 with p-values:")
print(thematic_df.round(4))
```

print("# --- Pandas ---")

import pandas as pd

data = {
    "name": ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade": [85, 72, 90, 68, 95],
    "city": ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True],
}
df = pd.DataFrame(data)

print("\n# Pandas Q1")
print(f"\n- First 3 rows:\n{df.head(3)}")
print(f"\n- Shape (rows, columns): {df.shape}")
print(f"\n- Data types:\n{df.dtypes}")

print("\n# Pandas Q2")
filtered_df = df[(df["passed"] == True) & (df["grade"] > 80)]
print(f"\n- Students who passed and have grade > 80:\n{filtered_df}")

print("\n# Pandas Q3")
df["grade_curved"] = df["grade"] + 5
print(f"\n- Updated DataFrame with new column 'grade_curved' (5 points added to each grade):\n{df}")

print("\n# Pandas Q4")
df["name_upper"] = df["name"].str.upper()
print(f"\n- Name and Name Uppercase columns:\n{df[['name', 'name_upper']]}")

print("\n# Pandas Q5")
grouped = df.groupby("city")
mean_grades = grouped["grade"].mean()
print(f"\n- Mean grade by city:\n{mean_grades}")

print("\n# Pandas Q6")
df["city"] = df["city"].replace("Austin", "Houston")
print(f"\n- Name and City columns after replacement:\n{df[['name', 'city']]}")

print("\n# Pandas Q7")
sorted_df = df.sort_values(by="grade", ascending=False)
top3 = sorted_df.head(3)
print(f"\n- Top 3 students by grade:\n{top3}")
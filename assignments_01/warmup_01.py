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
print(
    f"\n- Updated DataFrame with new column 'grade_curved' (5 points added to each grade):\n{df}"
)

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

print("\n\n# --- NumPy ---")
import numpy as np

print("\n# NumPy Q1")
arr = np.array([10, 20, 30, 40, 50])
print(f"\n- Shape: {arr.shape}")
print(f"- Dtype: {arr.dtype}")
print(f"- Number of dimensions: {arr.ndim}")

print("\n# NumPy Q2")

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(f"\n- Shape: {arr.shape}")
print(f"- Size (total elements): {arr.size}")

print("\n# NumPy Q3")
top_left = arr[:2, :2]
print(f"\n- Sliced out the top-left 2x2 block : \n{top_left}")

print("\n# NumPy Q4")
zeros_array = np.zeros((3, 4))
ones_array = np.ones((2, 5))
print(f"\n- 3x4 array of zeros:\n{zeros_array}")
print(f"\n- 2x5 array of ones:\n{ones_array}")

print("\n# NumPy Q5")
arr = np.arange(0, 50, 5)
print(f"\n- Array: {arr}")
print(f"- Shape: {arr.shape}")
print(f"- Mean: {arr.mean()}")
print(f"- Sum: {arr.sum()}")
print(f"- Standard Deviation: {arr.std()}")

print("\n# NumPy Q6")
arr = np.random.normal(loc=0, scale=1, size=200)
# print(f"\n- Array: {arr}")
print(f"Mean: {arr.mean()}")
print(f"Standard Deviation: {arr.std()}")

print("\n\n# --- Matplotlib ---")
import matplotlib.pyplot as plt

print("\n# Matplotlib Q1")
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

print("\n# Matplotlib Q2")
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()

print("\n# Matplotlib Q3")
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.scatter(x1, y1, color="blue", label="Dataset 1")
plt.scatter(x2, y2, color="orange", label="Dataset 2")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

print("\n# Matplotlib Q4")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(x, y)
ax1.set_title("Squares")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

ax2.bar(subjects, scores)
ax2.set_title("Subject Scores")
ax2.set_xlabel("Subjects")
ax2.set_ylabel("Scores")
plt.tight_layout()
plt.show()

print("\n\n# --- Descriptive Statistics ---")

print("\n# Descriptive Stats Q1")
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
arr = np.array(data)
print(f"\n- Mean: {np.mean(arr)}")
print(f"- Median: {np.median(arr)}")
print(f"- Variance: {np.var(arr)}")
print(f"- Standard Deviation: {np.std(arr)}")

print("\n# Descriptive Stats Q2")
scores = np.random.normal(65, 10, 500)
plt.hist(scores, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

print("\n# Descriptive Stats Q3")
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Score")
plt.show()

print("\n# Descriptive Stats Q4")
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Value")
plt.show()

# Exponential distribution is more skewed (right-skewed).
# Mean is better for normal data, median is better for skewed data.

print("\n# Descriptive Stats Q5")
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

import statistics as stats

print("Data1 Statistics:")
print("Mode:", stats.mode(data1))
print("Mean:", np.mean(data1))
print("Median:", np.median(data1))

print("\nData2 Statistics:")
print("Mode:", stats.mode(data2))
print("Mean:", np.mean(data2))
print("Median:", np.median(data2))  
# Why are the median and mean so different for data2?
# The mean is sensitive to outliers (like the value 150), while the median is not.
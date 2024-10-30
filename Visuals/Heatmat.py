import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = r"C:\Users\BossJore\PycharmProjects\ML_steudent_sleep\data\processed\student_sleep_patterns_processed.csv"
data = pd.read_csv(file_path)


sns.set(style="whitegrid")


plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap')
plt.show()


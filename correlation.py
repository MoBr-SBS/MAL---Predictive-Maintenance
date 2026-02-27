import pandas as pd

path = "predictive_maintenance.csv"
data = pd.read_csv(path, delimiter=',')


correlations = data[data.columns].corr(numeric_only=True)
#print(correlations)
print('All correlations')
print('-' * 30)
correlations_abs_sum = correlations[correlations.columns].abs().sum()
print(correlations_abs_sum)
print('Weakest correlations')
print('-' * 30)
print(correlations_abs_sum.nsmallest(3))
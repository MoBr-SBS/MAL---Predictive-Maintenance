import pandas as pd

path = "Data/predictive_maintenance.csv"
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


#All correlations
#------------------------------
#UDI                      1.485271
#Air temperature K        2.126392
#Process temperature K    2.283306
#Rotational speed rpm     1.968000
#Torque Nm                2.100486
#Tool wear min            1.146806
#Target                   1.482350
#dtype: float64

#Weakest correlations
#------------------------------
#Tool wear min    1.146806
#Target           1.482350
#UDI              1.485271
#dtype: float64
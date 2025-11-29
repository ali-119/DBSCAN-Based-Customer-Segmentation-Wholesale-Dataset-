import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN

# Task: Execute the following cells to enter data and display the data frame.
df = pd.read_csv(r'F:\download\File\Wholesale customers data 0.csv')

print(df.head())
'''
   Channel  Region  Fresh  Milk  Grocery  Frozen  Detergents_Paper  Delicassen
0        2       3  12669  9656     7561     214              2674        1338
1        2       3   7057  9810     9568    1762              3293        1776
2        2       3   6353  8808     7684    2405              3516        7844
3        1       3  13265  1196     4221    6404               507        1788
4        2       3  22615  5410     7198    3915              1777        5185
'''

print(df.info())
'''
<class 'pandas.core.frame.DataFrame'>       
RangeIndex: 440 entries, 0 to 439
Data columns (total 8 columns):
 #   Column            Non-Null Count  Dtype
---  ------            --------------  -----
 0   Channel           440 non-null    int64
 1   Region            440 non-null    int64
 2   Fresh             440 non-null    int64
 3   Milk              440 non-null    int64
 4   Grocery           440 non-null    int64
 5   Frozen            440 non-null    int64
 6   Detergents_Paper  440 non-null    int64
 7   Delicassen        440 non-null    int64
dtypes: int64(8)
memory usage: 27.6 KB
'''

print(df.columns)
'''
['Channel', 'Region', 'Fresh', 'Milk', 'Grocery', 'Frozen',
       'Detergents_Paper', 'Delicassen']
'''

# Task: Create a scatterplot that shows the relationship between the 
# costs of MILK and GROCERY, colored based on the Channel column.
plt.figure(figsize=(8, 6), dpi=150)
sns.scatterplot(data=df, x='Milk', y='Grocery', hue='Channel')
plt.show()


# Task: Use seaborn to create a histogram of MILK costs colored by Channel.
plt.figure(figsize=(7, 5), dpi=150)
sns.histplot(data=df, x='Milk', hue='Channel')
plt.show()


# Task: Create a cluster map with annotations of the correlation between costs in different categories.
sns.clustermap(data=df.drop(['Region', 'Channel'], axis=1).corr(), annot=True, row_cluster=False)
plt.show()


# Task: Create a paired data frame chart colored by Region.
sns.pairplot(data=df, hue='Region', palette='viridis')
plt.show()


# Task: Since the feature values ​​are in different orders of magnitude,
# let's scale the data. Use the Standard Scaler to scale the data.
scale = StandardScaler()
scale_df = scale.fit_transform(df)


# Task: Use DBSCAN and a for loop to create different models by testing different values ​​of epsilon.
out_percent_list = []
for eps in np.linspace(0.001, 3, 50):
    db_model = DBSCAN(eps=eps)
    db_labels = db_model.fit_predict(scale_df)

    percent_of_outlier = 100 * np.sum(db_labels == -1) / len(db_labels)
    out_percent_list.append(percent_of_outlier)


# Task: Create a line graph of the percentage of outliers versus the choice of epsilon value.
plt.plot(np.linspace(0.001, 3, 50), out_percent_list, marker='o', markerfacecolor='darkblue', markersize=4)
plt.yticks(range(0, 101, 20))
plt.xticks(np.arange(0.0, 3.1, 0.5))
plt.ylabel("Percentage of Points Classified as Outlier")
plt.xlabel("Epsilon Value")
plt.title("Epsilon Value to Percent of Outlier")
plt.show()


# Task: Retrain a DBSCAN model with an appropriate epsilon value based on the graph created in the previous task.
db_model = DBSCAN(eps=2)
db_labels = db_model.fit_predict(scale_df)


# Task: Create a scatterplot of milk vs. groceries colored based on the labels discovered by the DBSCAN model.
plt.figure(figsize=(8, 6), dpi=150)
sns.scatterplot(data=df, x='Grocery', y='Milk', hue=db_labels, palette='viridis')
plt.show()


# Task: Create a scatter diagram of milk versus detergent and paper, colored based on the labels.
plt.figure(figsize=(8, 6), dpi=150)
sns.scatterplot(data=df, x='Detergents_Paper', y='Milk', hue=db_labels, palette='viridis')
plt.show()


# Task: Create a new column in the main data frame called Labels that contains the DBSCAN labels.
df['Labels'] = db_labels
print(df.head())
'''
   Channel  Region  Fresh  Milk  Grocery  Frozen  Detergents_Paper  Delicassen  Labels
0        2       3  12669  9656     7561     214              2674        1338       0
1        2       3   7057  9810     9568    1762              3293        1776       0
2        2       3   6353  8808     7684    2405              3516        7844       0
3        1       3  13265  1196     4221    6404               507        1788       1
4        2       3  22615  5410     7198    3915              1777        5185       0
'''


# Task: Compare the statistical means of clusters and bins for cost values ​​in categories.
cats = df.drop(['Region', 'Channel'], axis=1)
cat_means = cats.groupby('Labels').mean()

print(cat_means)
'''
               Fresh          Milk       Grocery        Frozen  Detergents_Paper   Delicassen
Labels
-1      30161.529412  26872.411765  33575.823529  12380.235294      14612.294118  8185.411765
 0       8200.681818   8849.446970  13919.113636   1527.174242       6037.280303  1548.310606
 1      12662.869416   3180.065292   3747.250859   3228.862543        764.697595  1125.134021
'''


# Task: Normalize the data frame from the previous task using MinMaxScaler so that the average costs go from 0 to 1 and create a heatmap of the values.
min_max_scale = MinMaxScaler()
mm_scale = min_max_scale.fit_transform(cat_means)
means_scale = pd.DataFrame(data=mm_scale, index=cat_means.index, columns=cat_means.columns)

print(means_scale)
'''
           Fresh      Milk   Grocery    Frozen  Detergents_Paper  Delicassen
Labels
-1      1.000000  1.000000  1.000000  1.000000          1.000000    1.000000
 0      0.000000  0.239292  0.341011  0.000000          0.380758    0.059938
 1      0.203188  0.000000  0.000000  0.156793          0.000000    0.000000
'''

sns.heatmap(means_scale)
plt.show()


# Task: Create a heat map similar to the one above but without the layers.
sns.heatmap(means_scale.loc[[0, 1]], annot=True)
plt.show()


# Task: Which cost category was most different in the two clusters?
# 1: Detergents_Paper
# 2: Grocery
# 3: Milk

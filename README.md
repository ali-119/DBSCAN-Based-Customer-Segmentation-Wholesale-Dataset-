# DBSCAN-Based Customer Segmentation (Wholesale Dataset)
A machine learning project for segmenting wholesale customers using the **DBSCAN density-based clustering algorithm**, with a focus on identifying natural customer groups, detecting outliers, and analyzing spending behaviors across various product categories.

<p align="center">
  <img src="https://img.shields.io/badge/ML-DBSCAN-green">
  <img src="https://img.shields.io/badge/Preprocessing-StandardScaler-blue">
  <img src="https://img.shields.io/badge/Visualization-Seaborn-orange">
  <img src="https://img.shields.io/badge/Category-Unsupervised_Learning-purple">
  <img src="https://img.shields.io/badge/Language-Python-yellow">
</p>

------

# Overview
This project applies **DBSCAN**, a density-based clustering algorithm, to the popular *Wholesale Customers Dataset*.  
The goal is to:

- Discover natural customer groups  
- Detect abnormal high-spending outliers  
- Analyze differences in purchase patterns  
- Understand spending behavior across categories such as Milk, Grocery, Fresh, Frozen, and Detergents_Paper  

This type of segmentation is widely used in real-world applications like customer profiling, market analysis, and retail intelligence.

------

# Dataset
- **Name:** Wholesale Customers Dataset  
- **Source:** UCI Machine Learning Repository  
- **Records:** 440 customers  
- **Features:**  
  - Channel (Hotel/Restaurant/Retail)  
  - Region  
  - Fresh  
  - Milk  
  - Grocery  
  - Frozen  
  - Detergents_Paper  
  - Delicassen  

These features represent annual spending levels for each customer.

------

# Project Workflow

## 1) Exploratory Data Analysis
- Previewed the dataset and inspected NaN values  
- Basic visualizations:
  - Scatterplots (Milk vs Grocery)
  - Histograms colored by Channel
  - PairPlot by Region  
  - Correlation heatmap  
  - Seaborn Clustermap for feature similarity  

## 2) Preprocessing
Since the spending categories vary greatly (some up to 100,000 units), **Standard Scaling** was applied.

## 3) DBSCAN Modeling
Tested multiple epsilon values:

<pre>for eps in np.linspace(0.001, 3, 50):
    DBSCAN(eps=eps).fit_predict(scaled_data)</pre>


For each model, recorded:
- Number of clusters  
- Number of outliers  
- Percentage of outliers  

This allowed identifying a stable and meaningful eps region.

**Final eps chosen:** `eps = 2`

## 4) Cluster Visualization
Created cluster-colored scatterplots:

- Grocery vs Milk  
- Milk vs Detergents_Paper  
- Other spending relationships  

## 5) Cluster Profiling
A new column *Labels* stores DBSCAN cluster assignments.

Then compared cluster means:

<pre>cat_means = df.groupby("Labels").mean()</pre>


Outlier cluster (-1) represented extremely high spenders.  
Clusters 0 and 1 represent medium and low spenders.

## 6) Normalized Cluster Heatmap
Used MinMaxScaler to normalize group means and produce a 0–1 heatmap.

This clearly reveals:

- Outlier cluster has extremely high spending in all categories  
- Cluster 0: Average spending, moderate levels of Detergents_Paper  
- Cluster 1: Low-volume grocery/milk customers  

------

# Libraries Used
- numpy  
- pandas  
- seaborn  
- matplotlib  
- scikit-learn (DBSCAN, StandardScaler, MinMaxScaler)

------

# How to Run

## Clone the repository:
[github](https://github.com/ali-119/DBSCAN-Based-Customer-Segmentation-Wholesale-Dataset)

## Install dependencies:
<pre>pip install -r requirements.txt</pre>
- requirements.txt → [file](https://github.com/ali-119/DBSCAN-Based-Customer-Segmentation-Wholesale-Dataset/blob/main/requirements.txt)

or directly:
<pre>pip install numpy pandas seaborn matplotlib scikit-learn</pre>

Run all cells to train and evaluate the model.

> This project is implemented as a Python script.  
> (No Jupyter Notebook version yet)

------

# Results Summary
The DBSCAN model revealed **three meaningful behavioral groups**:

## Cluster -1 (Outliers)
- Extremely high spenders across all categories  
- Represent unique customer behavior  
- Most likely wholesale distributors or very large retailers  

## Cluster 0
- Medium-to-high grocery and detergent spending  
- Likely restaurants or hotels  

## Cluster 1
- Low spending across all features  
- Small shops or low-volume buyers  

------

# Conclusion
This project demonstrates the power of **density-based clustering** for customer segmentation:

- No need to predefine K  
- Natural detection of non-linear clusters  
- Automatic identification of extreme-value customers  
- Clear interpretation via cluster means and heatmaps  

DBSCAN provides robust segmentation for noisy, high-variance datasets like wholesale customer data.

------
 
# Author ✍️
**Author:** Ali  
**Field:** Data Science & Machine Learning Student  
**Email:** ali.hz87980@gmail.com  
**GitHub:** [ali-119](https://github.com/ali-119)

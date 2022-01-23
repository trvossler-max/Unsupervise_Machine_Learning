# Cryptocurrency Clusters - Unsupervised machine learning

## Background

* This project creates a report that includes what cryptocurrencies are on the trading market and determine whether they can be grouped to create a classification system for a new crypto investment.

* The raw data was processed to fit the machine learning models. Since there is no known classification system, unsupervised machine learing was used. Several clustering algorithms were used to explore whether the cryptocurrencies can be grouped together with other similar cryptocurrencies. A data visualization was created in order to display the findings as well as a written analysis.

### Data Preparation - The following steps were taken to prepare the data for machine learning.

* The `crypto_data.csv` was read into Pandas. The dataset was obtained from [CryptoCompare](https://min-api.cryptocompare.com/data/all/coinlist).

* All cryptocurrencies that are not being traded we discarded by using a query to filter.

* Removed all rows that had at least one null value.

* Filter for cryptocurrencies that have been mined. That is, the total coins mined should be greater than zero.

* Get_dummies was used to convert non-numeric data to numeric in order to make the dataset comprehensible to a machine learning algorithm. Since the coin names do not contribute to the analysis of the data, the `CoinName` from the original dataframe was deleted.

* The dataset was then scaled using StandardScaler so that columns that contain larger values do not unduly influence the outcome.

### Dimensionality Reduction

* Creating dummy variables above dramatically increased the number of features in the dataset. Perform dimensionality reduction with PCA was used to reduce the dataset and preserving 90% of the explained variance.

* The dataset dimensions were further reduced with t-SNE and visually inspect the results. A scatter plot was created of the t-SNE output. 

### Cluster Analysis with k-Means

* An elbow plot was created to identify the best number of clusters. This used a for-loop to determine the inertia for each `k` between 1 through 10. 

### Running the script
The script can be run in Jupyter Lab or Colab by loading the crypto_data.csv and executing the script.

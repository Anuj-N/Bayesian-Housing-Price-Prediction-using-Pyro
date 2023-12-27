# Bayesian-Housing-Price-Prediction-using-Pyro
The goal is to predict house prices using a Bayesian linear regression model implemented in Pyro and compare it with Simple Linear Regression

## Software Architecture

- The architecture involves data preprocessing using pandas, PyTorch for tensor operations, Pyro for probabilistic modeling, and matplotlib/seaborn for result visualization.
- No explicit client-server architecture; it's a standalone script.
- Testing is conducted locally, assessing model performance using R-squared and visualization.
- No database is involved; the dataset is fetched using scikit-learn.(We have also provided the same dataset as a csv file in [Dataset](Dataset/california_housing_data.csv)).

### Overview

1. **Limitations of Simple Linear Regression:**
   - Simple linear regression provides only point estimates for coefficients.
   - Bayesian regression is crucial for generating coefficient distributions and calculating uncertainty.

2. **Ease of Implementation with Pyro:**
   - Implementing Bayesian Regression in Python can be challenging.
   - Pyro, with its inbuilt functionalities, streamlines the process, making it more accessible.

3. **Utilizing Pyro for Bayesian Models:**
   - Pyro is equipped with features that simplify the implementation of Bayesian models.
   - We chose Pyro as our framework for mapping and implementing Bayesian models.

4. **Flexibility in Sampling Algorithms:**
   - Pyro facilitates the implementation of complex sampling algorithms like MCMC (Markov Chain Monte Carlo) and NUTS (No U-turn Sampling).
   - This allows for more robust and accurate probabilistic modeling.

5. **Addressing Dataset Assumptions:**
   - Linear regression assumes a normally distributed dataset.
   - Pyro allows us to assume priors of distributions of our choice, providing flexibility for different datasets.

6. **Incorporating Prior Knowledge:**
   - Probabilistic programming with Pyro allows us to include more prior knowledge about our problem.
   - This flexibility surpasses traditional linear regression, contributing to more informed predictions.
     
## Results and Tests

- **Result R^2 :** At same R^2 value (0.49) we were able to implement all three models - Bayesian Regression (with Gamma and Normal both) and linear regression and we were able to generate relevant distributions through bayesian regression in pyro.
- **Dataset:** California housing dataset is used, split into training and testing sets.
- **Benchmark:** R-squared is calculated to assess model performance. Visualizations include histograms of posterior distributions and scatter plots comparing predicted and true house prices.
- **Validation:** The comparison with traditional linear regression acts as a validation point, demonstrating the benefits of the Bayesian approach in capturing uncertainty.

## Potential for Future Work

- **Hyperparameter Tuning:** Explore sensitivity to priors and hyperparameters for better model performance.
- **Feature Engineering:** Experiment with additional features or transformations to improve predictive accuracy.
- **Ensemble Methods:** Investigate ensemble methods or model averaging to enhance robustness.
- **Online Learning:** Explore possibilities for online learning and continuous model improvement.
- **Integration with External Data:** Incorporate external data sources for richer feature sets.
- **Deployment:** Consider deployment strategies for the model, possibly as a web service or API (Application programming interface).
- **Explanability:** Integrate tools or techniques for explaining model predictions to end-users.


  ## File Organization:
- [Dataset](Dataset/california_housing_data.csv)
- [code-external](code-external/Test_code.ipynb)
- [code-orig](code-orig/BayesianRegression.ipynb)
- [result](result)

## How to Run

1. **Open the Python Notebook:**
   - Open the provided Python Notebook on any Python environment, ideally Google Colab.
   - For the Bayesian Regression model,Run the [Bayesian](code-orig/BayesianRegression.ipynb).
   - For the Gamma Model,Run the [Gamma](code-orig/BayesianRegression(with_Gamma).ipynb).

2. **Load the Dataset:**
   - Download the dataset from [Dataset](Dataset/california_housing_data.csv).
   - Load the dataset into the notebook for testing.

3. **Test the Code and Generate Graphs:**
   - Run the code cells in the notebook to execute the provided code.
   - Explore the generated graphs and results [result](result).

Please Note: Make sure to install any required dependencies mentioned in the notebook before running the code.

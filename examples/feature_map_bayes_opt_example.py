#!/usr/bin/env python3
"""
Example script demonstrating the use of BayesianOptimization for tuning
feature map parameters in ATHENA.
"""
import numpy as np
from athena.kas import KernelActiveSubspaces
from athena.feature_map import FeatureMap
from athena.utils import CrossValidation, average_rrmse

def main():
    # Set up random seed for reproducibility
    np.random.seed(42)
    
    # Generate some sample data
    input_dim = 2
    output_dim = 1
    n_samples = 30
    n_features = 10
    n_params = 1
    
    # Sample inputs from a uniform distribution
    inputs = np.random.uniform(-1, 1, (n_samples, input_dim))
    
    # Create some simple quadratic function outputs for this example
    outputs = np.sum(inputs**2, axis=1).reshape(-1, 1)
    
    # Compute analytical gradients for this simple function
    gradients = np.zeros((n_samples, output_dim, input_dim))
    for i in range(n_samples):
        for j in range(input_dim):
            gradients[i, 0, j] = 2 * inputs[i, j]
    
    # Create a feature map with Laplace distribution
    fm = FeatureMap(distr='laplace',
                    bias=np.random.uniform(0, 2 * np.pi, n_features),
                    input_dim=input_dim,
                    n_features=n_features,
                    params=np.zeros(n_params),
                    sigma_f=outputs.var())
    
    # Create a Kernel Active Subspace with the feature map
    kss = KernelActiveSubspaces(feature_map=fm, dim=1, n_features=n_features)
    
    # Set up cross-validation for parameter tuning
    csv = CrossValidation(inputs=inputs,
                          outputs=outputs,
                          gradients=gradients,
                          folds=3,
                          subspace=kss)
    
    print("Tuning feature map parameters using Bayesian Optimization...")
    
    # Tune the hyperparameters using Bayesian Optimization
    best = fm.tune_pr_matrix(func=average_rrmse,
                           bounds=[slice(-2, 1, 0.2) for _ in range(n_params)],
                           fn_args={'csv': csv},
                           method='bso',  # 'bso' uses BayesianOptimization
                           maxiter=20,
                           save_file=False)
    
    print(f"Optimization complete!")
    print(f"Best score: {best[0]}")
    print(f"Best parameters: {fm.params}")
    
    # Fit the model with the optimal parameters
    kss.fit(inputs=inputs, gradients=gradients, outputs=outputs)
    
    # Transform the inputs to the active subspace
    active_vars = kss.transform(inputs)[0]
    
    print(f"Shape of active variables: {active_vars.shape}")
    print("First 5 active variables:")
    print(active_vars[:5])

if __name__ == "__main__":
    main()

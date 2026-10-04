# Gradient Descent from Scratch

Predicting exam marks from study hours by building gradient descent with NumPy only, then validating it against scikit-learn.

## What this project shows
- Gradient descent implemented from scratch, learning **both** the slope `m` and intercept `b`
- Validation against scikit-learn `LinearRegression` and the normal equation (results match to 4 decimals)
- **Learning-rate experiment:** too small (slow), appropriate (converges), too large (diverges), including the theoretical stability limit computed from the Hessian
- **Iterations experiment:** distance from the exact solution vs. number of epochs
- **Feature scaling:** same accuracy in ~100 epochs instead of thousands

## Results

| Method | m | b |
|---|---|---|
| scikit-learn (exact, OLS) | 7.1225 | 12.0626 |
| Gradient descent, 5000 epochs, lr=0.01 | 7.1225 | 12.0626 |
| Gradient descent, scaled feature, 100 epochs, lr=0.1 | 7.1225 | 12.0626 |

Data is synthetic: `marks = 12 + 7.5 * hours + noise`, 10 points, seed 42.

## Key concepts
- **Loss:** mean squared error between predictions and true marks
- **Gradient:** direction in which the loss increases fastest; we step the opposite way
- **Learning rate:** step size; stable only if `lr < 2 / lambda_max` of the loss Hessian
- **Convergence:** loss drops fast, then flattens (diminishing returns)

## Why it matters
Linear regression has a closed-form answer, so it is the ideal place to verify the algorithm. Models without one, such as neural networks and large language models, are trained with variants of this same loop (SGD, Adam).

## Run it
```bash
pip install numpy matplotlib scikit-learn jupyter
jupyter notebook Gradient_Descent_Project.ipynb
```

## Possible extensions
Stochastic / mini-batch gradient descent, multiple features, logistic regression, momentum and Adam.

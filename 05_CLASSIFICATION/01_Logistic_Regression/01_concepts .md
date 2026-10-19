# Logistic Regression


>## Contents
>- [What is Logistic Regression?](#what-is-logistic-regression)
>- [Intuition](#intuition)
>- [Mathematical Understanding](#mathematical-understanding)
>
>    - [Cost Function](#cost-function)
>    - [Gradient Descent](#gradient-descent)
>
>- [Building Linear a Regression Model](./02_building_model.ipynb)
## What is logistic Regression


>Logistic Regeression is a classification algorithm that uses a **Sigmoid function** instead of the usual equation of a line. 
>$$
>\hat{y} = S = \frac{1}{1 + e^{-\hat{z}}}
>$$
>
>where: 
>$$
>Z = w.X + b \\
>$$
>
>$\hat{y} = $ P( Y=1 | X)
>
>The S-curve squashes any real-valued number into a probability range between 0 and 1 with the mid-point at y=0.5, such that, all values below it are classified as zero and >those above, 1.
>Logistic is the standard baseline model for binary classification tasks.
## Intuition


## Mathematical Understanding


### Loss Function 
>
>The loss function for logistic regression is known as the **Binary Cross Entropy Function** or **Log Loss**.
>
>$$ \text{L}(y, \hat{y}) = -(y \log \hat{y} + (1 - y)\log(1 - \hat{y}))$$
>
>When  y = 1,
>$$ \text{L}(1, \hat{y}) = -(1 \log \hat{y} + (1 - 1)\log(1 - \hat{y})) = -(y\log\hat{y})$$
>
>loss is negative. Since we want the make Loss function value as small as possible, we must make $\hat{y}$ very large, so that (-log $\hat{y}$) will be a large negative number
>
>When y = 0,
>
>$$ \text{L}(0, \hat{y}) = -(0 \log \hat{y} + (1 - 0)\log(1 - \hat{y})) = y\log\hat{y}$$
>loss is positive, so we make $\hat{y}$ very small, so that (-log $\hat{y}$) will be a large negative number
>
>We remember that the cost function is the aggregated loss over a collection of examples (i.e aggregation of loss functions). So, for logistic regression, the cost function >would be:
>
>$$\text{J(w,b)} = \frac{1}{m} \text{L}(y_i, \hat{y}_i) = -\frac{1}{m} \sum [y_i \log \hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i)]^2$$


### Gradient Descent

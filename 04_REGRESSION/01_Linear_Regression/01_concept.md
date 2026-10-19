# Linear Regression

## Contents

>- [What is Linear Regression?](#what-is-linear-regression)
>- [Core intuition](#intuition--how-does-it-work)
>
>- [Mathematics](#mathematics--key-equations)
>
>    - [Cost Function](#cost-function)
>    - [Gradient Descent](#gradient-descent)
>
>- [Important Points](#important-points)

     - [Scikit learn](#scikit-learn)
    
>- [When to use it](#when-to-use-it)
>- [Limitations](#limitations)
>- [Key Takeaways](#key-takeaways)
>- [Defer](#defer)
>

## What is Linear Regression?

>Regression refers to a predictive modelling technique used to technique analyzes the relationship between a dependent variable, **y**  (also known as the target) and one or >more independent variables/features/influencing factors. An example is seen in the prediction of house price given factors like **sqft**, **room size**, etc.
>It involves predicting infinitely many possible outputs. 
>
>Apart from fitting a straight line, other learning algorithms include using a function slightly more complicated. Linear >Regression uses the equation of a straight line.
>

## Intuition — How does it work?

>This Algorithm takes a training set $(x,y)$ and produces some function, $f$ (_previous hypothesis_) which takes $x$ and outputs an estimate/prediction $\hat{y}$. When the >output is $y$, it is the **true** value.
>To sell the house,  the model takes the i/p, $\mathbb{x}$ and makes an estimate of what the price would be.
>
>In designing an algorithm, we decide how to represent $\textit{f}_{w,b}$:
>
>$$
>\textit{f}_{w,b}(x) = wx + b
>$$
>
>This is a linear function (i.e. there no exponential terms or polynomial terms of higher degrees)
>
>### Types
>- **Univariate** (one variable):  $\textit{f}_{w,b}(x) = wx + b$
>- **Multivariate** (many variables): Multiple linear regression is a model for predicting the value of one dependent variable based on two or more independent variables:
>
>$$
>\textit{f}_{w,b}(x,y,z) = w_{1}x + w_{2}x + w_{3}x+, ..., w_{n} + b,  \; \; \;\; where \;\;\;n \in \mathbb{R}^n
>$$

### Advantages

>- Very simple to implement
>- Performs well on data with linear relationsship


## Mathematics — Key Equations

### Cost Function

> Cost function (also knows as _Loss function_ for a single training set) is a mathematical function that measures how far estimated values are from their true values. It helps > to determine and improve the performance of models by optimizing parameters (selecting the best parameters to minimize loss)
> 
> $$
> \text{Cost Function} = J(\omega) = \frac{1}{n} \sum_{i=1}^2 (L(y_i, \hat{y}_{i})) = \frac{1}{n} \sum_{i=1}^2 (y_{i}-\hat{y}_{i})^2
> $$
> where:
> $$
> \begin{aligned}
> \hat{y}_{i} = \textit{estimated / predicted} \ value \\
> y = true \; value \\
> (y_{i}-\hat{y}_{i})^2 = (error)^2 = L(y_i, \hat{y}_{i}) = \textit{Loss function of one sample}\\
> \sum = Summation  \\
> \end{aligned}
> $$

### Gradient Descent

> Gradient Descent is an iterative optimization algorithm used to find the minimum of a function, commonly appllied to train ML models and neutral networks by minimizing the loss > function. It is used for updating  the parameters of the learning model.
> 
> The gradient descent is defined by:
> 
> $$
> \omega_{\text{new}} = \omega_{\text{old}} - \alpha {\frac{dJ}{d\omega}} \\[1em]
> b_{\text{new}} = b_{\text{old}} - \alpha {\frac{dJ}{d\omega}} 
> $$
> We see how the losss function changes w.r.t. each parameter
> Or generally 
> $$
> \theta_{new} = \theta_{old} - \alpha \nabla J 
> $$
> where:
> - $\theta$ = model parameters/ parameter vector: 
>$$
>\theta = 
>\begin{bmatrix}
>\omega_{1} \\
>\omega_{2} \\
>\omega_{3} \\
>\vdots
>\end{bmatrix}
>$$
>
> - $\alpha$ = learning rate; a postive number  $(\text{i.e} > 0)$
> - $\nabla J(\theta)$ = gradient of the loss function:
>$$
>\nabla J (\theta) = 
>\begin{bmatrix}
>\dfrac{\partial J}{\partial \omega_1} \\[1em] 
>\dfrac{\partial J}{\partial \omega_2} \\[1em]
>\dfrac{\partial J}{\partial \omega_3} \\ 
>\vdots
>\end{bmatrix}
>$$
> 
> Our goal is to find the value of $\theta$ that minimizes $J$
>
> **Learning Rate**:
> 
> Learning rate is a tuning parameter in an optimization algorithm that determines the step size at each iteration while moving toward a minimum of a loss function.

## Important Points

### Scikit-learn

## When to use it

## Limitations

>- Not suitable for data having non-linear relationship
>- Underfitting issue
>- Sensitive to Outliers.

## Key Takeaways

## Defer
- 
- 


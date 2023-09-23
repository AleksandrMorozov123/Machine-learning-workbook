# Machine-learning-workbook
Key question from machine learning course Latex template

\documentclass{article}
\usepackage{graphicx} % Required for inserting images
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{tabularx} % required for tables
\usepackage{parskip} 
\usepackage{tikz,lipsum,lmodern}
\usepackage[most]{tcolorbox}
\usepackage{amsmath}
\usepackage[paperheight=6in,
   paperwidth=5in,
   top=8mm,
   bottom=20mm,
   left=10mm,
   right=8mm]{geometry}
\usepackage{blindtext}

\begin{document}
\fontsize{5}{7}\selectfont

\fontsize{7}{8}\selectfont

\fontsize{10}{11.5}\selectfont

\title{Machine learning}
\author{Morozov Aleksandr }
\date{August 2023}

\maketitle

\section{Introduction}

\Large  1. Naive Bayesian Classifier

\normalsize In this section, we discuss how to classify vectors of discrete-valued features, $ x \in {1,…, K}^D $, where K is the number of values for each feature, and $D$ is the number of features. We will use a generative approach. This requires us to specify the class conditional distribution, $ p(x | y = c) $. The simplest approach is to assume the features are conditionally independent given the class label. This allows us to write the class conditional density as a product of one dimensional densities:
$$ p(x|y = c, \theta) = \prod_{j=1}^D p(x_j|y = c, \theta_{jc}) $$
The resulting model is called “naïve” since we do not expect the features to be independent, even conditional on the class label. However, even of the naïve Bayes assumption is not true, it often results in classifiers that work well. One reason for this is that the model is quite simple (it only has $ O (CD) $ parameters, for $C$ classes and $D$ features), and hence it is relatively immune to overfitting.

\Large 2. Linear regression

\normalsize Our definition of a machine learning algorithm as an algorithm that is capable of improving a computer program’s performance at some task via experience is somewhat abstract. To make this more concrete, we present an example of a simple machine learning algorithm: linear regression. 
As name implies, linear regression solves a regression problem. In other words, the goal is to build a system that can take a vector $ x \in \mathbb{R}^n $ as input and predict the value of a scalar $ y \in \mathbb {R} $ as its output. In the case of linear regression, the output is a linear function of the input. Let $ \hat{y} $ be the value that our model predicts $ y $ should take on. We define the output to be 
$$ \hat {y} = w^T x $$
where $ w \in \mathbb {R}^n $ is a vector of parameters.
Paramaters are values that control the behavior of the system. In this case, $ w_i $ is the coefficient that we multiply by feature $ x_i $ before summing up the contributions from all the features. We can think of $w$ as a set of weights that determine how each feature affects the prediction. If a feature $ x_i $ receives a positive weight $ w_i $, then increasing the value of that feature increases the value of our prediction $ \hat {y} $. If a feature receives a negative weight, then increasing the value of that feature decreases the value of our prediction. If a feature’s weight is zero, it has no effect on the prediction.
We thus have a definition of our task $T$: to predict $y$ from $x$ by outputting $ \hat{y} = w^T x $. Next we need a definition of our performance measure, P.
Suppose that we have a design matrix of $m$ example inputs that we will not use for training, only for evaluating how well the model performs. We also have a vector of regression targets providing the correct value of $y$ for each of these examples. Because this dataset will only be used for evaluation, we call it the test set. We refer to the design matrix of inputs as $ X^{(test)}$ and the vector of regression targets as $ y^{(test)} $.
One way of measuring the performance of the model is to compute the mean squared error of the model on the test set. If $ \hat{y}^{(test)} $ gives the predictions of the model on the test set, then the mean squared error is given by
$$ MSE_{test} = \frac {1}{m} \sum_i (\hat{y}^{(test)} – y^{(test)})_i^2 $$
Intuitively, one can see that this error measure decreases to 0 when $ \hat{y}^{(test)} = y^{(test)} $. We can also see that 
$$ MSE_{test} = \frac{1}{m} || \hat{y}^{(test)} – y^{(test)} ||_2^2 $$
So the error increases whenever the Euclidean distance between the predictions and the target increases.
To make a machine learning algorithm, we need to design an algorithm that will improve the weights $w$ in a way that reduces $ MSE_{test} $ when the algorithm is allowed to gain experience by observing a training set $ (X^{(train)}, y^{(train)}) $. One intuitive way of doing this is just to minimize the mean squared error on the training set, $ MSE_{(train)} $.
To minimize $ MSE_{train} $, we can simply solve for where its gradient is 0:
$$ \nabla_w MSE_{train} = 0$$
$$ \Rightarrow \nabla_w \frac {1}{m} || \hat{y}^{(train)} – y^{(train)} ||_2^2 = 0 $$
$$ \Rightarrow \frac{1}{m} \nabla_w || X^{(train)} w – y^{(train)} ||_2^2 = 0 $$
$$ \Rightarrow \nabla_w (X^{(train)} w – y^{(train)})^T (X^{(train)} w – y^{(train)}) = 0 \quad \boxed{(5.9)} $$
$$ \Rightarrow \nabla_w (w^T X^{(train)T} X^{(train)} w – 2 w^T X^{(train)T} y^{(train)} + y^{(train)T} y^{(train)}) = 0 $$
$$ \Rightarrow 2 X^{(train)T} X^{(train)} w – 2 X^{(train)T} y^{(train)} = 0 $$
$$ \Rightarrow w = (X^{(train)T} X^{(train)})^{-1} X^{(train)T} y^{(train)} \quad \boxed {(5.12)}  $$
The system of equations whose solution is given by equation 5.12 is known as the normal equations. Evaluating equation 5.12 constitutes a simple learning algorithm. For an example of the linear regression learning algorithm in action, see figure 5.1. 
\begin{figure}[htp]
    \centering
    \includegraphics[width=10cm]{Linear regression.jpg}
    \caption{Linear regression}
\end{figure}

It is worth noting that the term linear regression is often used to refer to a slightly more sophisticated model with one additional parameter – an intercept term $b$. In this model
$$ \hat{y} = w^T x + b $$
So the mapping from parameters to predictions is still a linear function but the mapping from features to predictions is now an affine function. This extension to affine functions means that the plot of the model’s predictions still looks like a line, but it need not pass through the origin. Instead of adding the bias parameter $b$, one can continue to use the model with only weights but augment $x$ with an extra entry that is always set to 1. The weight corresponding to the extra 1 entry plays the role of the bias parameter. We will frequently use the term “linear” when referring to affine functions.
The intercept term $b$ is often called the bias parameter of the affine transformation. This terminology derives from the point of view that the output of the transformation is biased toward being $b$ in the absence of any input. This term is different from the idea of a statistical bias, in which a statistical estimation algorithm’s expected estimate of a quantity is not equal to the true quantity.

\Large 3. How to measure quality in regression: MSE, MAE, R2.\\
\normalsize In order to evaluate the performance of a statistical learning method on a given data set, we need some way to measure how well its predictions actually match the observed data. That is, we need to quantify the extent to which the predicted response value for a given observation is close to the true response value for that observation. In the regression setting, the most commonly-used measure is the mean-squared error (MSE), given by
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{f} (x_i))^2  \quad \boxed {(2.5)}
$$
Where $\hat{f} (x_i) $ is the prediction that $\hat{f} $ gives the ith observation. The MSE will be small if the predicted responses are very close to the true responses, and will be large if for some of the observations, the predicted and the true responses differ substantially. 
The MSE in (2.5) is computed using the training data that was used to fit the model, and so should more accurately be referred to as the training MSE. But, in general, we do not really care how well the method works on the training data. Rather, we are interested in the accuracy of the predictions that we obtain when we apply our method to previously unseen test data. Why is this what we care about? Suppose that we are interested in developing an algorithm to predict a stock’s price based on previous stock returns. We can train the method using the stock returns from the past 6 months. But we don’t really care about how well it will predict tomorrow’s price or next month’s price. 
Residual Standard Error
Recall from the model (3.5) that associated with each observation is an error term $\epsilon$ Due to the presence of these error terms, even if we knew the true regression line (i.e. even if $\beta_0$ and $\beta_1$ were known), we would not be able to perfectly predict Y from X. The RSE is an estimate of the standard deviation of $\epsilon$. Roughly speaking, it is the average amount that the response will deviate from the true regression line. It is computed using the formula
$$
RSE = \sqrt {\frac{1}{n-2}RSS} = \sqrt {\frac {1}{n-2} \sum_{i=2}^{n} (y_i - \hat {y_i})^2}
$$
Note that RSS was defined in Section 3.1.1., and is given by the formula
$$
RSS = \sum_{i=1}^{n} (y_i - \hat {y_i})^2  \quad \boxed {(3.16)}
$$

$$ R^2 statistic $$
The RSE provides an absolute measure of lack of fit of the model (3.5) to the data. But since it is measured in the units of Y, it is not always clear what constitutes a good RSE. The $R^2$ statistic provides an alternative measure of fit. It takes the form of a proportion – the proportion of variance explained – and so it always takes on a value between 0 and 1, and is independent of the scale of Y.
To calculate $R^2$, we use the formula
$
R^2 = \frac{TSS-RSS}{TSS} = 1-\frac{RSS}{TSS}
$
Where $ TSS = \sum (y_i - \bar {y})^2 $ is the total sum of squares, and RSS is defined in (3.16).
 MAE (Mean absolute error)
Depending on the specific loss function we use, the statistical risk of an estimator can take different names:
1. When the absolute error is used as a loss function, then the risk
$$
R(\hat{\theta}) = E[\Vert \hat{\theta} - \theta_0 \Vert]
$$
Is called the mean absolute error of the estimator.
Bias-variance decomposition
\\The dataset $X = (x_i, y_i)_{i=1}^{\ell} $ with $y_i \in \mathbb{R} $ for regression problem.
Denote loss function $L (y, a) = (y - a(x))^2 $
The empirical risk takes form:
$$R(a) = \mathbb {E}_{x, y} [(y-a(x))^2] = \int_{\mathbb{X}} \int_{\mathbb{Y}} p(x, y) (y - a(x))^2 dxdy$$
Let's show that
$$ a_*(x) = \mathbb{E} [y|x] = \int_{\mathbb{Y}} yp(y|x)dy = argmin_a R(a)$$
$$L(y, a(x)) = (y - a(x))^2 = (y - \mathbb{E} (y|x) + \mathbb{E} (y|x) - a(x))^2=$$ 
$$= (y - \mathbb{E} (y|x))^2 + 2(y - \mathbb{E} (y|x)) (\mathbb{E} (y|x) - a(x))+$$ 
$$ + (\mathbb{E}(y|x) - a(x))^2 $$
Returning to the risk estimation
$$ R(a) = \mathbb{E}_{x, y} L(y, a(x)) = \mathbb{E}_{x, y} (y - \mathbb{E} (y|x))^2 + \mathbb{E}_{x,y}\times$$
$$\times (\mathbb{E} (y|x) - a(x))^2 + 2\mathbb{E}_{x,y} (y - \mathbb{E} (y|x)) 2(\mathbb{E}_{x,y} (y - \mathbb{E}(y|x))(\mathbb{E} (y|x) - a(x))$$ 

\Large 4. Logistic regression

\normalsize We can generalize linear regression to the (binary) classification setting by making two changes. First, we replace the Gaussian distribution for $y$ with a Bernoulli distribution, which is more appropriate for the case when the response is binary, $ y \in {0, 1} $. That is, we use
$$ p(y | x, w) = Ber (y | \mu (x)) $$
Where $ \mu (x) = \mathbb {E} [y | x] = p (y = 1 | x) $ Second, we compute a linear combination of the inputs, as before, but then we pass this through a function that ensures $ 0 \leq \mu (x) \leq 1 $ by defining 
$$ \mu (x) = sigm (w^T x)      (1.9) $$
Where $ sigm (\eta) $ refers to the sigmoid function, also known as the logistic or logit function. This is defined as
$$ sigm (\eta) \triangleq \frac {1}{1 + exp (- \mu)} = \frac {e^{\eta}} {e^{\eta} + 1} $$
The term “sigmoid” means S-shaped: see Figure 1.19 (a) for a plot. It is also known as a squashing function, since it maps the whole real line to [0, 1], which is necessary for the output to be interpreted as a probability.
\begin{figure}[htp]
    \centering
    \includegraphics[width=10cm]{Logistic regression.jpg}
    \caption{Logistic regression}
    \label{fig:galaxy}
\end{figure}

Putting these two steps together we get
$$ p(y | x, w) = Ber (y | sigm (w^T x)) $$
This is called logistic regression due to its similarity to linear regression (although it is a form of classification, not regression!)
A simple example of logistic regression is shown in Figure 1.19(b), where we plot
$$ p(y_i = 1 | x_i, w) = sigm (w_0 + w_1 x_i)  $$
Where $ x_i $ is the SAT score of student $i$ amd $y_i$ is whether they passed or failed a class. The solid black dots show the training data, and the red circles plot $ p (y = 1 | x_i, \hat{w}) $, where $ \hat {w} $ are the parameters estimated from the training data.
If we threshold the output probability at 0.5, we can induce a decision rule of the form
$$ \hat{y} (x) = 1 \Leftrightarrow p(y = 1 | x) > 0.5 $$
By looking at Figure 1.19 (b), we see that  $ sigm (w_0 + w_1 x) = 0.5$ for $ x \approx 545 = x^* $. We can imagine drawing a vertical line at $ x = x^* $; this is known as a decision boundary. Everything to the left of this line is classified as a 0, and everything to the right of the line is classified as a 1. 
We notice that this decision rule has a non-zero error rate even on the training set. This is because the data is not linearly separable, i.e., there is no straight line we can draw to separate the 0s from 1s. We can create models with non-linear decision boundaries using basis function expansion, just as we did with non-linear regression.

\Large 5. Logistic loss.

\normalsize The negative log-likelihood for logistic regression is given by
$$ NLL(w) = - \sum_{i=1}^N \log [\mu_i^{\mathbb{I}(y_i = 1)} \times  (1 - \mu_i)^{\mathbb{I}(y_i = 0)}] = $$
$$ - \sum_{i=1}^{N} [y_i \log \mu_i + (1 – y_i) log (1 - \mu_i)] $$
This is also called the cross-entropy error function
Another way of writing this is as follows. Suppose $ \tilde{y}_i \in { - 1, + 1} $ instead of $ y_i \in {0, 1} $ We have $ p (y = 1) = \frac {1} {1 + \exp (-w^T x)}$ and 
$$p (y = 1) = \frac {1}{1 + exp (+w^T x)}$$ 
Hence
$$ NLL (w) = \sum_{i=1}^N \log (1 + \exp ( - \tilde{y}_i w^T x_i)) $$
Unlike linear regression, we can no longer write down the MLE in closed form. Instead, we need to use an optimization algorithm to compute it. For this, we need to derive the gradient and Hessian.
In the case of logistic regression, one can show that the gradient and Hessian of this are given by the following
$$ g = \frac {d} {dw} f(w) = \sum_i (\mu_i – y_i) x_i = X^T (\mu – y) $$
$$ H = \frac {d} {dw} g(w)^T = \sum_i (\nabla_{w \mu_i}) x_i^T = \sum_i \mu_i (1 - \mu_i) x_i x_i^T = X^T S X $$
Where $ S \triangleq diag (\mu_i (1 - \mu_i)) $. One can also show that H is positive definite. Hence the NLL is convex and has unique global minimum. 
The loss function for linear regression is squared loss. The loss function for logistic regression is Log Loss, which is defined as follows:
$$ LogLoss = \sum_{(x,y) \in D} - y \log (y') - (1 - y) \log (1 - y') $$
where:
\begin{itemize}
    \item $(x,y) \in D$ is the data set containing many labeled examples, which are $(x,y)$ pairs
    \item $y$ is the label in a labeled example. Since this is logistic regression, every value of $y$ must be either be 0 or 1
    \item $y'$ is the predicted value (somewhere 0 and 1), given the set of features in $x$
\end{itemize}

\Large 6. Maximum Likelihood Estimation (MLE)

\normalsize The goal of maximum likelihood is to find the optimal way to fit a distribution to the data. The reason you want to fit a distribution to your data is
it can be easier to work with and it is also more general - it applies to every experiment of the same type. We want the location that "maximizes the likelihood"
of observing the weights we measured. For the normal distribution that has been "fit" to the data by using the maximum likelihood estimations for the mean and
the standard deviation.\\
In everyday conversation, "probability" and "likelihood" mean the same thing. However, in statistics, "likelihood" specifically refers to this situation we've
covered here; where you are trying to find the optimal value for the mean or standard deviation for a distribution given a bunch of observed measurements.\\
In statistics, maximum likelihood estimation (MLE) is a method of estimating the parameters of a statistical model given observations, by finding the parameter
values that maximize the likelihood of making the observations given the parameters. MLE can be seen as a special case of the maximum a posteriori estimation (MAP)
that assumes a uniform prior distribution of the parameters, or as a variant of the MAP that ignores the prior and which therefore is unregularized.\\
Previously, we have seen some definitions of common estimators and analyzed their properties. But where did these estimators come from? Rather than guessing that some function might make a good estimator and then analyzing its bias and variance, we would like to have some principle from which we can derive specific functions that are good estimators for different models.\\ 
The most common such principle is the maximum likelihood principle.\\
Consider a set of $m$ examples $\mathbb {X} = {x^{(1)}, …, x^{(m)}}$ drawn independently from the true but unknown data generating distribution $p_{data} (x)$ \\
Let $ p_{model} (x; \theta) $ be a parametric family of probability distributions over same space indexed by $\theta$. In other words, $p_{model} (x; \theta) $ maps any configuration $x$ to a real number estimating the true probability $p_{data} (x) $. \\
The maximum likelihood estimator for $ \theta $ is then defined as 
$$ \theta_{ML} = \arg\max_{\theta} p_{model} (\mathbb{X}; \theta) = \arg\max_{\theta} \prod_{i=1}^m p_{model} (x^{(i)}; \theta) $$
This product over many probabilities can be inconvenient for a variety of reasons, For example, it is prone to numerical underflow. To obtain a more convenient but equivalent optimization problem, we observe that taking the logarithm of the likelihood does not change its argmax but does conveniently transform a product into a sum:
$$ \theta_{ML} = \arg\max_{\theta} \sum_{i=1}^m \log p_{model} (x; \theta) \quad  \boxed{(5.59)} $$
One way to interpret maximum likelihood estimation is to view it as minimizing the dissimilarity between the empirical distribution $ \hat {p}_{data}$ defined by the training set and the model distribution, with the degree of dissimilarity between the two measured by the KL divergence. The KL divergence is given by
$$ D_{KL} (\hat {p}_{data} \Vert p_{model} ) = \mathbb {E}_{X \sim \hat {p}_{data}} [\log \hat {p}_{data} (x) - \log p_{model} (x)] $$
The term on the left is a function only of the data generating process, not the model. This means when we train the model to minimize the KL divergence, we need only minimize 
$$ - \mathbb {E}_{X \sim \hat {p}_{data}} [\log p_{model} (x)] $$
Which is of course the same as the maximization in equation 5.59

\Large 7. Cross-validation

\normalsize The problem: So far, we’ve simply been told which points are the Training data and which points are the Testing data. However, usually no one tells us what is for Training and what is for Testing. How do we pick the best points for Training and the best points for Testing.
A solution: When we’re not told which data should be used for Training and for Testing, we can use Cross validation to figure out which is which in an unbiased way. Rather than worry too much about which specific points are best for Training and best for Testing, Cross Validation uses all points for both in an iterative way, meaning that we use them in steps.
Often we use about 80 % of the data for the training set, and 20 % for the validation set. But if the number of training cases is small, this technique runs into problems, because the model won’t have enough data to train on, and we won’t have enough data to make a reliable estimate of the future performance.
A simple but popular solution to this is to use cross validation (CV). The idea is simple: we split the training data into K folds, then, for each fold k in {1,.., K} we train on all the folds 
but the k’th, and the test on the k’th, in a round-robin fashion, as sketched in Figure 1.21 (b).

\begin{figure}[htp]
    \centering
    \includegraphics[width=10cm]{Cross validation.jpg}
    \caption{Cross validation}
    \label{fig:galaxy}
\end{figure}

We then compute the error averaged over all the folds, and use this as a proxy for the test error. (Note that each point gets predicted only once, although it will be used for training $K – 1$ times.) It is common to use $ K = 5 $; this is called 5-fold CV. Of we set K = N, then we get a method called leave-one out cross validation, or LOOCV, since in fold $i$, we train on all the data cases except for $i$, and then test on $i$. 
Choosing $K$ for a KNN classifier is a special case of a more general problem known as model selection, where we have to choose between models with different degrees of flexibility.


\Large 8. Overfitting and underfitting \\
\normalsize When a machine learning method fits the Training data really well but makes poor predictions, we say that it is Overfit to the Training data. Overfitting a machine learning method is related to something called the Bias-Variance Tradeoff.
The factors determining how well a machine learning algorithm will perform are its ability to:
1. Make the training error small
2. Make the gap between training and test error small.
These two factors correspond to the two central challenges in machine learning: underfitting and overfitting. Underfitting occurs when the model is not able to obtain a sufficiently low error value on the training set. Overfitting occurs when the gap between the training error and test error is too large.
We can control whether a model is more likely to overfit or underfit by altering its capacity. Informally, a model’s capacity is its ability to fit a wide variety of  functions. Models with low capacity may struggle to fit the training set. Models with high capacity can overfit by memorizing properties of the training set that do not serve them well on the test set.


\Large 9. L1 and L2 regularization

\normalsize Regualarization techniques plays a vital role in the development of machine learning models. Especially complex models, like neural networks, prone to overfitting the
training data. Broken down, the word "regularize" states that we're making regular. In a marthematical or ML context, we make something regular by adding information
which creates a solution that prevents overfitting. The "something" we're making regular in our ML context is the "objective function", something we try to minimize
during the optimization problem. 
Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error. 
$\ell_1$ regularization\\
When we have many variables, it is computationally difficult to find the posterior mode of $ p(\gamma | \mathcal{D})$. And although greedy algorithms often work well, they can of course get stuck in local optima.
Part of the problem is due to the fact that the $\gamma_j$ variables are discrete, $\gamma_j \in {0, 1}$. In the optimization community, it is common to relax hard constraints of this form by replacing discrete variables with continuous variables. We can do this by replacing spike-and-slab style prior, that assigns finite probability mass to the event that $w_j = 0$, to continuous priors that “encourage” $w_j = 0$ by putting a lot of probability density near the origin, such as a zero-mean Laplace distribution. There we exploited the fact that the Laplace has heavy tails. Here we exploit the fact that it has a spike near $\mu = 0 $ More precisely, consider a prior of the form
$$ p(w|\lambda) = \prod_{j=1}^D Lap(w_j | 0 , 1 /{\lambda}) \propto \prod_{j=1}^D e^{-\lambda |w_j|} $$
We will use a uniform prior on the offset term, $p(w_0) \propto 1$. Let us perform MAP estimation with this prior. The penalized negative log likelihood has the form
$$ f(w) = - \log p(\mathcal(D) | w) - \log p (w | \lambda) = NLL (w) + \lambda \Vert w \Vert_1 $$
Where $ \Vert w \Vert_1 = \sum_{j=1}^D |w_j|$ is the $\ell_1$ norm of $w$. For suitably large $\lambda$, the estimate $ \hat {w}$ will be sparse, for reasons we explain below. Indeed, this can be thought if a convex approximation to the non-convex $\ell_0$ objective.
$$ \arg\min_w NNL(w) + \lambda \Vert w \Vert_0 $$
In the case of linear regression, the $\ell_1$ objective becomes
$$ f(w) = \sum_{i=1}^N - \frac {1}{2 \sigma^2} (y_i –(w_0 w^T x_i))^2 + \lambda \Vert w \Vert_1 = \\
RSS (w) + \lambda’ \Vert w \Vert_1 $$ 
Where $ \lambda’ = 2 \lambda \sigma^2 $. This method is known as basis pursuit denoising or BPDN. In general, the technique of putting a zero-mean Laplace prior on the parameters and performing MAP estimation is called $\ell_1$ regularization.\\
$ell_2$ regularization\\
$ell_2$ regulariztion, or L2 norm, or Ridge (in regression problems), combats overfitting by forcing weights to be small, but not making them exactly 0.\\
So, if we're predicting house prices again, this means the less significant features for predicting the house price would still have some influence over the final prediction,
but it would only be a small influence.\\
The regularization term that we add to the loss function when performing $ell_2$ regularization is the sum of squares of all of the feature weights:
$$ LossFucntion = \frac{1}{N} \sum_{i=1}^N (\hat{y} - y)^2 \lambda \sum_{i-1}^N \theta_i^2 $$
So, $\ell_2$ regularization returns a non-sparse solution since the weights will be non-zero (although some may be close to 0).
A major snag to consider  when using $\ell_2$ regularization is that it's not robust to  outliers. The squared terms will blow up the differences in the error of outliers. 
The regularization would then attempt to fix this by penalizing the weights.\\

The difference between $\ell_1$ and $\ell_2$ regularization:\\
\begin{itemize}
    \item $\ell_1$ regularization penalizes the sum of absolute values of the weights, whereas $\ell_2$ regularization penalizes the sum of squares of the weights
    \item The $\ell_1$ regularization is sparse. The $\ell_2$ regularization is non-sparse.
    \item $\ell_2$ regularization doesn't perform feature selection, since weights are only reduced to values near 0 instead of 0. $\ell_1$ regularization has built-in feature selection
    \item $\ell_1$ regularization is robust to outliers, $\ell_2$ regularization is not
\end{itemize}
 
\Large 10. Precision and recall\\
\normalsize When trying to detect a rare event (such as retrieving a relevant document  or finding a face in an image), the number of negatives is very large. Hence comparing $  TPR = TP / N_+ $ to 
$FPR = FP / N_- $ is not very informative, since the FPR will be very small. Hence all the “action” in the ROC curve will occur on the extreme left. In such cases, it is common to plot the TPR versus the number of false positives, rather than vs the false positive rate.
However, in some cases, the very notion of  “negative” is not well-defined. For example, when detecting objects in images, if the detector works by classifying patches, then the number of patches examined – and hence the number of true negatives – is a parameter of the algorithm, not part of the problem definition. So we would like to use a measure that only talks about positives. 
The $ \mathbf {precision}$ is defined as $ TP/ \hat{N}_+ = p (y  = 1 | \hat{y} = 1) $ and the $ \mathbf {recall} $ is defined as $ TP / N_+ = p (\hat {y} | y = 1) $. Precision measures what fraction of our detections are actually positive, and recall measures what fraction of the positives we actually detected. If $ \hat {y}_i  \in {0, 1} $ is the predicted label, and  $ y_i \in {0, 1} $ is the true label, we can estimate precision and recall using
$$ P = \frac {\sum_i y_i \hat{y}_i}{\sum_i \hat{y}_i}, \quad R = \frac {\sum_i y_i \hat {y}_i}{\sum_i y_i} $$
A precision recall curve is a plot of precision vs recall as we vary the threshold $ \tau $. See Figure 5.15 (b). Hugging the top right is the best one can do.
\begin{figure}[htp]
    \centering
    \includegraphics[width=10cm]{Precsion and recall.jpg}
    \caption{Precision and recall}
    \label{fig:galaxy}
\end{figure}

This curve can be summarized as a single number using the mean precision (averaging over recall values), which approximates the area under the curve. Alternatively, one can quote the precision for a fixed recall level, such as the precision of the first K = 10 entities recalled. This is called the average precision at K score. This measure is widely used when evaluating information retrieval systems.

\Large 11. Support vector machine 

\normalsize One of the most influential approaches to supervised learning is the support vector machine. This model is similar to 
logistic regression in that it driven by as linear function $w^T x + b$. Unlike logistic regression, the support vector machine 
does not provide probabilities, but only outputs a class identity. The SVM predicts that the positive class is present when
$w^T x + b$ is positive. Likewise, it predicts that the negative class is present when $w^T x + b$ is negative. 

One key innovation associated with support vector machines is the $\mathbf {kernel \quad trick}$. The kernel trick consists of observing
that many machine learning algorithms can be written exclusively in terms of dot products between examples. For example, it can be 
shown that the linear function used by the support vector machine can be re-written as 

$$w^T x + b = b + \sum_{i=1}^m \alpha_i x^T x^{(i)} \quad  \boxed {(5.82)} $$
where $\mathbf{x^{(i)}}$ is a training example and $\mathbf{\alpha}$ is a vector of coefficients. Rewriting the learning algorithm this way allows us
to replace $\mathbf{x}$ by the output of a given feature function $\phi(x)$ and the dot product with a function $k(x, x^{(i)}) = \phi(x) \cdot \phi(x^{(i}))$
called a $\mathbf{kernel}$. The operator $\cdot$ operator represents an inner product analogous to $\phi(x)^T \phi(x^{(i)})$. For some feature spaces,
we may not use literally the vector inner product. In some infinite dimensional spaces, we need to use other kinds of inner products, for
example, inner products based on integration rather than summation.

After replacing dot products with kernel evaluations, we can make predictions using the function
$$f(x) = b + \sum_i \alpha_i k(x, x^{(i)}) \quad \boxed {(5.83)} $$
This function is nonlinear with respect to $\mathbf{x}$, but the relationship between $\mathbf{\phi(x)}$ and $\mathbf{f(x)}$  is linear. Also, the 
relationship between $\mathbf{\alpha}$ and $\mathbf{f(x)}$ is linear. The kernel-based function is exactly equivalent to preprocessing the data
by applying $\phi(x)$ to all inputs, then learning a linear model in the new transformed space.

The kernel trick is powerful for two reasons. First, it allows us to learn models that are nonlinear as a function of $\mathbf{x}$ using
convex optimization techniques that are guaranteed to converge efficiently. This is possible because we consider $\phi$ fixed and optimize
only $\mathbf{\alpha}$, i.e., the optimization algorithm can view the decision function as being linear in a different space. Second, the kernel
function k often admits an implementation that is significantly more computational efficient than naively constructing two $\phi(x)$  vectors
and explicitly taking their dot product.

In some cases, $\phi(x)$ can even be infinite dimensional, which would result in infinite computational cost for the naive, explicit approach. In
many cases, $k(x, x')$ is a nonlinear, tractable function of $\mathbf{x}$ even then $\phi(x)$ is intractable. As an example of infinite-dimensional
feature space with a tractable kernel, we construct a feature mapping $\phi(x)$ over the non-negative integers x. Suppose that this mapping
returns a vector containing x ones followed by infinitely many zeros. We can write a kernel function $k(x, x^{(i)})=min (x, x^{(i)})$ that
is exactly equivalent to the corresponding infinite-dimensional dot product.

The most commonly used kernel is the Gaussian kernel

$$k(u, v) = \mathcal {N} (u-v; 0; \sigma^2 \mathbf{I}) \quad \boxed{(5.84)} $$
where $\mathcal{N}(x; \mu; \Sigma)$ is the standard normal density. This kernel is also known as the radial basis fucntion (RBF) kernel, because
its value decreases along lines in $v$ space radiating outward from $u$. The Gaussian kernel corresponds to a dot product in an infinite-dimensional
space, but the derivation of this space is less straightforward than in our example of the min kernel over the integers.

We can think of the Gaussian kernel as performing a kind of template matching. A training example x associated with training label y
becomes a template for class y. When a test point $x'$ is near x according to Euclidean distance, the Gaussian kernel has a large response, indicating
that $x'$ is very similar to the x template. The model then puts a large weight on the associated training label y. Overall, the prediction
will combine many such training labels weighted by the similarity of the corresponding training examples.

Support vector machines are not only algorithm that can enhanced using the kernel  trick. Many other linear models can be enhanced in this way.
The category of algorithms that employ the kernel trick is known as kernel machines or kernel methods

\Large 12. Bootstrap and Bagging

\normalsize The $\mathbf {bootstrap}$ is a simple Monte Carlo technique to approximate the sampling distribution. This is particularly useful in cases where the estimator is a complex function of the true parameters. 
The idea is simple. If we knew the true parameters $ \theta^* $, we could generate many (say S) fake datasets, each of size N, from the true distribution, $ x_i^s \sim p (\cdot | \theta^* )$, for $ s = 1 : S, I = 1 : N $. We could then compute our estimator from each sample, $ \hat {\theta}^s = f (x_{1:N}^S) $ and use the empirical distribution of the resulting samples as our estimate of the sampling distribution. Since $\theta$ is unknown, the idea of the parametric bootstrap is to generate the samples using $ \hat {\theta} (\mathcal {D}) $ instead. An alternative, called the non-parametric bootstrap, is to sample the $ x_i^s $ (with replacement) from the original data $ \mathcal {D} $, and then compute the induced distribution as before. 
Figure 6.1 shows an example where we compute the sampling distribution of the MLE for a Bernoulli using the parametric bootstap. (Results using the non-parametric bootstrap are essentially the same). We see that the sampling distribution is asymmetric, and therefore quite far from Gaussian, when $ N = 10 $; when $ N = 100 $, the distribution looks more Gaussian, as theory suggests (see below).
\begin{figure}[htp]
    \centering
    \includegraphics[width=10cm]{Bootstrap.png}
    \caption{Bootstrap}
    \label{fig:galaxy}
\end{figure}
A natural question is: what is the connection between the parameter estimates $ \hat {\theta}^s = \hat {\theta} (x_{1:N}^s)$ computed by the bootstrap and parameter values sampled from the posterior,
$$ \theta^s \sim p (\cdot | \mathcal {D}) $$ ? 
Conceptually they are quite different. But in the common case that the prior is not very strong, they can be quite similar. For example, Figure 6.1 shows an example where we compute the posterior using a uniform Beta (1, 1) prior, and then sample from it. We see that the posterior and the sampling distribution are quite similar. So one can think of the bootstrap distribution as a “poor man’s” posterior.
However, perhaps surprisingly, bootstrap can be slower than posterior sampling. The reason is that the bootstrap has to fit the model S times, whereas in posterior sampling, we usually only fit the model once (to find a local mode), and then perform local exploration around the mode. Such local exploration is usually much faster than fitting a model from scratch.

Bagging (short for bootstrap aggregating) is a technique for reducing generalization error by combining several models. The idea is to train several different models separately, then have all of the models vote on the output for test examples. This is an example of a general strategy in machine learning called model averaging. Techniques employing this strategy are known as ensemble methods.
The reason that model averaging works is that different models will usually not make all the same errors on the test set.
Consider for example a set of k regression models. Suppose that each model makes an error $epsilon_i$ on each example, with the errors drawn from a zero-mean multivariate normal distribution with variances $ \mathbb{E}[\epsilon_i^2] = v $  and covariances $ \mathbb{E} [\epsilon_i  \epsilon_j] = c $ Then the error made by the average prediction of all the ensemble models is $ \frac{1}{k} \sum_i \epsilon_i $ The expected squared error of the ensemble predictor is 
$$ \mathbb{E} [(\frac{1}{k} \sum_i \epsilon_i)^2]  = \frac {1}{k^2} \mathbb{E} [ \ sum_i (\epsilon_i^2  + \sum_{j \neq i} \epsilon_i \epsilon_j)] =\\
= \frac {1}{k} v + \frac {k – 1}{k} c $$
In the case where the errors are perfectly correlated and $ c = v $, the mean squared error reduces to v, so the model averaging  does not help at all. In the case where the errors are perfectly uncorrelated and $c = 0$, the expected squared error of the ensemble decreases linearly with the ensemble size. In other words, on average, the ensemble will perform at least as well as any of its members, and if the members make independent errors, the ensemble will perform significantly better than its members. 

\Large 13. Random Forest

\normalsize There are two ensemble learning paradigms: bagging and boosting. Bagging consists of creating many “copies” of the training data (each copy is slightly different from another) and then apply the weak learner to each copy to obtain multiple weak models and then combine them. The bagging paradigm is behind the random forest learning algorithm.
The “vanilla” bagging algorithm works like follows. Given a training set, we create B random samples $ S_b $ (for each $ b = 1, …, B $) of the training set and build a decision tree model $ f_b $ using each sample  $ S_b $ as the training set. To sample $ S_b $ for some $b$, we do the sampling with replacement. This means that we start with an empty set, and then pick at random an example from the training set and put its exact copy to $S_b$ by keeping the original example in the original training set. We keep picking examples at random until the $ |S_b| = N $.
After training, we have $B$ decision trees. The prediction for a new example $x$ is obtained as the average of $B$ predictions:
$$ y \leftarrow \hat {f} (x) \stackrel{\text{def}}{\scalebox{2}[1]{=}} \frac {1}{B} \sum_{b=1}^B f_b (x) $$
In the case of regression, or by taking the majority vote in the case of classification.
The random forest algorithm is different from the vanilla bagging in just one way. It uses a modified tree learning algorithm that inspects, at each split in the learning process, a random subset of the features. The reason for doing this is to avoid the correlation of the trees: if one or a few features are very strong predictors for the target, these features will be selected to split examples in many trees. This would result in many correlated trees in our “forest”. Correlated predictors cannot help in improving the accuracy of prediction. The main reason behind a better performance of model ensembling is that models that are good will likely agree on the same prediction, while bad models will likely disagree on different ones. Correlation will make bad models more likely to agree, which will hamper the majority vote of the average.
The most important hyperparameters to tune are the number of trees, $B$, and the size of the random subset of the features to consider at each split.
Random forest is one of the most widely used ensemble learning algorithms. Why is it so effective? The reason is that by using multiple samples of the original dataset, we reduce the variance of the final model. Remember that the low variance means low overfitting. Overfitting happens when our model tries to explain small variations in the dataset because our dataset is just small sample of the population of all possible examples of the phenomenon we try to model. If we were unlucky with how our training set was sampled, then it could contain some undesirable (but unavoidable) artifacts: noise, outliers and over- or underrepresented examples. By creating multiple random samples with replacement of our training set, we reduce the effect of these artifacts.


\Large 14. Principal Component Analysis

\normalsize Often a dataset has many columns, perhaps tens, hundreds, thousands and more. Modelling data with many features is challenging, and models built from data that include irrelevant features are often less skillful than models trained from the most relevant data. It is hard to know which features of the data are relevant and which are not. Methods for automatically reducing the number of columns of a dataset are called dimensionality reduction, and perhaps the most popular is method is called the principal component analysis or PCA for short. This method is used in machine learning to create projections of high-dimensional data for both visualization and for training models. The core of the PCA method is a matrix factorization method from linear algebra. The eigendecomposition can be used and more robust implementations may use the singular-value decomposition or SVD.
Principal Component Analysis, or PCA for short, is a method for reducing the dimensionality of data. It can be thought of as a projection method where data with m-columns (features) is projected into a subspace with m or fewer columns, whilst retaining the essence of the original data. The PCA method can be described and implemented using the tools of linear algebra.
PCA is an operation applied to a dataset, represented by an $ n \times m $ matrix $A$ that results in a projection of $A$ which we will call $B$. Let’s walk through the steps of this operation.
$$ A = \begin {pmatrix} a_{1,1} & a_{1,2} \\ a_{2,1} & a_{2,2} \\ a_{3,1} & a_{3,2} \end {pmatrix} $$
$$ B = PCA (A) $$
The first step is to calculate the mean values of each column.
$$ M = mean (A) $$
Next, we need to center the values in each column by subtracting the mean column value
$$ C = A - M $$ 
The next step is to calculate the covariance matrix of the centered matrix C. Correlation is a normalized measure of the amount and direction (positive or negative) that two columns change together. Covariance is a generalized and unnormalized version of correlation across multiple columns. A covariance matrix is a calculation of covariance of a given matrix with covariance scores for every column with very other column, including itself
$$ V = cov(C) $$
Finally, we calculate the eigendecomposition of the covariance matrix $V$. This results in a list of eigenvalues and a list of eigenvectors
$$ values,vectors = eig(V) $$
The eigenvectors represent the directions or components for the reduced subspace of $B$, whereas the eigenvalues represent the magnitudes for the directions. The eigenvectors can be sorted by the eigenvalues in descending order to provide a ranking of the components or axes of the new subspace for $A$. If all eigenvalues have a similar value, then we know that the existing representation may already be reasonably compressed or dense and that the projection may offer little. If there are eigenvalues close to zero, they represent components or axes of $B$ that may be discarded. A total of $m$ or less components must be selected to comprise the chosen subspace. Ideally, we would select $k$ eigenvectors, called principal components, that have the $k$ largest eigenvalues
$$ B = select(values, vectors) $$
Other matrix decomposition methods can be used  such as Singular-Value Decomposition, or SVD. As such, generally the values are referred to as singular values and the vectors of the subspace are referred to as principal components. Once chosen, data can be projected into the subspace via matrix multiplication
$$ P = B^T \cdot A $$
Where $A$ is the original data that we wish to project, $B^T$ is the transpose of the chosen principal components and $P$ is the projection of $A$. This is called the covariance method for calculating the PCA, although there are alternative ways to calculate it.

15. ROC curves and all that

Suppose we are solving a binary decision problem, such as classification, hypothesis testing, object detection, etc. Also, assume we have a labeled data
set, $\mathcal{D} = {(x_i, y_i)}$. Let $\delta(x) = \mathbb{I} (f(x) > \tau)$ be our decision rule where $f(x)$ is a measure of confidence that $y=1$ 
(this should be monotonically related to $p(y=1|x)$), but does not need to be probability), and $\tau$ is some threshold parameter. For each given
value of $\tau$, we can apply our decision rule and count the number of true positives, false positives, true negatives, and false negatives that occur,
as shown in table 5.2. This table of errors is called a confusion matrix.\\
From this table, we can compute the true positive rate (TPR), also known as the sensitivity, recall or hit rate, by using 
$$TPR=TP/N_+ \approx p(\hat{y}=1|y=1$$
We can also compute the false positive rate (FPR), aslo called false alarm rate, or the type I error rate, by using 
$$FPR=FP/N_- \approx p(\hat{y} = 1|y=0)$$
These and other definitions are summarized in Tables 5.3 and 5.3. We can combine these errors in any way we choose to compute a loss function.

However, rather than computing the TPR and FPR for a fixed threshold $\tau$, we can run our detector for a set of thresholds, and then
plot the TPR vs FPR as an implicit function of $\tau$. This is called a receiver operator characteristic or ROC curve. Any system
can achieve the point on the bottom left, (FPR = 0, TPR = 0), by setting $\tau=1$ and thus classifying everything as negative;
similarly any system can achieve the point on the top right, (FPR = 1, TPR =1), by setting $\tau =0$ and thus classifying 
everything as positive. If a system is performing at chance level, then we can achieve any point on the diagonal line 
TPR = FPR by choosing an appropriate threshold, A system that perfectly separates the positives from negatives
has a threshold that can achieve the top left corner, (FPR = 0, TPR = 1); by varying the threshold such a system will
"hug" the left axis and the top axis.

The quality of a ROC curve is often summarized as a single number using the area under the curve or AUC. Higher AUC 
scores are better; the maximum is obviously 1. Another summary statistic that is used is the equal error rate or EER,
also called the cross over rate, defined as the value which satisfies $FPR = FNR$. Since $FNR = 1 - TPR$, we can compute
the EER by drawing a line from the top left to the bottom right and seeing where it intersects the ROC curve. Lower
EER scores are better; the minimum is obviously 0. 


16. Machine learning classification\\
The problem: we have a big pile of data, and we want to use it to make classifications. For example, we meet this person and want to Classify them as someone who will like StatQuest or not.
A Solution: We can use our data to build a Classification Tree to Classify a person as someone who will like StatQuest or not.

2 main ideas:
First, we use Testing data to evaluate machine learning methods.
Second, just because a machine learning method fits the Training data well, it doesn’t mean it will perform well with the Testing data.

When a machine learning method fits the Training data really well but makes poor predictions, we say that it is Overfit to the Training data. Overfitting a machine learning method is related to something called the Bias-Variance Tradeoff.

Independent and dependent variables
So far, we’ve been  predicting Height from Weight measurements and the data have all been displayed on a nice graph. However, we can also organize the data in a nice table.
Now, regardless of whether we look at the data in the graph or in the table, we can see that Weight varies from person to person, and thus, Weight is called a Variable.
Likewise, Height varies from person to person, so Height is also called a Variable.
That being said, we can be more specific about the types of Variables that Height and Weight represent.
Because our Height predictions depend on Weight measurements, we call Height a Dependent variable.
In contrast, because we’re not predicting Weight, and this, Weight does not depend on Height, we call Weight an Independent Variable. Alternatively, Weight can be called a Feature.

Cross validation
The problem: So far, we’ve simply been told which points are the Training data and which points are the Testing data. However, usually no one tells us what is for Training and what is for Testing. How do we pick the best points for Training and the best points for Testing.
A solution: When we’re not told which data should be used for Training and for Testing, we can use Cross validation to figure out which is which in an unbiased way. Rather than worry too much about which specific points are best for Training and best for Testing, Cross Validation uses all points for both in an iterative way, meaning that we use them in steps. 

\Large 17. Linear regression

\normalsize Linear regression is a model of the form
$$p (y|x, \theta) = \mathcal {N} (y| w^T x, \sigma^2) $$
Linear regression can be made to model non-linear relationship by replacing x with some non-linear function of the inputs, $\phi(x)$. That is we use 
$$p (y|x, \theta) = \mathcal{N} (y | w^T \phi(x), \sigma^2)$$
This is known as basis function expansion. (Note that the model is still linear in the parameters w, so it is still called linear regression; the importance of this will become clear below). A simple example are polynomial basis functions, where the model has the form
$$ \phi (x) =  1, x, x^2,..., x^d $$

Maximum likelihood estimation (least squares)

A common way to estimate the parameters of a statistical model is to compute the MLE, which is defined as
$$\hat {\theta} \triangleq \arg\max_{\theta} \log p (\mathcal {D} | \theta) $$ 
It is common to assume the training examples are independent and identically distributed, commonly abbreviated to iid. This means we can write the log-likelihood as follows:
$$\ell (\theta) \triangleq log p (\mathcal {D} |\theta) = \sum_{i=1}^N \log p (y_i|x_i, \theta) $$
Instead of maximizing the log-likelihood, we can equivalently minimize the negative log likelihood or NLL:
$$NLL (\theta) \triangleq - \sum_{i=1}^N \log p (y_i | x_i, \theta) $$
The NLL formulation is sometimes more convenient, since many optimization software packages are designed to find the minima of functions, rather than maxima.
Now let us apply the method of MLE to the linear regression setting. Inserting the definition of the Gaussian into the above, we find that the log likelihood is given by 
$$\ell (\theta) = \sum_{i=1}^N \log (\frac {1}{2 \pi \sigma^2)^{1/2}} 
\exp (- \frac{1}{2 \sigma^2} (y_i – w^T x_i)^2) =$$
$$ = \frac {-1}{2 \sigma^2} RSS (w) - \frac {N}{2} \log (2 \pi \sigma^2) $$

RSS stands for residual sum of squares and is defined by
$$ RSS (w) \triangleq \sum_{i=1}^N (y_i – w^T x_i)^2 $$

The RSS is also called the sum of squared errors, or SSE, and SSE/N is called the mean squared error or MSE. It can also be written as the square of the $ \ell_ 2 $ norm of the vector of  residual errors:
$$ RSS (w) = ||\epsilon||_2^2 = \sum_{i=1}^N \epsilon_i^2 $$
Where $ \epsilon_i = (y_i – w^T x_i) $ 
We see that the MLE for w is the one that minimizes the RSS, so this method is known as least squares. The training data $ (x_i, y_i) $ are shown as red circles, the estimated values $ (x_i, \hat {y_i}) $ are shown as blue crosses, and the residuals $ \epsilon_i = y_i - \hat {y}_i  $ are shown as vertical blue lines. The goal is to find the setting of the parameters (the slope $w_i$ and intercept $ w_0 $) such that the resulting red line minimizes the sum of squared residuals (the length of the vertical blue lines). 

Derivation of the MLE 

First, we rewrite the objective in a form that is more amenable to differentiation:
$$ NLL (w) = \frac {1}{2} (y – X w)^T (y – Xw) = \frac {1} {2} w^T (X^T X)w – w^T (X^T y) $$

Where

$$ X^T X = \sum_{i=1}^N x_i x_i^T = \sum_{i=1}^N  
\begin{pmatrix} 
x_{i, 1}^2 & \ldots & x_{i, 1}, x_{i, D} \\
 \ldots & \ddots  & \ldots \\
x_{i, D} x_{i, 1} & \ldots & x_{i, D}^2  
\end {pmatrix}$$
is the sum of squares matrix and
$$X^T y = \sum_{i=1}^N x_i y_i $$
Using results from Equation 4.10, we see that the gradient of this is given by
$$ g(w) = [X^T X w – X^T y] = \sum_{i=1}^N x_i (w^T x_i – y_i) $$
Equating to zero we get
$$ X^T X w = X^T y $$
This is known as the normal equation. The corresponding solution $ \hat {w} $ to this linear system of equations is called the ordinary least squares or OLS solution, which is given by
$$ \hat {w}_{OLS} = (X^T X)^{-1} X^T y $$

\Large 18. The sum of the Squared residuals

\normalsize The problem – we have a model that makes predictions. In this case, we’re using Weight to predict Height. However, we need to quantify the quality of the model and its predictions.\\
A solution – One way quantify the quality if a model and its predictions is to calculate the Sum of the squared residuals. As the name implies, we start by calculating Residuals, the differences between the Observed values and the values Predicted by the model.
$$ Residual = Observed – Predicted $$
Since in general, the smaller the Residual, the better the model fits the data, it’s tempting to compare models by comparing the sum of their Residuals, but the Residuals below the blue line would cancel out the ones above it. So, instead of calculating the sum of the Residuals, we square the Residuals first and calculate the Sum of the Squared Residuals (RSS).
The Sum of Squared Residuals (SSR) is usually defined with fancy Sigma notation and the right-hand side reads: “The sum of all observations of the squared difference between the observed and predicted values.”
$$ SSR = \sum_{i=1}^n (Observed_i – Predicted_i)^2 $$
Where $n$ - the number of Observations, $i$ - the index for each Observation\\

The problem – sum of the squared residuals (SSR), although awesome, is not super easy to interpret because it depends, in part, on how much data you have. \\
For example, if we start with a simple dataset with 3 points, the Residuals are, from left to right, 1, -3, and 2, and SSR = 14. Now, if we have a second dataset that includes 2 more data points added to the first one, and the Residuals are -2 and 2, then the SSR increases to 22.\\
However, the increase in the SSR from 14 to 22 does not suggest that the second model, fit to the second, larger dataset, is worse than the first. In only tells is that the model with more data has more Residuals.\\
A solution – one way to compare the two models that may be fit to different-sized datasets is to calculate the Mean Squared Error (MSE), which is simply the average of the SSR
$$ MSE = \frac {SSR} {Number \quad of \quad observations, n} = \sum_{i=1}^n  \frac {( Observed_i – Predicted_i)^2} {n} $$

 $R^2$: main ideas\\
The problem – as we just saw, the MSE, although totally cool, can be difficult to interpret because it depends, in part, on the scale of the data. In this example, changing the units from millimeters to meters reduced the MSE by a lot ($ MSE = 4.7$ and $ MSE = 0.0000047 $) \\
A solution: $R^2$, pronounced R squared, is a simple, easy-to-interpret metric that does not depend on the size of the dataset or its scale. Typically, $R^2$ is calculated by comparing the SSR or MSE around the mean y-axis value. In this example, we calculate the SSR or MSE around the average Height and compare it to the SSE or MSE around the model we’re interested in. In this case, that means we calculate the SSR or MSE around the blue line that uses Weight to predict Height. $R^2$ then gives us a percentage of how much the predictions improved by using the model we’re interested in instead of just the mean. \\
In this example, $R^2$ would tell us how much better our predictions are when we use the blue line, which uses Weight to predict Height, instead of predicting that everyone has the average Height. \\
$R^2$ values go from 0 to 1 and are interpreted as percentages, and the closer the value is to 1, the better the model fits the data relative to the mean y-axis value.\\
First, we calculate the Sum of the Squared Residuals for the mean. We’ll call SSR the SSR (mean). In this this example, the mean Height is 1.9 and the SSR (mean) is 1.6
$$ SSR(mean) = (2.3 – 1.9)^2 + (1.2 – 1.9)^2 +$$
$$+ (2.7 -1.9)^2 +(1.4-1.9)^2 +(2.2 -1.9)^2 = 1.6 $$
Then, we calculate the SSR for the fitted line, SSR (fitted line), and we get 0.5
Note: The smaller Residuals around the fitted line, and the smaller SSR given the same dataset, suggest the fitted line does a better job making predictions than the mean.
$$ SSR (fitted \quad line) = (1.2 – 1.1)^2 + (2.2 – 1.8)^2 +$$
$$ +(1.4 – 1.9)^2 + (2.7 – 2.4)^2 + (2.3 – 2.5)^2 = 0.5 $$
Now we can calculate the $R^2$ value using a surprisingly simple formula
$$ R^2 = \frac {SSR(mean) – SSR (fitted \quad line)}{SSR (mean)} = \frac {1.6 – 0.5}{1.6} = 0.7 $$
And the result, 0.7, tells us that there was a 70% reduction in the size of the Residuals between the mean and the fitted line.

In general, because the numerator for $R^2$ 
$$ SSR(mean) – SSR (fitted \quad line) $$

Is the amount by which the SSRs shrank when we fitted the line, $R^2$ values tell us the percentage the Residuals around the mean shrank when we used the fitted line.\\
When SSR (mean) = SSR (fitted line), the both models predictions are equally good (or equally bad), and $R^2 = 0$
$$ \frac{SSR(mean) – SSR(fitted \quad line)}{SSR(mean)} = \frac {0}{SSR(mean)} = 0 $$
When SSR (fitted line) = 0, meaning that the fitted line fits the data perfectly, then $R^2 =1$
$$ \frac {SSR(mean) – 0}{SSR(mean)} = \frac {SSR(mean)}{SSR(mean)} = 1 $$

\Large 19. Gradient Descent\\
\normalsize  The problem -  a major part of machine learning is optimizing a model’s fit to the data. Sometimes this can be done with an analytical solution, but it’s not always possible. \\
For example, there is no analytical solution for Logistic regression, which fits an s-shaped squiggle to data. \\
Likewise, there is no analytical solution for Neural Networks, which fit fancy squiggles to data.\\
A solution – when there’s no analytical solution, Gradient Descent save the day. Gradient Descent is an iterative solution that incrementally steps toward an optimal solution and is used in a very wide variety of situations.
Gradient Descent starts with an initial guess and then improves the guess, one step at a time, until it finds an optimal solution or reaches a maximum number of steps. \\
Note: even though there’s an analytical solution for Linear regression, we’re using it to demonstrate how Gradient Descent works because we can compare the output from Gradient Descent to the known optimal values. \\

\Large Write down a gradient descent step. How to adjust it for huge datasets?\\
\normalsize Perhaps the simplest algorithm for unconstrained optimization is gradient descent, also known as steepest descent. This can be written as follows:
$$ \theta_{k+1} = \theta_k - \eta_k g_k $$ 
Where $\eta_k$ is the step size or learning rate. The main issue in gradient descent is: how should we set the step size? This turns out to be quite tricky. If we use a constant learning rate, but make it too small, convergence will be very slow, but if we make it too large, the method can fail to converge at all. This is illustrated in Figure 8.2, where we plot the following (convex) function
$$ f(\theta) = 0.5 (\theta_1^2 - \theta_2)^2 + 0.5 (\theta_1 – 1)^2 $$

We arbitrarily decide to start from (0, 0). In Figure 8.2(a), we use a fixed step size of $ \eta = 0.1 $ ; we see that it moves slowly along the valley. In Figure 8.2(b), we use a fixed step size of $ \eta = 0.6 $; we see that the algorithm starts oscillating up and down the sides of the valley and never converges to the optimum. \\
Let us develop a more stable method for picking the step size, so that the method guaranteed to converge to a local optimum no matter where we start (This property is called global converge, which should not be confused with convergence to the global optimum!). By Taylor’s theorem, we have
$$ f(\theta + \eta d) \approx f(\theta) + \eta g^T d $$
Where $d$ is our descent direction. So if $\eta$ is chosen small enough,  then 
$ f (\theta + \eta d) < f (\theta) $, since the gradient will be negative. But we don’t want to choose  the step size $ \eta $ too small, or we will move very slowly and may not reach the minimum. So let us puck $ \eta $ to minimize:
$$ \phi (\eta) = f (\theta_k + \eta d_k) $$
This is called line minimization or line search. There are various methods for solving this 1d optimization problem\\
Figure 8.3 a) demonstrates that line search does indeed work for our simple problem. However, we see that the steepest descent path with exact line searches exibists a characteristic zig-zag behavior. To see why, note that an exact line search satisfies $ \eta_k = argmin_{\eta>0} \phi (\eta) $ A necessary condition for the optimum is $ \phi’ (\eta) = 0 $ By the chain rule, $ \phi’ (\eta) = d^T g $, where $ g = f’ (\theta + \eta d) $ is the gradient at the end of the step. So we either have $ g = 0 $, which means we have found a stationary point, or $ g \perp d $, which means that exact search stops at a point where the local gradient is perpendicular to the search direction. Hence consecutive directions will be orthogonal (see Figure 8.3 (b)) This explains the zig-zag behavior. \\


One simple heuristic to reduce the effect of zig-zagging is to add a momentum term, $ (\theta_k - \theta_{k-1}) $, as follows:
$$ \theta_{k+1} = \theta_k - \eta_k g_k + \mu_k (\theta_k - \theta_{k-1}) $$
Where $ 0 \leq \mu_k \leq 1 $ controls the importance of the momentum term. In the optimization community, this is known as the heavy ball method.

\Large 20. Stochastic Gradient Descent \\
\normalsize  Nearly all deep learning is powered by one very important algorithm: stochastic gradient descent or SGD. Stochastic gradient descent is an extension of the gradient descent algorithm. \\
A recurring problem in machine learning is that large training sets are necessary for good generalization, but large training sets are also more computationally expensive.\\
The cost function used by a machine learning algorithm often decomposes as a sum over training examples of some per-example loss function. For example, the negative conditional log-likelihood of the training data can be written as
$$ J(\theta) = \mathbb {E}_{x,y \sim \hat{p}_{data}} L (x, y, \theta) = \frac {1}{m} \sum_{i=1}^m L (x^{(i)}, y^{(i)}, \theta) $$
Where L is the per-example loss $ L (x, y, \theta) = - \log p (y| x; \theta)$ \\
For these additive cost functions, gradient descent requires computing 
$$ \nabla_{\theta} J (\theta) = \frac {1}{m} \sum_{i=1}^m \nabla_{\theta} L (x^{(i)}, y^{(i)}, \theta) $$
The computational cost of this operation is O (m). As the training set size grows to billions of examples, the time to take a single gradient step becomes prohibitively long. \\ 
The insight of stochastic gradient descent id that the gradient is an expectation. The expectation may be approximately estimated using a small set of samples. Specifically, on each step of the algorithm, we can sample a minibatch of examples $ \mathbb {B} = {x^{(i)}, ….., x^{(m’)}} $ drawn uniformly from the training set. The minibatch size $m’$ is typically chosen to be relatively small number of examples, ranging from 1 to a few hundred. Crucially, $m’$ is usually held fixed as the training set size m grows. We may fit a training set with billions of examples using updates computed on only a hundred examples.\\
The estimate of the gradient is formed as
$$ g = \frac {1}{m’} \nabla_{\theta} \sum_{i=1}{m’} L(x^{(i)}, y^{(i)}, \theta) $$
Using examples from the minibatch $\mathbb {B}$. The stochastic gradient descent algorithm then follows the estimated gradient downhill:
$$ \theta \leftarrow \theta - \epsilon g $$
Where $\epsilon$ is the learning rate. \\

\Large 21. What is the likelihood? Where is Maximum Likelihood Estimation (MLE) usually used?\\
\normalsize Previously, we have seen some definitions of common estimators and analyzed their properties. But where did these estimators come from? Rather than guessing that some function might make a good estimator and then analyzing its bias and variance, we would like to have some principle from which we can derive specific functions that are good estimators for different models.\\ 
The most common such principle is the maximum likelihood principle.\\
Consider a set of $m$ examples $\mathbb {X} = {x^{(1)}, …, x^{(m)}}$ drawn independently from the true but unknown data generating distribution $p_{data} (x)$ \\
Let $ p_{model} (x; \theta) $ be a parametric family of probability distributions over same space indexed by $\theta$. In other words, $p_{model} (x; \theta) $ maps any configuration $x$ to a real number estimating the true probability $p_{data} (x) $. \\
The maximum likelihood estimator for $ \theta $ is then defined as 
$$ \theta_{ML} = \arg\max_{\theta} p_{model} (\mathbb{X}; \theta) =\\
= \arg\max_{\theta} \prod_{i=1}^m p_{model} (x^{(i)}; \theta) $$
This product over many probabilities can be inconvenient for a variety of reasons, For example, it is prone to numerical underflow. To obtain a more convenient but equivalent optimization problem, we observe that taking the logarithm of the likelihood does not change its $\arg\max$ but does conveniently transform a product into a sum:
$$ \theta_{ML} = \arg\max_{\theta} \sum_{i=1}^m \log p_{model} (x; \theta)  (5.59) $$
One way to interpret maximum likelihood estimation is to view it as minimizing the dissimilarity between the empirical distribution $ \hat {p}_{data}$ defined by the training set and the model distribution, with the degree of dissimilarity between the two measured by the KL divergence. The KL divergence is given by
$$ D_{KL} (\hat {p}_{data} \Vert p_{model} ) = \mathbb {E}_{X \sim \hat {p}_{data}} [\log \hat {p}_{data} (x) - \log p_{model} (x)] $$
The term on the left is a function only of the data generating process, not the model. This means when we train the model to minimize the KL divergence, we need only minimize 
$$ - \mathbb {E}_{X \sim \hat {p}_{data}} [\log p_{model} (x)] $$
Which is of course the same as the maximization in equation 5.59

\Large What is a regularization? What is the difference between L1 and L2 regularization in linear models? Is it the only way to constrain the solution? \\
\normalsize Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error. \\
$\ell_1$ regularization\\
When we have many variables, it is computationally difficult to find the posterior mode of  $p (\gamma | \mathcal{D})$. And although greedy algorithms often work well, they can of course get stuck in local optima.\\
Part of the problem is due to the fact that the $\gamma_j$ variables are discrete, $\gamma_j \in {0, 1}$. In the optimization community, it is common to relax hard constraints of this form by replacing discrete variables with continuous variables. We can do this by replacing spike-and-slab style prior, that assigns finite probability mass to the event that $w_j = 0$, to continuous priors that “encourage” $w_j = 0$ by putting a lot of probability density near the origin, such as a zero-mean Laplace distribution. There we exploited the fact that the Laplace has heavy tails. Here we exploit the fact that it has a spike near $\mu = 0 $ More precisely, consider a prior of the form
$$ p(w|\lambda) = \prod_{j=1}^D Lap(w_j | 0,1 /\lambda) \propto \prod_{j=1}^D e^{-\lambda |w_j|} $$
We will use a uniform prior on the offset term, $p(w_0) \propto 1$. Let us perform MAP estimation with this prior. The penalized negative log likelihood has the form
$$ f(w) = - \log p(\mathcal(D) | w) - \log p (w | \lambda) = NLL (w) + \lambda \Vert w \Vert_1 $$
Where $ \Vert w \Vert_1 = \sum_{j=1}^D |w_j|$ is the $\ell_1$ norm of $w$. For suitably large $\lambda$, the estimate $ \hat {w}$ will be sparse, for reasons we explain below. Indeed, this can be thought if a convex approximation to the non-convex $\ell_0$ objective.
$$ \arg\min_w NNL(w) + \lambda \Vert w \Vert_0 $$
In the case of linear regression, the $\ell_1$ objective becomes
$$ f(w) = \sum_{i=1}^N - \frac {1}{2 \sigma^2} (y_i –(w_0 w^T x_i))^2 + \lambda \Vert w \Vert_1 = \\
RSS (w) + \lambda’ \Vert w \Vert_1 $$ 
Where $ \lambda’ = 2 \lambda \sigma^2 $. This method is known as basis pursuit denoising or BPDN. In general, the technique of putting a zero-mean Laplace prior on the parameters and performing MAP estimation is called $\ell_1$ regularization. 

\Large Why is it a good idea to normalize data before applying linear model?

\normalsize Normalization is the process of converting an actual range of values which a numerical feature can take, into a standard range of values, typically in the interval $[-1, 1]$ or $[0, 1]$.
For example, suppose the natural range of a particular feature is 350 to 1450. By substracting 350 from every value of the feature, and dividing the result by 110, one can normalize those
values into the range $[0, 1]$

More generally, the normalization formula looks like this:
$$ \bar{x} = \frac {x^{(j)} - min^{(j)}}{max^{(j)} - min^{(j)}} $$
where $min^{(j)}$ and $max^{(j)}$ are, respectively, the minimum and the maximum value of the feature $j$ in the dataset

Why do we normalize? Normalizing the data is not a strict requirement. However, in practice, it can lead to an increased speed of learning. Remember the gradient descent example from the previous chapter.
Imagine you have a two-dimensional feature vector. When you update the parameters of $w^{(1)}$ and $w^{(2)}$, you use partial derivatives of the average squared error with respect to $w^{(1)}$ and $w^{(2)}$ If $x^{(1)}$ is in the range $[0, 1000]$ and $x^{(2)}$ the range $[0, 0.0001]$, then the derivative with respect to a larger feature will dominate the update.

Additionally, it's useful to ensure that our inputs are roughly in the same relatively small range to avoid problems which computers have when working with very small or very big numbers (known as numerical overflow).

\Large Provide a linear classification problem statement. What is a margin?\\
\normalsize Linear regression is a popular method for solving high-dimensional function-fitting problems, but we can still apply the idea of linear regression to classification problems as well. As will be shown later, the advantage of linear regression is that model estimation is relatively simple, and it can be solved by a closed-form solution without using an iterative algorithm.
The basic idea in linear regression is to establish a linear mapping from input feature vectors $x$ to output targets $y = w^T x$. The only difference for two-class classification is that output targets are binary: $ y = \pm 1$. The popular criterion to estimate the mapping function is to minimize the total square error in a given training set, as follows:
$$ w^* =\arg\min_w E(w) = \arg\min_w \sum_{i=1}^N (w^T x_i - y_i)^2    (6.8) $$
where the objective function $E (w)$ measures the total reconstruction error in the training set when the linear model is used to construct each output from corresponding input.
By constructing the following two matrices:
$$ X = \begin {bmatrix}
x_1^T \\
x_2^T \\
\cdots \\
x_N^T \\
\end {bmatrix}_{N \times d} 
y = \begin {bmatrix} 
y_1 \\
y_2 \\
\cdots \\
y_N \\
\end {bmatrix}_{N \times 1} $$
we can represent the objective function $E(w)$ as follows:
$$ E(w) = \Vert Xw - y \Vert^2 = (Xw - y)^T (Xw - y) =\\
= w^T X^TXw - 2 w^T X^T y + y^T y $$
By diminishing the gradient $ \frac {\partial E(w)}{\partial w} = 0$, we have
$$ 2 X^T X w - 2 X^T y = 0 $$
Next, we derive the following closed-form solution for the linear regression problem:
$$ w^* = (X^T x)^{-1} X^T y $$
where we need to invert a $d \times d$ matrix $X^T X$, which is expensive for high-dimensional problems.
Once the linear model $w^*$ is Estimated in Eq. (6.9), we can assign a label to any new data $x$ based on the sign of the linear function:
$$y = (w^{*T} x) = \begin{cases} + 1 & \mbox {if} \quad w^{*T} x > 0 \\ -1, & \mbox{otherwise} \end{cases}$$

Liner regression can be easily solved by a closed-form solution, but it may not yield good performance for classification problems. The main reason is that the square error used in training models does not match well with our goal in classification. For classification problems, our primary concern is to reduce the misclassification errors rather than the reconstruction error. 

In machine learning, a margin classifier is a classifier which is able to give an associated distance from the decision boundary for each example. For instance, if a linear classifier (e.g. perceptron or linear discriminant analysis) is used, the distance (typically euclidean distance, though others may be used) of an example from the separating hyperplane is the margin of that example.

\Large Assume the dataset for binary classification is imbalanced, so 95 % of data belong to the first class. 
How to adjust the classification quality measures, to work with such data? Why?\\
\normalsize In many practical situations, your labeled dataset will have underrepresented the examples of some class. This is the case, for example, when your classifier has to distinguish between genuine and fraudulent e-commerce transactions: the examples of geniune transactions are much more frequent. If we use SVM with soft margin, you can define a cost for misclassified examples. Because noise is always present in the training data, there are high chances that many eaxmples of geniune transactions would end up on the wrong side of the decision boundary by contributing to the cost.
The SVM algorithm will try to move the hyperplane to avoid as much as possible misclassified examples. Because noise is always presernt in the training data, there are high chances that many examples of geniune transactions would end up on the wrong side of the decision boundary by contributing to the cost.
The SVM algorithm will try to move the hyperplane to avoid as much as possible misclassified examples. The "fraudulent" examples, which are in the minority, risk being misclassified in order to classify more numerous examples of the majority class correctly. This problem is observed for most learning algorithms applied to imbalanced datasets.
Each binary subproblem is likely suffer from the class imbalance problem. To see this, suppose we have 10 equally represented classes. When training $f_1$, we will have 10% positive examples and 90% negative examples, which can hurt performance. It is possible to devise ways to train all $C$ clssifiers simultaneously, but the resulting method takes $ O (C^2 N^2) $ time to train and $O (C^2 N_{sv})$ time.
Another approach is to use the one-versus-one or OVO approach, also called all pairs, in which we train $C(C-1)/2$ classifiers to discriminate all pairs $f_{c,c'}$ We then classify a point into the class which has the highest number of votes, However, this can also result in ambiguities. Also, it takes $O (C^2 N^2)$ time to train and $ O (C^2 N_{sv})$ to test each data point, where $N_{sv}$ is the number of support vectors. 

\Large Logistic loss function. How is related to Maximum likelihood estimation?\\
\normalsize In machine learning applications, such as neural networks, the loss functions is used to assess the goodness of fit of model. For instance, consider a simple neural net with one neuron and linear (identity) activation that has one input $x$ and one output $y$:
$$ y = b + wx $$
We train this NN on the sample dataset: $ (x_i, y_i) $ with $ i = 1,..., n $ observations. The training is trying different values of parameters $b, w$ and checking how good is the fit using the loss function.
$$ C (e) = e^2 $$
Then we have the following loss:
$$ Loss (b, w|x, y) = \frac {1}{n} \sum_{i=1}^n C(y_i - b - wx_i) $$
Learning means minimizing this loss:
$$ \min_{b, w} Loss (b, w|x, y) $$
MLE connection
You can pick the loss function which ever way you want, or fits your problem. However, sometimes the loss function choice follows the MLE approach to your problem. For instance, the quadratic cost and the above loss function are natural choices if you deal with Gaussian linear regression. Here's how.
Suppose that somehow you know that the true model is
$$ y = b + wx + \varepsilon $$
with $\varepsilon \sim \mathcal {N} (0, \sigma_2) $ - random Gaussian error with a constant variance. If this is truly the case then it happens so that the MLE of the parameters $b , w$ is the same as the optimal solution using the above NN with quadratic cost (loss).
Note, that in NN you're not obliged to alwasys pick cost (loss) function that mathches some kind of MLE approach. Also, although I described this approach using the neural networks, it applies to other statistical learning techniques in machine learning and beyond.

Loss function for Logistic Regression
The loss function for linear regression is squared loss. The loss function for Logistic regression is Log Loss, which is defined as follows:
$$ Log Loss = \sum_{(x, y) \in D} - y \log (y') - (1 - y) \log (1 - y') $$
where:
$\bullet$ $(x, y) \in D $ is the data set containing many labeled examples, which are $(x, y)$ pairs
$\bullet$ $y$ is the label in a labeled example. Since this is logistic regression, every value of $y$ must either be 0 or 1
$\bullet$ $y'$ is the predicted value (somewhere between 0 and 1), given the set of features in $x$

Surrogate loss functions
Minimizing the loss in the ERM/ RRM framework is not always easy. For example, we might want to optimize the AUC, or F1 scores. Or more simply, we might just want to minimize the 0-1 loss, as is common in classification. Unfortunately, the 01- risk is very non-smooth objective and hence is hard to optimize. One alternative is to use maximum likelihood estimation instead, since log-likelihood is a smooth convex upper bound on the 0-1 risk, as we show below.
To see this, consider binary logistic regression, and let $ y_i \in {-1, +1} $ Suppose our decision function computes the log-odds ratio,
$$ f(x_i) = \log \frac {p (y =1 | x_i, w)}{p (y = -1 |x_i, w)} = w^T x_i = \eta_i $$
Then the corresponding probability distribution on the output label is
$$ p(y_i| x_i, w) = sigm (y_i, \eta_i) $$
Let us define the log-loss as as
$$ L_{NLL} (y, \eta) = - \log p (y|x, w) = \log (1 + e^{-y \eta})$$
Log-loss is an example of a surrogate loss fucntion. Another example is the hinge loss:
$$ L_{hinge} (y, \eta) = \max (0, 1 - y \eta) $$

\Large Multiclass classification. One-vs-one, one-vs-all, their properties\\
\normalsize In many practical applications of pattern recognition, the number of classes, $c$, may be more than two. For example, the number of classes is $c=26$ in hand-written alphabet recognition.  
One way to reduce a multiclass classification problems is the one-versus-all method, which considers $c$ binary classification problems of one class versus the other classes. More specifically, for
$ y = 1,..., c$ the yth binary classification problem assigns label +1 to samples in class $y$ and -1 to samples in all other classes. Let $ \hat{y}_y (x) $ be a learned decision function for yth binary
classification problem. Then, test sample $x$ is classified into class $\hat{y}$ that gives the highest score:
$$\hat{y} = \arg\max_{y = 1,...,c} \hat {f}_y (x) $$
Another way to reduce a multiclass classification problem into binary classification problems is the one-versus-one method, which considers $c (c - 1)/2$ binary classification problems of one class 
versus another class. More specifically, for $y, y' = 1,..., c$, the $(y, y')th$ binary classification problem assigns label + 1 to samples in class y and  -1 to samples in class $y'$. Let 
$\hat{f}_{y, y'} (x)$ be a learned decision function for the $ (y, y')th$ binary classification problem. Then, test sample $x$ is classified into class $\hat{y}$ that gathers the highest votes:
$$ \hat{y}_{y, y'}(x) \geq 0 \Rightarrow Vote \quad for \quad class \quad y$$
$$ \hat{y}_{y, y'}(x) < 0 \Rightarrow Vote \quad for \quad class \quad y' $$
One-versus-all consists only of $c$ binary classification problems, while one-versus-one consists of $c(c - 1)/2$ binary classification problems. Thus, one-versus-all is more compact then
one-versus-one. However, a binary classification problem is one-versus-one involves samples in all classes. Thus, each binary classification problem is one-versus-one may
be solved more efficiently than that in one-versus-all. Suppose that each class contains $n/c$ training samples, and the computational complexity if classifier training is
linear with respect to the number of training samples. Then the total computational complexity for one-versus-all and one-versus-one is both $O(cn)$. However, if classifier
training takes super-linear time, which is often the case in many classification algorithms, one-versus-one is computationally more efficient then one-versus-all.
One-versus-one would also be advantegeous in that each binary classification problem is balanced. More specifically, when each class contains $n/c$ training samples, each binary
classification problem is one-versus-one contains $n/c$ positive and $n/c$ negative training samples, while each binary classification problem is one-versus-all contains $n/c$
 positive and $n(c - 1)/c$ negative training samples. The latter situation is often referred to as class imbalance, and obtaining high classification  accuracy in the 
 imbalanced case is usually more challenging than the balanced case. Furthermore, each binary classification problem is one-versus-one would be simpler than one-versus-rest,
 because the "all" class in one-versus-all usually possesses multimodality, which makes classifier training harder.\\
 On the other hand, one-versus-one has a potential limitation that voting can be tied: for example, when $c = 3$,
 $$ \hat{y}_{1,2} (x) \geq 0 \Rightarrow Vote \quad for \quad class \quad 1 $$
 $$ \hat{f}_{2,3}(x) \geq 0 \Rightarrow Vote \quad for \quad class \quad 2 $$
$$ \hat{y}_{1,3} (x) < 0 \Rightarrow Vote \quad for \quad class \quad 3 $$
In such a situation, weighted voting according to the value of $\hat{y}_{y, y'}(x) $ could be a practical option. However, it is not clear what the best way to weight is. 
As discussed above, both one-versus-all and one-versus-ove approaches have pros and cons. The direct method does not necessarily perform better than the reduction approaches, 
because multiclass classification problems are usually more complicated than binary classification problems. In practice, the best approach should be
selected depending on the target problem and other constraints.

\Large Bias-variance tradeoff\\
\normalsize Although using an inbiased estimator seems like a good idea, this is not always the case. To see why, suppose we use quadratic loss. As we showed above, the 
corresponding risk is the MSE. We now derive a very useful decomposition of the MSE (All expectations and variances are wrt the true distribution $p(\mathcal{D}|\theta^*)$,
but we drop the explicit conditioning for notational brevity). Let $\hat{\theta}= \hat{\theta}(\mathcal{D}) $ denote the estimate, and $\Bar{\theta} = \mathbb{E}[\hat{\theta}]$
denote the expected value of the estimate (as we wary $\mathcal{D}$). Then we have
$$ \mathbb {E} [(\hat{\theta} - \theta^*)^2] = \mathbb {E} [[(\hat{\theta} - \bar {\theta}) + (\bar{\theta} - \theta^*)]^2] $$ 
$$ = \mathbb {E} [(\hat{\theta} - \bar{\theta})^2] + 2(\bar{\theta} - \theta^*) \mathbb {E} [\hat{\theta} - \bar {\theta}] + (\bar{\theta} - \theta^*)^2 $$
$$ =\mathbb {E} [(\hat{\theta} - \bar{\theta})^2] + (\bar {\theta} - \theta^*)^2 $$
$$ = var [\hat{\theta}] +bias^2 (\hat{\theta}) $$
In words 
$$\boxed{MSE = variance + bias^2}$$
This is called the bias-variance tradeoff. What it means is that it might be wise to use a biased estimator, so long as it reduces our variance, assuming
our goal is to minimize squared error.

\Large Boosting and gradient boosting. Main idea, gradient derivation.\\
\normalsize Boosting is a greedy algorithm for fitting adaptive basis-function models of the form in equation
$$ f(x) = w_0 + \sum_{m=1}^M w_m \phi_m(x) \quad \boxed{(16.3)}$$
where $\phi_m$ are generated by alogrithm called a weak learner or a base learner. The algorithm works by applying the weak learner sequentially
to weighted versions of the data, where more weight is given to examples that were misclassified by earlier rounds.\\
The goal of boosting is to solve the following optimization problem:
$$ \min_f \sum_{i=1}^N L(y_i, f(x_i)) $$
and $L(y, \hat{y})$ is some loss function, and $f$ is assumed to be an ABM model as in Equation 16.3\\

Rather than deriving new versions of boosting for every different loss function, it is possible to derive a generic version, known
as gradient boosting. To explain this, imagine minimizing 
$$ \hat{f} = \arg\min_f L(f) $$
where $ f=(f(x_1), ..., f(x_N)) $ are the "parameters". We will solve this stagewise, using gradient descent. At step $m$, let $g_m$ be the gradient
of $L(f)$ evaluated at $f = f_{m-1}$:
$$ g_{im} = [\frac {\partial L (y_i, f(x_i))}{\partial f(x_i)}]_{f=f_{m-1}} $$
We then make the update
$$ f_m = f_{m-1} - \rho_m g_m $$
where $\rho_m$ is the step length, chosen by
$$ \rho_m = \arg\min_{\rho} l(f_{m-1} - \rho g_m) $$
This is called functional gradient descent.
In its current form, this is not much use, since it only optimizes $f$ at a fixed set of $N$ points, so we do not learn a function that can generalize.
However, we can modify the algorithm by fitting a weak learner to approximate the negative gradient signal, That is, we use this update
$$ \gamma_m = \arg\min_{\gamma} \sum_{i=1}^N (-g_{im} - \phi (x_i; \gamma))^2 $$

Algorithm 16.4: Gradient boosting\\
 1. Initialize $f_0 (x) = \arg\min_{\gamma} \sum_{i=1}^N L(y_i, \phi(x_i; \gamma)) $;\\
 2. for $ m = 1 : M $ do\\
     Compute the gradient residual using $ r_{im} = - [\frac{\partial L (y_i, f(x_i))}{\partial f(x_i)}]_{f(x_i) = f_{m-1}(x_i)} $;\\
    Use the weak learner to compute $\gamma_m$ which minimizes 
    $$ \sum_{i=1}^N (r_{im} - \phi(x_i; \gamma_m))^2 $$;\\
    Update $f_m (x) = f_{m-1} (x) + \nu \phi (x; \gamma_m) $;\\
Return $f(x) = f_M (x) $ \\
If we apply this algorithm using squared loss, we recover L2Boosting. If we apply this algorithm to log-loss, we get an algorithm known as BinomialBoost.
The advantage of this over LogitBoost is that it does not need to be able to do weighted fitting: it just applies any black-box regression model to the
gradient vector. Also, it is relatively easy to extend to the multi-class case. We can also apply this algorithm to other loss functions, such as
the Huber loss, which is more robust to outliers than squared error loss.

\Large Linear regression model for MSE minimization problem. Write down the formula and the derivative of the loss function w.r.t. weights
\normalsize Regression models 
$$ \hat{Y}_i = b_0 + b_1 X_i $$
Linear regression problem statement:
\begin{itemize}
    \item Dataset $\mathcal{L} = {x_i, y_i}_{i=1}^N$, where $x_i \in  \mathbb{R}, \quad y_i \in \mathbb{R}$ 
    \item The model is linear:
    $$ \hat{y} = w_0 + \sum_{k=1}^p x_k \cdot w_k = // x = [1, x_1, x_2, ...., x_p]// = x^T \mathbf{w} $$
    where$\mathbf {w} = (w_0, w_1,..., w_n), \quad w_0 is bias term$
    \item Least squares method (MSE minimization) provides a solution:
    $$ \mathbf{\hat{w}} = \arg\min_w ||Y - \hat{Y} ||_2^2 = \arg\min_w || Y - X \mathbf{w} ||_2^2 $$
\end{itemize}

Denote quadratic loss function:
$$ Q(\mathbf{w}) = (Y - X \mathbf{w})^T (Y - C \mathbf{w})= ||Y - X \mathbf{w}||_2^2 $$
where $ X = [x_1,..., x_n], \quad x_i \in \mathbb{R}^p \quad Y = [y_1, ..., y_n], \quad y_i \in \mathbb{R} $
To find optimal solution let's equal to zero the derivative if equation above:
$$ \nabla_{\mathbf{w}} Q(\mathbf{w}) = \nabla_{\mathbf{w}} [Y^T Y - Y^T X \mathbf{w}- \mathbf{w} - \mathbf{w}^T X^T Y + \mathbf{w}^T X^T X \mathbf{w}] = $$
$$ = 0 - X^T Y - X^T Y + (X^T X + X^T X) \mathbf{w} = 0 $$
$$ \mathbf{\hat{w}} = (X^T X)^{-1} X^T Y $$
Regularization \\
To make the matrix nonsingular, we can add a diagonal matrix:
$$ \mathbf{\hat{w}} = (X^T X + \lambda I)^{-1} X^T Y $$
where $I = diag [1_1, ..., 1_p]$
Actually, it's a solution for the following loss function:
$$ Q (\mathbf{w}) = ||Y - X \mathbf{w}\Vert_2^2 + \lambda^2 ||w||_2^2 $$

Gauss-Markov theorem\\
Suppose target values are expressed in following form:
$$ Y = X \mathbf{w} + \varepsilon $$, where $ \varepsilon = [\varepsilon_1,... , \varepsilon_N] $ are random variables \\
Gauss-Markov assumptions: 
\begin {itemize}
    \item $\mathbb{E} (\varepsilon_i) = 0 \quad \forall i$
    \item $Var (\varepsilon_i) = \sigma^2 < inf \quad \forall i$
    \item $Cov (\varepsilon_i, \varepsilon_j) = 0 \quad \forall i \neq j$
\end {itemize}

\begin{tcolorbox}[colback=red!5!white,colframe=red!75!black]
  $\mathbf{\hat{w}} = (X^T X)^{-1} X^T Y$
\end{tcolorbox}
delivers Best Linear Unbiased Estimator \\

Different norms \\
Once more: loss functions: 
$$ MSE = \frac{1}{n} \Vert x^T \mathbf{w} - y \Vert_2^2  $$
$$ MAE = \frac{1}{n} \Vert x^T \mathbf{w} - y \Vert_1 $$
Regularization terms:
\begin{itemize}
    \item $L_2: \Vert \mathbf{w} \Vert_2^2$
    \item  $L_2: \Vert \mathbf{w} \Vert_1 $
\end{itemize}

What's the difference? \\
MSE $(L_2)$
\begin{itemize}
    \item delivers BLUE (Best Linear Unbiased Estimator) according to Gauss-Markov theorem
    \item differentiable
    \item sensitive to noise
\end{itemize}

MAE ($L_1$)
\begin{itemize}
    \item non-differentiable (not a problem)
    \item much more prone to noise
\end{itemize}

$L_2$ regularization
\begin{itemize}
    \item constraints weights 
    \item delivers more stable solution
    \item differentiable
\end{itemize}

$L_1$ regularization
\begin{itemize}
    \item non-differentiable (not a problem)
    \item select features
\end{itemize}

Other functions to measure the quality in regression:
\begin{itemize}
    \item R2 score
    \item MAPE
    \item SMAPE
    \item ....
\end{itemize}

Supervised learning problem statement\\
Let's denote: 

\begin{enumerate}
   \item Training set $\mathcal{L} = {x_i, y_i}_{i=1}^n$, where
   \begin{enumerate}
     \item $(x \in \mathbb{R}^p, y \in \mathbb{R})$ for regression
     \item $x_i \in \mathbb{R}^p, \quad y_i \in {+1, -1}$ for binary classification
    \item Model $f(x)$ predicts some value for every object
    \item Loss function $Q (x, y, f)$ that should be minimized
     \end{enumerate}
   \end{enumerate}

Overfitting vs. underfitting
\begin{itemize}
    \item We can control overfitting / underfitting by altering model's capacity (ability to fit a wide variety of functions)
    \item select appropriate hypothesis space
\end{itemize}

Linear models are simple yet quite effective models \\
Regularization incorporates some prior assumptions / additional constraints \\
Trust your validation \\

\Large Linear classification
\normalsize Linear classification:
\begin{itemize}
    \item margin
    \item loss functions
\end{itemize}

Logistic regression
\begin{itemize}
    \item sigmoid derivation 
    \item Maximum Likelihood Estimation
    \item Logistic loss
    \item probability calibration
\end{itemize}
Multiclass aggregation strategies\\
Metrics in classification \\
Classification problem:
$$ X \in R^{n \times p} $$
$$ Y \in C^n \quad e.g. C = {-1, +1} $$
$$ |C| < + \infty $$
The most simple linear classifier
$$c(x) = \begin{cases} 1, & \mbox{if } f(x) \geq 0 \quad \mbox{Why cutoff value is fixed?} \\ -1, & \mbox{if } f(x) <0 \quad \mbox{(bias term is implied)} \end{cases}$$
or equivalently 
$$ c(x) = sign(f(x)) = sign (x^T w) $$
Geometrical interpretation: hyperplane dividing space into two subspaces \\
Let's define linear model's Margin as
$$ M_i = y_i \cdot f(x_i) = y_i \cdot x_i^T w $$
main property: negative margin reveals misclassification 
$$ M_i > 0 \Leftrightarrow y_i = c (x_i) $$
$$ M_i \leq 0 \Leftrightarrow y_i \neq c (x_i) $$

Weights choice:\\
Remembering old paradigm 
$$ Empirical \quad risk = \sum_{by \quad objects} Loss \quad on \quad objcet \leftarrow min_{model \quad params} $$
Essential loss is misclassification
$$ L_{mis} (y_i^t, y_i^p) = [y_i^t \neq y_i^p] = [M_i \leq 0]$$
Iverson bracket:
$$ [P] = \begin{cases} 1, & \mbox{if  P is true} \\ 0, & \mbox{otherwise} \end{cases}$$ 
Disadvantages \\
\begin{itemize}
    \item Not differentiable
    \item Overlooks confidence
\end{itemize}
Square loss\\
Let's treat classification problem as regression problem: $Y \in {-1, 1} \mapsto Y \in R $ \\
thus we optimize MSE
$$ L_{MSE} = (y_i - x_i^T w)^2 = \frac{(y_i^2 - y_i \cdot x_i^T w)^2}{y_i^2} = (1 - y_i \cdot x_i^T w)^2 = (1 - M_i)^2 $$
Advantage: already solved              Disadvantage: penalizes for high confidence
Other losses:
\begin{itemize}
    \item square loss $Q(M) = (1 - M)^2$
    \item hinge loss $V(M) = (1 - M)_+$
    \item savage loss $S(M) = 2(1 + e^M)^{-1} $
    \item logistic loss $ L(M) = \log_2 (21 + e^{-M+}) $
    \item exponential loss $ E(M) = e^{-M} $
\end{itemize}
Intuition: \\
I. Let's try to predict porbability of an object to have positive class
$$ p_+ = P(y = 1 |x) \in [0, 1] $$
II. But all we can predict is a real number 
$$ y = x^T w \in R $$
III. Time for some tricks
$$ \frac{p_+}{1-p_+} \in [0, + \infty) $$
$$ \log \frac{p_+}{1-p_+} \in R $$
Here is the match\\
IV. Reverse the closed form 
$$ \frac{p_+}{1-p_+} = exp (x^T w) $$
$$ p_+ = \frac{1}{1 + exp (-x^T w)} = \sigma (x^T w) $$
$$ \mathbf{Sigmoid \quad (aka \quad logistic) \quad function}$$
$$ \sigma(x) = \frac{1}{1 + exp(-x)} $$
Sigmoid is odd relative to (0, 0.5) point \\
Symmetric property
$$ 1 - \sigma(x) = \sigma (-x) $$
Derivative: $ \sigma' (x) = \sigma (x) \cdot (1 - \sigma (x)) $
Maximum likelihood estimation\\
Just to remind
$$ \log L (w | X, Y) = \log P (X, Y | w) = \log \prod_{i=1}^n P(x_i, y_i | w) $$
Calculating probabilities for objects
$$ if \quad y_i = 1: P(x_i, 1|w) = \sigma_w (x_i) = \sigma_w (M_i) $$
$$ if \quad y_i = -1: P(x_i, -1 | w) = 1 - \sigma_w (x_i) = \sigma_w (-x_i) = \sigma_w (M_i) $$
$$ \log L (w |X, Y) = \sum_{i=1}^n \log \sigma_w (M_i) = - \sum_{i=1}^n \log (1 + exp (-M_i)) \Rightarrow \min_w $$
Logistic loss $L_{Logistic} = \log (1 + exp (-M_i))$ \\
Accuracy = $ \frac {1}{n} \sum_{i=1}^n [y_i^t = y_i^p] $ \\
Balanced accuracy = $ \frac{1}{C} \sum_{k=1}{C} \frac{\sum_i [y_i^t = k \quad and y_i^t = y_i^p]}{\sum_i [y_i^t = k]} $\\
Precision = $ \frac{TP}{TP + FP} $       Recall = $ \frac{TP}{TP + FN} $ \\
F-score - Harmonic mean of precision and recall. Closer to smaller one
$$ F_1 = \frac{2}{precion^{-1} + recall^{-1}} = 2 \frac{precision \cdot recall}{precision + recall} $$
Generalization to different ratio between Precision and Recall
$$ F_{\beta} = (1 + \beta^2 \frac{precision \cdot recall}{\beta^2 precision + recall} $$
Receiver Operating Characteristic (ROC)
$$ FPR = \frac{FP}{FP + TN} $$
$$ TPR = \frac{TP}{TP + FN}(=Recall) $$
Classifier needs to predict probabilities. Objects get sorted by positive probability. Baseline is random predictions. Always above duiiagonal (for reasonable classifier). If below - change sign of predictions. Strictly higher curve means better classifier. Number of steps (thresholds) not bigger than dataset.\\
ROC Area Under Curve (ROC-AUC) - effectively lays in (0.5, 1). Bigger ROC-AUC doesn't imply higher curve everywhere.\\
Precision-Recall Curve - AUC is in (0, 1).\\

Multiclass metrics
\begin{center}
\begin{tabular}{||c c c c||} 
 \hline
 average & Precision & Recall & $F_{\beta}$ \\ [0.5ex] 
 \hline\hline
 "micro" & $ P(y, \hat{y}) $ & $ R(y, \hat{y}) $ & $ F_{\beta} (y, \hat{y}) $ \\ 
 \hline
 "samples" & $ \frac{1}{|S|} \sum_{s \in S} P (y_s. \hat{y}_S) $ & $ \frac{1}{|S|} \sum_{s \in S} R (y_s, \hat{y}_s) $ & $ \frac{1}{|S|} \sum_{s \in S} F_{\beta} (y_s, \hat{y}_s)$ \\
 \hline
 "macro" & $ \frac{1}{|L|} \sum_ {l \in L} P (y_l, \hat{y}_l) $ & $ \frac{1}{|L|} \sum_{l \in L} R(y_l, \hat{y}_l) $ & $ \frac{1}{|L|} \sum_{l \in L} F_{\beta} (y_l, \hat{y}_l) $ \\
 \hline
 "weighted & $ \frac{1}{\sum_{l \in L} |\hat{y}_l|} \sum_{l \in L} |\hat{y}_l| P (y_l, \hat{y}_l) $ & $ \frac{1}{\sum_{l \in L} |\hat{y}_l|} \sum_{l \in L} |\hat{y}_l| R (y_l, \hat{y}_l) $ & $ \frac{1}{\sum_{l \in L} |\hat{y}_l|} \sum_{l \in L} |\hat{y}_l| F_{\beta} (y_l, \hat{y}_l) $ \\
  \\ [1ex] 
 \hline
\end{tabular}
\end{center}

\Large Margin
\normalsize $$ y \in {1, -1} $$
$$ y_i = 1: w^T x_i - c > 0 $$
$$ y_i = -1: w^T x_i - c < 0 $$
$$ c_+ (w) = \underset{y_i = 1}{min} (w^T x_i) $$
$$ c_- (w) = \underset{y_i = -1}{max} (w^T x_i) $$
$$ \rho (w) = \frac{c_+ (w) - c_- (w)}{2} $$
$$ \rho (\frac{w_0}{||w_0||}) = \frac{1}{||w_0||} $$
Optimization problem
$$ y_i = 1: w^T x_i - c > 0 \quad \rho (w) = \frac{1}{||w||} \rightarrow \underset{w, c}{max}$$
$$ y_i = -1: w^T x_i - c < 0  $$
$$ M_i = y_i \cdot (w^T x_i - c) \quad s.t. \quad y_i (w^T x_i - c) \geq 1 $$
$$ L(w, c, \alpha) = \frac{1}{2} w^T w - \sum_i \alpha_i (y_i (w^T x_i - c) - 1) $$
Hinge loss
$$ L (w, c, \alpha) = \frac{1}{2} w^T w - \sum_i \alpha_i (y_i( w^T x_i - c)  - 1) $$
$$ L^{hinge} = (1 - M)_+ $$
$$ L(w, c, \alpha) = \frac{1}{2} ||w||_2^2 + \sum_i \alpha_i L_i^{hinge} $$

Kernel trick
$$ y_i =1: w^T x_i - c > 0 $$
$$ y_i = -1: w^T x_i - c < 0 $$
$$ x \mapsto \phi (x) , w \mapsto \phi (w) \Rightarrow \langle w, x \rangle \mapsto \langle \phi(w), \phi (x) \rangle $$
$$ K (w, x) = \langle \phi (w), \phi (x) \rangle $$
Kernel types \\
Linear: $K (w, x_ = \langle w, x \rangle$ \\
Polynomial: $ K (w, x) = ( \gamma \langle w, x \rangle + r)^d $ \\
Gaussian radial basis function: $ K (w, x) = e^{- \gamma ||w - x||^2} $ \\
\Large Principal Component Analysis
\normalsize $$x_1,...., x_n \rightarrow g_1, ..., g_k, k \leq n$$
$$ U: U U^T = I, G = X U $$
$$ \hat{X} = G U^T \approx X $$
$$ ||F U^T - X || \rightarrow \underset{G, U}{min} \quad s.t. \quad rank (G) \leq k $$

Singular value decomposition (Eckart-Young-Mirsky theorem)
$$ ||GU^T - X|| \rightarrow \underset{G, U}{min} s.t. rank (G) \leq k $$
$$ X = V \Sigma U^T : ||GU^T - V \Sigma U^T ||_2 = ||G - V \Sigma ||_2 $$
$$ G = V \Sigma' : || V \Sigma' - V \Sigma ||_2 = || |Sigma' - \Sigma ||_2 $$
$$ ||A||_2 = \sigma_{max} (A): || \Sigma' - \Sigma ||_2 = \sigma_k {\Sigma} = \sigma_k (X) $$
Another approach - residual variance maximization. Take new basis vectors greedy. Same result for G and U. Always normalize data before PCA.\\
Get rid of low-variance components
$$ E_m = \frac{||GU^T - F||^2}{||F||^2} = \frac{\lambda_{m+1} + ... + \lambda_n}{\lambda_1 + ...+ \lambda_n} \leq \varepsilon $$
Validation strategies - special case: Leave One Our (LOO) - good for tiny datasets

\Large Decision tree
\normalsize How to split data properly?
$$ \frac{|L|}{|Q|} H(L) + \frac{|R|}{|Q|} H(R) \rightarrow \underset{j, t}{min} $$
$H (R)$ is measure of "heterogeneity" of our data. Consider binary classsification problem: \\
Obvious way: Misclassification criteria: $ H(R) = 1 - \max {p_0, p_1} $
1. Entropy criteria: $ H(R) = - p_0 \log p_0 - p_1 \log p_1 $
2. Gini impurity: $ H(R) = 1 - p_0^2 - p_1^2 = 1 - 2 p_0 p_1 $
$H(R)$ is measure of "heterogeneity" of our data. Consider multiclass classification problem:
Obvious way: Misclassification criteria: $ H(R) = 1 - \underset{k}{\max {p_k}}$
1. Entropy criteria: $ H(R) = - \sum_{k=0}^K p_k \log p_k $
2. Gini impurity: $ H(R) = 1 - \sum_k (p_k)^2 $
Information criteria: \\
Entropy: $ S = - M \sum_{k=0}^K p_k \log p_k $
$$ \begin{cases} 
p_k $\in$ [0; 1], & \mbox {$\forall$ k} \\ 
$\sum_k$ p_k = 1 & $\leftarrow$ 
\end{cases}$$
$$ L(p, \lambda) = - \sum_k p_k \log p_k + \lambda (\sum_k p_k -1) $$
$$ \frac{\partial L}{\partial \lambda}; \frac{\partial L}{\partial p_k} = 0 \quad \forall k $$
$$ \frac{\partial L}{\partial p_k} = - \log p_k - p_k \frac{1}{p_k} + \lambda = 0 $$
$$ \forall h \quad \lambda = \log p_k + 1  $$
$$ p_k = e^{\lambda - 1} $$
$$ \sum_k p_k = 1; \sum_{k=0}^K e^{\lambda - 1} = 1 $$
$$ e^{\lambda - 1} = \frac{1}{k + 1} $$
$$ \forall k \quad p_k = \frac{1}{k + 1} $$
Information criteria: Entropy in binary case $N = 2$
$$ S = - p_+ \log_2 p_+ - p_- \log_2 p_- = - p_+ \log_2 p_+ - (1 - p_+) \log_2 (1 - p_+) $$
$$ - \frac{1}{N} \sum_{i=1}^N \sum_{k=0}^K [y)i = k] \log c_k \rightarrow \underset{c_k}{\max} $$
$$ c_k \in [0; 1] \forall k \quad \sum_k c_k = 1 $$
$$ L(c, \lambda) = - \frac{1}{N} \sum_i \sum_k [y_i = k] C_k + \lambda (\sum_k c_k - 1) $$
$$ \frac{\partial L}{\partial c_k} = - \frac{1}{N} \sum_i [y_i = k] \cdot \frac{1}{c_k} + \lambda = 0 $$
$$ \frac{\partial L}{\partial c_k} = - \frac{p_k}{c_k} + \lambda c_k = \frac{p_k}{\lamda} $$
$$ \sum_k = 1 \Rightarrow \sum_k p_k = 1 \Rightarrow 1 = \frac{1}{\lambda} \Rightarrow \lambda = 1 $$
$$ \boxed{c_k = p_k} $$

Information criteria: Gini impurity
$$ G = 1 - \sum_k (p_k)^2 $$
$$ \sum_k p_k \cdot \sum_{j \neq k} p_j = \sum_k p_k \cdot ( 1 - p_k) = \sum_k (p_k - (p_k)^2) = 1 - \sum_k p_k $$
Gini impurity in benary case $N = 2$:
$$ G = 1 - p_+^2 - p_-^2 = 1 - p_+^2 - (1 - p_+)^2 = 2p_+ (1 - p_+) $$
Information criteria: $H(R)$ is measure of "heterogeneity" of our data. Consider regression problem:\\
1. Mean squared error $ H(R) = \underset{c}{\min} \frac{1}{|R|} \sum_{(x_i, y_i) \in R} (y_i - c)^2 $ \\
What is constant - $ c^* = \frac{1}{|R|} \sum_{y_i \in R} y_i $
\Large Pruning
\normalsize Pre-pruning: constrain the tree before construction \\
Post-pruning: simplify constructed tree\\
Missing values in Decision trees - if the value is missing, one might use both sub-trees average their predictions
$$ \hat{y} = \frac{|L|}{|Q|} \hat{y}_L + \frac{|R|}{|Q|} \hat{y}_R $$
Let J be the subspace of the original feature space, corresponding to the leaf of the tree. Prediction takes form
$$ \hat{y} = \sum_j w_j [x \in J_j] $$
Construction algorithms: overview \\
\begin{itemize}
    \item ID-3 - entropy criteria; stops when no more gain available 
    \item C4.5 - normalised entropy criteria; stops depending on leaf size; incorporates pruning
    \item C5.0 - some updates on C4.5
    \item CART - Gini criteria; cost-complexity pruning; surrogate predicates for missing data
\end{itemize}

\Large Bootstrap
\normalsize Consider dataset X containbing m objects. Pick m objects with return from X and repeat in N times to get N datasets. Error of model trained on $X_j$:
$$ \varepsilon_j (x) = b_j (x) - y(x), \quad j = 1, ...., N $$
Then $ \mathbb{E} (b_j(x) - y(x))^2 = \mathbb{E}_x \varepsilon_j^2 (x) $ \\
The mean error of N models: $ E_1 = \frac{1}{N} \sum_{j=1}^N \mathbb{E}_x \varepsilon_j^2 (x) $ \\
Consider the errors unbiased and uncorrelated:
$$ \mathbb{E}_x \varepsilon_j (x) = 0 $$
$$ \mathbb{E}_x \varepsilon_i (x) \varepsilon_j (x) = 0, \quad i \neq j $$
$$ E_N = \mathbb{E}_x (\frac{1}{N} \sum_{j=1}^n b_j (x) - y(x))^2 = $$
$$ = \mathbb{E}_x (\frac{1}{N} \sum_{j=1}^N \varepsilon_j (x))^2 $$
$$ \frac{1}{N^2} \mathbb{E}_x (\sum_{j=1}^N \varepsilon_j^2 (x) + \underbrace{\ssum_{i \neq j} \varepsilon_i (x) \varepsilon_j (x)})= \frac{1}{N} E_1 $$
The final model averages all predictions:
$$ a (x) = \frac{1}{N} \sum_{j=1}^N b_j (x) $$
\Large Random forest
\normalsize
\begin{itemize}
    \item One of the greatest "universal" models
    \item  There are some modifications: extremely randomized isolation forest, etc
    \item Allows to use train data for validation: OOB
\end{itemize}
$$ OOB = \sum_{i=1}^{\ell} L (y_i, \frac{1}{\sum_{n=1}^N [x_i \notin X_n]} \sum_{n=1}^N [x_i \notin X_n] b_n (x_i)) $$

\Large Neural networks 
\normalsize Logistic regression $ X \rightarrow Wx + b \rightarrow P(y) $
$$ P(y|x) = \sigma (w \cdot x + b) $$
$$ L = - \sum_i y_i \log P (y|x_i) + (1 - y_i) \log (1 - P(y|x_i)) $$
Activation functions: nonlinearities\\
Sigmoid: $ f(a) = \frac{1}{1 + e^a} $ \\
Tanh: $ f(a) = tanh (a) $ \\
ReLU: $ f(a) = max (0, a) $ \\
Softplus: $ f(a) = \log (1 + e^a) $
Some generally accepted terms \\
Layer - a building block for NNs:
\begin{itemize}
    \item Dense/Linear/FC layer: $ f(x) = Wx + b $
    \item Nonlinearity layer: $ f(x) = \sigma (x) $
    \item Input layer, output layer 
    \item A few more we will cover later
\end{itemize}
Activation function - function applied to layer output
\begin{itemize}
    \item Sigmoid
    \item tanh
    \item ReLU
    \item Any other function to get nonlinear intermediate signal in NN
\end{itemize}
Backpropogation - a fancy word for "chain rule". Backpropogation and chain rule:\\
Chain rule is just a simple math: $ \frac{\partial L}{\patrial x} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial x}$\\
Backpropogation example: $f (x, y, z) = (x + y)z$ e.g. $x = -2, y = 5, z = -4$
$$ \boxed{q = x + y \quad \frac{\partial q}{\partial x} = 1, \frac{\partial q}{\partial y} = 1}$$
$$ \boxed{f = qz \quad \frac{\partial f}{\partial q} = z, \frac{\partial f}{\partial z} = q} $$
Want: $ \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z}$ \\
Chain rule:
$$ \boxed{\frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \frac{\partial q}{\partial x}} $$
Backpropogation example:
$$ L(w, x) = \frac{1}{1 + exp (- (x_0 w_0 + x_1 w_1 + w_2))} $$
Backpropogation: matrix form
$$\begin{matrix}
y_1 = f_1 (x) = x_1\\
y_2 = f_2 (x) = x_2\\
\vdots\\
y_n = f_n (x) = x_n
\end{matrix}$$

$$ \frac{\partial y}{\partial x} = \begin{bmatrix}
\nabla f_1 (x)\\
\nabla f_2 (x)\\
\cdots\\
\nabla f_m (x)
\end{bmatrix} = 
\begin{bmatrix}
\frac{\partial}{\partial x} f_1 (x)\\
\frac{\partial}{\partial x} f_2 (x)\\
\cdots\\
\frac{\partial}{\partial x} f_m (x)
\end{bmatrix} = 
\begin{bmatrix}
\frac{\partial}{\partial x_1} f_1 (x) & \frac{\partial}{\partial x_2} f_1 (x) & \cdots & \frac{\partial}{\partial x_n} f_1 (x)\\
\frac{\partial}{\partial x_1} f_2 (x) & \frac{\partial}{\partial x_2} f_2 (x) & \cdots & \frac{\partial}{\partial x_n} f_2 (x)\\
\cdots\\
\frac{\partial}{\partial x_1} f_m (x) & \frac{\partial}{\partial x_2} f_m (x) & \cdots & \frac{\partial}{\partial x_n} f_m (x)
\end{bmatrix} $$
$$ \frac{\partial y}{\partial x} = \begin{bmatrix}
\frac{\partial}{\partial x} f_1 (x)\\
\frac{\partial}{\partial x} f_2 (x)\\
\cdots\\
\frac{\partial}{\partial x} f_m (x)
\end{bmatrix} =
\begin{bmatrix}
\frac{\partial}{\partial x_1} f_1 (x) & \frac{\partial}{\partial x_2} f_1 (x) & \cdots & \frac{\partial}{\partial x_n} f_1 (x)\\
\frac{\partial}{\partial x_1} f_2 (x) & \frac{\partial}{\partial x_2} f_2 (x) & \cdots & \frac{\partial}{\partial x_n} f_2 (x)\\
\cdots\\
\frac{\partial}{\partial x_1} f_m (x) & \frac{\partial}{\partial x_2} f_m (x) & \cdots & \frac{\partial}{\partial x_n} f_m (x)
\end{bmatrix} $$
$$ = \begin{bmatrix} 
\frac{\partial}{\partial x_1} x_1 & \frac{\partial}{partial x_2} x_1 & \cdots & \frac{\partial}{\partial x_n} x_1 \\
\frac{\partial}{\partial x_1} x_2 & \frac{\partial}{partial x_2} x_2 & \cdots & \frac{\partial}{\partial x_n} x_2 \\
\cdots \\
\frac{\partial}{\partial x_n} x_2 & \frac{\partial}{partial x_n} x_2 & \cdots & \frac{\partial}{\partial x_n} x_n 
\end{bmatrix}$$
(and since $\frac{\partial}{\partial x_j} x_i = 0$ fro $j \neq i$
$$ \begin{bmatrix}
\frac{\partial}{\partial x_1} x_1 & 0 & \cdots & 0 \\
0 & \frac{\partial}{\partial x_2} x_2 & \cdots  & 0 \\
\ddots \\
0 & 0 & \cdots & 1 
\end{bmatrix}$$ 
$$ = I$$ ($I$ is the identity matrix with ones down the diagonal) \\
Stochastic gradient descent (and variations) is good to optimize NN parameters 
$$ x_{t+1}= x_t - learning \quad rate \cdot dx $$
Activation functions: Sigmoid $ f(a) = \frac{1}{1 + e^{-a}}$\\
\begin{itemize}
    \item Maps R to (0, 1)
    \item Historically popular, one of the first approximations of neuron activations
\end{itemize}
Problems:\\
\begin{itemize}
    \item Almost zero gradients on the both sides (saturation)
    \item Shifted (not zero-centered) output
    \item Expensive computation of the exponent
\end{itemize}
Activation functions: tanh $ f(a) = tanh (a) $ \\
\begin{itemize}
    \item Maps R to (-1, 1)
    \item Similar to the Sigmoid in other ways
\end{itemize}
Problems: \\
\begin{itemize}
    \item Almost zero gradients on the both sides (saturation)
    \item Expensive computation of the exponent
\end{itemize}
Activation functions: ReLU $ f(a) = max (0, a) $\\
\begin{itemize}
    \item Very simple to compute (both for forward and backward) - up to 6 times faster than Sigmoid
    \item Does not saturate when $x>0$ - so the gradients are not 0
\end{itemize}
Problems: \\
\begin{itemize}
    \item Zero gradients when $x>0$
    \item Shifted (not zero-centered) output
\end{itemize}
Activation functions: Leaky ReLU $ f(a) = max (0.01a, a) $ \\
\begin{enumerate}
    \item Very simple to compute (both forward and backward) - up to 6 times faster than sigmoid
    \item Does not saturate
\end{enumerate}
Problem - shifted, but not so much output \\
Activation fucntions: ELU 
$$ \begin{cases}
    a, & \mboxed {a > 0} \\
    \alpha (exp(a) - 2), & \mboxed {a \leq 0}
\end{cases} $$
\begin{itemize}
    \item Similar to ReLU
    \item Does not saturate
    \item Close to zero mean outputs
\end{itemize}
Problem - requires exponent computation \\
Activation functions: sum up \\
\begin{itemize}
    \item Use ReLU as baseline approach
    \item Be careful with the learning rates
    \item Try out Leaky ReLU or ELU
    \item Try out tanh, but do not expect nuch from it
    \item Do not use Sigmoid                  
\end{itemize}
Outro \\                                  
\begin{itemize}
    \item Neural networks are great, especially for data with specific structure 
    \item All operations should be differentiable to use backpropogation mechanics (and still it is just basic differentiation) 
    \item Many techniques in Deep Learning are inspired by nature (or general sense)
\end{itemize}

\Large Convolutional Neural Networks
\normalsize Outline:
\begin{enumerate}
    \item Convolutional layer structure
    \item Pooling layers
    \item Top architectures overview
\end{enumerate}
1 number://
the result of taking a dot product between the filter and small 5x5x3 chunk of the image (i.e. 5*5*3 = 75-dimensional dot product + bias):
$$ w^T x + b $$
We call the layer convolutional because it is related to convolution of two signals:
$$ f[x, y] * g[x, y] = \sum_{n_1 = - \infty}^{\infty} \sum_{n_2 = - \infty}^{\infty} f[n_1, n_2] \cdot g [x - n_1, y - n_2] $$
where $ f [n_1, n_2] $ - elementwise multiplication and sum of a filter and the signal (image)
Output size:
$$ (N - F) / stride + 1 $$
$$ e.g. N = 7, F = 3:$$
$$ stride \quad 1 \Rightarrow (7 - 3)/1 + 1 = 5 $$
$$ stride \quad 2 \Rightarrow (7 - 3)/2 + 1 = 3 $$
$$ stride \quad 3 \Rightarrow (7 - 3)/3 + 1 = 2.33 $$
In practice: Common to zero pad the border
e.g. input 7x7 \\
3x3 filter, applied with stride 1 pad with 1 pixel border $\Rightarrow$ what is the output? //
7x7 output! //
In general, common to see CONV layers with stride 1, filters of size FxF, and zero-padding with $(F-1)/2$ (will preserve size spatially) //
e.g. $F = 3$ $\Rightarrow$ zero pad with 1 \\
$F = 5$ $\Rightarrow$ zero pad with 2 //
$F = 7$ $\Rightarrow$ zero pad with 3 //
Examples time: //
Input volume: 32x32x3 - 10 5x5 filters with stride 1, pad 2 //
Output volume size: (32 + 2*2 - 5)/1 + 1 = 32 spatially, so 32x32x10                                                     
Pooling layer //
\begin{itemize}
    \item Makes the representations smaller and more manageable 
    \item Operates over each activation map independently
\end{itemize}
Summary: //
\begin{itemize}
    \item ConvNets stack convolutional, pooling and dense layers
    \item Trend towards smaller filters and deeper architectures
    \item 1x1 convolutions are meaningful
    \item Humanity is already beaten on ImageNet
\end{itemize}

\Large Recurrent Neural Networks and Language Models
\normalsize Outline: \\
\begin{enumerate}
    \item RNN intuitions
    \item Language models
    \item Memory concept: LSTM
    \item RNN as encoder for sequential data
    \item Vanishing gradient
\end{enumerate}
Recurrent neural network: with formulas
$$ h_0 = \bar{0} $$
$$ h_1 = \sigma (\langle W_{hid} [h_0, x_0]\rangle + b) $$
$$ h_2 = \sigma (\langle W_{hid} [h_1, x_1]\rangle + b)_ = \sigma (\langle W_{hid} [\sigma ( \langle W_{hid} [h_0, x_0]\rangle + b, x_1]\rangle + b) $$
$$ h_{i+1} = \sigma (\langle W_{hid} [h_i, x_i]\rangle + b) $$
$$ P(x_{i+1}) = softmax (\langle W_{out}, h_i \rangle + b_{out}) $$
LSTM: quick overview 
$$ i_t = \sigma (W_i \cdot [h_{t-1}, x_t] + b_i) $$
$$ \tilde {C}_t = \tanh (W_C \cdot [h_{t-1}, x_t] + b_C) $$
$$ C_t = f_t * C_{t-1} + i_t * \tilde{C}_t $$
$$ o_t = \sigma (W_o [h_{t-1}, x_t] + b_0) $$
$$ h_t = o_t * \tanh{(C_t)} $$
$$ C_t = f_t * C_{t-1} + i_t * \tilde {C}_t $$
LSTM with formulas:\\
$\underline {Sigmoid \quad function}$: all gate values are between 0 and 1 \\

\Large Basics of Neural Network Programming - Binary Classification
\normalsize Notation
$$ (x, y) \quad x \in \mathbb{R}^{n_x} \quad y \in {0, 1} $$
m training examples: ${(x^{(1)}, y^{(1)}, (x{(2)}, y^{(2)},\cdots, (x^{(m)}, y^{(m)}}$
$$ M = M_{train}, M_{test} = Number test \quad examples $$
$$\begin{bmatrix}
\vdots & \vdots & \vdots \\
x^{(1)} & x^{(2)} & x^{(m)}\\
\vdots & \vdots & \vdots
\end{bmatrix}
$$
$$ Y = [y^{(1)}, y^{(2)}, \cdots, y^{(m)}] $$
$$ Y \in \mathbb {R}^{1 \times m} \quad Y \cdot shape = (1, m) $$
$$ X \in \mathbb {R}^{n \times m} \quad X \cdot shape = (n_x, m) $$
\Large Logistic Regression \\
\normalsize Given $x$, want $ \hat{y} = P(y=1 | x), 0 \leq \hat{y} \leq 1, \quad x \in \mathbb{R}^{n_x} $//
Parameters: $ \boxed{w} \in \mathbb{R}^{n_x} $, $\boxed{b} \in \mathbb{R} $
Output $\hat{y} = \sigma (w^T x + b) $
$$ x_0 = 1, \quad x \in \mathbb {R}^{n_x + 1} $$
$$ \hat{y} = \sigma (\theta^T x) $$
$$\begin{bmatrix}
\theta_0 \\
\theta_1 \\
\theta_2 \\
\vdots \\
\theta_{n_x}
\end{bmatrix}
$$
$$ \sigma (z) = \frac{1}{1 + e^{-z}} $$
If $z$ is large then $ \sigma (z) \approx \frac{1}{1 + 0} = 1$ \\
If $z$ large negative number $ \sigma (z) = \frac{1}{1 + e^{-z}} \approx \frac{1}{bignum} \approx 0$ \\
Logistic Regression cost function
$$ \hat{y}^{(i)} = \sigma (w^T x^{(i)} + b), where \quad \sigma(z^{(i)}) = \frac{1}{1 + e^{-z(i)}} $$
$$ z^{(i)} = w^T x_{(i)} + b $$
Given $ {(x^{(1)}, y^{(1)}), \cdots, (x^{(m)}, y^{(m)})}$, want $ \hat{y}^{(i)} \approx y^{(i)} $\\
Loss (error) function: $ \mathcal{L} (\hat{y}, y) = \frac{1}{2} (\hat{y} - y)^2 $
$$ \mathcal{L} (\hat{y}, y) = - (y \log \hat{y} + (1 - y) \log (1 - \hat{y})) $$
If $y = 1$: $\mathcal{y} (\hat{y}, y) = - \log \hat{y} \leftarrow$ want $ \log \hat{y}$ large, want $\hat{y}$ large\\
If $y = 0$: $\mathcal{L} (\hat{y}, y) = - \log (1 - \hat{y}) \leftarrow$ want $ \log (1 -  \hat{y}) $ large, want $\hat{y}$ small\\
Cost function: $ J (w, b) = \frac{1}{m} \sum_{i=1}^m \mathcal{L} (\hat{y}^{(i)}, y^{(i)} ) = - \frac{1}{m} \sum_{i=1}^m [y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)})]$ \\
\Large Gradient Descent
\normalsize Recap: $\hat{y} = \sigma (w^T x + b), \sigma (z) = \frac{1}{1 + e^{-z}}$
$$ J(w, b) = \frac{1}{m} \sum_{i=1}^m \mathcal{L} (\hat{y}^{(i)}, y^{(i)}) = - \frac{1}{m} \sum_{i=1}^m y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)}) $$
Want to find $w,b$ that minimize $J(w, b)$
$$w := w - \alpha \frac{d J (w)}{d w} \Rightarrow w := w - \alpha dw $$
$$ \frac{d J (w)}{d w} = ? $$
$$ w := w - \alpha \frac{d J (w, b)}{dw} \quad \boxed{\frac{\partial J (w, b)}{\partial w}}$$
$$ b := b - \alpha \frac{d J (w, b}{d b} \quad \boxed{\frac{\partial J (w, b)}{\partial b}}$$

\Large Backpropagation\\
\normalsize For effective calculation of loss function gradient we apply backpropagation method
$$ if \quad f (x) = g_m (g_{m-1} (...(g_1(x)...)) \quad when $$ 
$$\frac{\partial f}{\partial x} = \frac{\partial g_m}{\partial g_{m-1}} \frac{\partial g_{m-1}}{\partial g_{m-2}} \cdots \frac{\partial g_2}{\partial g_1} \frac{\partial g_1}{\partial x} $$
Backpropagation in 1 dimension. Assume $w_0$ - variable when function
$$ f(w_0) = g_m (g_{m-1} (...g_1 (w_0)...)) $$
where $g_i$ scalars. When
$$ f'(w_0)= g'_m (g_{m-1} (...g_1 (w_0)...)) \cdot g'_{m-1} (g_{m-2} (...g_1(w_0)...)) \cdot ... \cdot g'_1 (w_0)  $$
Gradient:
$$ [D_{x_0} (u \circ v)](h) = [D_{v(x_0)} u]([D_{x_0} v] (h)) $$
Backpropagattion algorithm: Assume we have values of weights $ W_0^i $ and we want step SGD on mini-batch $X$
\begin{enumerate}
    \item Forward pass, calculate and memorize all hidden $ X= X^0, X^1, ..., X^m = \hat{y} $
    \item Calculate all gradients with backward pass
    \item With claculated gradients get step SGD
\end{enumerate}

Initialization methods:\\
0. Initialization to 0 - only collapse. For example, we have neural network with 2 layers
$$ X^1 $$
$$ X^2 = X^1 W $$
where $W = 0$ - matrix of weights with zero initialization
$$ X^3 = h (X^2) $$
where $h$ - elementwise nonlinearity
$$ X^4 = X^3 U $$
where $U = 0$ matrix with zero initialization \\
What is going in forward pass
$$ X^2 = X^1 \cdot W = 0 $$
We getting matrix with zero columns with continued
$$ X^3 = h(0) $$
$$ X^4 = X^3 U = 0 $$
Where we see in back-propagation. Assume we calculated gradient $ \nabla_{X^4} \mathcal{L} $
$$ \nabla_U \mathcal{L} = (X^3)^T \nabla_{X^4} \mathcal{L} = 0 $$
i.e. matrix $U$ not renewed. Then
$$ \nabla_{X^3} \mathcal{L} = \nabla_{X^4} \mathcal{L} U^T = 0 $$
$$ \nabla_{X^2} \mathcal{L} = \nabla_{X^3} \mathcal{L} \odot h' (X^2) = 0 $$
$$ \nabla_W \mathcal{L} = (X^1)^T \nabla_{X^2} \mathcal{L} = 0 $$
i.e., weights of $W$ not renewed, therefore learning is not doing.\\
1. Random numbers initiliazation
$$ Var (y) = Var (w^T x) = \sum_{i=1}^n [\mathbb{E} [x_i]^2 Var (w_i) + \mathbb{E} [w_i]^2 Var (x_i) + Var (w_i) Var(x_i)]$$
$$ Var (w^T x) = n_{in} Var (w) Var (x) $$
where $ Var (x) $ - is variance everything component of $x$ and $Var (w) = \sigma^2 $ - variance component of $w$  
$$ \forall i, Var (w_i) = \frac{1}{n_in} $$
is calibrated random numbers initialization \\

2. Xavier and Normalized Xavier initialization
$$ \forall i, Var (w_i) = \frac{2}{n_{in} + n_{out}} $$
$$ \forall i, Var(w_i) \sim U [- \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}] $$


\end{document}


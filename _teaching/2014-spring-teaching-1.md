---
title: "Teaching experience 2"
collection: teaching
type: "Workshop"
permalink: /teaching/2015-spring-teaching-1
venue: "University 1, Department"
location: "City, Country"
---



\newcommand{\matr}{\boldsymbol}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\mathbb{Var}}
\newcommand{\bvec}{\matr{\beta}}
\newcommand{\xb}{\matr{x}^\prime\bvec}




# 1. Introduction - Roadmap for this semester

---------

What are Generalized Linear Models? Generalizations of the linear model,Duh!

Linear Model $y = \mathbf x^\prime \boldsymbol\beta + \epsilon, \epsilon \sim \mathbb{N}(0,1)$

What are we doing here?

*  Explain $y$ as a function of $\mathbf x$.
*  Assume a distribution for $y$, conditional on $\mathbf x$ (Gaussian)
*  The role of $\mathbf x$ is to model the mean of the resulting Gaussian, i.e. $y \sim \mathbb{N}(\mu(\mathbf x), \sigma^2)$, where e.g. $\mu(\mathbf x) = \mathbf x^\prime \boldsymbol \beta$ 

What do we want to do in this course in the upcoming few months?

*Relax assumption that*

1. $y$ is conditionally Gaussian. (Ch. 2, 3)
2. $x$ only affects the mean of $y$ (explicitly in Ch. 4, but remember in case of e.g. Poisson distribution $\mathbb{E}=\mathbb{Var}=\lambda$, which is the parameter we model).
3. Influence is linear in $x$ (Ch. 5, leave the realm of GLMs and open up tp GAMs and STAR).

Mixed models (Ch.6) tackle the correlation structure (think of repeated measurements) and Duration time analysis (Ch.7) comes with a very different way of looking at a model. This will become clear in the end.

---------

# 2. GLM for non-normal responses 


## Binary models

<img style="float: right;" src="../images/01.jpg">


**Binary regression is classification!**


### Linear probability model

$$
\begin{aligned}
y_i =0 &\rightarrow \epsilon_i = y_i - x_i^\prime \beta = - x_i^\prime \beta\\
y_i =1 &\rightarrow \epsilon_i = 1- x_i^\prime\beta\\
\end{aligned}
$$
$\rightarrow \epsilon_i$ is discrete

$$
x_i^\prime \beta \in \mathbb{R} \rightarrow \hat{y}_i = x_i^\prime \hat{\beta} \in \mathbb{R}
$$

In a GLM: 

$$
\begin{aligned}
E(y_i) &= 1 \cdot P(y_i=1) + 0\cdot P(y_i=0) \\
       &= P(y_i = 1) = \pi_i
\end{aligned}
$$
$\rightarrow$ Model is continuous.

$$
\begin{aligned}
h: \mathbb{R} &\rightarrow [0,1]\\
   x_i^\prime \beta &\rightarrow \pi_i = h(x_i^\prime \beta)
\end{aligned}
$$

---------


### Link/response functions
 
Logistic link:
$$
\begin{aligned}
h(\eta) &= \frac{\exp(\log(\frac{\pi}{1-\pi}))}{1+ \exp(\log(\frac{\pi}{1-\pi]}))} \\
&= \frac{\frac{\pi}{1-\pi}}{\frac{1-\pi +\pi}{1-\pi}} = \frac{\pi}{1} = \pi
\end{aligned}
$$

```{r ,echo=TRUE, fig.align="center"}
eta <- seq(-4, 4, length=100)
qplot(eta, plogis(eta), geom="line",  main="Logistic link")
```

Important concept and easy to be confused about: **odds** : $\frac{\pi}{1-\pi}= \frac{\text{success prob.}}{\text{failure prob.}}$

If odds

* $=1 \rightarrow \text{success and failure equally likely}$
* $>1 \rightarrow \text{success more likely than failure}$
* $<1 \rightarrow \text{success less likely than failure}$

**Interpretation relative:** $x_{j+1} / x_j$ (to enable comparison)


$$
\frac{\frac{\pi(x_{j}+1)}{1-\pi(x_{j}+1)}}{\frac{\pi(x_{j})}{1-\pi(x_{j})}} = \exp(\beta_j)
$$

---------


### Response functions

All response functions are cumulative distribution functions (cdfs).

```{r ,echo=TRUE, fig.align="center"}
eta <- seq(-4, 4, length=100)

pcll <- function(x) 1-exp(-exp(x))

ggdat <-tibble("eta"=eta, "logit"=plogis(eta), "probit"=pnorm(eta), "cll"=pcll(eta) )
ggdat %>% pivot_longer(c("logit", "probit", "cll"), names_to="link") %>% 
  ggplot + geom_line(aes(x=eta, y=value, colour=link)) + ggtitle("Response functions")

```

Because $h$ are cdfs, naturally there is an associatted distribution (with expectation and variance).

* **logistic**: $\mathrm{E}(x) = 0, \mathrm{Var}(x) = \frac{\pi^2}{3}$
* **probit**: $\mathrm{E}(x) = 0, \mathrm{Var}(x) = 1$
* **cloglog**: $\mathrm{E}(x) = -0.5772, \mathrm{Var}(x) = \frac{\pi^2}{6}$

You can always standardize the links/effects to make them comparable.

```{r ,echo=TRUE, fig.align="center"}
eta <- seq(-4, 4, length=100)

ggdat <-tibble("eta"=eta, "logit"=plogis(eta*pi/sqrt(3)), "probit"=pnorm(eta), "cll"=1-exp(-exp(eta*pi/sqrt(6)-0.5772)) )
ggdat %>% pivot_longer(c("logit", "probit", "cll"), names_to="link") %>% mutate(value = scale(value)) %>%
  ggplot + geom_line(aes(x=eta, y=value, colour=link)) + ggtitle("Standardized response functions")

```

* \rightarrow differences in link functions can be canceled to a large extent by linear operations in the predictor (linear operation on the  scale of the predictor lead to nonlinear transformation on the scale of $h$! (as cdfs are nonlinear with very few exceptions ($\mathbb{U}$))
* since a (GLM) does exactly that, influence of link functions is smaller than on the first glance

---------


### Latent utility model

Why do we do this? It is just the general approach towards binary response models. Think the other way round: we started with links and stated later that they are cdfs. Now, we derive that they are cdfs (through $\epsilon$) and then think about what kind of links they could represent.
<!-- Binary responses $y_i \in \{0,1\}$ -->
<!-- Utility differences $\tilde{y}_i = \boldsymbol{x}_i^\prime \boldsymbol{\beta} + \epsilon_{i}$ -->
<!--  $$ -->
<!-- \begin{aligned} -->
<!--  P(y_i = 1) &= P(\text{"differences in utilities is positive"})  \\ -->
<!--  &= P(\tilde{y}_i >0) \\ -->
<!--  &= P(\boldsymbol{x}^\prime_i + \epsilon_i >0) -->
<!-- \end{aligned} -->
<!-- $$ -->
$$
\begin{aligned}
P(y_i = 1) &= P(\tilde{y}_i >0) \\
&= 1- P(\tilde{y}_i \leq 0) \\
& = 1-P(\boldsymbol{x}_i^\prime \boldsymbol{\beta} + \epsilon \leq 0)\\
& = 1- P(\epsilon_i \leq - \boldsymbol x_i^\prime \boldsymbol \beta) \\
&= 1- h(-\boldsymbol{x}_i^\prime \boldsymbol \beta)\\
&= h(\boldsymbol {x}_i^\prime \boldsymbol \beta)
\end{aligned}
$$

$\rightarrow$ $h$ corresponds to a symmetric distribution function $F$ (e.g. logit, probit), if $F(\eta) = 1-F(-\eta)$ and we can write the model in the general form $\pi=F(\eta)$ and $\eta = F^{-1}(\pi)$.

If assymmetric (e.g cloglog): $\eta = - F^{-1}(1-\pi)$.

----
### Interpretation of parameters in the logit model

The influence of the explanatory variables on the probability $\pi =P(y=1)$ is nonlinear and quite obscure. **Direct interpretability only via the signs of $\beta$** ,i.e. if $\beta>0$, then $\pi$ increases and vice versa. We need the odds ratio for more exact interpretations.



Don't be confused:

* "success" probability $\pi=P(y=1)$, "failure" proability $1-\pi = P(y=0)$
* **odds** are defined as the "success" probability divided by the "failure" probability: $\frac{\pi}{1-\pi}$, e.g. if the chance of me winning a game is $\pi=0.75$ and the chance of losing is $1-\pi=0.25$, then the odds are $0.75/0.25= 3$ to $1$ that I will win the game. Therefore, note that **odds are no probabilities**, but a ratio of "successes" divided by the "failures" (you get a probability by dividing by "failures" + "successes"). Odds follow a multiplicative model $\frac{\pi}{1-\pi} = \exp(\beta_0)\exp(x_1\beta_1)...$
* **odds ratio** is a ratio of odds and since odds are ratios, odds ratios are ratios of ratios (sorry). **odds ratios are different from odds** and can help interpreting regression coefficients, because they "isolate" their respective impact. Thus, they indicate relationships between two different configurations, e.g. odds of being sick with treatment ($x =1$) vs. odds of being sick without treatment ($x=0$) $\rightarrow$ compare with slide 41/box 5.2from regression book.

$$
\begin{aligned}
\text{Odds ratio} &= \frac{\text{odds}_{\text{treatment}}}{\text{odds}_{\text{no treatment}}} = \frac{\frac{P(\text{sick}|\text{treatment})}{P(\text{not sick}|\text{treatment})}}{\frac{P(\text{sick}| \text{no treatment})}{P(\text{not sick}|\text{no treatment})}} = \exp(\beta) \\
\end{aligned}
$$
The odds ratio of variable $x \in \{1=\text{treatment}, 0 = \text{no treatment}\}$ is the respective change in odds, when variable $x$ changes (increases from, if $x$ is cont.) $0$ to $1$:
$$
\begin{aligned}
\text{odds}_{\text{treatment}} = \exp(\beta)\cdot \text{odds}_{\text{no treatment}}
\end{aligned}
$$
This would still hold if there were other covariables as their impact would cancel in this ratio.

### Grouped data
$$
\begin{aligned}
y_g \sim \text{Bin}(\pi_g, n_g) \rightarrow &\mathrm{E}(y_g) = n_g \pi_g \\  
                                          &\mathrm{Var}(y_g) = n_g \pi_g (1-\pi_g)\\
\bar{y}_g \sim \text{Bin}(\pi_g, n_g)/n_g \rightarrow & \mathrm{E}({y}_g / n_g) = \pi_g\\
                                                    &\mathrm{Var}(\bar{y}_g) = \mathrm{Var}({y_g /n_g}) =\mathrm{Var}({y_g})/n_g^{2} = n_g/n_g^2 \pi_g (1-\pi_g)=\pi_g    (1-\pi_g)/n_g
\end{aligned}
$$

### Overdispersion

Dispersion means variability and refers to variance. Overdispersion is a situation, where the empirical variance exceeds the "theoretical" variance that is expected from the model. Happens very often. Becomes clearer with count regression.

How do positively correlated observations lead to overdispersion? In few words: because the independence assumption inherent in binomially distributed rvs is not given anymore. Let $Y_i \sim \mathrm{Ber}(\pi)$, then, if $Y_i$ are independent

\begin{align}
Y &=\sum_{i=1}^n Y_i \sim \mathrm{Bin}(n,\pi) \\
\E &= n\pi & \Var(Y) = n\pi(1-\pi) \label{eq:iidbin}\tag{1}
\end{align}


However, if there is correlation, the variance of a sum isn't the sum of the variances anymore, i.e. $\Var(\sum_{i=1}^n Y_i) \neq \sum_{i=1}^n \Var(Y_i)$ which was used in eq. 1 ($\sum_{i=1}^n \Var({Y_i}) =\sum_{i=1}^n \pi(1-\pi) = n\pi(1-\pi)$ ) 

For correlated obs. $Corr(Y_i, Y_j)=\rho > 0$ expectation is the same (invariant) $\E = n\pi$, but the variance becomes
$$
\Var(Y)=\Var(\sum_{i=1}^n Y_i) = n\pi(1-\pi)(1 + \rho(n-1))
$$
which is certainly $> n\pi(1-\pi)$, the variance we would expect under the binomial model. ($\rho <0$ would lead to underdispersion).  Does the left term look familiar, maybe to the scaling factor $\phi$ on slide 49? 
Take a look at
https://en.wikipedia.org/wiki/Variance#Sum_of_correlated_variables
if you want and try to derive it from there (just plug in our $\Var$ for $\sigma^2$)
Think of overdispersion not simply of "too much variance". It means more variance than expected. There is a difference!


### Maximum Likelihood for binary GLM


Log-likelihood
$$
\begin{aligned}
l(\matr{\beta}) &= \log(L(\matr{\beta} \beta))\\
&= \sum_{i=1}^{n}l_i(\matr{\beta})\\
&= \sum_{i=1}^{n}y_i \log(\pi_i) + (1-y_i)\log(1-\pi_i)
\end{aligned}
$$
"Score contribution" (returns vector of length of $\matr{\beta}$) 


$$
\begin{aligned}
\matr{s}_i(\matr{\beta}) &= \frac{\partial}{\partial \matr{\beta}} l_i (\matr{\beta})\\
&= \frac{\partial}{\partial \matr{\beta}} \frac{\partial \eta_i}{\partial \eta_i} \frac{\partial \pi_i}{\partial \pi_i} l_i(\matr{\beta}) \text{   (chain rule)}\\
&= \frac{\partial \eta_i}{\partial \matr{\beta}} \frac{\partial \pi_i}{\partial \eta_i} \frac{\partial}{\pi_i} l_i(\matr{\beta})
\end{aligned}
$$
Score function is the sum of score contributions (as with the log-likelihood)
$$
\begin{aligned}
\matr{s}(\matr{\beta}) = \sum_{i=1}^{n}\matr{s}_i(\matr{\beta})
\end{aligned}
$$

Newton-Raphson

![](../images/animation.gif)



## Count regression

<img style="float: right;" src="../images/CountVonCount.png">

In general, similar issues with $\matr{x}_i^\prime \matr{\beta}$ as before (not discrete and positivity not guaranteed), i.e.

* $y_i$ discrete but $\matr{x}_i^\prime \matr{\beta} \text{ continuous.}$
* $y_i \geq 0$ but $\matr{x}_i^\prime \matr{\beta} \in \mathbb{R}$

Why not try a similar idea as in logistic regression? That is, assume a distribution for $y_i$ and model $\mathrm{E}(y_i) = h(\matr{x}_i^\prime \matr{\beta})$. This idea, i.e. assuming a distribution for $y_i$ and model the respective parameter(s) by transforming a linear predictor $\matr{x}_i^\prime \matr{\beta}$ such that it fulfills the respective properties (e.g. positivity for $\lambda$ or \{0,1\} for $\pi$ etc.) is applied to many problems in GLMs.

## Log-Normal model

What kind of random variable is log-normal distributed? An rv that is normal, when logarithmized. 
Note that
$$ \mathrm{E}(y_i) \neq \exp(\mathrm{E}(\tilde{y}_i)), \text{ where } \tilde{y}_i=\log(y_i)$$
In other words, this can result in (sometimes huge) bias! 
Furthermore, this means that the log-normal model is not a GLM in definition of the next section! 

&nbsp;
&nbsp;

## Generalized Linear Models


**Exponential family** 
Why is it important? Because if we can show that a response distribution belongs to the exponential family, we immediately know its inferential properties and can present it in a unified framework. Probably most of the distributions you have worked with so far belong to the exponential family.


<img style="float: center;" src="../images/tab_expf.png">

(Fahrmeir, Ludwig, et al. "Regression models." Regression. Springer, Berlin, Heidelberg, 2013. 303)
**Table shows the canonical links!**

* $\theta$ is called the canonical parameter; think of what we do here:
  1. specify linear predictor $\eta = \matr{x}^\prime \bvec$ to model the expectation $\mu$ via the link function $g$: $\mu=g^{-1}(\eta)$ or  $g(\mu) = \eta$
  2. often $\mu$ cannot be $=\xb$,but requires to be positive, between zero and one etc. so, we need to transform $\eta$ in a function $\mu (\matr{x})$
  3. $\theta$ is defined on the scale of $y$ and needs to be retransformed via $\theta = (b^\prime)^{-1}(\mu)$ see below;\

     **If the link function relates $\theta$, $\mu$ and $\eta$ such that $\eta=\theta$, we call it the canonical link, given by $g=(b^\prime)^{-1}$**. $b$ is related to the distribution and governs what is canonical.\
     
* every exponential family member has a canonical link function, resulting in $\theta_i = \eta_i = \matr{x}_i^\prime \bvec$ E.g. for Bernoulli, it is the logistic link function, **not probit or cll**! The canonical link guarantees properties that guarantee a smooth estimation process.
* function $b()$ becomes clear when brought into the exponential family form; enters expectation + variance via

\begin{align}
\E(y) &= \mu= b^\prime(\theta)  & \Var(y) =\phi b^{\prime\prime}(\theta)
\end{align}

* $\phi$ is dispersion parameter; often assumed to be known
* $c()$ is a normalizing constant, independent of $\theta$
* $w$ is a weight


A GLM is fully defined by

1. the **random component** specified in terms of the conditional exponential family density $f(y|\cdot)$
2. the **non-random component** in form of the linear predictor $\xb$
3. the **link function**  $g(\mu)=\eta=\matr{x}^\prime \bvec$ or $\mu = g^{-1}(\eta)$; maps to required space, see above

### Bernoulli

$$
\begin{aligned}
\log(f(y_i|\pi_i)) &= y_i \log(\pi_i) - y_i \log(1-\pi_i) + \log(1-\pi) \\
&= y_i \log(\frac{\pi_i}{1-\pi_i}) + \log(1-\pi_i),
\end{aligned}
$$
 where
 
* $\log(\frac{\pi_i}{1-\pi_i}) = \theta_i$
* $\log(1-\pi)=\log(\frac{1}{1+\exp(\theta_i)})=-\log(1+\exp(\theta_i)) = - b(\theta_i)$

Connection to expectation and variance:
$$
\begin{aligned}
b^\prime(\theta_i) &= \frac{\partial}{\partial \theta_i} \log(1+\exp(\theta_i))\\
&= \frac{1}{1+\exp(\theta_i)} \exp(\theta_i)\\
&= \pi_i =\mathrm{E}_i\\
b^{\prime\prime}(\theta_i) &= \frac{\partial}{\partial \theta_i} \frac{\exp(\theta_i)}{1+\exp(\theta_i)}\\
&= \frac{\exp(\theta_i)(1+\exp(\theta_i)) - \exp(\theta_i)\exp(\theta_i)}{(1+\exp(\theta_i))^2}\\
&=\frac{\exp(\theta_i)}{1+\exp(\theta_i)} \frac{1}{1+\exp(\theta_i)}\\
&= \pi_i(1-\pi_i) = \Var
\end{aligned}
$$


### Scaled binomial
$$
\begin{aligned}
\log(f(\bar{y}_i)) &= n_i \bar{y}_i \log(\pi_i) + (n_i -n_i \bar{y})\log(1-\pi_i) + {\log{n_i} \choose {n_i\bar{y}_i}}\\
&= n_i \bar{y}_i \log \frac{\pi_i}{1-\pi_i} + n_i\log(1-\pi) + \log {n_i \choose n_i \bar{y}_i},
\end{aligned}
$$
where

* $n_i \bar{y}_i \log \frac{\pi_i}{1-\pi_i} = \theta_i$
* $n_i\log(1-\pi) = - \log(1+\exp(\theta_i))$

### Showing that a distribution is a member of the exponential family

There is no clear step-by-step manual, but the following approach often makes it easier to see what is what:
**Take the exp of the log of the given density you want to associate with the exp. family.**

I.e.,e.g. for Bernoulli:

\begin{aligned}
p(y|\pi) &= \pi^y(1-\pi)^{1-y} &\text{ oh, no this looks nothing like the exp family form}\\
&= \exp \left( \log \left(\frac{\pi}{1-\pi}\right)y + \log(1-\pi)\right) &\text{ much better!}\\
\end{aligned}

This works often, because the exp. family form is an $\exp$ of a sum/difference, which results from taking the $\log$.
What's left is that you have to be really sure about what is linear in which parameter and especially what term can contain what (e.g. that the normalizing constant cannot contain $\theta$ etc.)

### Maximum Likelihood for GLMs

Advantages of the canonical link:

* always concave log-likelihood leading to unique ML estimators
* $\matr{F}(\bvec) = \matr{H}(\bvec)$

$$
\begin{aligned}
s(\bvec) &= \sum_{i=1}^n s_i(\bvec) \\
s_i(\bvec) &= \frac{\partial}{\partial \bvec} l_i(\bvec) \\
l_i(\bvec) &= \frac{y_i \theta_i - b(\theta_i)}{\phi}\omega_i + c(y_i, \phi, \omega_i)\\
s_i(\bvec) &= \frac{\partial\eta_i}{\partial \bvec} \frac{\partial \mu_i}{\partial \eta_i} \frac{\partial \theta_i}{\partial \mu_i} \frac{\partial}{\partial \theta_i} l_i(\bvec) & (2)\\ 
\end{aligned}
$$
What do those terms mean?

* $\frac{\partial\eta_i}{\partial \bvec} = \frac{\partial}{\partial \bvec} \matr{x}_i^\prime \bvec = \matr{x}_i$
* $\frac{\partial \mu_i}{\partial \eta_i} =\frac{\partial}{\partial \eta_i} h(\eta_i) = h^\prime(\eta_i)$
* $\frac{\partial \theta_i}{\partial \mu_i} = \left(\frac{\partial \mu_i}{\partial \theta_i} \right)^{-1} = \left(\frac{\partial b^\prime(\theta_i)}{\partial \theta_i}\right)^{-1} = (b^{\prime\prime}(\theta_i))^{-1} = \frac{1}{b^{\prime\prime}(\theta_i)}$
* $\frac{\partial}{\partial \theta_i} l_i(\bvec) = (y_i - b^\prime(\theta_i))\frac{\omega_i}{\phi}, \text{ where } b^\prime(\theta_i) =\mu_i$


Plug into eq. (2):

$$
s_i(\bvec) = \matr{x}_i \frac{h^\prime (\eta_i)}{b^{\prime\prime}(\theta_i)} \frac{\phi}{\omega_i}(y_i - \mu_i)
$$

#### Expected Fisher Information

\begin{align}
\matr{F}(\bvec) &= \E(\matr{F}^\star(\bvec))\\
&= -\frac{\partial}{\partial\bvec} \matr{s}(\bvec)\\
&= \text{Cov}(\matr{s}(\bvec))\\
&=E(\matr{s(\bvec)}s(\bvec)^\prime)\\
&=\sum_{i=1}^n \E(\matr{s}_i(\bvec)s_i(\bvec)^\prime)\\
&=\sum_{i=1}^n \matr{F}_i(\bvec)\\
\matr{F}_i(\bvec) &= \matr{x}_i\left(\frac{h^\prime(\eta_i)}{\sigma_i^2}\right)^2\E((y_i - \mu_i)^2)\matr{x}_i^\prime\\
&= \matr{x}_i\matr{x}_i^\prime \frac{(h^\prime(\eta_i))^2}{\sigma_i^2}\\
\end{align}


### Model Fit and Model Choice

#### Pearson statistic

$$\chi^2 = \sum_{i=1}^G \frac{(y_i-\hat{\mu})^2}{v(\hat{\mu_i})/w_i}$$

#### Deviance

$$D = - 2 \sum_{i=1}^G(l_g(\hat{\mu}_g) - l_g(\bar{y}))$$
Both statistics are approx. $\sim \phi \chi^2(G-p)$, where $p$ is the number of parameters and $G$ is the number of groups. The corresponding test of model fit compares the estimated model fit to that of the saturated mode. We assume that under $H_0$ the estimated (not the saturated) model is true.

Important concepts:

* Saturated model: a model very **every** observation has its own parameter estimated, resulting in a "perfect" fit; we need this concept as a benchmark of good model fit, we can compare our models to (e.g in Pearsons statistic and deviance).
* **perfect model fit does not mean perfect or even good model (quite the contrary)** Its just says that our model is very adapted to the data (in-sample). Pearson an deviance are very naive approaches for evaluating a model, as they only capture this kind of adaption. Selection criteria such as the AIC are also in-sample, but try to penalize too high adaptivity. 
* "Too good" a model fit leads to overfitting, take a look at
![](../images/overfitting.png)
(Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006.)
Which model fits better to the data? Which model to you think has better predictive performance? Will be treated in much more detail later. Just remember that it is very easy to construct a model that interpolates your data, but predicts really badly.



# 3. Categorical regression

**GOAL: Estimate probabilities e.g. "What is the probability that person i has infection type r?"**

$$\pi_{ir} = P(Y_i = r)=P(y_{ir}=1)$$

# 4. Quantile regression

In linear regression, we want to model the conditional mean $E(y|x) =x \beta$. This worked by minimizing the residual sum of squares $E((y-\mu)^2|x)$ by assuming $\mu=x\beta$. BUT modelling, e.g. the median $x_{med} = x\beta$ could be even more interesting. In this case we minimize $E(|y-x_{med}||x)$ by assuming $x_{med} = x\beta_{med}$. But why stop at the median? In general, we may want to establish a relationship between the covariates and quantiles of response distribution
$$
q_\tau(y|x) = x\beta_\tau
$$

That is the goal of quantile regression. In comparison to e.g. normal regression, we can model the whole distribution of the response in conjunction with the covariates without assuming a fixed distribution (as in e.g. normal, poisson ... regression; a distribution is fully caputured by its quantiles if you align them close enough e.g. $\tau \in (0.05, 0.1, ..., 0.95, 1.0))$. Think of CONDITIONAL quantiles, i.e. condition on a covariate value (usually a set of values) and look at the quantiles of the resulting distribution of $y$. 

![](../images/qr_munich.png)

# 5. Mixed Models


Intra-class correlation coefficient for random intercept model:
\[
\rho(y_{ij}, y_{ik}) = \frac{Cov(y_{ij},y_{ik})}{\sqrt{Var(y_{ij})} \sqrt{Var(y_{ik})}}
\]
\begin{align}
Var(y_{ij}) = Var(y_{ik})&= Var(\matr{x}^T\bvec + b_i + \epsilon_{ij})\\
&= Var(b_i + \epsilon_{ij}) &\text{$\bvec$ is not random!}\\
&= Var(b_i) + Var(\epsilon_{ij}) + 2Cov(b_i + \epsilon_{ij})\\
&= \sigma^2 + \tau^2\\
\end{align}

\begin{align}
Cov(y_{ij},y_{ik}) &= Cov(\matr{x}^T\bvec + b_i + \epsilon_{ij}, \matr{x}^T\bvec + b_i + \epsilon_{ik}) \\
&= Cov(b_i +\epsilon_{ik}, b_i + \epsilon_{ik})\\
&= Cov(b_i, b_i) + Cov(b_i, \epsilon_{ik}) + Cov(\epsilon_{ik}, b_i) + Cov(\epsilon_{ij}, \epsilon_{ik})\\
&= \tau^2
\end{align}

\[
\rho(y_{ij}, y_{ik}) = \frac{\tau^2}{\sqrt{\tau^2+\sigma^2}\sqrt{\tau^2+\sigma^2}} = \frac{\tau^2}{\tau^2+\sigma^2}
\]

Think of how you would derive the intra-class correlation coefficient for the random slopes model!

Note that both the random intercept and the random slope model assume that the covariate effects are the same for each individual or cluster.


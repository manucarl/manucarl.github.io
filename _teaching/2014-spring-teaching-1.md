---
title: "Teaching experience 2"
collection: teaching
type: "Workshop"
permalink: /teaching/2015-spring-teaching-1
venue: "University 1, Department"
location: "City, Country"
---




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

# Independent Component Analysis (ICA): Algorithms and Applications

- [Independent Component Analysis (ICA): Algorithms and Applications](#independent-component-analysis-ica-algorithms-and-applications)
  - [Abstract](#abstract)
  - [Motivation](#motivation)
  - [Independent Component Analysis](#independent-component-analysis)
    - [Definition of ICA](#definition-of-ica)
    - [Ambiguities of ICA](#ambiguities-of-ica)
    - [Illustration of ICA](#illustration-of-ica)
  - [Independence](#independence)
    - [Definition and fundamental properties](#definition-and-fundamental-properties)
    - [Uncorrelated variables are only partly independent](#uncorrelated-variables-are-only-partly-independent)
    - [Why Gaussian variables are forbidden](#why-gaussian-variables-are-forbidden)
  - [Principles of ICA estimation](#principles-of-ica-estimation)
    - [Nongaussian is independent](#nongaussian-is-independent)
    - [Measures of nongaussianity](#measures-of-nongaussianity)
      - [Kurtosis](#kurtosis)
      - [Negentropy](#negentropy)
      - [Approximations of negentropy](#approximations-of-negentropy)
    - [Minimization of Mutual Information](#minimization-of-mutual-information)
      - [Mutual Information](#mutual-information)
      - [Defining ICA by Mutual Information](#defining-ica-by-mutual-information)
    - [Maximum Likelihood Estimation](#maximum-likelihood-estimation)
      - [The likelihood](#the-likelihood)
      - [The Infomax Principle](#the-infomax-principle)
      - [Connection to mutual information](#connection-to-mutual-information)
    - [ICA and Projection Pursuit](#ica-and-projection-pursuit)
  - [Preprocessing for ICA](#preprocessing-for-ica)
    - [Centering](#centering)
    - [Whitening](#whitening)
    - [Further Preprocessing](#further-preprocessing)
  - [The FastICA Algorithm](#the-fastica-algorithm)
    - [FastICA for one unit](#fastica-for-one-unit)
    - [FastICA for several units](#fastica-for-several-units)
    - [FastICA and maximum likelihood](#fastica-and-maximum-likelihood)
    - [Properties of the FastICA Algorithm](#properties-of-the-fastica-algorithm)
  - [Applications of ICA](#applications-of-ica)
    - [Separation of Artifacts in MEG Data](#separation-of-artifacts-in-meg-data)
    - [Finding Hidden Factors in Financial Data](#finding-hidden-factors-in-financial-data)
    - [Reducing Noise in Natural Images](#reducing-noise-in-natural-images)
    - [Telecommunications](#telecommunications)
  - [Conclusion](#conclusion)

## Abstract

- Problem: finding usable representation of multivariante data (random data)
- for simplicity often as linear transformation of the original data, each component is a linear combination of the original variables
- e,g, principal component analysis, factor analysis and projection pursuit
- Idea of ICA: finding linear representations of nongaussian data with all components being statistically independent or as independent as possible

## Motivation

- scenario (_cocktail-party problem_):
  - two people in a room are talking simultaneously
  - they are being recorded by two microphones
    - each in a different location
  - both microphones produce their own time signal
    - i.e. $x_1(t)$ and $x_2(t)$
  - each signal is a weighted sum of the spoken signals from each speaker 
    - i.e. $s_1(t)$ and $s_2(t)$ as original signals
  - this can be expressed in the following linear equation

$$
\begin{equation}
x_1(t) = a_{11}s_1 + a_{12}s_2
\end{equation}
$$

$$
\begin{equation}
x_2(t) = a_{21}s_1 + a_{22}s_2
\end{equation}
$$

-  
  - $a_{ij}$ are unknown parameters which depend on the distances between microphones and speakers 
  - goal is to estimate the original signals $s_1(t)$ and $s_2(t)$ from only th recorded signals $x_1(t)$ and $x_2(t)$
- the big problem is that $a_{ij}$ are unknown
  - if they were known the linear equation could be solved using known methods
- one possible approach is estimating $a_{ij}$ through statistical properties
  - here it is enough go with the assumption that $s_1(t)$ and $s_2(t)$ are statistically independent at every time step $t$
  - this is exactly what ICA is doing with fairly accurate results
    - interestingly the signs are reversed, however
- ICA has lots of area of application
  - originally used for similar problem to the _cocktail-party problem_
  - also useful for EEG (brain activity scan, many electrode mixing with brain activity)
  - feature extraction

## Independent Component Analysis

### Definition of ICA

- statistical "latent variables" model, assumption $n$ linear combinations of $x$ with $n$ independent components
- statistical "latent variables" model, assumption $n$ linear combinations of $x$ with $n$ independent components

$$
\begin{equation}
x_j = a_{j1}s_1 + a_{j2}s_2 + \dots a_{jn}s_n \text{ , for all $j$}
\end{equation}
$$

- time index $t$ is dropped, because it is assumed that each combination is a random interval
- it can also be assumed that this is a zero-mean model
- more convenient to vector-matrix notation
- new notation:
  - **x**, the random vector of elements $x_1$, $\dots$ ,$x_n$
  - **s**, the random vector of elements $s_1$, $\dots$ ,$s_n$
  - **A**, the matrix with elements $a_{ij}$
- with the new notation the mixing model from above can be simplified to
  - this is the statistical model aka. ICA model

$$
\begin{equation}
\bold{x} = \bold{A}\bold{s}
\end{equation}
$$

- sometime the columns of **A** are needed, denoted $\bold{a}_j$ or as model

$$
\begin{equation}
\bold{x} = \sum^n_{i=1} \bold{a}_i\bold{s}_i
\end{equation}
$$

- the ICA model is generative model
  - describing how data is observed to be generated in a process of missing the components $s_i$
- the independent components cannot be directly observed, latent variables
- the mixing matrix **A** is assumed to be unknown
- only the random vector **x** can be observed and must be used to estimate **A** and **s**
- starting point is the assumption that the components $s_i$ are statistically independent
  - must also have a nongaussian distribution
- for simplicity it is also assumed that **A** is square (not necessary)
- once **A** has been estimated, its inverse can be computed, calling it **W** and giving the updated formula

$$
\begin{equation}
\bold{s} = \bold{W}\bold{x}
\end{equation}
$$

### Ambiguities of ICA

in the ICA model two ambiguities will hold

1. the variance (energies) of the independent components cannot be determined
   
   - since both **s** and **A** are unknown one component **S**_i can always be cancelled by dividing the corresponding column of **a**_i of **A** by the same scalar
   - since the values are random it can be assumed each component has a variance of $E\{s^2_i\} = 1$, fixing the magnitude
   - that still leave the problem of the flipped sign, but multiplying by -1 wouldn't affect the model
   - this ambiguity is insignificant in most applications

2. the order of the independent components cannot be determined

    - since both **s** and **A** are unknown the order of terms in the sum can be freely rearranged
    - meaning a permutation matrix **P** and its inverse can be integrated into the model, $\bold{x} = \bold{A}\bold{P}^{-1}\bold{P}\bold{s}$
    - **Ps** being the original independent variables in a new order
    - $\bold{AP}^{-1}$ being the new unknown mixing matrix

### Illustration of ICA

- using these two independent components with uniform distributions

$$
\begin{equation}
p(s_i) = 
    \begin{cases}
        \frac{1}{2\sqrt{3}} & \text{ , if }\lvert s_i \rvert \le \sqrt{3} \\
        0 & \text{ , otherwise}
    \end{cases}
\end{equation}
$$

- they are chosen to have mean of zero and variance of one
- their joint density is uniform on a square, product of their marginal densities
- next, mixing the independent components with following mixing matrix

$$
\begin{equation}
    \bold{A}_0 = \left(
    \begin{matrix}
        2 & 3 \\
        2 & 1  
    \end{matrix} \right)
\end{equation}
$$

- the mixing results in the mixed variables $x_1$ and $x_2$
  - these are no longer independent
  - their uniform distribution is on a parallelogram
- the problem now is to estimate $\bold{A}_0$ of the ICA model using only information from $x_1$ and $x_2$
- the edges of the parallelogram go in directions of the columns of **A**
  - one possibility is to estimate the ICA model by
    - estimating the joint density of $x_1$ and $x_2$ and then locating the edges
  - not a realistic approach because this only works when the variables have a uniform distribution and computational very complicated

## Independence

### Definition and fundamental properties

- two values are independent from each other when there is no way to gather any information about one of the values from the the other and vice versa
- this is true for the original signals $s_i$ but not for the mixed signals $x_i$
- this can also be defined with probability densities
- using the joint probability density function (pdf) of $y_1$ and $y_2$ as $p(y_1, y_2)$
- the marginal pdf of $y_1$ (and analogue $y_2$) is $p_1(y_1)$

$$
\begin{equation}
  p_1(y_1) = \int p(y_1, y_2)dy_2
\end{equation}
$$

- $y_1$ and $y_2$ are only independent if the joint pdf is factorable as
  - this applies for any number of random factors

$$
\begin{equation}
  p(y_1, y_2) = p_1(y_1)p_2(y_2)
\end{equation}
$$

- the most important property can be derived from this definition
  - given two function $h_1$ and $h_2$, following will always be true

$$
\begin{equation}
  E\{h_1(y_1)h_2(y_2)\} = E\{h_1(y_1)\}E\{h_2(y_2)\}
\end{equation}
$$

- proof:

$$
\begin{equation}
  \begin{split}
    E\{h_1(y_1)h_2(y_2)\} &= \iint h_1(y_1)h_2(y_2)p(y_1,y_2)dy_1dy_2 \\
    &= \iint h_1(y_1)p_1(y_1)h_2(y_2)p_2(y_2)dy_1 dy_2 \\
    &= \int h_1(y_1)p_1(y_1)dy_1 \int h_2(y_2)p_2(y_2)dy_2 \\
    &= E\{h_1(y_1)\}E\{h_2(y_2)\}
  \end{split}
\end{equation}
$$

### Uncorrelated variables are only partly independent

- two random variables $y_1$ and $y_2$ are uncorrelated when their covariance is zero

$$
\begin{equation}
  E\{y_1,y_2\} - E\{y_1\}E\{y_2\} = 0
\end{equation}
$$

- independent variables are also uncorrelated
  - uncorrelated variables are necessarily independent
- many ICA methods limit the estimation process to always result in uncorrelated estimates of the independent components

### Why Gaussian variables are forbidden

- fundamental restriction is independent components must be nongaussian
  - otherwise ICA is not possible
- reason: using a orthogonal mixing matrix and $s_i$ are gaussian
  - $x_i$ will also be gaussian, uncorrelated and of unit variance
  - their joint density is

$$
\begin{equation}
  p(x_1,x_2) = \frac{1}{2\pi} \exp\left( -\frac{x^2_1 + x^2_2}{2} \right)
\end{equation}
$$

- the density is completely symmetric
  - no information about he direction of the columns of the mixing matrix **A** can be gathered
  - and by that it cannot be estimated

## Principles of ICA estimation

### Nongaussian is independent

### Measures of nongaussianity

#### Kurtosis

#### Negentropy

#### Approximations of negentropy

### Minimization of Mutual Information

#### Mutual Information

#### Defining ICA by Mutual Information

### Maximum Likelihood Estimation

#### The likelihood

#### The Infomax Principle

#### Connection to mutual information

### ICA and Projection Pursuit

## Preprocessing for ICA

### Centering

### Whitening

### Further Preprocessing

## The FastICA Algorithm

### FastICA for one unit

### FastICA for several units

### FastICA and maximum likelihood

### Properties of the FastICA Algorithm

## Applications of ICA

### Separation of Artifacts in MEG Data

### Finding Hidden Factors in Financial Data

### Reducing Noise in Natural Images

### Telecommunications

## Conclusion

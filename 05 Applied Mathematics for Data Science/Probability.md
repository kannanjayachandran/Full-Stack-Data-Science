<!-- 
    Author : Kannan Jayachandran
    File : Probability.md
    Section : Applied Mathematics for Data Science     
 -->

<h1 align="center"> PROBABILITY </h1>

---

## Table of Contents   

1. [Experiment](#experiment)
1. [Outcome](#outcome)
1. [Sample Space](#sample-space)
1. [Event](#event)
1. [Event Space](#event-space)
1. [Random Variable](#random-variable)
1. [Calculating Probability](#calculating-probability)
1. [Probability Distribution](#probability-distribution)
1. [Density functions](#density-functions)
1. [Probability Density Function (PDF)](#probability-density-function-pdf)
1. [Probability Mass Function (PMF)](#probability-mass-function-pmf)
1. [Cumulative Density Function (CDF)](#cumulative-density-function-cdf)
1. [Symmetric Distribution and Skewness](#symmetric-distribution-and-skewness)
1. [Kurtosis](#kurtosis)
1. [Gaussian Distribution or Normal Distribution](#gaussian-distribution-or-normal-distribution)
1. [Empirical rule or 68–95–99.7 rule](#empirical-rule-or-68–95–99.7-rule)
1. [Standard Normal Variate](#standard-normal-variate)
1. [Quantile-Quantile (QQ) Plot](#quantile-quantile-qq-plot)
1. [Student's t-distribution or t-distribution](#students-t-distribution-or-t-distribution)
1. [Uniform Distribution](#uniform-distribution)
1. [Standard uniform distribution](#standard-uniform-distribution)
1. [Binomial Distribution](#binomial-distribution)
1. [Bernoulli Distribution](#bernoulli-distribution)
1. [Lognormal Distribution](#Log-Normal-Distribution)
1. [Pareto Distribution](#pareto-distribution)

---

**Probability is the mathematical framework for quantifying uncertainty and making predictions about data. It forms the theoretical foundation for statistical inference, machine learning algorithms, and data-driven decision making.**

## Experiment 

An **experiment** is any process or procedure that produces a well-defined outcome. In data science, experiments can be deterministic (always produce the same result) or stochastic (produce random outcomes).

Think of an experiment as *any action or observation that generates data*. In machine learning contexts, this could be:
- A single prediction by a model
- Drawing a sample from a population
- A user interaction with a system
- A measurement or sensor reading

---

## Outcome 

An **outcome** is a single possible result of an experiment. It represents one specific realization from the set of all possibilities.

> An outcome is what actually happens when you perform an experiment once.

**Mathematical Notation**: We typically denote outcomes as $\omega \in \Omega$, where $\Omega$ is the sample space.

**Examples**:
- Rolling a die: outcome could be 3
- Email classification: outcome is "spam" or "not spam"
- Stock price tomorrow: outcome is a specific price value

---

## Sample Space ($\Omega$ or S) 

The **sample space** is the set of all possible outcomes of an experiment. It represents the complete universe of what could happen.

**Mathematical Definition**:
$$\Omega = \{\omega_1, \omega_2, \omega_3, \ldots\}$$

where each $\omega_i$ is a possible outcome.

- For example, in a coin toss, the sample space is {head, tail}. For a roll of a six-sided die, the sample space is {1, 2, 3, 4, 5, 6}.

### Types of Sample Spaces

| Type | Description | Example |
|:----:|:------------|:--------|
| **Finite** | Countable number of outcomes | Die roll: $\{1, 2, 3, 4, 5, 6\}$ |
| **Countably Infinite** | Infinite but enumerable | Number of emails until spam: $\{1, 2, 3, \ldots\}$ |
| **Uncountably Infinite** | Continuous range | Time until system failure: $[0, \infty)$ |

### Best Practices
- Always clearly define your sample space before calculating probabilities
- Ensure outcomes are mutually exclusive (no overlap)
- Verify sample space is collectively exhaustive (covers all possibilities)

---

## Event (A)

An **event** is a subset of the sample space to which we assign a probability. Events represent outcomes or combinations of outcomes we're interested in measuring.

**Mathematical Definition**:
$$A \subseteq \Omega$$

- For example, in a coin toss, the event of getting a head is a subset of the sample space

### Set Operations on Events

| Operation | Notation | Meaning | Data Science Example |
|:---------:|:--------:|:--------|:---------------------|
| **Union** | $A \cup B$ | Either A or B occurs | Model predicts positive OR ground truth is positive |
| **Intersection** | $A \cap B$ | Both A and B occur | True positive: prediction AND truth are positive |
| **Complement** | $A^c$ or $\bar{A}$ | A does not occur | False negative: NOT predicted positive |
| **Difference** | $A \setminus B$ | A occurs but not B | False positive cases |

**Visual Representation**:

![Ven diagram representation of probability](./img/ven.png)

### Common Pitfalls
- Confusing $P(A \cup B)$ with $P(A) + P(B)$ (only true if events are mutually exclusive)
- Forgetting that $A \cap A^c = \emptyset$ (empty set)
- Not checking if events are independent before multiplying probabilities

---

## Event space ($\mathcal{F}$)

The **event space** (also called $\sigma$-algebra or $\sigma$-field) is the collection of all events to which we can assign probabilities. It's a formal mathematical structure that ensures probability is well-defined.

**Mathematical Definition**: A σ-algebra $\mathcal{F}$ on sample space $\Omega$ must satisfy:

1. **Contains the sample space**: $\Omega \in \mathcal{F}$
2. **Closed under complementation**: If $A \in \mathcal{F}$, then $A^c \in \mathcal{F}$
3. **Closed under countable unions**: If $A_1, A_2, \ldots \in \mathcal{F}$, then $\bigcup_{i=1}^{\infty} A_i \in \mathcal{F}$

**Intuition**: The event space is the "*legal*" collection of sets we're allowed to ask probability questions about. Not every subset of $\Omega$ needs to be measurable.

**Example**:
For a coin toss with $\Omega = \{H, T\}$:
$$\mathcal{F} = \{\emptyset, \{H\}, \{T\}, \{H, T\}\}$$

This contains:
- The impossible event: $\emptyset$
- Individual outcomes: $\{H\}$ and $\{T\}$
- The certain event: $\{H, T\}$

**Practical Note**: In most data science applications with finite sample spaces, the event space is simply the power set (all possible subsets) of $\Omega$. The σ-algebra formalism becomes critical in measure theory and continuous probability spaces.

---

## Probability (P)

**Probability** quantifies the likelihood of an event occurring, expressed as a real number between 0 and 1.

A **probability function** $P: \mathcal{F} \rightarrow [0, 1]$ assigns to each event $A \in \mathcal{F}$ a number $P(A)$ representing its likelihood.

### Classical (Frequentist) Definition
$$P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}$$

**Interpretation Scale**:
- $P(A) = 0$: Event $A$ is **impossible**
- $P(A) = 0.5$: Event $A$ is **equally likely** to occur or not
- $P(A) = 1$: Event $A$ is **certain**

> *Probability is fundamentally a measure of the relative size of a set.*
>
> When we write $P(A)$, we're measuring how "large" event $A$ is relative to the sample space $\Omega$.

**Example Calculation**:
Consider rolling a fair six-sided die. What is the probability of rolling an even number?

- **Sample space**: $\Omega = \{1, 2, 3, 4, 5, 6\}$, $|\Omega| = 6$
- **Event**: $A = \{2, 4, 6\}$, $|A| = 3$
- **Probability**: 
$$P(A) = \frac{|A|}{|\Omega|} = \frac{3}{6} = 0.5$$

### Complement Rule

The probability of an event not occurring is:

$$P(A^c) = 1 - P(A)$$

This is extremely useful in practice. Often it's easier to calculate the probability of something NOT happening. For example, When calculating prediction errors:

$$P(\text{error}) = 1 - P(\text{correct}) = 1 - \text{accuracy}$$

### Best Practices

- Always verify probabilities sum to 1 across exhaustive, mutually exclusive events
- Use simulation to validate analytical probability calculations
- For rare events, consider using complement: $P(A) = 1 - P(A^c)$ if $P(A^c)$ is easier to compute
- In classification, ensure class probabilities sum to 1: $\sum_{i=1}^{k} P(y = c_i) = 1$

### Common Pitfalls

- Applying classical definition to non-uniform sample spaces (e.g., loaded dice)
- Confusing empirical frequency with true probability (especially with small samples)
- Forgetting that probability is a model assumption, not always ground truth
- Misapplying frequentist probability to one-time events

---

## Axioms of Probability

The **axioms of probability** (Kolmogorov's axioms) provide the mathematical foundation for all probability theory. These three simple rules ensure probability behaves consistently and meaningfully.

### Kolmogorov's Three Axioms

For a probability function $P$ defined on event space $\mathcal{F}$:

**Axiom 1: Non-negativity**
$$P(A) \geq 0 \quad \forall A \in \mathcal{F}$$

Probabilities cannot be negative. The likelihood of any event is at minimum zero.

**Axiom 2: Normalization (Unitarity)**
$$P(\Omega) = 1$$

The probability of the entire sample space (something must happen) is 1.

**Axiom 3: Countable Additivity (σ-Additivity)**

For any countable collection of mutually exclusive events $A_1, A_2, A_3, \ldots$ where $A_i \cap A_j = \emptyset$ for $i \neq j$:

$$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

If events cannot occur simultaneously, the probability of any one occurring is the sum of their individual probabilities.

### Derived Properties

From these three axioms, we can derive all other probability rules:

**1. Probability of Empty Set**:
$$P(\emptyset) = 0$$

**Proof**: Let $A_1 = \Omega$ and $A_i = \emptyset$ for $i > 1$. These are mutually exclusive.
$$P(\Omega) = P\left(\bigcup_{i=1}^{\infty} A_i\right) = P(\Omega) + \sum_{i=2}^{\infty} P(\emptyset)$$
$$1 = 1 + \sum_{i=2}^{\infty} P(\emptyset)$$
Therefore $P(\emptyset) = 0$.

**2. Complement Rule**:
$$P(A^c) = 1 - P(A)$$

**Proof**: $\Omega = A \cup A^c$ and $A \cap A^c = \emptyset$
$$1 = P(\Omega) = P(A \cup A^c) = P(A) + P(A^c)$$

**3. Monotonicity**:
If $A \subseteq B$, then $P(A) \leq P(B)$

**4. Finite Additivity**:
For mutually exclusive events $A_1, \ldots, A_n$:
$$P(A_1 \cup \cdots \cup A_n) = P(A_1) + \cdots + P(A_n)$$

### Best Practices

- Always normalize model outputs (e.g., using `softmax` for multi-class)
- Verify probability distributions before using them in decision systems
- Use log-probabilities for numerical stability in deep learning
- Remember that probabilities are a model of uncertainty, not absolute truth

### Common Pitfalls

- Using unnormalized scores as probabilities (e.g., raw SVM outputs)
- Forgetting to handle numerical precision in probability sums
- Assuming independence without checking (violates proper probability calculations)
- Mixing probability with odds or log-odds without conversion

---

## Rules of Probability

Building on the axioms, we derive fundamental rules for calculating probabilities of complex events.

### Addition Rule

The **addition rule** (or sum rule) calculates the probability that at least one of two events occurs.

**General Addition Rule** (for any two events):
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

![Addition rule](./img/PB-addition-rule.png)

**Intuition**: We add individual probabilities but subtract the overlap (intersection) because it's counted twice.

**Special Case** (mutually exclusive events): If $A \cap B = \emptyset$:
$$P(A \cup B) = P(A) + P(B)$$

**Extended Form** (three events):
$$P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)$$

This follows the inclusion-exclusion principle.

**Example**: In binary classification confusion matrix:
- Event A: Model predicts positive
- Event B: True label is positive
- $P(A \cup B)$: Probability model predicts positive OR true label is positive

$$P(\text{predicted positive OR true positive}) = P(\text{pred+}) + P(\text{true+}) - P(\text{TP})$$

### Multiplication Rule

The **multiplication rule** (or product rule) calculates the probability that two events both occur.

**General Multiplication Rule**:
$$P(A \cap B) = P(A) \cdot P(B|A) = P(B) \cdot P(A|B)$$

where $P(B|A)$ is the conditional probability of B given A.

**Special Case** (independent events): If A and B are independent:
$$P(A \cap B) = P(A) \cdot P(B)$$

**Extended Form** (chain rule):
$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \cdot P(A_2|A_1) \cdot P(A_3|A_1 \cap A_2) \cdots P(A_n|A_1 \cap \cdots \cap A_{n-1})$$

**ML Application**: In naive Bayes classifier:
$$P(y, x_1, x_2, \ldots, x_n) = P(y) \prod_{i=1}^{n} P(x_i|y)$$

assuming conditional independence of features given the class.

### Summary Table

| Rule | Formula | When to Use |
|:-----|:--------|:------------|
| **Addition** | $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ | Finding "at least one" probabilities |
| **Addition (Mutually Exclusive)** | $P(A \cup B) = P(A) + P(B)$ | Events can't occur together |
| **Multiplication** | $P(A \cap B) = P(A) \cdot P(B\|A)$ | Finding "both occur" probabilities |
| **Multiplication (Independent)** | $P(A \cap B) = P(A) \cdot P(B)$ | Events don't affect each other |
| **Complement** | $P(A^c) = 1 - P(A)$ | "Not A" is easier to calculate |

### Best Practices

- Always check if events are mutually exclusive before using simplified addition rule
- Verify independence assumption before using simplified multiplication rule (very common mistake in ML)
- Use complement rule when direct calculation is difficult
- In ML, log-space calculations prevent numerical underflow: $\log P(A \cap B) = \log P(A) + \log P(B|A)$

---

## Conditional Probability and Independence

**Conditional probability** measures the likelihood of an event occurring given that another event has already occurred. It's fundamental to causal reasoning, Bayesian inference, and nearly all ML algorithms.

**Definition**:
$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

Read as "probability of A given B."

**Intuition**: When we know B has occurred, we're restricting our sample space to only those outcomes in B. We then measure what fraction of B also contains A.

**Geometric Interpretation**: 
- Original sample space: $\Omega$
- New "reduced" sample space: $B$
- We want the proportion of $B$ that overlaps with $A$

### Independence

Two events A and B are **independent** if knowing one occurred doesn't change the probability of the other.

**Definition** (equivalent forms):

1. $P(A \cap B) = P(A) \cdot P(B)$
2. $P(A|B) = P(A)$
3. $P(B|A) = P(B)$

**Intuition**: Information about B gives us no information about A. They don't influence each other.

**Important Distinction**:
- **Independence** ≠ **Mutual Exclusivity**
- Mutually exclusive events are usually *dependent*!
- If $A \cap B = \emptyset$ and both have positive probability, then $P(A|B) = 0 \neq P(A)$

**Conditional Independence**: Events A and B are conditionally independent given C if:
$$P(A \cap B | C) = P(A|C) \cdot P(B|C)$$

This is crucial for naive Bayes and graphical models.

**Naive Bayes Assumption**:

In naive Bayes classification, we assume features are conditionally independent given the class:

$$P(x_1, x_2, \ldots, x_n | y) = \prod_{i=1}^{n} P(x_i | y)$$

This is usually **violated in practice**, but the classifier often works well anyway!

### Law of Total Probability

If $B_1, B_2, \ldots, B_n$ partition the sample space (mutually exclusive and exhaustive), then:

$$P(A) = \sum_{i=1}^{n} P(A | B_i) \cdot P(B_i)$$

**Intuition**: We can calculate the probability of A by considering all possible "scenarios" $B_i$ that could lead to A.

**Continuous Version**: If B is continuous with PDF $f_B(b)$:
$$P(A) = \int P(A|B=b) \cdot f_B(b) \, db$$

### Best Practices

- Always verify independence assumption before using $P(A \cap B) = P(A) \cdot P(B)$
- Use chi-square test or mutual information to test independence empirically
- Be cautious with naive Bayes: conditional independence is rarely true but often "good enough"
- When calculating marginal probabilities, use law of total probability to sum over all scenarios

### Common Pitfalls

- Assuming independence without justification (extremely common!)
- Confusing independence with correlation: uncorrelated ≠ independent (except for Gaussian)
- Ignoring conditional dependencies in causal reasoning
- Forgetting that pairwise independence doesn't imply mutual independence

---


<!-- Edited -->

# Break
















## Random Variable (X) 

A random variable is a variable that can assume various values, and the specific value it takes is subject to chance or randomness. They are of two types; `discrete` and `continuous`. **Discrete random variables** can be represented by a **probability distribution table**, while **continuous random variables** can be represented by a **probability density function**.

| Discrete Random Variable | Continuous Random Variable |
| :----------------------: | :------------------------: |
| A random variable which represents outcomes that can be counted or enumerated and are typically associated with countable or distinct values (Typically whole numbers) .| A random variable which represents outcomes that can be measured and can take on any value in a given range (Any real number) . | 
|       Whole number       |         Real number        |
|       Countable set      |       Uncountable set      |
|       Finite set         |         Infinite set       |




## Probability Distribution

A probability distribution is a mathematical function that describes how the values of a random variable are distributed or spread out. It can be used to represent both discrete (discrete probability distribution) and continuous (continuous probability distribution) random variables. 

> The distribution of individual data points is referred to as the `data distribution`, while the distribution of a sample statistic is known as the `sampling distribution`.

## Probability Density Function (PDF)

The Probability Density Function, or **PDF** also called (Density Function), calculates the probability of observing a given outcome within a continuous probability distribution. PDF is calculated for continuos random variables. PDF's always satisfy the following two conditions;

1. Always be non-negative $\color{#F99417}f(x) \ge 0$ for all $\color{#F99417}x$

2. The total area under the PDF curve over the entire range of possible values is equal to $\color{#F99417}1$.

The probability of a  random variable ($\color{#F99417}X$) falling within a small interval, say $\color{#F99417}[a, b]$, is given by the integral of the PDF over that interval.

$$P(a\le x \le b) = \int_{a}^{b}f(x)dx$$

![PDF of Gaussian distribution image from wikipedia](./img/pdf.png)
> PDF of Gaussian distribution

- The integral of the density function over a specific interval gives the probability that the random variable falls within that interval.

The PDF curve represents the probability density of the random variable at each point. This means that the area under the PDF curve between two points represents the probability of the random variable taking on a value within that range.

**We can calculate PDF using the following steps**;

1. Find the probability distribution and the parameters of the distribution.

2. Get the PDF formula for the distribution and calculate.

## Probability Mass Function (PMF)

Probability mass function (PMF) is a function that gives the probability of a discrete random variable taking on a specific value. It is denoted as $\color{#F99417}P(X=x)$. It provides the probability of $\color{#F99417}X$ taking on a specific value.

The PMF must satisfy the following two conditions:

1. For all possible values of $\color{#F99417}x$, $\color{#F99417}P(X=x)\ge 0$.

2. The sum of the PMF over all possible values of $\color{#F99417}x$ is equal to $\color{#F99417}1$.

![PMF diagram of a fair die](./img/pmf.png)

## Cumulative Density Function (CDF)

Calculates the probability of an observation equal or less than a value. It is also called  `cumulative distribution function`. It calculates the cumulative likelihood.

$$F(x) = P(X\le x)$$

We can calculate CDF of a continuous random variable by integrating the PDF of the random variable and for a discrete random variable, we can calculate CDF by summing up the probabilities of all possible outcomes that are less than or equal to $\color{#F99417}x$.

![CDF image](./img/cdf.png)
> CDF of Gaussian distribution 

- CDF is a monotonically increasing function (non-decreasing). That is; as $\color{#F99417}x$ increases, $\color{#F99417}F(x)$ can only increase or remain the same.

- As shown in the above diagram, CDF in a normal distribution is a smooth `S` shaped curve.

## Symmetric Distribution and Skewness

A distribution is considered **symmetric** when the right half of the distribution is a mirror image of the left half. This means that the number of data points on one side is equal to the number on the other. An example of a symmetric distribution is the `normal distribution`.

A distribution is said to be **skewed** if the right half of the distribution is not a mirror image of the left half. In this case, the number of data points on one side is not equal to the other side.  For example, the `exponential distribution` is skewed. There are two types of skewness;

1. **Right Skewed or Positive Skewed** : If the right tail is longer than the left tail, the distribution is said to be right skewed. Here **mean** is greater than the **median**. 

2. **Left Skewed or Negative Skewed** : If the left tail is longer than the right tail, the distribution is said to be left skewed. In this case, the **mean** is less than the **median**.

![Skewness image from wikipedia](./img/Skewness.png)

> Skewness is a measure asymmetry of the distribution.

## Kurtosis

Kurtosis provides insights into the peakedness of a distribution and measures the heaviness of the tails of a distribution.

![Kurtosis image from wikipedia](./img/Kurtosis.png)

From the image above, you can observe that higher kurtosis indicates a greater peakedness in the distribution.

- The normal distribution has a kurtosis of 3, making it a **mesokurtic** distribution.

- A distribution with a kurtosis of less than 3 is classified as a **platykurtic** distribution.

- A distribution with a kurtosis greater than 3 is referred to as a **leptokurtic** distribution.

> We compare Kurtosis coefficient, to determine the kurtosis of a distribution (Meso, Platy, Lepto).

## Gaussian Distribution or Normal Distribution ($\color{#F99417}X\sim N(\mu, \sigma)$)

The Gaussian distribution, also known as the Normal distribution, is one of the most common and important probability distributions. It is characterized by a bell-shaped curve and described by two key parameters: the mean ($\color{#F99417}\mu$) and the standard deviation ($\color{#F99417}\sigma$).  Mathematically we would write a normal distribution as;

$$X\sim N(\mu, \sigma)$$

We denote normal distribution as $\color{#F99417}N(\mu, \sigma)$ or  $\color{#F99417}N(\mu, \sigma^2)$, where $\color{#F99417}\sigma^2$ is the variance of the distribution. It is a continuous distribution.

We can write the **PDF** of a normal distribution as; 

$$\large f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}$$


where $\color{#F99417}\large \frac{1}{\sqrt{2\pi}\sigma}$ is the normalization constant, $e$ is the base of the natural logarithm, $\color{#F99417}\large \frac{x-\mu}{\sigma}$ is the standard score or z-score.

> z-score is the standard score or standard value of a random variable. It is the number of standard deviations that a data point is above from the mean. It is denoted as $\color{#F99417}z$.

The **CDF** of a normal distribution is given by;

$$\large F(x) = {\displaystyle \Phi ({\frac {x-\mu }{\sigma }})={\frac {1}{2}}\left[1+\operatorname {erf} \left({\frac {x-\mu }{\sigma {\sqrt {2}}}}\right)\right]}$$

Where $\color{#F99417}erf$ is the error function. We can rewrite this equation as;

$$\large F(x) = \frac{1}{\sigma\sqrt{2\pi}}\int_{-\infty}^{x}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}dx$$
 
Gaussian distribution is important as it is the  distribution of many natural phenomena. For example, the height of people, the weight of people, the marks of students in a class, etc. are all normally distributed.

|Property | Notation/Formula |
|:---:|:---:|
|Mean | $\color{#F99417}\mu$ |
|Median | $\color{#F99417}\mu$ |
|Mode | $\color{#F99417}\mu$ |
|Standard Deviation | $\color{#F99417}\sigma$ |
|Variance | $\color{#F99417}\sigma^2$ |
|Skewness | $\color{#F99417}0$ |

## Empirical rule or 68–95–99.7 rule

The Empirical Rule is a statistical guideline that applies to a normal distribution, stating that almost all data points will fall within three standard deviations of the mean.

![Empirical formula image from wikipedia](./img/empirical_rule.png)

Let's consider a normal distribution $\color{#F99417}X \sim N(\mu, \sigma^2)$, with $\color{#F99417}\mu = 0$. Then we can say that;

1. Around $\color{#F99417}68\%$ of the data lies within $\color{#F99417}\mu \pm \sigma$ $\color{#F99417}[-1\sigma, 1\sigma]$.

2. Around $\color{#F99417}95\%$ of the values lie within $\color{#F99417}\mu \pm 2\sigma$ $\color{#F99417}[-2\sigma, 2\sigma]$.

3. Around $\color{#F99417}99.7\%$ of the values lie within $\color{#F99417}\mu \pm 3\sigma$ $\color{#F99417}[-3\sigma, 3\sigma]$.

<div align="center">

**[1 $\sigma$ = 68%]**

**[2 $\sigma$ = 95%]**

**[3 $\sigma$ = 99.7%]**

</div>

## Standard Normal Variate

A standard normal variate is a random variable that follows a normal distribution with a mean of zero and a standard deviation of one. It is denoted as $\color{#F99417}Z$. 

$$\large z\sim N(0, 1)$$

The process of converting a normal distribution to a standard normal distribution is called **standardization**. 

Let us consider the iris dataset, with `PetalLengthCm` as one feature. We can represent this feature as a normal distribution as follows; 

$$X \sim N(\mu, \sigma^2)$$ 

where $\color{#F99417}X$ can take any value in the range $\color{#F99417}[x_1, x_2, ..., x_{50}]$. We can standardize this distribution as;

$\large x_i$' = $\large \frac{x_i - \mu}{\sigma} \;\;\forall \;i= 1, 2, .., 50$ 

where $\color{#F99417}\large x_i'$ is the standardized value of $\color{#F99417}\large x_i$. Now we can write the standardized distribution as;

$$X` \sim N(0, 1)$$

where $\color{#F99417}X`$ can take any value in the range $\color{#F99417}[x_1`, x_2`, ..., x_{50}`]$.

## Quantile-Quantile (QQ) Plot

QQ plot is a graphical technique for easily determining whether a random variable is Gaussian or normally distributed. Consider the random variable $\color{#F99417}x$, with samples/observations $\color{#F99417}\bar x_1, \bar x_2, \bar x_3, ...\bar x_n$.

- First sort them in ascending order. $\color{#F99417}\bar x`_1, \bar x`_2, \bar x`_3, ...\bar x`_n$.

- Calculate the percentile. $\color{#F99417}x^{(1)}, x^{(2)}, x^{(3)}, ...x^{(n)}$. We get the first percentile at $\color{#F99417}\bar x`_{\frac{n}{100}}$

> If we have 100 data points, then the first percentile is the first data point. If we have 1000 data points, then the first percentile is the 10th data point.

- Create $Y \sim N(0, 1)$, where $\color{#F99417}Y$ is a random variable that follows a standard Gaussian distribution, with mean $\color{#F99417}\mu = 0$ and standard deviation $\color{#F99417}\sigma = 1$.

- Create $\color{#F99417}n$ observations from the $\color{#F99417}Y$ distribution. Sort them in ascending order and find the percentile. $\color{#F99417}y^{(1)}, y^{(2)}, y^{(3)}, ...y^{(n)}$. These are also called `theoretical quantiles`.

- Now plot the $\color{#F99417}x^{(i)}$ vs $\color{#F99417}y^{(i)}$.

If the plot is a straight line (approx.), then the random variable $\color{#F99417}x$ is normally distributed.

> Q-Q plot can also help us in determining whether we have same distributions; given two random variables $\color{#F99417}x$ and $\color{#F99417}y$.

## Student's t-distribution or t-distribution ($\color{#F99417}t(\nu)$)

The t-distribution is a continuous probability distribution that serves as a generalization of the normal distribution. It shares the symmetry around zero and bell-shaped characteristics with the normal distribution but exhibits heavier tails. The t-distribution commonly arises in statistical inference, particularly when estimating the mean of a normal distribution with samples of different sizes.

$$\large\text{data }= \frac{x - \mu(x)}{\frac{S}{\sqrt{n}}}$$

Where $\color{#F99417}S$ is the sample standard deviation, $\color{#F99417}n$ is the sample size and $\color{#F99417}\mu(x)$ is the sample mean. The degrees of freedom $\color{#F99417}\nu$ is equal to $\color{#F99417}n-1$.

We can write the PDF of t-distribution as;

$$\large f(x) = \frac{\Gamma(\frac{\nu + 1}{2})}{\sqrt{\nu\pi}\Gamma(\frac{\nu}{2})}(1 + \frac{x^2}{\nu})^{-\frac{\nu + 1}{2}}$$

where $\color{#F99417}\Gamma$ is the gamma function. $\color{#F99417}\Gamma$ function is a generalization of the factorial function to non-integer values. It is defined as;

$$\large \Gamma(z) = \int_{0}^{\infty}x^{z-1}e^{-x}dx$$

![PDF of t-distribution](./img/student_pdf.png)

> PDF of t-distribution

We can write the CDF of t-distribution as;

$$\large F(x) = \frac{\Gamma(\frac{\nu + 1}{2})}{\sqrt{\nu\pi}\Gamma(\frac{\nu}{2})}\int_{-\infty}^{x}(1 + \frac{x^2}{\nu})^{-\frac{\nu + 1}{2}}dx$$

where $\color{#F99417}\nu$ is the degrees of freedom, $\color{#F99417} x$ is the random variable and $\color{#F99417}F(x)$ is the CDF of t-distribution.

![CDF of t-distribution](./img/student_cdf.png)

> CDF of t-distribution

|Property | Notation/Formula |
|:---:|:---:|
|Mean | $\color{#F99417}0$ for $\color{#F99417}\nu > 1$ |
|Median | $\color{#F99417}0$ |
|Mode | $\color{#F99417}0$ |
|Standard Deviation | $\color{#F99417}\sqrt{\frac{\nu}{\nu - 2}}$ for $\color{#F99417}\nu > 2$ |
|Variance | $\color{#F99417}\frac{\nu}{\nu - 2}$ for $\color{#F99417}\nu > 2$ |
|Skewness | $\color{#F99417}0$ for $\color{#F99417}\nu > 3$ |
|Kurtosis | $\color{#F99417}\frac{6}{\nu - 4}$ for $\color{#F99417}\nu > 4$ |

**As freedom increases, t-distribution approaches normal distribution.**

## Uniform Distribution

Uniform distributions are probability distributions with equally likely outcomes. It has constant probability. It can be of two types; `discrete uniform distribution` and `continuous uniform distribution`.

In the **discrete uniform distribution**, outcomes are countable and equally likely. That is they have same probability. 

> For example if we toss a fair coin, we will get either heads or tails. Both of these outcomes have the same probability of occurring. We wont get any other outcome.

In continuous uniform distribution, outcomes are continuous and infinite. 

> An example for continuous uniform distribution is a random number generator. 

We can write uniform distribution as $\color{#F99417}X \sim U(a, b)$, where $a$ and $b$ are the two parameters of the uniform distribution ( the lower and upper limits of the distribution).  The probability density function of a uniform distribution is given by;

$$f(x) = \frac{1}{b-a}$$

where $\color{#F99417}a \le x \le b$

**PDF** of a continuous uniform distribution is a horizontal line, which is denoted by; 

$$f(x) = \begin{cases} \frac{1}{b-a} & \text{if } a \le x \le b \\ 0 & \text{otherwise} \end{cases}$$

In terms of $\color{#F99417}\mu$ and $\color{#F99417}\sigma^2$, we can write the PDF as;

$$f(x) = \begin{cases} \frac{1}{2\sigma\sqrt{2}} & \text{for }-\sigma\sqrt{3}\le x - \mu \le \sigma\sqrt{3}\\ 0 &\text{otherwise} \end{cases}$$

![PDF diagram of uniform continuous distribution](./img/PDF_Continuous_Uniform.png)

**CDF** of a continuous uniform distribution is given by;

$$F(x) = \begin{cases} 0 & \text{if } x < a \\ \frac{x-a}{b-a} & \text{if } a \le x \le b \\ 1 & \text{if } x > b \end{cases}$$

In terms of $\color{#F99417}\mu$ and $\color{#F99417}\sigma^2$, we can write the CDF as;

$$F(x) = \begin{cases} 0 & \text{for } x-\mu < - \sigma\sqrt{3} \\ \frac{x-\mu + \sigma\sqrt{3}}{2\sigma\sqrt{3}} & \text{for } - \sigma\sqrt{3} \le x - \mu <  \sigma\sqrt{3} \\ 1 & \text{for } x - \mu \ge \sigma\sqrt{3} \end{cases}$$

![CDF of continuous uniform distribution from wikipedia](./img/CDF-Uniform-Continuous.png)

|Property | Notation/Formula |
|:---:|:---:|
|Mean, Median | $\color{#F99417}\frac{a+b}{2}$ |   
| Mode | $\color{#F99417}\texttt{Any value x in (a, b)} $ |
| Variance | $\color{#F99417}\frac{1}{12}(b-a)^2$ |
| Std. Deviation | $\large\color{#F99417}\frac{b-a}{\sqrt{12}}$ |
| Skewness | $\color{#F99417}0$ |
| Number of outcomes (n) | $\color{#F99417}b-a+1$ |
| Probability of each outcome | $\color{#F99417}\frac{1}{n}$ |

## Standard uniform distribution

Continuous uniform distribution with parameters $\color{#F99417}a = 0$ and $\color{#F99417}b = 1$ is called the standard uniform distribution. It is denoted as $\color{#F99417}U(0, 1)$. One interesting property of this distribution is that if $\color{#F99417}u_1$ has a standard uniform distribution, then $\color{#F99417}u_2 = 1 - u_1$ also has a standard uniform distribution. This property is called inversion method, where we can use continuous standard distribution to generate random number $\color{#F99417}x$ from any continuous distribution with CDF $\color{#F99417}F$.

**PMF** of discrete uniform distribution is given by;

$$f(x) = \begin{cases} \frac{1}{n} & \text{for } x = 1, 2, ..., n \\ 0 & \text{otherwise} \end{cases}$$

![PMF of discrete uniform distribution](./img/PMF_Discrete_Uniform.png)

**CDF** of discrete uniform distribution is given by;

$$F(x) = \begin{cases} 0 & \text{for } x < 1 \\ \frac{x}{n} & \text{for } 1 \le x \le n \\ 1 & \text{for } x > n \end{cases}$$

![CDF of discrete uniform distribution](./img/CDF_Discrete_Uniform.png)

> Uniform distribution is a special case of beta distribution. This makes uniform distribution to be used for modelling a variety of random phenomena.

## Binomial Distribution

Binomial distribution is a _discrete probability distribution_ with parameters : number of trials ($\color{#F99417}n$) and probability of success in each trial ($\color{#F99417}p$), that describes the number of successes in a sequence of $\color{#F99417}n$ independent experiments, each asking a `yes–no` question, and each with its own Boolean-valued outcome: `success or failure`.

A single success/failure experiment is called a `Bernoulli trial` or `Bernoulli experiment`, and a sequence of outcomes is called a `Bernoulli process`. For a single trial, we can write the probability of success as $\color{#F99417}p$ and the probability of failure as $\color{#F99417}q = 1 - p$. And in this case where $\color{#F99417}n=1$, the binomial distribution is a Bernoulli distribution.

We can write binomial distribution as $\color{#F99417}X \sim B(n, p)$. The **PMF** of binomial distribution is given by;

$$f(x) = \begin{cases} \large\binom{n}{x}p^x(1-p)^{n-x} & \text{for } x = 0, 1, ..., n  \end{cases}$$

**where**; 

- $\color{#F99417}\binom{n}{x}$ is the binomial coefficient (which is the number of ways of picking $\color{#F99417}x$ (number of successes) unordered outcomes from $\color{#F99417}n$ possibilities (the number of trials), also known as a combination)

- $\color{#F99417}P$ is the probability of success on a single trial 

![PMF of binomial distribution](./img/PMF_Binomial.png)

**PMF in binomial distribution gives the probability of getting $\color{#F99417}x$ successes in $\color{#F99417}n$ trials.**

> For example if we flip a coin 5 times and want to know the probability of getting 3 heads (F(3)), we can use the PMF of binomial distribution.

**CDF** of binomial distribution is given by;

$$F(x) = \begin{cases} 0 & \text{for } x < 0 \\ \sum_{k=0}^{x}\binom{n}{k}p^k(1-p)^{n-k} & \text{for } 0 \le x \le n \\ 1 & \text{for } x > n \end{cases}$$

![CDF of binomial distribution](./img/CDF_binomial.png)

**For binomial distribution CDF gives the the number of successes will be less than or equal to a certain value.**

> For example if we flip a coin 5 times and want to know the probability of getting 3 or fewer heads (F(0) + F(1) + F(2) + F(3)), we can use the CDF of binomial distribution.

|Property | Notation/Formula |
|:---:|:---:|
|Mean & Median | $\color{#F99417}np$ |
|Mode | $\color{#F99417}\lfloor (n+1)p \rfloor$ |
|Variance | $\color{#F99417}npq$ |
|Std. Deviation | $\color{#F99417}\sqrt{np(1-p)}$ |
|Skewness | $\color{#F99417}\frac{q-p}{\sqrt{npq}}$ |

## Bernoulli Distribution

Bernoulli distribution is a `discrete probability distribution` that can have only two possible outcomes. It is a special case of binomial distribution where a single experiment is conducted. It is a discrete distribution with two possible outcomes, `success` and `failure`. It is also called a `two-point distribution` or `two-point random variable`.

We can write Bernoulli distribution as $\color{#F99417}X \sim Bernoulli(p)$, where $p$ is the probability of success. **PMF** of a Bernoulli distribution is given by;

$$f(x) = p^x(1-p)^{1-x}$$

where $x \in \{0, 1\}$.

This can also be interpreted as;

$$PMF = \begin{cases} q & \text{if } k = 0 \\ p & \text{if } k = 1 \end{cases}$$

where $\color{#F99417}q = 1-p$ 

![PMF of Bernoulli distribution](./img/PMF_bernoulli.png)

|Property | Notation/Formula |
|:---:|:---:|
|Mean | $\color{#F99417}p$ |
|Median | $\color{#F99417}\begin{cases} 1 & \text{if } p \ge \frac{1}{2} \\ [0, 1] & \text{if } p = \frac{1}{2} \\ 0 & \text{if } p < \frac{1}{2} \end{cases}$ |
|Mode | $\color{#F99417}\begin{cases} 1 & \text{if } p \ge \frac{1}{2} \\ 0, 1 & \text{if } p = \frac{1}{2} \\ 0 & \text{if } p < \frac{1}{2} \end{cases}$ |
|Variance | $\color{#F99417}pq$ |
|Skewness | $\color{#F99417}\frac{q-p}{\sqrt{pq}}$ |

## Log Normal Distribution

The log-normal distribution is a continuous probability distribution that describes a random variable whose natural logarithm follows a normal distribution. In other words, if the random variable $\color{#F99417}X$ follows a log-normal distribution, then the variable $\color{#F99417}Y = \ln(X)$ is normally distributed.

We can write log-normal distribution as $\color{#F99417}X \sim LN(\mu, \sigma^2)$, where $\color{#F99417}\mu$ and $\color{#F99417}\sigma^2$ are the mean and variance of the variable's natural logarithm. The **PDF** of a log-normal distribution is given by;

$$f(x) = \frac{1}{x\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{ln(x)-\mu}{\sigma})^2}$$

where $x > 0$.

> It is similar to normal distribution, other than the ln(x) term. The PDF curve of a log-normal distribution is skewed to the right.

![PDF of log normal distribution](./img/PDF_log_normal.png)

> As $\color{#F99417}\sigma$ increases, the distribution gets more skewed to the right.

**CDF** of a log-normal distribution is given by;

$$F(x) = \frac{1}{2} + \frac{1}{2}erf(\frac{ln(x)-\mu}{\sigma\sqrt{2}})$$

where $x > 0$.

![CDF of log normal distribution](./img/CDF_log_normal.png)

|Property | Notation/Formula |
|:---:|:---:|
|Mean | $\color{#F99417}e^{\mu + \frac{\sigma^2}{2}}$ |
|Median | $\color{#F99417}e^{\mu}$ |
|Mode | $\color{#F99417}e^{\mu - \sigma^2}$ |
|Variance | $\color{#F99417}(e^{\sigma^2}-1)e^{2\mu + \sigma^2}$ |
|Skewness | $\color{#F99417}\frac{e^{\sigma^2}+2}{\sqrt{e^{\sigma^2}-1}}$ |

The log-normal distribution finds applications in various fields, including human behavior, finance, biology, chemistry, hydrology, and social sciences. It is particularly useful in modeling variables like the length of social media comments, user dwell time on a website, or the duration of chess games, among others. 

## Pareto Distribution

Pareto distribution is a continuous probability distribution used to model phenomena where a small number of items or events account for the majority of the impact or occurrences. It is commonly referred to as the "80-20" rule, where approximately 80% of the effects come from 20% of the causes. It is based on `Power Law`.

### Power law

Power law is a functional relationship between two quantities, where a relative change in one quantity results in a proportional relative change in the other quantity, independent of the initial size of those quantities: _**one quantity varies as a power of another**_. For instance, considering the area of a square in terms of the length of its side, if the length is doubled, the area is multiplied by a factor of four.

![Power law](./img/Power_law.png)

- This is what `pareto principle` also says. It says that 80% of the effects come from 20% of the causes.

We can write a pareto distribution as $\color{#F99417}X \sim Pareto(\alpha, x_m)$, where $\color{#F99417}\alpha$ is the shape parameter and $\color{#F99417}x_m$ is the scale parameter. **PDF** of a pareto distribution is given by;

$$f(x) = \frac{\alpha x_m^\alpha}{x^{\alpha+1}}$$

where $x \ge x_m$ and $\alpha > 0$.

![PDF of Pareto distribution](./img/PDF_Pareto.png)

**CDF** of a pareto distribution is given by;

$$F(x) = 1 - (\frac{x_m}{x})^\alpha$$

where $x \ge x_m$ and $\alpha > 0$.

![CDF of pareto distribution](./img/CDF_Pareto.png)

|Property | Notation/Formula |
|:---:|:---:|
|Mean | $\color{#F99417}\begin{cases} \infty & \text{if } \alpha \le 1 \\ \frac{\alpha x_m}{\alpha - 1} & \text{if } \alpha > 1 \end{cases}$ |
|Median | $\color{#F99417}x_m(\frac{1}{2})^{\frac{1}{\alpha}}$ |
|Mode | $\color{#F99417}x_m$ |
|Variance | $\color{#F99417}\begin{cases} \infty & \text{if } \alpha \in (1, 2] \\ \frac{x_m^2\alpha}{(\alpha-1)^2(\alpha-2)} & \text{if } \alpha > 2 \end{cases}$ |
|Skewness | $\color{#F99417}\begin{cases} \text{undefined} & \text{if } \alpha \le 3 \\ \frac{2(1+\alpha)}{\alpha-3}\sqrt{\frac{\alpha-2}{\alpha}} & \text{if } \alpha > 3 \end{cases}$ |

> We can use log-log plot to check if a distribution follows pareto distribution. If the plot is linear, then the distribution follows pareto distribution most of the times (not always). 

---

## Box Cox Transformation

Data transformation involves applying mathematical operations to a dataset without changing its essential characteristics. There are two main types: `linear transformations` and `non-linear transformations`.

**Linear Transformation**: Preserves the linearity of the data and only scales it.

**Non-Linear Transformation**: Alters the shape and distribution of the data using mathematical functions like logarithmic or exponential.

`Power transformation` is a type of non-linear transformation. It involve using a parameter $\color{#F99417}\lambda$ to perform a range of transformations to stabilize the variance of a non-normal distribution. For example taking the square root and the logarithms of observation in order to make the distribution normal belongs to the class of power transforms.

### Box Cox transformation

The Box-Cox transformation is a non-linear technique used to convert a log-normal distribution into a normal distribution and stabilize variance. It exposes the underlying normal distribution in the data without changing its fundamental nature.

We can describe box cox transformation as;

$$Y' = \frac{Y^λ - 1}{\lambda} $$

where $\color{#F99417}Y$ is the response variable and $\color{#F99417}λ$ is the transformation parameter.

Mathematically we can write box cox transformation as;

$$y(\lambda) = \begin{cases} \frac{y^\lambda - 1}{\lambda} & \text{if } \lambda \ne 0 \\ \ln(y) & \text{if } \lambda = 0 \end{cases}$$

where $\color{#F99417}y > 0$.

Selecting the optimal value of $\color{#F99417}\lambda$ is important. We may use maximum likelihood estimation to find the optimal value of $\color{#F99417}\lambda$. Generally we select the value of $\color{#F99417}\lambda$ to be between [$\color{#F99417} -5, 5$]. Common values for $\color{#F99417}\lambda$ are;

| value | Transformation |
|:---:|:---:|
| $\color{#F99417}\lambda = -1$ | reciprocal transformation  |
| $\color{#F99417}\lambda = -0.5$ | reciprocal square root transformation  |
| $\color{#F99417}\lambda = 0$ | log transformation  |
| $\color{#F99417}\lambda = 0.5$ | square root transformation  |
| $\color{#F99417}\lambda = 1$ | no transformation  |
| $\color{#F99417}\lambda = 2$ | square transformation  |

> See the notebook for implementation of box cox transformation using python.

**Limitations of box cox transformation**

- Defined only for strictly positive values of $\color{#F99417}y$.

- It may not fully address situations where data variance is not constant (homoscedasticity assumption).

- It may not always be applicable.

- Selecting an optimal $\color{#F99417}\lambda$ and interpreting the transformed values can be challenging.

---

## Law of Large Numbers

Law of large numbers is a foundational theorem to both probability and statistics. It states that the average result from repeating an experiment multiple times will better approximate the true or expected underlying result. This theorem has important implications in applied machine learning, such as the law of large numbers is critical for understanding the selection of training datasets, test datasets, and in the evaluation of model skill.

Some important terms related to law of large numbers are;

- **Independent and Identically Distributed (IID)** : A sequence of random variables is independent and identically distributed if each random variable has the same probability distribution as the others and all are mutually independent (The result of one trial does not depends on another one).

- **Regression to the mean** : The phenomenon that if a variable is extreme on its first measurement, it will tend to be closer to the average on its second measurement. It is also called **reversion to the mean**.

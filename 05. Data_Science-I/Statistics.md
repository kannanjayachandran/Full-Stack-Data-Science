<!-- 
    Author : Kannan Jayachandran
    File : Statistics.md
 -->

 <h1 align="center" style="color: orange"> STATISTICS </h1>

 ## Population vs Sample  

- The entire group of objects or events under study, often denoted as  $\color{#F99417}N$ 

- A subset of the population, typically represented as $\color{#F99417}n$.   

![population and sample](./img/population.png)

- **Univariate analysis** : Analysis of a single variable.

## Mean

The average of data, calculated by summing all data points and dividing by the number of data points.

$$\mu = \frac{\sum_{i=0}^{n}x_i}{n}$$

## Median

The middle value in data, found by sorting the data and selecting the middle value. For even-sized data, it's the average of the two middle values.

$$median = \frac{n+1}{2}^{th} value$$

## Mode

The most frequent value in the data, determined by counting occurrences.

## Variance ($\sigma^2$)

A measure of data spread, computed as the average of squared differences of data points from the mean. 

$$var = \frac{1}{n}\sum_{i=0}^{n}(x_i-\mu)^2$$

## Standard Deviation ($\sigma$)

The square root of variance, often used to measure data spread (Average distance of data points from the mean). The coefficient of variation (cv) is the ratio of standard deviation to the mean.

$$\sigma = \sqrt{\frac{1}{n}\sum_{i=0}^{n}(x_i-\mu)^2}$$

$$cv = \frac{\sigma}{\mu}$$

> Median absolute deviation is similar to standard deviation but it is more robust to outliers. It is calculated by taking the median of the absolute values of the deviations from the median. **MAD = median(|x_i - median(x)|)** { `statmodels.robust.mad()` }

## Range 

The difference between the highest and lowest values in a dataset, showing data spread.

$$range = max(x) - min(x)$$

## Percentage

A ratio of count of an event to the total number of events, expressed as a number between 0 and 1 or between 0 and 100.

$$percentage = \frac{count}{total}$$

## Percentile

A value below which a certain percentage of data falls. The 50th percentile is the median. Percentiles like 25th, 75th, and 100th are called "_Quantiles_".

$$\text{Percentile} = \frac{\text{Number of values below the given value}}{\text{Total number of values}}$$

## Inter Quartile Range (IQR)

The range between the 75th and 25th percentiles (middle 50% data), used to identify outliers.

$$IQR = Q_3 - Q_1$$

## Sampling

The process of selecting a subset of individuals from a population for estimating population characteristics. It should aim to be 

- representative 

- Have an appropriate sample size

- Randomly selected

- Chosen without replacement.

> These might not always be possible, but we should try to achieve as many as possible.

### Types of Sampling

1. **Simple Random Sampling**: Every member of the population has an equal chance of selection.

2. **Stratified Sampling**: Population divided into non-overlapping groups.

3. **Systematic Sampling**: Members selected at regular intervals.

4. **Convenience Sampling**: Selecting individuals with expertise.

5. **Cluster Sampling**: Population divided into clusters, with random cluster selection.

## Selection Bias

When a sample misrepresents the populations we call it a **`selection bias`** or **`sampling bias`**. Extensive hunting through data in search of something interesting is called **`data snooping`**. Bias or non-reproducibility resulting from repeated data modeling, or modeling data with large numbers of predictor variables is called **`vast search effect`**. 

## Sampling Distribution

The probability distribution of sample statistics from random samples of a population. The process of sampling distribution is as follows;

- Consider a population with any distribution from which samples are drawn.

- Select random sample of size `n` from the population.

- Calculate the mean of the sample. 

- Repeat the above steps `M` times.

- Develop a frequency distribution of the sample means.

- Plot the frequency distribution of the sample statistic.

**If we have `M` sample means $\color{#F99417}\bar x_i = \bar x_1, \bar x_2, \bar x_3, ...\bar x_n$ then the sampling distribution of the sample means is the probability distribution of these sample means.**

---
> Various distributions and their properties are covered in the [Probability](Probability.md) section.
---

## Central Limit Theorem (CLT)

CLT is one of the most important theorems in statistics. It states that, if we have a population with finite mean $\color{#F99417}\mu$ and variance $\color{#F99417}\sigma^2$, with sufficiently large random samples of size $\color{#F99417}n$  with replacement, taken $\color{#F99417}m$ times, then the distribution of sample means will approximate a normal distribution.

$$OR$$

**_As the size of the sample increases, the distribution of the mean across multiple samples will approximate a Gaussian distribution. OR ; The distribution of errors from estimating the population mean will be normally distributed_.**

$$ \bar x_i \sim N(\mu, \frac{\sigma^2}{n})\;\;as\;\; {n\rightarrow \infin}$$ 

where $\color{#F99417}\bar x_i$ is the sampling distribution of the sample means, $\color{#F99417}N$ is the normal distribution with mean $\color{#F99417}\mu$ (which is same as the population mean) and variance $\color{#F99417}\frac{\sigma^2}{n}$ (where $\color{#F99417}\sigma^2$ is the population variance and $\color{#F99417}n$ is the sample size).

> We generally consider CLT to be valid if the sample size ($\color{#F99417}n$) is greater than 30.

## Statistical Significance

In both probability and statistics, we are dealing with uncertainty and our attempt to comprehend it. Hence it is extremely important to quantify the uncertainty. We want to determine whether the results observed in an experiment or study are likely due to a real effect or simply due to random chance. We use **statistical significance tests** to make this determination. That is to conclude whether the observed results are **statistically significant** or not.

_**Statistical significance testing involves assessing whether the differences or relationships observed in a sample are likely to exist in the population from which the sample is drawn.**_

We use the **CLT** and our understanding about **a well known distribution** (like _normal distribution_) to make inferences about the population. 

<!-- mention hypothesis testing here -->

## Confidence Interval

A statistical concept used to estimate the range within which a population parameter (mean or proportion) is likely to fall. 

- It provides a range of values rather than a single point estimate.

- It quantifies the uncertainty associated with the estimate.   

**A range of values, derived from sample data, that is used to estimate an unknown population parameter. It has a specified level of confidence associated with it, (something like 95%) that the parameter lies within the interval.**

## Statistical Hypothesis Testing

Hypothesis testing is a framework used to provide confidence or likelihood about making inference from a sample to a larger population. The assumptions we make are called **hypothesis** and the statistical tests used for this are called **statistical hypothesis tests**. Hypothesis testing is something which we will be spending a lot of time as it is crucial to understand it thoroughly. Before diving deeper into hypothesis testing and leaning how to perform it, we need to understand the following terms;

### Proof by Contradiction

A mathematical proof that establishes the truth of a statement by first assuming that its opposite is true and then showing that this assumption leads to a contradiction. Hypothesis testing is practical application of this methodology.

### Hypothesis

We make an assumption and then investigate about that assumption and collect evidence to either prove that assumption or to discard it. It is the method of **Science**. Hypothesis in hypothesis testing also follows the same ideology. We make an assumption about the population and then collect evidence from the sample to either prove or discard the assumption.













<!-- 

## Hypothesis Testing

We use _statistical hypothesis testing_ or _significance tests_ to provide confidence or likelihood about making inferences from a sample of data to a larger population. We calculate some quantity under some assumption, then we interpret the result and either reject or fail to reject the assumption.

The assumption is called the _null hypothesis_ ($\color{#F99417}H_0$). The violation of the null hypothesis is known as the alternative hypothesis ($\color{#F99417}H_1$). 

> We use the `proof by contradiction` method to test the hypothesis. We assume that the null hypothesis is true and then we try to prove that it is false. If we fail to prove that the null hypothesis is false, then we accept the null hypothesis











Key Steps in significance testing
Formulate Hypotheses:

Null Hypothesis (H₀): It assumes there is no effect or no difference.
Alternative Hypothesis (H₁): It states there is a significant effect or difference.
Select Significance Level (α):

This is the threshold for deciding whether the results are statistically significant. Common values are 0.05 or 5%.
Collect Data:

Gather and analyze the data relevant to the study.
Perform Statistical Test:

Popular tests include t-tests, chi-square tests, and ANOVA, depending on the nature of the data.
Calculate P-Value:

The p-value represents the probability of obtaining the observed results or more extreme, assuming the null hypothesis is true. A low p-value (typically less than α) suggests evidence against the null hypothesis.
Make a Decision:

If the p-value is less than α, reject the null hypothesis in favor of the alternative hypothesis. Otherwise, fail to reject the null hypothesis.



Key Elements:
Point Estimate:

This is the single value calculated from the sample data that serves as the best guess for the population parameter. For example, the sample mean for estimating the population mean.
Margin of Error:

The margin of error is the range added to and subtracted from the point estimate to create the interval. It is influenced by the variability in the data and the chosen level of confidence.
Level of Confidence:

This is the probability that the interval will contain the true parameter. Common choices are 90%, 95%, and 99%.
Formula:
The general formula for a confidence interval is:
Confidence Interval
=
Point Estimate
±
Margin of Error
Confidence Interval=Point Estimate±Margin of Error

The margin of error depends on the standard error of the estimate and the critical value associated with the chosen confidence level.











.

1. First we must interpret the statistical test result in order to make claims. There are two common ways to interpret the result of a statistical test;

- **p-value**: The probability of observing a test statistic as extreme as the one observed, by chance, given that the null hypothesis is true. If the p-value is less than the significance level ($\color{#F99417}\alpha$), then we reject the null hypothesis. A common value used for $\color{#F99417}\alpha$ is 0.05 or 5% (arbitrary value).

$$\text{p-value} \le \color{#F99417}\alpha \text{ significant result, reject null hypothesis}$$

$$\text{p-value} > \color{#F99417}\alpha \text{ not significant result, fail to reject the null hypothesis}$$

 Given the observed sample data, we can find the confidence level of the hypothesis by subtracting the significance level from 1.

$$\text{Confidence Level} = 1 - \alpha$$ 

- **Critical Value**: The value of the test statistic that separates the region of acceptance from the region of rejection. If the test statistic is less than the critical value, then we fail to reject the null hypothesis. If the test statistic is greater than the critical value, then we reject the null hypothesis.

$$\text{test statistic} <  \color{#F99417}\text{ critical value: fail to reject null hypothesis}$$

$$\text{test statistic} \ge \color{#F99417}\text{ critical value: reject null hypothesis}$$



## Errors in Hypothesis Testing

- **Type I Error**: Rejecting the null hypothesis when it is true. The probability of making a type I error is $\color{#F99417}\alpha$. Also known as a false positive.

- **Type II Error**: Failing to reject the null hypothesis when it is false. The probability of making a type II error is $\color{#F99417}\beta$. Also known as a false negative.

## Degree of Freedom

While calculating statistics, we must include some information about the population. One way to do this is via degrees of freedom. It is the number of independent information that go into the estimate of a parameter from the sample data. 

$$\texttt{df or }\nu \le n $$

Where $\color{#F99417}n$ is the sample size.

If we calculate the mean of the sample data using the following formula; 

$$\bar x = \frac{\sum_{i=0}^{n}x_i}{n}$$

Then the degree of freedom is $\color{#F99417}n$.

But if we calculate the variance of the sample data using the following formula;

$$s^2 = \frac{\sum_{i=0}^{n}(x_i-\bar x)^2}{n-1}$$

Then the degree of freedom is $\color{#F99417}n-1$.
 -->

<!-- Append current work -->

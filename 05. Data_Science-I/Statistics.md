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

## Percentile

A value below which a certain percentage of data falls. The 50th percentile is the median. Percentiles like 25th, 75th, and 100th are called "_Quantiles_".

## Inter Quartile Range (IQR)

The range between the 75th and 25th percentiles (middle 50% data), used to identify outliers.

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

> When a sample misrepresents the populations we call it a **sample bias** or **sampling bias**.

## Sampling Distribution

The probability distribution of sample statistics from random samples of a population. The process of sampling distribution is as follows;

- Consider a population with any distribution from which samples are drawn.

- Select random sample of size `n` from the population.

- Calculate the mean of the sample. 

- Repeat the above steps `M` times.

- Develop a frequency distribution of the sample means.

- Plot the frequency distribution of the sample statistic.

**If we have `M` sample means $\color{#F99417}\bar x_i = \bar x_1, \bar x_2, \bar x_3, ...\bar x_n$ then the sampling distribution of the sample means is the probability distribution of these sample means.**

## Central Limit Theorem (CLT)

CLT is one of the most important theorems in statistics. It states that, if we have a population with finite mean $\color{#F99417}\mu$ and variance $\color{#F99417}\sigma^2$, with sufficiently large random samples of size $\color{#F99417}n$  with replacement, taken $\color{#F99417}m$ times, then the distribution of sample means will approximate a normal distribution.

$$OR$$

**_As the size of the sample increases, the distribution of the mean across multiple samples will approximate a Gaussian distribution. OR ; The distribution of errors from estimating the population mean will be normally distributed_.**

$$ \bar x_i \sim N(\mu, \frac{\sigma^2}{n})\;\;as\;\; {n\rightarrow \infin}$$ 

where $\color{#F99417}\bar x_i$ is the sampling distribution of the sample means, $\color{#F99417}N$ is the normal distribution with mean $\color{#F99417}\mu$ (which is same as the population mean) and variance $\color{#F99417}\frac{\sigma^2}{n}$ (where $\color{#F99417}\sigma^2$ is the population variance and $\color{#F99417}n$ is the sample size).

> We generally consider CLT to be valid if the sample size ($\color{#F99417}n$) is greater than 30.

<!-- Append current work -->

<!-- 
    Author : Kannan Jayachandran
    File : Statistics.md
    Section : Mathematics for data science and machine learning
 -->

 <h1 align="center"> STATISTICS </h1>

 ### 1. Introduction
 - [What is statistics](#statistics)
 - [Types: Descriptive vs Inferential](#descriptive-vs-inferential-statistics)

---

### 2. Population vs Sample
- [Definition](#population-vs-sample)

---

### 3. Descriptive Statistics
**A. Measures of Central Tendency**
- [Mean](#mean)
- [Median](#median)
- [Mode](#mode)

**B. Measures of Dispersion**
 - [Variance](#variance-)
 - [Standard Deviation](#standard-deviation-)
 - [Coefficient of Variation](#coefficient-of-variation-cv)
 - [Median Absolute Deviation](#median-absolute-deviation-mad)
 - [Range](#Range)
 - [Percentage](#Percentage)
 - [Percentile](#Percentile)
 - [Inter Quartile Range](#inter-quartile-range-iqr)
---

### 4. Data Types Scales of measurement

- [Quantitative Data : Continuous and Discrete](#quantitative-data)
- [Qualitative Data : Nominal and Ordinal](#qualitative-data)
- [Interval Data](#interval-data)
- [Ratio Data](#ratio-data)

---

## Statistics

Statistics helps you understand data you already have.

## Descriptive vs Inferential Statistics

**Descriptive statistics** involves summarizing and presenting the characteristics of a dataset through numerical and visual form like mean, median, mode, and standard deviation, as well as charts and graphs, focusing solely on the data at hand.

**Inferential statistics**, on the other hand, uses data from a sample to make broader conclusions, predictions, or generalizations about a larger population, often employing techniques like hypothesis testing and confidence intervals to account for the uncertainty inherent in sampling.

 ## Population vs Sample  

- **Population** : The entire group of objects or events under study. Denoted by $\color{#F99417}N$ 

- **Sample** : A subset of the population used to represent the group. Denoted by $\color{#F99417}n$.   

![population and sample](./img/population.png)

> _Example_ : Studying the height of all adults in a country (population) by measuring the height of 1,000 randomly chosen adults (sample).

## Measure of Central Tendency

They describe where the most data lies. Common measures include:

- Mean – The arithmetic average.

- Median – The middle value.

- Mode – The most frequent value.

## Mean

The average value of a dataset, calculated by summing all data points and dividing by the number of observations.

$$\mu = \frac{\sum_{i=0}^{n}x_i}{n}$$

- Sensitive to outliers.

- Commonly used in symmetric distributions.

## Median

The value that separates the higher half from the lower half of the dataset. 

- Sort the data in ascending order and find the middle value.

- If the number of data points is even, take the average of the two central values.

$$\text{median position} = \frac{n+1}{2}^{th} value$$

- More robust to outliers compared to the mean.

- Ideal for skewed distributions.

## Mode

The value that appears most frequently in a dataset.

- A dataset can have no mode, one mode (unimodal), two modes (bimodal), or multiple modes (multimodal).

- Useful for categorical and nominal data.

## Measures of Dispersion (Spread)

These metrics quantify the variability or spread in a dataset. They help assess how much the data deviates from the central tendency. It includes:

- Variance : Measure of how far numbers are spread out from their mean.

- Standard Deviation : Square root of variance, indicating average distance from the mean.

- Range : Difference between the maximum and minimum values.

- Interquartile Range (IQR) : Difference between the 75th and 25th percentiles, indicating the spread of the middle 50% of data.

## Variance ($\sigma^2$)

A measure of how much the values in the dataset deviate from the mean, squared to give weight to larger deviations (Without squaring values less than mean and those values greater than mean will cancel out each other) .

$$var = \frac{1}{n}\sum_{i=0}^{n}(x_i-\mu)^2$$

where $\mu$ is the mean value.

- Units are squared (e.g., cm² if data is in cm).

- Higher variance indicates more spread.

## Standard Deviation ($\sigma$)

The square root of variance, giving a measure of average deviation from the mean in the original units.

$$\sigma = \sqrt{\frac{1}{n}\sum_{i=0}^{n}(x_i-\mu)^2}$$

- Most commonly used measure of spread.

- Used in constructing confidence intervals and Z-scores.

## Coefficient of Variation (CV)

A normalized measure of dispersion, showing standard deviation relative to the mean.

$$cv = \frac{\sigma}{\mu}$$

- Useful for comparing variability between datasets with different units or scales.

## Median Absolute Deviation (MAD)

A robust measure of variability that is less affected by outliers.

$$\text{Median Absolute Deviation} = median(x_i - median(x))$$

- Recommended when data contains extreme values.

- Used in robust statistics.

> In Python: `statsmodels.robust.mad(x)`

## Range 

The range is the simplest measure of dispersion. It represents the spread of a dataset by calculating the difference between the maximum and minimum values.

$$range = max(x) - min(x)$$

- Sensitive to outliers, since it only considers the extreme values.

- Gives a quick sense of the data spread but does not provide information about the distribution within that spread.

## Percentage

A percentage expresses a proportion or a fraction relative to a total, often used to compare quantities.

$$Percentage = \frac{Count}{Total}\times 100$$

- Values are usually expressed between 0 and 100.

- Commonly used to report data summaries like success rates, error rates, and proportions.

## Percentile

A percentile indicates the value below which a given percentage of observations in a group falls.


$$\text{Percentile} = \frac{\text{Number of values below a given value}}{\text{Total number of values}}\times100$$

- 50th percentile is equivalent to the median.

- 25th percentile ($Q_1$), 75th percentile ($Q_3$), and other such cut-offs are called _quantile_.

- **Quartiles**: Special percentiles that divide the data into four equal parts (Q1, Q2, Q3).

- **Deciles**: Divide the data into ten equal parts.

- **Percentiles**: Divide the data into one hundred equal parts.

> For example; For the dataset `[7 7 8 4 3 3 5 2 1 4 5 4 2 3 6 1]`, the 25th percentile value is 2.75. It means 25% of the values in the dataset lies below 2.75.

## Inter Quartile Range (IQR)

The interquartile range measures the spread of the middle 50% of the data. It is useful for identifying the presence of outliers.

$$IQR = Q_3 - Q_1$$

Where:

- $Q_1$ = 25th percentile and $Q_3$ = 75th percentile

- Lower Bound for outlier detection: $Q_1 - 1.5 \times IQR$

- Upper Bound for outlier detection: $Q_3 + 1.5 \times IQR$

If any data point falls outside these bounds, it is considered a potential outlier.

> Example: If $Q_1 = 20$ and $Q_3 = 40$:
> 
> $IQR = 40 - 20 = 20$
> 
> Outlier thresholds would be:
> 
> Lower Bound = $20 - 1.5(20) = -10$
>
> Upper Bound = $40 + 1.5(20) = 70$
> 
> Thus, any data point below -10 or above 70 would be considered an outlier.

<!-- End of descriptive statistics -->

## Data Types & Scales of measurement

Data refers to a collection of values, measurements, or observations that can be analyzed statistically. It is broadly categorized into two main types: `Quantitative` (Numerical) and `Qualitative` (Categorical) data.

### Quantitative data

Quantitative data consists of numerical values that represent measurable quantities. It is further divided into:

- **Discrete data**: Represents countable values that occur in whole numbers.
> Example: Number of students in a class, number of goals scored.

- **Continuous data**: Represents measurable quantities that can take any value within a range, including fractions and decimals.
> Example: Height of students, time taken to complete a task.

### Qualitative data

Qualitative data (also called categorical data) describes characteristics or attributes that cannot be measured numerically. It is classified into:

- **Nominal data**: Represents categories or labels without any inherent order.
> Example: Gender, car brands, blood types.

- **Ordinal data**: Represents categories with a meaningful order or ranking, but the differences between adjacent ranks are not consistent or measurable.
> Example: Movie ratings (good, better, best), education levels (high school, undergraduate, postgraduate).

![Quantitative vs Qualitative data](./img/Types_of_data.png)

In statistical measurement, data can also be classified based on scales of measurement, which include:

### Interval Data

Interval data has ordered categories with equal intervals between values but lacks a true (absolute) zero point. Zero does not imply "none" or the absence of quantity. Arithmetic operations like addition and subtraction are valid, but ratios (multiplication/division) are not meaningful.

> Example, temperature in Celsius or Fahrenheit, IQ score, calendar years.

### Ratio Data

Ratio data includes all the properties of interval data plus a meaningful zero point, which indicates the complete absence of the quantity. All mathematical operations are valid, including ratios.

> Example : Age, height, weight, income, distance, temperature in Kelvin, number of items produced.

<!-- Inferential Statistics -->

## Sampling

Sampling is the process of selecting a subset (called a sample) from a larger group (called a population) in order to make statistical inferences about the population. It is a core concept in statistics because collecting data from an entire population is often impractical or impossible.

A good sample should ideally be:

- Representative of the population

- Of an appropriate size

- Randomly selected

- Selected without replacement (each member has one chance to be selected)

> While it's not always possible to meet all these criteria, striving for them helps ensure the validity of conclusions drawn from the data.

### Census vs Sampling

| Aspect      | Census                               | Sampling                           |
|-------------|--------------------------------------|------------------------------------|
| Definition  | Collects data from the entire population | Collects data from a subset of the population |
| Time        | Time-consuming                       | Time-efficient                     |
| Cost        | Expensive                            | Economical                         |
| Accuracy    | High (if conducted properly)         | Depends on sample quality          |
| Feasibility | Often impractical for large populations | Practical and scalable            |

### Types of Sampling

1. **Simple Random Sampling**: Each individual in the population has an equal and independent chance of being selected.
> Example: Using a random number generator to pick participants from a list.

2. **Stratified Sampling**: The population is divided into strata (non-overlapping subgroups) based on a specific characteristic (e.g., age, gender), and random samples are drawn from each stratum.
> Useful when certain subgroups need adequate representation.

3. **Cluster Sampling**: The population is divided into clusters (usually geographically), and entire clusters are randomly selected for sampling.
> Example: Randomly selecting schools and surveying all students within those schools.

4. **Systematic Sampling**: Every kth element is selected from an ordered list after choosing a random starting point.
> Example: Selecting every 10th person on a voter registration list.

5. **Convenience Sampling**: Sample is drawn from a group that is easy to access or contact.
> Example: Surveying students outside a classroom. Not ideal; prone to bias and lacks generalizability.

6. **Snowball Sampling** : Initially selected participants recruit additional participants from their acquaintances. Common in studies of hidden or hard-to-reach populations.
> Example: Research on people with rare diseases or specific social networks.

## Sampling Errors and Bias

In any statistical study, sampling error refers to the difference between a sample statistic and the actual population parameter it is intended to estimate. 

While random sampling error is expected and quantifiable, **sampling bias** is a systematic error that leads to an unrepresentative sample and can severely compromise the validity of conclusions.

## Types of Sampling Bias

1. **Selection Bias (Sampling Bias)** : Occurs when the method of selecting participants causes certain groups to be overrepresented or underrepresented. This results in a sample that does not accurately reflect the population.
> Example: Surveying only internet users to understand general public opinion.

2. **Non-Response Bias** : Arises when a significant portion of selected individuals fail to respond, and those who do respond differ meaningfully from non-respondents.

> Example: In a health survey, people with health issues may be less likely to respond.

3. **Under-coverage** : Happens when some groups within the population are left out of the sampling process entirely.

> Example: Using a landline-based survey might under-represent younger populations who mostly use mobile phones.

4. **Survivorship Bias** : Occurs when only "survivors" or successful outcomes are considered, ignoring those that dropped out or failed.

> Example: Analyzing only profitable companies to assess industry trends ignores those that went bankrupt.

### Other Sources of Bias

1. **Data snooping or P-Hacking** : Involves extensively searching through data to find statistically significant results, even if they occurred by chance. It increases the risk of `Type I errors` (false positives).

> Example: Testing multiple variables without predefining a hypothesis and reporting only those with significant p-values.

2. **Vast Search Effect (Model Overfitting Bias)** : It emerges when multiple models or large numbers of predictors are tested on the same dataset. The more models you test, the more likely you are to find spurious patterns that don't generalize.

> Example: Testing 100 different models on the same dataset and choosing the one that fits best — even if it fits by coincidence.


| Type of Bias             | Description                                                      | Example                                              |
|--------------------------|------------------------------------------------------------------|------------------------------------------------------|
| Selection Bias           | Non-random selection method                                      | Only urban participants surveyed                     |
| Non-Response Bias        | Differences between respondents and non-respondents              | Sick individuals less likely to participate          |
| Undercoverage            | Entire subgroup excluded                                         | No representation of rural areas                     |
| Survivorship Bias        | Focus only on successful observations                            | Only successful startups are analyzed                |
| Data Snooping            | Searching for patterns without predefining hypotheses            | Testing many correlations to find a significant one  |
| Vast Search Effect       | Bias from trying many models/variables without validation        | Overfitting with many features                       |

## Sampling Distribution

A **sampling distribution** is the **probability distribution of a given statistic based on repeated random samples from a population**.

> A probability distribution describes the likelihood of each possible outcome of a random variable. It can be a list, table, or function showing what values the variable can take and how often each value (or range of values) is expected to occur.
>
> statistic can be anything like the mean, variance, proportion, etc.

- It tells us how the statistic (e.g., sample mean) would behave if we repeated the sampling process many times.

It is easier to understand this conceptually;  

*Suppose we have a population with any distribution (normal, skewed, etc.), and we repeatedly draw **random samples of the same size** from this population. For each sample, we compute a statistic — like the **sample mean** $\color{#F99417}\bar{x}$. If we plot the values of this statistic from all the samples, the resulting distribution is the **sampling distribution** of that statistic.*

- The **mean** of the sampling distribution of the sample mean is equal to the **population mean** ($\color{#F99417}\mu$).

- The **standard deviation** of the sampling distribution of the sample mean is called the **standard error** ($\color{#F99417}SE$), calculated as:

$$SE = \frac{\sigma}{\sqrt{n}}$$

Where:
- $\sigma$ = population standard deviation  
- $n$ = sample size  

- As `n` increases, the sampling distribution of the mean becomes **narrower and more symmetric** (approaching normality by the **Central Limit Theorem**).

## Central Limit Theorem (CLT)

The Central Limit Theorem is one of the foundational results in statistics. It describes the behavior of the sampling distribution of the sample mean, particularly when the sample size becomes large.

If we repeatedly draw random samples of size $\color{#F99417}n$ (with replacement) from a population with a finite mean $\color{#F99417}\mu$ and finite variance $\color{#F99417}\sigma^2$, then as $\color{#F99417}n$ becomes large, the sampling distribution of the sample means will approach a normal distribution, regardless of the shape of the original population distribution.

$$\large \bar x_i \sim N(\mu, \frac{\sigma^2}{n})\;\;as\;\; {n\rightarrow \infin}$$

Where:

$\bar{x}_i$ = mean of the $i^{th}$ sample

$N(\mu, \frac{\sigma^2}{n})$ = normal distribution with mean $\mu$ and variance $\frac{\sigma^2}{n}$

$\mu$ = population mean

$\sigma^2$ = population variance

$n$ = sample size

> ***Intuition**: As sample size increases, the distribution of the sample means becomes more and more bell-shaped, even if the underlying population distribution is skewed, uniform, or otherwise non-normal.*

This property is powerful because it allows us to use normal distribution techniques (like $z$-scores, confidence intervals, hypothesis tests) even when dealing with unknown or non-normal populations — as long as we have sufficiently large sample sizes.

### Conditions for CLT

- The population must have a **finite mean** and **finite variance**.

- Samples must be independent and identically distributed (i.i.d.).

- Sampling is typically assumed with replacement (or from a large population if without replacement).

- The sample size $\color{#F99417}n$ should generally be ≥ 30 for CLT to be reliable, though for near-normal populations, smaller $\color{#F99417}n$ may suffice.

### Key Implications

- The mean of the sampling distribution equals the population mean:

$$\mu_{\bar x} = \mu$$

- The standard deviation of sampling distribution (standard error) is: 

$$\sigma_{\bar x} = \frac{\sigma}{\sqrt{n}}$$

- Enables us to perform inferential statistics (e.g., confidence intervals, hypothesis testing) using the normal distribution.

**Example**

Suppose you measure the average height of adult men in a country. The population distribution is slightly right-skewed. If you collect 1,000 random samples of 50 men and compute the mean height in each sample, the histogram of those 1,000 sample means will closely resemble a normal distribution, even though the original population is skewed.

## Statistical Significance

In statistics, statistical significance helps us quantify uncertainty and decide whether the patterns observed in data are due to chance or reflect real effects in the population.

It answers the core question:

> "**Are the observed results likely due to random variation, or are they significant enough to generalize to the population?**"

<!-- Sampling introduces variability. Statistical tests help determine whether observed differences or associations exceed what we would expect due to random chance alone. One among them is **Significance Testing**. It uses probability theory (often the normal, t, or chi-square distributions) to evaluate whether the sample statistic differs significantly from a hypothesized value or another group.

If a test statistic falls in the critical region (i.e., the tail of the distribution under a given significance level), we reject the null hypothesis, suggesting that the result is unlikely due to chance alone. -->

## Confidence Interval

A confidence interval estimates an unknown population parameter (like mean, proportion) by providing a range of plausible values calculated from sample data.

> A 95% confidence interval for the mean means:
>
> "If we repeated the sampling process many times, approximately 95% of the calculated intervals would contain the true population mean."
>
> It does NOT mean there is a 95% chance that the true mean lies within one particular computed interval — that’s a common misinterpretation.

**Example**

Imagine you're trying to estimate the average height of all adults in your city. Since it's impractical to measure every individual, you select a sample—a smaller group of people—and calculate their average height.

Now, instead of assuming this sample average is exactly the true population average, you calculate a confidence interval. This gives you a range—from a lower limit to an upper limit—within which you believe the true average height likely falls. For example you might say;

> Based on this sample, I am 95% confident that the average height of all adults in the city lies between 165 cm and 172 cm.

The "95% confidence" part means this:

> If you repeated this sampling process 100 times, each time drawing a different sample and computing a new confidence interval, then approximately 95 out of those 100 intervals would capture the true average height of the population.

In essence, a confidence interval does not guarantee that the true mean lies within any one specific interval, but rather reflects the reliability of the method used to estimate it.

<!-- EDITED |^ -->

## Statistical Significance

In probability and statistics, our focus is on understanding and dealing with **uncertainty**. Quantifying this uncertainty is crucial. The goal is to "*determine if the outcomes observed in an experiment or study are likely a result of a genuine effect or mere random chance*". This determination is made through statistical significance tests to conclude whether the observed results are indeed **statistically significant**.

- **Statistical significance testing** involves evaluating whether the differences or relationships observed in a sample are likely to exist in the population from which the sample is drawn. 

- The **Central Limit Theorem** (CLT) and our knowledge of well-known distributions, such as the normal distribution, guide us in making inferences about the population. 

## Confidence Interval

The confidence interval is a statistical concept used to estimate the range within which a population parameter (mean or proportion) is likely to lie:

- It provides a range of values instead of a single point estimate.

- It quantifies the uncertainty associated with the estimate.

**It's a range derived from sample data, aiming to estimate an unknown population parameter with a specified level of confidence usually expressed in percentage**

## Statistical Hypothesis Testing

Hypothesis testing is a framework used to provide confidence or likelihood about making inference from a sample to a larger population. The assumptions we make are called **hypothesis** and the statistical tests used for this are called **statistical hypothesis tests**. Hypothesis testing is something which we will be spending a lot of time as it is crucial to understand it thoroughly. Before diving deeper into hypothesis testing and leaning how to perform it, we need to understand the following terms;

### Proof by Contradiction

A mathematical proof that establishes the truth of a statement by first assuming that its opposite is true and then showing that this assumption leads to a contradiction. Hypothesis testing is practical application of this methodology.

### Hypothesis

We make an assumption and then investigate about that assumption and collect evidence to either prove that assumption or to discard it. It is the method of **Science**. Hypothesis in hypothesis testing also follows the same ideology. We make an assumption about the population and then collect evidence from the sample to either prove or discard the assumption.



## Hypothesis Testing

We use _statistical hypothesis testing_ or _significance tests_ to provide confidence or likelihood about making inferences from a sample of data to a larger population. We calculate some quantity under some assumption, then we interpret the result and either reject or fail to reject the assumption.

The assumption is called the _null hypothesis_ ($\color{#F99417}H_0$). The violation of the null hypothesis is known as the alternative hypothesis ($\color{#F99417}H_1$). 

> We use the `proof by contradiction` method to test the hypothesis. We assume that the null hypothesis is true and then we try to prove that it is false. If we fail to prove that the null hypothesis is false, then we accept the null hypothesis

One of the common measures of statistical significance is the **p-value**. It is the probability of observing a test statistic as extreme as the one observed, by chance, given that the null hypothesis is true. If the p-value is less than the significance level ($\color{#F99417}\alpha$), then we reject the null hypothesis. A common value used for $\color{#F99417}\alpha$ is 0.05 or 5% (arbitrary value).










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


<!-- 
    Author : Kannan Jayachandran
    File : Data_visualization.md
 -->

<h1 align="center" style="color: orange"> Data Visualization </h1>

Data visualization is the process of making sense of data through visual representations. Data visualization serves two fundamental purposes:

1. **Understanding the data**

2. **Communicating the results**

> No code/low code data visualization tools would be covered in detail in the [Data Engineering Tools section](../12.%20Data%20Engineering%20and%20Big%20Data%20tools/Readme.md).

## Line plot

A plot that shows the data as a collection of points.

- We use a line plot to present observations collected at regular intervals.

**[plt.plot()](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)**

```python
x = [x*0.1 for x in range(100)]
y = np.sin(x)

plt.plot(x, y)
plt.show()
```

![Line plot of sine wave](./img/Sin_wave.png)

## Bar chart

A bar plot is a type of plot that shows the relative quantities for multiple categories. This is useful when we have a large number of values and we want to see which ranges most of the values fall into. The bars can be either vertical or horizontal and they can be stacked or grouped. **[sns.barplot()](https://seaborn.pydata.org/generated/seaborn.barplot.html)** or **[plt.bar()](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html)**

![Bar-chart image](./img/barplot.png)

## Histogram

It is used to summarize the distribution of  data. It is a type of bar plot that shows the frequency of each value. The height of the bar represents the frequency of the value. The width of the bar represents the range of values. Histograms are density estimates. **[sns.histplot()](https://seaborn.pydata.org/generated/seaborn.histplot.html) | [sns.displot()](https://seaborn.pydata.org/generated/seaborn.displot.html#seaborn-displot)** or **[plt.hist()](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html)**

![Histogram](./img/histogram.png)

## Box and Whisker plot (Boxplot)

It is also used to summarize the distribution of the data sample. The box represents the inter-quartile range (IQR), which is the range between the 25th and 75th percentile of the data. The line in the middle of the box is the median. The whiskers represent the rest of the distribution. The points outside the whiskers are outliers. **[sns.boxplot()](https://seaborn.pydata.org/generated/seaborn.boxplot.html)** or **[plt.boxplot()](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html)**

![Image of boxplot](./img/boxPlot.png)

![Explanation of Whisker plot](./img/Whisker_plot.png)

## Scatter plot

A scatter plot is a type of plot that shows the data as a collection of points. The position of a point depends on its two-dimensional value, where each value is a position on either the horizontal or vertical dimension. They are useful to show association or correlation between two variables. **[sns.scatterplot()](https://seaborn.pydata.org/generated/seaborn.scatterplot.html)** or **[plt.scatter()](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html)**

![Scatter plot image](./img/Scatter_plot.png)

## Pair plot

A pair plot is a type of plot that shows the distribution of a variable (univariate distribution) and the relationship between multiple variables (multivariate distribution). If we have $x$ features, we will get $x^2$ plots.

![Pair plot image](./img/PairPlot.png)

It is good to use a pair plot if we are dealing with a small number of features. If we have a large number of features, we can use a correlation matrix to see the correlation between the features. **[sns.pairplot()](https://seaborn.pydata.org/generated/seaborn.pairplot.html)**

## Violin plot

A violin plot is similar to a box plot, except that it also shows the probability density of the data at different values. It is a combination of a box plot and a kernel density plot (a histogram).

![Violin plot](./img/Violin.png)

## Heatmap

A heatmap is a two-dimensional representation of data in which values are represented by colors. It is useful to visualize the correlation between features. **[sns.heatmap()](https://seaborn.pydata.org/generated/seaborn.heatmap.html)**

<!-- Add a section that explains what type of data visualization to use for what type of data. -->

# Import necessary libraries
import math  # For mathematical operations, including NaN values
import statistics  # For basic statistical calculations
import numpy as np  # For numerical computations and matrix operations
import scipy.stats  # For advanced statistical analyses
import pandas as pd  # For working with structured data (e.g., DataFrames)

# Define data lists
x = [8.0, 1, 2.5, 4, 28.0]  # A list of numeric data
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]  # A list with NaN (missing value)

# Print original lists
print(x)
print(x_with_nan)

# Convert data to numpy arrays and pandas Series
y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)

# Print the converted structures
print(y)
print(y_with_nan)
print(z)
print(z_with_nan)

# Calculate the mean (average) using different methods
mean_ = sum(x) / len(x)  # Pure Python method
print(mean_)

mean_ = statistics.mean(x)  # Using statistics.mean
print(mean_)

mean_ = np.mean(x)  # Using numpy.mean
print(mean_)

# Handling NaN values in numpy
print(np.mean(y_with_nan))  # Returns NaN
print(y_with_nan.mean())  # Returns NaN
print(np.nanmean(y_with_nan))  # Ignores NaN

# Handling NaN values in pandas
print(z_with_nan.mean())  # Ignores NaN by default

# Weighted mean calculation
w = [0.1, 0.2, 0.3, 0.25, 0.15]  # Weights for the data
wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)  # Pure Python approach
print(wmean)

# Alternative weighted mean calculation
wmean = sum(x_ * w_ for (x_, w_) in zip(x, w)) / sum(w)
print(wmean)

# Weighted mean using numpy
y, w = np.array(x), np.array(w)
wmean = np.average(y, weights=w)
print(wmean)

# Weighted mean using pandas
z = pd.Series(x)
wmean = np.average(z, weights=w)
print(wmean)

# Using numpy sum for weighted mean
print((w * y).sum() / w.sum())

# Handling datasets with NaN for weighted mean
w = np.array([0.1, 0.2, 0.3, 0.0, 0.2, 0.1])
h = (w * y_with_nan).sum() / w.sum()
print(h)  # Result may contain NaN
print(np.average(y_with_nan, weights=w))  # Handles NaN
print(np.average(z_with_nan, weights=w))  # Handles NaN

# Harmonic mean calculation
hmean = len(x) / sum(1 / item for item in x)  # Pure Python method
print(hmean)

# Using statistics.harmonic_mean
hmean = statistics.harmonic_mean(x)
print(hmean)

# Handling NaN with statistics.harmonic_mean
# Uncommenting the next line will raise an error due to NaN
# print(statistics.harmonic_mean(x_with_nan))

# Geometric mean calculation
gmean = 1
for item in x:  # Pure Python method
    gmean *= item
gmean **= 1 / len(x)
print(gmean)

# Using statistics.geometric_mean
gmean = statistics.geometric_mean(x)
print(gmean)

# Handling NaN with statistics.geometric_mean
# Uncommenting the next line will raise an error due to NaN
# print(statistics.geometric_mean(x_with_nan))

# Using scipy.stats.gmean
print(scipy.stats.gmean(y))  # Ignores NaN
print(scipy.stats.gmean(z))  # Ignores NaN

# Median calculation
median_ = statistics.median(x)  # Using statistics.median
print(median_)

# Median with numpy
median_ = np.median(y)
print(median_)

# Handling NaN with numpy
print(np.nanmedian(y_with_nan))  # Ignores NaN

# Pandas Series handles NaN by default
print(z_with_nan.median())

# Mode calculation
u = [2, 3, 2, 8, 3, 2, 1, 3]
mode_ = statistics.mode(u)  # Using statistics.mode
print(mode_)

# Variance calculation
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)  # Pure Python approach
print(var_)

# Using statistics.variance
var_ = statistics.variance(x)
print(var_)

# Standard deviation
std_ = var_ ** 0.5  # Pure Python method
print(std_)

# Using numpy
print(np.std(y, ddof=1))  # With numpy
print(z.std(ddof=1))  # With pandas Series (ignores NaN)

# Skewness
print(scipy.stats.skew(y, bias=False))  # Using scipy.stats.skew

# Percentiles
print(np.percentile(y, [25, 50, 75]))  # Using numpy.percentile
print(np.nanpercentile(y_with_nan, [25, 50, 75]))  # Handles NaN

# Range calculation
print(np.ptp(y))  # Range using numpy

# Summary statistics
result = scipy.stats.describe(y, ddof=1, bias=False)  # Using scipy.stats.describe
print(result)

# Visualizations with matplotlib
import matplotlib.pyplot as plt

# Simple line plot
plt.plot([1, 2, 3], [5, 7, 4])
plt.show()

## box plot 
from plotnine.data import huron
from plotnine import ggplot, aes, geom_boxplot

(
  ggplot(huron)
  + aes(x="factor(decade)", y="level")
  + geom_boxplot()
)

## all plot 
#hwy: Miles per gallon
#displ: Engine size
#class: Vehicle class
#year: Model year

from plotnine.data import mpg
from plotnine import ggplot, aes, facet_grid, labs, geom_point

(
    ggplot(mpg)
    + facet_grid(facets="year~class")
    + aes(x="displ", y="hwy")
    + labs(
        x="Engine Size",
        y="Miles per Gallon",
        title="Miles per Gallon for Each Year and Vehicle Class",
    )
    + geom_point()
)

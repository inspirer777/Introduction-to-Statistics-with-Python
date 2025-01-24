import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd

x = [8.0, 1, 2.5, 4, 28.0]
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]
print(x)
print(x_with_nan)
y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
print(y)
print(y_with_nan)
print(z)
print(z_with_nan)
mean_ = sum(x) / len(x)
print(mean_)

mean_ = statistics.mean(x)
print(mean_)



mean_ = np.mean(x)
print(mean_)

print(np.mean(y_with_nan))

print(y_with_nan.mean())

print(np.nanmean(y_with_nan))
#pandsa 
print(z_with_nan.mean())

x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]
wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)
print(wmean)
wmean = sum(x_ * w_ for (x_, w_) in zip(x, w)) / sum(w)
print(wmean)

## np.average () wmean
y, z, w = np.array(x), pd.Series(x), np.array(w)
wmean = np.average(y, weights=w)
print(wmean)

wmean = np.average(z, weights=w)
print(wmean)
## np.sum weight mean
print((w * y).sum() / w.sum())

# However, be careful if your dataset contains nan values:
w = np.array([0.1, 0.2, 0.3, 0.0, 0.2, 0.1])
h = (w * y_with_nan).sum() / w.sum()
print(h)
print(np.average(y_with_nan, weights=w))

print(np.average(z_with_nan, weights=w))
# harmonic mean
hmean = len(x) / sum(1 / item for item in x)
print(hmean)

#statistics.harmonic_mean
hmean = statistics.harmonic_mean(x)
print(hmean)
#statistics.StatisticsError
print(statistics.harmonic_mean(x_with_nan))
# nan
print(statistics.harmonic_mean([1, 0, 2]))


## third way
scipy.stats.hmean(y)
scipy.stats.hmean(z)
## geometric mea
# pure python
gmean = 1
for item in x:
              gmean *= item
gmean **= 1 / len(x)
print(gmean)
#statistics.geometric_mean
gmean = statistics.geometric_mean(x)
print(gmean)

gmean = statistics.geometric_mean(x_with_nan)
print(gmean)
#scipy.stats.gmean()
print(scipy.stats.gmean(y))
print(scipy.stats.gmean(z))
## median

#statistics.median
median_ = statistics.median(x)
print(median_)
#The
median_ = statistics.median(x[:-1])
print(median_)

print(statistics.median(x_with_nan))
print(statistics.median_low(x_with_nan))
print(statistics.median_high(x_with_nan))

#can also get the median with np.median():
median_ = np.median(y)
print(median_)
median_ = np.median(y[:-1])
print(median_)

 #If this behavior is not what you want, then you can use nanmedian() 
print(np.nanmedian(y_with_nan))
print(np.nanmedian(y_with_nan[:-1]))
#Pandas Series objects ignores nan values by default:
print(z.median())
print(z_with_nan.median())
### Mode

#You can obtain the mode with statistics.mode() and statistics.multimode()
mode_ = statistics.mode(u)
mode_
mode_ = statistics.multimode(u)
print(mode_)
#


# get the mode with scipy.stats.mode():
u, v = np.array(u), np.array(v)
mode_ = scipy.stats.mode(u)
print(mode_)
mode_ = scipy.stats.mode(v)
print(mode_)
#NumPy 
print(mode_.mode)
print(mode_.count)

#pandas
u, v, w = pd.Series(u), pd.Series(v), pd.Series([2, 2, math.nan])
print(u.mode())
print(v.mode())
print(w.mode())

# calculate the sample variance with pure Python:
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
print(var_)

# function statistics.variance():
var_ = statistics.variance(x)
print(var_)
##  nan values among your data, then statistics.variance()
print(statistics.variance(x_with_nan))
##  function np.var()
var_ = np.var(y, ddof=1)
print(var_)
var_ = y.var(ddof=1)
print(var_)
# If you have nan values in the dataset, then np.var() and .var() will return nan:
print(np.var(y_with_nan, ddof=1))
# nan
print(y_with_nan.var(ddof=1))
# skips nan values by default:
print(z.var(ddof=1))
print(z_with_nan.var(ddof=1))

# Standard Deviation
#pure Python:
std_ = var_ ** 0.5
print(std_)
#use statistics.stdev():
std_ = statistics.stdev(x)
print(std_)
## numpy
print(np.std(y, ddof=1))
print(y.std(ddof=1))
print(np.std(y_with_nan, ddof=1))
#nan
print(y_with_nan.std(ddof=1))
#nan
print(np.nanstd(y_with_nan, ddof=1))

#pd.Series   |(method .std() that skips nan by default:
print(z.std(ddof=1))
print(z_with_nan.std(ddof=1))

##Skewness 

#  sample skewness with scipy.stats.skew():
y, y_with_nan = np.array(x), np.array(x_with_nan)
print(scipy.stats.skew(y, bias=False))

print(scipy.stats.skew(y_with_nan, bias=False))
#nan

# Pandas Series
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
print (z.skew())

print(z_with_nan.skew())

## Percentiles
#statistics.quantiles():
x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]


# np.percentile()
y = np.array(x)
print(np.percentile(y, 5))
print(np.percentile(y, 95))

#sequence number
print(np.percentile(y, [25, 50, 75]))
print(np.median(y))

# nan values, then use np.nanpercentile()
y_with_nan = np.insert(y, 2, np.nan)
print( y_with_nan)
print(np.nanpercentile(y_with_nan, [25, 50, 75]))

#numbers between 0 and 1 instead of percentiles:
print(np.quantile(y, 0.05))
print(np.quantile(y, 0.95))
print(np.quantile(y, [0.25, 0.5, 0.75]))
print(np.nanquantile(y_with_nan, [0.25, 0.5, 0.75]))

#pd.Series objects have the method .quantile():
z, z_with_nan = pd.Series(y), pd.Series(y_with_nan)
print(z.quantile(0.05))
print(z.quantile(0.95))
print(z.quantile([0.25, 0.5, 0.75]))
print(z_with_nan.quantile([0.25, 0.5, 0.75]))

# Ranges
#function np.ptp():
print(np.ptp(y))
print(np.ptp(z))
print(np.ptp(y_with_nan))
print(np.ptp(z_with_nan))


## Summary of Descriptive Statistics
## use scipy.stats.describe() like this:
result = scipy.stats.describe(y, ddof=1, bias=False)
print(result)



result.minmax[0]  # Min
result.minmax[1]  # Max

## pandas
#method .describe():

result = z.describe()
print(result)
print(result['std'])
print(result['75%'])

## simple plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy

plt.plot([1, 2, 3],[5, 7, 4])
plt.show()





a = np.array([[1,1,1],
              [2,3,1],
              [4,9,2],
             [8,27,4]
              ,[16,1,1]])
print (a)
np.mean(a)
a.mean()
np.median(a)
a.var(ddof=1)
df = pd.DataFrame(a)
print(df)
## box plot
np.random.seed(seed=0)
x = np.random.randn(1000)
y = np.random.randn(100)
z = np.random.randn(10)
fig, ax = plt.subplots()
ax.boxplot((x, y, z), vert=False, showmeans=True, meanline=True,
           labels=('x', 'y', 'z'), patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'yellow'},
           meanprops={'linewidth': 2, 'color': 'red'})
plt.show()
### how import dataset  the same of the R
from plotnine.data import economics

economics
## bar plot with mpg data
from plotnine.data import mpg
from plotnine import ggplot, aes, geom_bar

mpg
ggplot(mpg) + aes(x="class") + geom_bar()


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




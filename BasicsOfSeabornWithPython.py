import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

sbnInbuildDF = sbn.get_dataset_names()

# Loading the Datsets from the Seaborn Inbuild DataFrames
crash_dataset = sbn.load_dataset('car_crashes')
crash_dataset.head()

tips_dataset = sbn.load_dataset('tips')
tips_dataset.head()

iris_dataset = sbn.load_dataset('iris')
iris_dataset.head()

flights_dataset = sbn.load_dataset('flights')
flights_dataset.head()

attention_dataset = sbn.load_dataset('attention')
attention_dataset.head()

"""
Distribution Plot
It is used basically for univariant set of observations and visualizes it through a histogram.
 i.e. only one observation and hence we choose one particular column of the dataset.
"""
sbn.distplot(crash_dataset['alcohol'])
sbn.distplot(crash_dataset['speeding'], kde = False, bins = 25)

"""
Joint Plot
It is used to draw a plot of two variables with bivariate and univariate graphs. 
It basically combines two different plots.
"""
sbn.jointplot(x = 'speeding', y = 'not_distracted', data = crash_dataset, kind = 'hex')
sbn.jointplot(x = 'speeding', y = 'ins_premium', data = crash_dataset, kind = 'reg', height = 10)

"""
Kde Plot
Kdeplot is a Kernel Distribution Estimation Plot which depicts the probability density function 
of the continuous or non-parametric data variables i.e. we can plot for the univariate or multiple 
variables altogether.
"""
sbn.kdeplot(crash_dataset['speeding'])
sbn.kdeplot(crash_dataset['speeding'], crash_dataset['ins_losses'])

"""
Pair Plot
A pairplot plot is a pairwise relationships in a dataset. The pairplot function creates a grid of 
Axes such that each variable in data will by shared in the y-axis across a single row and in 
the x-axis across a single column. 
"""

sbn.pairplot(tips_dataset)
sbn.pairplot(tips_dataset, hue = 'smoker', palette = 'rainbow')

"""
Rug Plot
A rugplot is a graph that places a dash horizontally with each occurrence of an item in a dataset.
Areas where there is great occurrence of an item see a greater density of these dashes. 
"""
sbn.rugplot(tips_dataset['total_bill'])
sbn.rugplot(crash_dataset['alcohol'], height = 0.8, axis = 'y')

"""
Styling 
"""
sbn.set_style('darkgrid')
#plt.figure(figsize = (10, 4))
sbn.set_context('paper', font_scale = 1.6, rc={"lines.linewidth": 2.5})
sbn.jointplot(x = 'alcohol', y = 'speeding', data = crash_dataset, kind = 'reg')
sbn.despine(left = True, bottom = True)

# CATEGORICAL PLOTS
# The categorical plots plot the values in the categorical column against another 
# categorical column or a numeric column.
"""
Bar Plot
It  is used to display the mean value for each value in a categorical column, 
against a numeric column.
"""
sbn.barplot(x = 'total_bill', y = 'day', data = tips_dataset)
sbn.barplot(x = 'total_bill', y = 'day', data = tips_dataset, estimator = np.var)

"""
Count Plot
iI displays the count of the categories in a specific column.
"""
sbn.countplot(y = 'smoker', data = tips_dataset)
sbn.countplot(x = 'smoker', data = tips_dataset, orient = 'v', hue = 'day', dodge = True)

"""
Box Plot
Box Plot is the visual representation of the depicting groups of numerical data through their
quartiles. It is also used for detect the outlier in data set. It captures the summary of the 
data efficiently with a simple box and whiskers and allows us to compare easily across groups. 
Boxplot summarizes a sample data using Minimum, 25th (First Quartile), 50th (Median 
{Second Quartile}), 75th percentiles (Third Quartile) and Maximum.
These percentiles are also known as the lower quartile, median and upper quartile.
"""
sbn.boxplot(x = 'day', y = 'total_bill', data = tips_dataset, hue = "sex", orient = 'v')

"""
Violin Plot
Violinplots summarize numeric data over a set of categories. They are essentially a box plot 
with a kernel density estimate (KDE) overlaid along the range of the box and reflected to make 
it look nice. They provide more information than a boxplot because they also include information 
about how the data is distributed within the inner quartiles.
"""
sbn.violinplot(x = 'sepal_length', y = 'species', data = iris_dataset)
sbn.violinplot(x = 'day', y = 'total_bill', data = tips_dataset, hue = 'sex', split = True)

"""
Strip Plot
A strip plot is a scatter plot where one of the variables is categorical. 
They can be combined with other plots to provide additional information. 
"""
sbn.stripplot(x = 'total_bill', y = 'day', data = tips_dataset)
sbn.stripplot(x = 'total_bill', y = 'day', data = tips_dataset, jitter = True, hue = 'sex', dodge = True)

"""
Swarm Plot

"""
sbn.swarmplot(x = 'day', y = 'total_bill', data = tips_dataset)

# with violin Plot
sbn.violinplot(x = 'total_bill', y = 'day', data = tips_dataset, palette = 'rainbow')
sbn.swarmplot(x = 'total_bill', y = 'day', data = tips_dataset, color = 'red')

# MATRIX PLOT
"""
Heatmap

"""
carsh_matrix = crash_dataset.corr()
#plt.figure(figsize = (8, 6))
#sbn.set_context('paper', font_scale = 1.4)
ax = sbn.heatmap(carsh_matrix, annot = True, cmap = 'Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

# Using the pivot Table
flights_table = flights_dataset.pivot_table(index = 'month', columns = 'year', values = 'passengers')
sbn.heatmap(flights_table, cmap = 'Greens', linecolor = 'white', linewidth = 1)

"""
Cluster Map
A clustered heatmap is different from an ordinary heatmap on the following terms:
    - The heatmap cells are all clustered using a similarity algorithm.
    - Dentograms are drawn for the columns and the rows of the heatmap.
"""
species = iris_dataset.pop('species')
sbn.clustermap(iris_dataset)

sbn.clustermap(iris_dataset, figsize=(7, 5), row_cluster=False, dendrogram_ratio=(.1, .2), cbar_pos=(0, .2, .03, .4))

"""
Pair Grid
This class maps each variable in a dataset onto a column and row in a grid of multiple axes.
Different axes-level plotting functions can be used to draw bivariate plots in the upper 
and lower triangles, and the the marginal distribution of each variable can be shown on the diagonal.
"""
iris_pairgrid = sbn.PairGrid(iris_dataset, hue = 'species')
# iris_pairgrid.map(plt.scatter)
iris_pairgrid.map_diag(plt.hist)
iris_pairgrid.map_offdiag(plt.scatter)
iris_pairgrid.map_upper(plt.scatter)
iris_pairgrid.map_lower(sbn.kdeplot)
iris_pairgrid.add_legend(fontsize=11)

"""
Facet Grid

"""
tips_FG = sbn.FacetGrid(tips_dataset, col = 'time', hue = 'smoker', aspect = 1.3,
                        col_order = ['Dinner', 'Lunch'], palette = 'Set1')
tips_FG.map(plt.scatter, 'total_bill', 'tip')
tips_FG.add_legend()

# Different ways 

kws = dict(s = 50, linewidth = 0.5, edgecolor = 'w')
tips_FG = sbn.FacetGrid(tips_dataset, col = 'sex', hue = 'smoker', aspect = 1.3,
                        hue_order = ['Yes', 'No'], hue_kws = dict(marker = ['^', 'v']))
tips_FG.map(plt.scatter, 'total_bill', 'tip', **kws)
tips_FG.add_legend()

attention_FG = sbn.FacetGrid(attention_dataset, col = 'subject', col_wrap = 5, height = 1.5)
attention_FG.map(plt.plot, 'solutions', 'score', marker = '.')
attention_FG.add_legend()

print
mport numpy as np
import matplotlib.pyplot as plt
import caltech101.file_reader as fread
import analyzer

fr = fread.FileReader()
cats = fread.getCategories()
neuron_cat = analyzer.get_act_category_count()

# Let's try to plot for category humans
idx = cats.index("humans")
fr_neurons = neuron_cat[:, idx]
x_axis = range(0, len(fr_neurons))
plt.bar(x_axis, fr_neurons)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import caltech101.file_reader as fread
import analyzer
import seaborn as sns

# fr = fread.FileReader()
# cats = fread.getCategories()
# neuron_cat = analyzer.get_act_category_count()

# # Let's try to plot for category humans
# idx = cats.index("humans")
# fr_neurons = neuron_cat[:, idx]
# x_axis = range(0, len(fr_neurons))
# plt.bar(x_axis, fr_neurons)
# plt.show()

def group_line_plot(xval, yval, item_names, xlabel, ylabel, title, file_name):
    fig=plt.figure(figsize=(12,10))
    sns.set_style("darkgrid")
    for i in np.arange(yval.shape[0]):
        plt.plot(xval,yval[i,:],'-o')
    plt.legend(item_names, loc='lower left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(file_name)
    plt.close()

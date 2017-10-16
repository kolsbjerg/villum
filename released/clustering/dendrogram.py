from matplotlib.pyplot import *
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from ase.ga.clustering_v3 import AgglomerativeClustering
from ase.ga.comparators_v3 import BagOfBonds
from ase.io import read,write
from ase.visualize import view
from math import ceil
import numpy as np

data = read('data_test.traj@:')
n_samples = len(data)

# Comparator
comp = BagOfBonds(excluded_types=[1])

# Cluster data
ac = AgglomerativeClustering(comp=comp, linkage='average', unique_data=False)
feature_matrix, similarity_matrix, linkage_matrix = ac.grow_tree(data=data)

# Cut tree
labels, clusters, branches, centroids, cluster_energies, avg_width = \
    ac.cut_tree(t=1.0, criterion='inconsistent', cluster_min=2)

n_clusters = len(branches[0])

## FIGURE ##
params={'lines.linewidth':1.5,
        'legend.fontsize':8,
        'xtick.labelsize':8,
        'ytick.labelsize':8,
        'axes.labelsize':8,
        'axes.linewidth':0.5}

rcParams.update(params)

# Get colors for dendrogram branches
color_palette = ['#a50f15','#084594']
set_link_color_palette(color_palette)
n_colors = len(color_palette)
color_palette = ['grey']+color_palette*int(ceil(n_clusters/float(n_colors)))

colors = ['grey']*(2*n_samples-1)
for cluster, label in zip(branches[0],branches[1]):
    twigs = ac.get_leaves(cluster)[1]
    for twig in twigs:
        colors[int(twig)] = color_palette[label]

# Dendrogram
figure(figsize=(7.5,2))

R = dendrogram(linkage_matrix, p=58, truncate_mode=None,
               show_contracted=True,
               count_sort=False,
               distance_sort=True,
               get_leaves=True,
               no_labels=False,
               leaf_font_size=10,
               link_color_func=lambda k: colors[k])

ylabel('Average similarity')
xs = R[('icoord')]
xi = [xs[-1][0],xs[1][0],xs[0][0],xs[0][-1],xs[3][0],xs[2][0],xs[2][-1]]
xticks(xi,[1,2,3,4,5,6,7])
yticks([0.0,0.10,0.20,0.30,0.40,0.50])

savefig('dendrogram_sample.png',bbox_inches='tight')

show()

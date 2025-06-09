# !/usr/bin/env python
#
# Script info
# -----------
# __author__ = 'Ryan Godin'
# __copyright__ = '© His Majesty the King in Right of Canada, as represented by the Minister of Agriculture and Agri-Food,' \
#                 '2025-'
# __credits__ = 'Ryan Godin, Etienne Lord'
# __email__ = ''
# __license__ = 'Open Government Licence - Canada'
# __maintainer__ = 'Ryan Godin'
# __status__ = 'Development'
# __version__ = '1.0.1'

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import argparse
from sklearn.manifold import TSNE

class tSNE:

    def __init__(self, filepath):

        # Load Data | Charger les données
        self.filepath = filepath
        self.data = np.load(filepath, allow_pickle=True)

        self.x = self.data['data']
        self.y = self.data['labels']

        # Reshape the data to be 2D
        if self.x.ndim > 2:
            n_samples = self.x.shape[0]
            n_features = np.prod(self.x.shape[1:])
            self.x = self.x.reshape(n_samples, n_features)

        # Label Crops
        self.crop_labels = self.data['crops']
        self.label_map = {i: name for i, name in enumerate(self.crop_labels)}

        # Check self.y contains only keys present in self.label_map
        valid_y_indices = np.isin(self.y, list(self.label_map.keys()))
        self.labels = np.array(
            [self.label_map[label] for label in self.y[valid_y_indices]]
            )
        self.x = self.x[valid_y_indices]

        # Color Map Labels
        unique_labels = np.unique(self.labels)
        cmap = plt.colormaps.get_cmap('jet')
        colors = cmap(np.linspace(0, 1, len(unique_labels)))
        self.color_map = {label: colors[i] for i,
                          label in enumerate(unique_labels)}

        # Sample points
        self.n_samples = self.x.shape[0]
        self.time = round(time.time())

        # Only plot data points
        if self.n_samples > 0:
            self.indices = np.random.choice(self.x.shape[0], self.n_samples,
                                            replace=False)
            self.sampled_data = self.x[self.indices]
            self.sampled_labels = self.labels[self.indices]

            # Run t-SNE
            tsne = TSNE(n_components=2, random_state=self.time)
            self.sampled_data_tsne = tsne.fit_transform(self.sampled_data)

            # Plot
            self._plot()
        else:
            print('[WARN] Not enough data points to sample for t-SNE.')


    def _plot(self):
      """
      Plots the t-SNE when parent is called
      """
      plt.figure(figsize=(8, 6))
        # Make sure 'sampled_labels' is not empty
      if self.sampled_labels.size > 0:
            for label in np.unique(self.sampled_labels):
                label_indices = (self.sampled_labels == label)
                plt.scatter(
                    self.sampled_data_tsne[label_indices, 0],
                    self.sampled_data_tsne[label_indices, 1],
                    label=label,
                    c=[self.color_map[label]],
                    alpha=0.8,
                    edgecolors='w'
                )

            plt.title('t-SNE Visualization')
            plt.xlabel('Axis 1')
            plt.ylabel('Axis 2')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'tSNE_{self.time}.svg')
            plt.show()
      else:
            print('[WARN] No data points to plot after sampling.')
   
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data", type=str, required=True, 
                      help="Path to the dataset NPZ file")
    args = parser.parse_args()
    tsne = tSNE(args.data)
    
    
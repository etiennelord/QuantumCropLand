# !/usr/bin/env python
#
# Script info
# -----------
# __author__ = 'Ryan Godin'
# __copyright__ = '© His Majesty the King in Right of Canada, as represented by the Minister of Agriculture and Agri-Food Canada,' \
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

    def visualize_train_vs_test_tsne(self):
        """
        Performs t-SNE on combined train, test, and validation sets
        """
        # Reshape train, test, and validation data if necessary
        x_train_reshaped = self.x_train
        if x_train_reshaped.ndim > 2:
            n_samples_train = x_train_reshaped.shape[0]
            n_features_train = np.prod(x_train_reshaped.shape[1:])
            x_train_reshaped = x_train_reshaped.reshape(n_samples_train, n_features_train)

        x_test_reshaped = self.x_test
        if x_test_reshaped.ndim > 2:
            n_samples_test = x_test_reshaped.shape[0]
            n_features_test = np.prod(x_test_reshaped.shape[1:])
            x_test_reshaped = x_test_reshaped.reshape(n_samples_test, n_features_test)

        x_valid_reshaped = self.x_valid
        if x_valid_reshaped.ndim > 2:
            n_samples_valid = x_valid_reshaped.shape[0]
            n_features_valid = np.prod(x_valid_reshaped.shape[1:])
            x_valid_reshaped = x_valid_reshaped.reshape(n_samples_valid, n_features_valid)

        # Combine train, test, and validation data
        x_combined = np.vstack((x_train_reshaped, x_test_reshaped, x_valid_reshaped))

        # Create labels for train, test, and validation sets
        labels_combined = np.array(
            ['Train'] * x_train_reshaped.shape[0] +
            ['Test'] * x_test_reshaped.shape[0] +
            ['Validation'] * x_valid_reshaped.shape[0]
        )

        if x_combined.shape[0] > 0:
            # Perform t-SNE on combined data
            tsne_combined = TSNE(n_components=2, random_state=self.time)
            x_combined_tsne = tsne_combined.fit_transform(x_combined)

            # Plotting
            plt.figure(figsize=(8, 6))

            # Plot train points
            train_indices = (labels_combined == 'Train')
            plt.scatter(
                x_combined_tsne[train_indices, 0],
                x_combined_tsne[train_indices, 1],
                label='Train',
                c='blue',
                alpha=0.6,
                edgecolors='w'
            )

            # Plot test points
            test_indices = (labels_combined == 'Test')
            plt.scatter(
                x_combined_tsne[test_indices, 0],
                x_combined_tsne[test_indices, 1],
                label='Test',
                c='red',
                alpha=0.6,
                edgecolors='w'
            )

            # Plot validation points
            valid_indices = (labels_combined == 'Validation')
            plt.scatter(
                x_combined_tsne[valid_indices, 0],
                x_combined_tsne[valid_indices, 1],
                label='Validation',
                c='green',  # You can choose a different color
                alpha=0.6,
                edgecolors='w'
            )


            plt.title('t-SNE Visualization of Train, Test, and Validation Sets')
            plt.xlabel('Axis 1')
            plt.ylabel('Axis 2')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'tSNE_train_test_valid_{self.time}.svg') # Save as SVG
            plt.show()
        else:
            print('[WARN] Not enough data points in combined train, test, and validation sets for t-SNE.')



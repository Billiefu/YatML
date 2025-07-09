"""
Copyright (C) 2024 Fu Tszkok

:module: Exercises3
:function: Precision-Recall curve visualization
:author: Fu Tszkok
:date: 2024-10-31
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import matplotlib.pyplot as plt

# Precision and recall data points
# The data represents performance at different classification thresholds
recalls = [1 / 7, 2 / 7, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 5 / 7, 6 / 7, 6 / 7, 1]  # Recall values (TP/(TP+FN))
precisions = [1, 1, 2 / 3, 3 / 4, 4 / 5, 5 / 6, 5 / 7, 6 / 8, 6 / 9, 7 / 10]  # Precision values (TP/(TP+FP))

# Set figure size
plt.figure(figsize=(8, 6))

# Plot PR curve with circle markers
plt.plot(recalls, precisions, marker='o')

# Configure plot appearance
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)

# Set axis limits and ticks
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks([i / 10 for i in range(11)])  # x-axis ticks at 0.1 intervals
plt.yticks([i / 10 for i in range(11)])  # y-axis ticks at 0.1 intervals

# Display the plot
plt.tight_layout()  # Adjust layout to prevent label clipping
plt.show()

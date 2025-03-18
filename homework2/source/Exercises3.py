import matplotlib.pyplot as plt

# Recall and precision data
recalls = [1/7, 2/7, 2/7, 3/7, 4/7, 5/7, 5/7, 6/7, 6/7, 1]
precisions = [1, 1, 2/3, 3/4, 4/5, 5/6, 5/7, 6/8, 6/9, 7/10]

# Plotting the PR curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, marker='o')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(True)
plt.xticks([i/10 for i in range(11)])  # Set x-axis ticks
plt.yticks([i/10 for i in range(11)])  # Set y-axis ticks

# Display the graph
plt.show()

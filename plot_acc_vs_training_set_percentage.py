import matplotlib.pyplot as plt
import numpy as np

training_examples = np.array([1, 0.5, 0.25, 0.125, 0.0625])
training_examples = training_examples * 50000
print (training_examples)

train_accuracy = np.array([98.19, 97.9, 97.68, 97.97, 97.58])
test_accuracy = np.array([98.85, 98.72, 98.39, 97.82, 97.56])

plt.loglog(training_examples, train_accuracy, label='train  accuracy')
plt.loglog(training_examples, test_accuracy, label='test accuracy')
plt.xticks(training_examples, training_examples)
plt.legend()

plt.show()
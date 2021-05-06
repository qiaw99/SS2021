import numpy as np

prediction = np.load("./prediction.npy")
exact = np.load("./prediction_exact.npy")

print(prediction.shape)
print(exact)
print(np.sum(prediction == exact))
print(np.sum(prediction == exact)/prediction.shape[0] * 100)
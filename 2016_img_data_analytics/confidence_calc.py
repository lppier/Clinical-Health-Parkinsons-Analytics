import numpy as np

const = 1.96 # 95% confidence interval
error = 1 - 0.988
n = 4361
interval = const * np.sqrt((error * (1-error)) / n)
print("Static Spiral Test")
print(str(round(error, 3)) + "+/- " + str(round(interval, 3)))

error = 1 - 0.996
n = 3489
interval = const * np.sqrt((error * (1-error)) / n)
print("Dynamic Spiral Test")
print(str(round(error, 3)) + "+/- " + str(round(interval, 3)))

error = 1 - 0.949
n = 3940
interval = const * np.sqrt((error * (1-error)) / n)
print("Stability Test on Point")
print(str(round(error, 3)) + "+/- " + str(round(interval, 3)))
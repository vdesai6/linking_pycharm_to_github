import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,10,100)
y = np.cos(x)

plt.figure()
plt.scatter(x, y)
plt.show()
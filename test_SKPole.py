import math
from SKPole import SKPole
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

pos = np.array([1.0 + 2.5/150.0, 0, 1.0/150.0])
mu = 3.036e-6

calc_pole = SKPole(pos, mu)
my_pole = calc_pole.get_pole()

x = np.linspace(-2*math.pi, 2*math.pi, 1000)

d_a_ls = []

for angle in x:
    d_a_ls.append((calc_pole.get_d_a_l(angle, 100000/1.496e8) * 1.496e8 * 1e9)/(np.power(3.147e7,2)))

plt.figure()
plt.plot(x, d_a_ls)
plt.xlabel("Angle Alpha")
plt.ylabel("Differential Lateral Acceleration")

# Creating x-values and x-labels for x-axis of plot
xvals = []

for i in range(-2, 3):
    xvals.append(math.pi*i)

xlabels = []
for i in range(- 2, 3):
    xlabels.append(str(i) + r'$\pi$')

plt.xticks(xvals,xlabels)
plt.title("Differential Lateral Acceleration vs. Angle from Pole")

"""
print((calc_pole.scuff_d_a_l(np.array([1, 0, 0]), 100000/1.496e8) * 1.496e8 * 1e9)/(np.power(3.147e7,2)))
print((calc_pole.scuff_d_a_l(np.array([0, 1, 0]), 100000/1.496e8) * 1.496e8 * 1e9)/(np.power(3.147e7,2)))
print((calc_pole.scuff_d_a_l(np.array([1/math.sqrt(2), 1/math.sqrt(2), 0]), 100000/1.496e8) * 1.496e8 * 1e9)/(np.power(3.147e7,2)))
print((calc_pole.scuff_d_a_l(np.array([0, 0, 1]), 100000/1.496e8) * 1.496e8 * 1e9)/(np.power(3.147e7,2)))
"""

plt.show()
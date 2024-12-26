import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Plot example subplots with different scales
ax0 = plt.subplot(2, 2, 2)
ax0.plot([1, 2, 3], [1, 2, 3])
ax3= plt.subplot(2, 2, 4)
ax3.plot([1, 2, 3], [10, 20, 30])
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Title')
plt.tight_layout()
plt.show()

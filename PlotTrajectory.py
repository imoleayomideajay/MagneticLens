import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def round_to_1(x):
    return round(x, -int(np.floor(np.log10(abs(x)))))

dists = np.loadtxt('dists.txt') * 1e3
slopes = np.loadtxt('slopes.txt') * 1e3
phi_tot = np.loadtxt('phi_total.txt')
phi_tot[dists[:, 1] < 0] += np.pi       # add pi to azimuthal angle when trajectory intersects with optical axis

r_max = np.max(dists[:, 1])

fig = plt.figure()

# z distance vs r distance from optical axis
ax1 = fig.add_subplot(221)
ax1.plot(dists[:, 0], dists[:, 1], lw=1.0)

ax1.set_title('Distance from optical axis', fontsize=8)
ax1.set_xlabel('Z distance [mm]', fontsize=6)
ax1.set_ylabel('R distance [mm]', fontsize=6)

r_lim1 = 1.05 * r_max
ax1.set_ylim(-r_lim1, r_lim1)
ax1.spines['bottom'].set_position(('data', 0))
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# polar plot of trajectory projected on x-y plane
ax2 = fig.add_subplot(222, projection='polar')
ax2.plot(phi_tot[:, 1], np.abs(dists[:, 1]), lw=1.0)

ax2.set_title('Trajectory projected on x-y plane', fontsize=8)
# ax2.title.set_position([0.5, 1.1])
# ax2.text(1.1, 0.48, 'x', fontsize=6, transform=ax2.transAxes)
# ax2.text(0.4, 1.04, 'y', fontsize=6, transform=ax2.transAxes)

ax2.set_ylim(0.0, r_max)
ax2.yaxis.set_major_locator(MultipleLocator(round_to_1(r_max/4.0)))
# ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
ax2.grid(True)

ax1.tick_params(axis='both', labelsize=4)
ax2.tick_params(axis='x', labelsize=4, pad=-5)
ax2.tick_params(axis='y', labelsize=4)

# fig.subplots_adjust(hspace=0.5)
fig.savefig('trajectory.png', dpi=300, bbox_inches='tight', pad_inches=0)

ax1.cla()
ax2.cla()
fig.clf()
plt.close(fig)
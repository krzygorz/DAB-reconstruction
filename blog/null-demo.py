import numpy as np
import matplotlib.pyplot as plt

N_frame = 196608
N_null = 2656

plt.style.use('grayscale')

plt.figure(figsize=(8, 4), dpi=200)
samples = np.fromfile('DAB+_20240712_105527Z_2048000_176640000_float32_iq.raw', np.complex64)
plt.plot(np.abs(samples[:N_frame]), '.', ms=1)
plt.xlim(0,N_frame)
plt.ylim(0,1)
plt.xlabel("time [samples]")
plt.ylabel("|x(t)| (normalized)")


plt.savefig("null.png")

fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
power = np.correlate(np.abs(samples[:N_frame+N_null])**2, np.ones(N_null)/(N_null), mode='valid')
ax.plot(power)
ax.set_xlabel("window start [samples]")
ax.set_ylabel("power inside window")

null_start = np.argmin(power)

x1 = null_start-50
x2 = null_start+50
zoom_x = np.arange(x1,x2)
# inset Axes....
y1, y2 = 0, np.max(power[zoom_x])  # subregion of the original image
axins = ax.inset_axes(
    [0.4, 0.2, 0.47, 0.47],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
axins.plot(zoom_x, power[zoom_x])
axins.axvline(null_start, linestyle='--')
axins.text(null_start+1,y2*0.9, f"$t_{{\\min}}$ = {null_start}", rotation=90, verticalalignment='top', size='small')

ax.indicate_inset_zoom(axins, edgecolor="black")
ax.set_xlim(0,N_frame)

plt.savefig("power.svg")
#plt.show()

import numpy as np
import matplotlib.pyplot as plt
import yaml
from temperature_generator import TemperatureGenerator


# create the generator
T = TemperatureGenerator("heating_instructions.yaml")

print(T.t_max)

# times to calculate temperature at
t = np.linspace(0, T.t_max, 1000)

# figure for plotting
fig, ax = plt.subplots(
    1,
    1,
    figsize=(6.4, 4.8),
    constrained_layout=True,
)

# plot the time-dependent temperatures
ax.plot(
    t,
    T(t),
    "-",
    zorder=2,
)

# limit the x-axis range
ax.set_xlim(t.min(), t.max())

# label the axes
ax.set_xlabel(r"$t$ (h)")
ax.set_ylabel(r"$T$ (K)")

# display the heating instructions next to the axes
ax.text(
    1.025,
    1.0,
    "Instructions:\n\n" + yaml.dump(T.instructions, Dumper=yaml.Dumper),
    ha="left",
    va="top",
    transform=ax.transAxes,
    size="x-small",
    clip_on=False,
)

# save the figure
fig.savefig("generated-T-dependence.pdf")

# show the plot
plt.show()

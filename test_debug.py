import sys
sys.path.insert(0, '.')
import numpy as np

from xrd_crystal.crystal import Crystal
from xrd_crystal.diffraction import diffraction_simulation

crystal = Crystal(
    a=5.4309, b=5.4309, c=5.4309,
    alpha=90, beta=90, gamma=90,
    space_group="Fd-3m",
    space_group_number=227,
    name="Si"
)
crystal.add_atom("Si", 0.0, 0.0, 0.0, 1.0, 0.5)
crystal.add_atom("Si", 0.25, 0.25, 0.25, 1.0, 0.5)

peaks = diffraction_simulation(crystal, wavelength=1.5406, two_theta_min=10, two_theta_max=80)

print(f"Total peaks: {len(peaks)}")
print(f"{'hkl':>12} {'2theta':>10} {'d':>10} {'I':>10}")
for p in peaks:
    print(f"({p.h:>3},{p.k:>3},{p.l:>3}) {p.two_theta:>10.3f} {p.d:>10.4f} {p.intensity:>10.2f}")

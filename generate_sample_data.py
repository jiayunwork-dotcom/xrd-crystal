"""
生成示例实验XRD数据
"""

import sys
sys.path.insert(0, '.')
import numpy as np

from xrd_crystal.crystal import Crystal
from xrd_crystal.diffraction import diffraction_simulation, powder_pattern_caglioti

crystal = Crystal(
    a=5.4309, b=5.4309, c=5.4309,
    alpha=90, beta=90, gamma=90,
    space_group="Fd-3m",
    space_group_number=227,
    name="硅 (Si)"
)
crystal.add_atom("Si", 0.0, 0.0, 0.0, 1.0, 0.5)
crystal.add_atom("Si", 0.25, 0.25, 0.25, 1.0, 0.5)

peaks = diffraction_simulation(
    crystal,
    wavelength=1.5406,
    two_theta_min=10,
    two_theta_max=80,
    use_symmetry=False
)

two_theta = np.linspace(10, 80, 1000)
pattern = powder_pattern_caglioti(
    peaks, two_theta,
    U=0.02, V=-0.01, W=0.015,
    eta=0.5,
    wavelength=1.5406
)

np.random.seed(42)
noise = np.random.normal(0, 0.5, len(pattern))
pattern_with_noise = pattern + noise
pattern_with_noise = np.maximum(pattern_with_noise, 0)

with open('examples/silicon_experimental.xy', 'w') as f:
    for tt, inten in zip(two_theta, pattern_with_noise):
        f.write(f"{tt:.4f}  {inten:.4f}\n")

print("示例实验数据已生成: examples/silicon_experimental.xy")
print(f"数据点数: {len(two_theta)}")
print(f"峰数: {len(peaks)}")

"""
调试hkl指标和消光问题
"""

import sys
sys.path.insert(0, '.')

from xrd_crystal.crystal import Crystal
from xrd_crystal.diffraction import diffraction_simulation, _canonicalize_hkl

crystal = Crystal(
    a=5.4309, b=5.4309, c=5.4309,
    alpha=90, beta=90, gamma=90,
    space_group="Fd-3m",
    space_group_number=227,
    name="硅 (Si)"
)
crystal.add_atom("Si", 0.0, 0.0, 0.0, 1.0, 0.5)
crystal.add_atom("Si", 0.25, 0.25, 0.25, 1.0, 0.5)

print("=" * 60)
print("调试hkl规范化")
print("=" * 60)

test_hkls = [(-1,-1,-1), (1,1,1), (-2,-2,0), (2,2,0), (-3,-1,-1), (3,1,1)]
for h, k, l in test_hkls:
    ch, ck, cl = _canonicalize_hkl(h, k, l, "cubic")
    print(f"({h}{k}{l}) -> ({ch}{ck}{cl})")

print("\n" + "=" * 60)
print("调试衍射峰计算")
print("=" * 60)

peaks = diffraction_simulation(
    crystal,
    wavelength=1.5406,
    two_theta_min=10,
    two_theta_max=80,
    use_symmetry=True
)

print(f"总峰数: {len(peaks)}")
print("\n峰列表:")
for i, peak in enumerate(peaks):
    print(f"  {i+1:2d}: ({peak.h:2d}{peak.k:2d}{peak.l:2d})  2θ={peak.two_theta:6.3f}°  I={peak.intensity:8.4f}%  d={peak.d:.4f}")

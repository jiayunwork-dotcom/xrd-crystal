import sys
sys.path.insert(0, '.')
from xrd_crystal.crystal import Crystal
from xrd_crystal.space_group import SpaceGroup
from xrd_crystal.diffraction import diffraction_simulation

crystal = Crystal(
    a=5.4309, b=5.4309, c=5.4309,
    alpha=90, beta=90, gamma=90,
    space_group="Fd-3m",
    space_group_number=227,
    name="硅 (Si)"
)
crystal.add_atom("Si", 0.0, 0.0, 0.0, 1.0, 0.5)
crystal.add_atom("Si", 0.25, 0.25, 0.25, 1.0, 0.5)

sg = SpaceGroup("Fd-3m", 227)
print(f"Space group symops count: {len(sg.symops)}")
for op in sg.symops:
    print(f"  R={op.rotation.tolist()}, t={op.translation.tolist()}")

peaks = diffraction_simulation(crystal, wavelength=1.5406, two_theta_min=10, two_theta_max=90)
print(f"\nPeaks found: {len(peaks)}")
for p in peaks:
    print(f"  ({p.h}{p.k}{p.l})  2θ={p.two_theta:.3f}°  I={p.intensity:.2f}%")

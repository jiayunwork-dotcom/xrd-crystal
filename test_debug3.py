import sys
sys.path.insert(0, '.')
from xrd_crystal.crystal import Crystal
from xrd_crystal.diffraction import diffraction_simulation, bragg_angle

crystal = Crystal(
    a=5.4309, b=5.4309, c=5.4309,
    alpha=90, beta=90, gamma=90,
    space_group="Fd-3m",
    space_group_number=227,
    name="硅 (Si)"
)

d_422 = crystal.d_spacing(4, 2, 2)
tt_422 = bragg_angle(d_422, 1.5406)
print(f"(422): d={d_422:.4f} Å, 2θ={tt_422:.3f}°")

peaks = diffraction_simulation(crystal, wavelength=1.5406, two_theta_min=10, two_theta_max=90)
print(f"\n2θ范围 10-90° 的峰列表:")
for p in peaks:
    print(f"  ({p.h}{p.k}{p.l})  2θ={p.two_theta:.3f}°  d={p.d:.4f} Å  I={p.intensity:.2f}%")

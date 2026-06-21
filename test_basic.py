"""
基本功能测试脚本
"""

import sys
sys.path.insert(0, '.')
import numpy as np

from xrd_crystal.crystal import Crystal, Atom
from xrd_crystal.space_group import SpaceGroup, multiplicity
from xrd_crystal.scattering import atomic_scattering_factor
from xrd_crystal.diffraction import diffraction_simulation, bragg_angle
from xrd_crystal.peak_fitting import pseudo_voigt, caglioti_fwhm
from xrd_crystal.indexing import find_peaks_derivative, index_pattern
from xrd_crystal.visualization import plot_crystal_3d

print("=" * 60)
print("XRD晶体衍射分析工具 - 基本功能测试")
print("=" * 60)

print("\n1. 测试晶体结构...")
crystal = Crystal(
    a=5.4309, b=5.4309, c=5.4309,
    alpha=90, beta=90, gamma=90,
    space_group="Fd-3m",
    space_group_number=227,
    name="硅 (Si)"
)
crystal.add_atom("Si", 0.0, 0.0, 0.0, 1.0, 0.5)
crystal.add_atom("Si", 0.25, 0.25, 0.25, 1.0, 0.5)

print(f"   晶体: {crystal.name}")
print(f"   晶系: {crystal.crystal_system}")
print(f"   体积: {crystal.volume:.4f} Å³")
print(f"   原子数: {len(crystal.atoms)}")
print("   ✓ 晶体结构创建成功")

print("\n2. 测试空间群...")
sg = SpaceGroup("Fd-3m", 227)
print(f"   空间群: {sg.symbol} (#{sg.number})")
print(f"   阶数: {sg.order}")
print(f"   晶系: {sg.crystal_system}")
print("   ✓ 空间群创建成功")

print("\n3. 测试原子散射因子...")
s = 0.3  # sin(theta)/lambda
f_si = atomic_scattering_factor("Si", s)
print(f"   Si 在 s={s}: f={f_si:.4f}")
print("   ✓ 原子散射因子计算成功")

print("\n4. 测试衍射谱计算...")
peaks = diffraction_simulation(
    crystal,
    wavelength=1.5406,
    two_theta_min=10,
    two_theta_max=80,
    use_symmetry=True
)
print(f"   计算得到 {len(peaks)} 个衍射峰")
if peaks:
    print(f"   最强峰: 2θ={peaks[0].two_theta:.3f}°, 强度={peaks[0].intensity:.2f}%")
    print(f"   峰数: {len(peaks)}")
print("   ✓ 衍射谱计算成功")

print("\n5. 测试峰形模拟...")
two_theta_range = np.linspace(10, 80, 500)
pv = pseudo_voigt(two_theta_range, 30.0, 0.15, 0.5)
fwhm_c = caglioti_fwhm(30.0, 0.02, -0.01, 0.015)
print(f"   Pseudo-Voigt 计算完成 (最大值: {np.max(pv):.4f})")
print(f"   Caglioti FWHM 在 30°: {fwhm_c:.4f}°")
print("   ✓ 峰形模拟成功")

print("\n6. 测试寻峰算法...")
test_peaks_positions = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]
test_intensity = np.zeros(1000)
tt = np.linspace(10, 80, 1000)
for pos in test_peaks_positions:
    test_intensity += pseudo_voigt(tt, pos, 0.2, 0.5) * np.random.uniform(0.5, 1.0)

test_intensity += np.random.normal(0, 0.01, len(tt))

found_peaks = find_peaks_derivative(tt, test_intensity, threshold=0.1, min_distance=1.0)
print(f"   已知峰数: {len(test_peaks_positions)}, 检出峰数: {len(found_peaks)}")
print("   ✓ 寻峰算法成功")

print("\n7. 测试3D可视化...")
try:
    fig = plot_crystal_3d(crystal, use_symmetry=True, show_unit_cell=True)
    print(f"   3D图轨迹数: {len(fig.data)}")
    print("   ✓ 3D可视化生成成功")
except Exception as e:
    print(f"   ✗ 3D可视化失败: {e}")

print("\n" + "=" * 60)
print("所有基本功能测试完成!")
print("=" * 60)

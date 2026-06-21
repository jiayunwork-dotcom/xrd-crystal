"""
验证修复：hkl正指标、消光规则、零强度峰过滤
"""

import sys
sys.path.insert(0, '.')
import numpy as np

from xrd_crystal.crystal import Crystal
from xrd_crystal.space_group import SpaceGroup, check_systematic_extinction
from xrd_crystal.diffraction import diffraction_simulation

print("=" * 60)
print("验证修复结果")
print("=" * 60)

print("\n1. 测试 Fd-3m (#227) 系统消光规则...")
sg = SpaceGroup("Fd-3m", 227)

test_cases = [
    (1, 1, 1, False, "111 - 应允许 (全奇)"),
    (2, 0, 0, True,  "200 - 应消光 (不全奇不全偶)"),
    (2, 2, 0, False, "220 - 应允许 (全偶, h+k+l=4, 4n)"),
    (3, 1, 1, False, "311 - 应允许 (全奇)"),
    (2, 2, 2, True,  "222 - 应消光 (全偶, h+k+l=6, 6%4!=0)"),
    (4, 0, 0, False, "400 - 应允许 (全偶, h+k+l=4, 4n)"),
    (3, 3, 1, False, "331 - 应允许 (全奇)"),
    (4, 2, 2, False, "422 - 应允许 (全偶, h+k+l=8, 4n)"),
    (5, 1, 1, False, "511 - 应允许 (全奇)"),
    (3, 3, 3, False, "333 - 应允许 (全奇)"),
    (4, 4, 0, False, "440 - 应允许 (全偶, h+k+l=8, 4n)"),
    (5, 3, 1, False, "531 - 应允许 (全奇)"),
    (6, 2, 0, False, "620 - 应允许 (全偶, h+k+l=8, 4n)"),
    (5, 3, 3, False, "533 - 应允许 (全奇)"),
    (4, 4, 4, False, "444 - 应允许 (全偶, h+k+l=12, 4n)"),
    (1, 1, 0, True,  "110 - 应消光 (不全奇不全偶)"),
    (2, 1, 0, True,  "210 - 应消光 (不全奇不全偶)"),
    (3, 2, 1, True,  "321 - 应消光 (不全奇不全偶)"),
]

all_correct = True
for h, k, l, expected_extinct, desc in test_cases:
    result = check_systematic_extinction(h, k, l, 227)
    status = "✓" if result == expected_extinct else "✗"
    if result != expected_extinct:
        all_correct = False
        print(f"   {status} ({h}{k}{l}): {desc} → 实际={'消光' if result else '允许'}")
    else:
        print(f"   {status} ({h}{k}{l}): {desc} → {'消光' if result else '允许'}")

if all_correct:
    print("   ✓ 所有消光规则测试通过!")
else:
    print("   ✗ 部分消光规则测试失败!")

print("\n2. 测试硅晶体衍射谱计算...")
crystal = Crystal(
    a=5.4309, b=5.4309, c=5.4309,
    alpha=90, beta=90, gamma=90,
    space_group="Fd-3m",
    space_group_number=227,
    name="硅 (Si)"
)
crystal.add_atom("Si", 0.0, 0.0, 0.0, 1.0, 0.5)

peaks = diffraction_simulation(
    crystal,
    wavelength=1.5406,
    two_theta_min=10,
    two_theta_max=80,
    use_symmetry=True
)

print(f"   有效峰数: {len(peaks)}")

print("\n   峰列表:")
all_positive = True
no_zero_intensity = True
for peak in peaks:
    sign = ""
    if peak.h < 0 or peak.k < 0 or peak.l < 0:
        all_positive = False
        sign = " ← 负指标!"
    if peak.intensity <= 0:
        no_zero_intensity = False
        sign += " ← 零强度!"
    print(f"   ({peak.h}{peak.k}{peak.l})  2θ={peak.two_theta:.3f}°  d={peak.d:.4f} Å  I={peak.intensity:.2f}%{sign}")

if all_positive:
    print("   ✓ 所有hkl指标均为正数!")
else:
    print("   ✗ 仍有负指标存在!")

if no_zero_intensity:
    print("   ✓ 没有零强度的消光峰!")
else:
    print("   ✗ 仍有零强度峰存在!")

expected_peaks = [(1,1,1), (2,2,0), (3,1,1), (4,0,0), (3,3,1), (4,2,2)]
print(f"\n3. 验证硅的期望峰...")
for h, k, l in expected_peaks:
    found = any(p.h == h and p.k == k and p.l == l for p in peaks)
    status = "✓" if found else "✗"
    print(f"   {status} ({h}{k}{l}) - {'找到' if found else '未找到'}")

extinct_peaks = [(2,0,0), (2,2,2), (4,2,0)]
print(f"\n4. 验证消光峰不存在...")
for h, k, l in extinct_peaks:
    found = any(p.h == h and p.k == k and p.l == l for p in peaks)
    status = "✓" if not found else "✗"
    print(f"   {status} ({h}{k}{l}) - {'正确消光' if not found else '不应出现!'}")

print("\n" + "=" * 60)
print("验证完成!")
print("=" * 60)

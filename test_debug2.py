import sys
sys.path.insert(0, '.')
import numpy as np
from xrd_crystal.crystal import Crystal
from xrd_crystal.space_group import SpaceGroup, check_systematic_extinction
from xrd_crystal.diffraction import _canonicalize_hkl, _get_unique_key

crystal = Crystal(
    a=5.4309, b=5.4309, c=5.4309,
    alpha=90, beta=90, gamma=90,
    space_group="Fd-3m",
    space_group_number=227,
    name="硅 (Si)"
)

for h in range(-8, 9):
    for k in range(-8, 9):
        for l in range(-8, 9):
            if h == 0 and k == 0 and l == 0:
                continue
            
            ch, ck, cl = _canonicalize_hkl(h, k, l, crystal.crystal_system)
            if (ch, ck, cl) == (4, 2, 2):
                extinct = check_systematic_extinction(h, k, l, 227)
                key = _get_unique_key(h, k, l, crystal.crystal_system)
                print(f"(hkl)=({h},{k},{l}) → canonical=({ch},{ck},{cl}), extinct={extinct}, key={key}")
                if not extinct:
                    d = crystal.d_spacing(h, k, l)
                    if 0.5 < d < 10:
                        print(f"  → d={d:.4f}, 应该出现在峰列表中!")
                    break

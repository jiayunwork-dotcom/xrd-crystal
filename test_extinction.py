import sys
sys.path.insert(0, '.')
from xrd_crystal.space_group import check_systematic_extinction, _check_fd3m_extinction, _all_even_or_all_odd

print("Testing _all_even_or_all_odd with negative indices:")
print(f"  (-1,-1,-1): {_all_even_or_all_odd(-1,-1,-1)}")
print(f"  (-2,0,0):   {_all_even_or_all_odd(-2,0,0)}")
print(f"  (1,1,1):    {_all_even_or_all_odd(1,1,1)}")
print(f"  (2,0,0):    {_all_even_or_all_odd(2,0,0)}")
print(f"  (1,0,0):    {_all_even_or_all_odd(1,0,0)}")

print("\nTesting _check_fd3m_extinction:")
test_hkl = [
    (1,1,1), (-1,-1,-1), (2,0,0), (-2,0,0), (2,2,0), (-2,-2,0),
    (3,1,1), (-3,-1,-1), (2,2,2), (-2,-2,-2), (4,2,0), (-4,-2,0),
    (4,0,0), (-4,0,0), (3,3,1), (-3,-3,-1),
]
for h, k, l in test_hkl:
    extinct = _check_fd3m_extinction(h, k, l)
    expected = "EXTINCT" if (h+k+l) % 2 != 0 or (h%2==0 and k%2==0 and l%2==0 and (h+k+l)%4!=0) else "allowed"
    match = "✓" if (extinct and expected=="EXTINCT") or (not extinct and expected=="allowed") else "✗"
    print(f"  ({h:>3},{k:>3},{l:>3}): extinct={extinct}, expected={expected} {match}")

print("\nTesting check_systematic_extinction for sg#227:")
for h, k, l in [(1,1,1), (2,0,0), (2,2,0), (3,1,1), (2,2,2), (4,2,0), (4,0,0), (3,3,1)]:
    extinct = check_systematic_extinction(h, k, l, 227)
    print(f"  ({h},{k},{l}): extinct={extinct}")

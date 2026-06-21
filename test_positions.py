import numpy as np
from xrd_crystal.diffraction import _get_point_group_symops, _get_lattice_translations
from xrd_crystal.space_group import _get_lattice_type

lattice_type = _get_lattice_type(227)
print(f'Lattice type: {lattice_type}')

pg_ops = _get_point_group_symops("cubic", 227)
print(f'Point group ops: {len(pg_ops)}')

lattice_trans = _get_lattice_translations(lattice_type)
print(f'Lattice translations: {len(lattice_trans)}')

diamond_trans = [np.array([0.0, 0.0, 0.0]), np.array([0.25, 0.25, 0.25])]
print(f'Diamond glide translations: {len(diamond_trans)}')

atom_pos = np.array([0.0, 0.0, 0.0])

full_positions = []
for dg in diamond_trans:
    for pg in pg_ops:
        rotated = pg.apply(atom_pos)
        shifted = (rotated + dg) % 1.0
        for lt in lattice_trans:
            pos = (shifted + lt) % 1.0
            full_positions.append(pos)

seen = set()
unique = []
for pos in full_positions:
    key = tuple(np.round(pos, 6))
    if key not in seen:
        seen.add(key)
        unique.append(pos)

print(f'Total unique positions: {len(unique)}')
for p in sorted([tuple(np.round(x, 4)) for x in unique]):
    print(f'  {p}')

# Check against standard 8a positions
standard_8a = [
    (0, 0, 0),
    (0, 0.5, 0.5),
    (0.5, 0, 0.5),
    (0.5, 0.5, 0),
    (0.25, 0.25, 0.25),
    (0.25, 0.75, 0.75),
    (0.75, 0.25, 0.75),
    (0.75, 0.75, 0.25),
]

print('\nStandard 8a:')
for p in standard_8a:
    print(f'  {p}')

# Now compute F(111)
for hkl in [(1,1,1), (2,2,0), (3,1,1)]:
    h, k, l = hkl
    F = sum(np.exp(2j * np.pi * (h*p[0] + k*p[1] + l*p[2])) for p in unique)
    print(f'F({h}{k}{l}) from my positions = {F:.4f}, |F|^2 = {abs(F)**2:.2f}')

for hkl in [(1,1,1), (2,2,0), (3,1,1)]:
    h, k, l = hkl
    F = sum(np.exp(2j * np.pi * (h*p[0] + k*p[1] + l*p[2])) for p in standard_8a)
    print(f'F({h}{k}{l}) from standard  = {F:.4f}, |F|^2 = {abs(F)**2:.2f}')

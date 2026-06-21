import numpy as np

positions_8a = [
    (0, 0, 0),
    (0, 1/2, 1/2),
    (1/2, 0, 1/2),
    (1/2, 1/2, 0),
    (1/4, 1/4, 1/4),
    (1/4, 3/4, 3/4),
    (3/4, 1/4, 3/4),
    (3/4, 3/4, 1/4),
]

print('Standard 8a positions for Fd-3m:')
for p in positions_8a:
    print(f'  {p}')

for hkl in [(1,1,1), (2,2,0), (3,1,1), (4,0,0), (3,3,1)]:
    h, k, l = hkl
    phases = [np.exp(2j * np.pi * (h*x + k*y + l*z)) for x, y, z in positions_8a]
    F = sum(phases)
    print(f'F({h}{k}{l}) = {F:.4f}, |F|^2 = {abs(F)**2:.2f}')

print()
print('My program positions:')
my_positions = [
    (0.0, 0.0, 0.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
    (0.75, 0.25, 0.75),
    (0.25, 0.75, 0.25),
    (0.25, 0.25, 0.25),
    (0.75, 0.75, 0.75),
]

for hkl in [(1,1,1), (2,2,0), (3,1,1), (4,0,0), (3,3,1)]:
    h, k, l = hkl
    phases = [np.exp(2j * np.pi * (h*x + k*y + l*z)) for x, y, z in my_positions]
    F = sum(phases)
    print(f'F({h}{k}{l}) = {F:.4f}, |F|^2 = {abs(F)**2:.2f}')

print()
print('Theoretical:')
print('For diamond structure Fd-3m:')
print('  F^2(111) = 32*f^2')
print('  F^2(220) = 64*f^2')
print('  Ratio = 2.0')
print()
print('With LP and multiplicity:')
wavelength = 1.5406
a = 5.431

for hkl in [(1,1,1), (2,2,0)]:
    h, k, l = hkl
    d = a / np.sqrt(h**2 + k**2 + l**2)
    two_theta = 2 * np.degrees(np.arcsin(wavelength / (2 * d)))
    theta = two_theta / 2
    LP = (1 + np.cos(np.radians(two_theta))**2) / (np.sin(np.radians(theta))**2 * np.cos(np.radians(theta)))
    F2_std = abs(sum(np.exp(2j * np.pi * (h*x + k*y + l*z)) for x, y, z in positions_8a))**2
    print(f'  ({h}{k}{l}): 2theta={two_theta:.2f}, LP={LP:.2f}, |F|^2={F2_std:.2f}')

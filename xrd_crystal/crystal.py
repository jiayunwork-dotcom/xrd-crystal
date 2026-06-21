"""
晶体结构数据模型
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Atom:
    """原子类"""
    element: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    b_iso: float = 1.0

    @property
    def frac_coord(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def __repr__(self):
        return f"Atom({self.element}, ({self.x:.4f}, {self.y:.4f}, {self.z:.4f}), occ={self.occupancy}, B={self.b_iso})"


@dataclass
class Crystal:
    """晶体结构类"""
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    space_group: str = "P1"
    space_group_number: int = 1
    atoms: List[Atom] = field(default_factory=list)
    name: str = "Untitled"

    @property
    def crystal_system(self) -> str:
        """确定晶系"""
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = np.deg2rad(self.alpha), np.deg2rad(self.beta), np.deg2rad(self.gamma)

        tol = 1e-3

        if abs(a - b) < tol and abs(b - c) < tol and abs(alpha - np.pi/2) < tol and abs(beta - np.pi/2) < tol and abs(gamma - np.pi/2) < tol:
            return "cubic"
        elif abs(a - b) < tol and abs(alpha - np.pi/2) < tol and abs(beta - np.pi/2) < tol and abs(gamma - np.pi/2) < tol:
            return "tetragonal"
        elif abs(a - b) < tol and abs(alpha - np.pi/2) < tol and abs(beta - np.pi/2) < tol and abs(gamma - 2*np.pi/3) < tol:
            return "hexagonal"
        elif abs(alpha - np.pi/2) < tol and abs(beta - np.pi/2) < tol and abs(gamma - np.pi/2) < tol:
            return "orthorhombic"
        elif abs(alpha - np.pi/2) < tol and abs(gamma - np.pi/2) < tol:
            return "monoclinic"
        else:
            return "triclinic"

    @property
    def volume(self) -> float:
        """计算晶胞体积"""
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = np.deg2rad(self.alpha), np.deg2rad(self.beta), np.deg2rad(self.gamma)

        V = a * b * c * np.sqrt(
            1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2
            + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
        return V

    @property
    def metric_tensor(self) -> np.ndarray:
        """计算度量张量"""
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = np.deg2rad(self.alpha), np.deg2rad(self.beta), np.deg2rad(self.gamma)

        G = np.array([
            [a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
            [a*b*np.cos(gamma), b**2, b*c*np.cos(alpha)],
            [a*c*np.cos(beta), b*c*np.cos(alpha), c**2]
        ])
        return G

    def d_spacing(self, h: int, k: int, l: int) -> float:
        """计算面间距d"""
        G_inv = np.linalg.inv(self.metric_tensor)
        hkl = np.array([h, k, l])
        d_inv_sq = hkl @ G_inv @ hkl
        return 1.0 / np.sqrt(d_inv_sq)

    def frac_to_cart(self, frac: np.ndarray) -> np.ndarray:
        """分数坐标转笛卡尔坐标"""
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = np.deg2rad(self.alpha), np.deg2rad(self.beta), np.deg2rad(self.gamma)

        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)

        ax = a
        ay = 0.0
        az = 0.0

        bx = b * cos_gamma
        by = b * sin_gamma
        bz = 0.0

        cx = c * cos_beta
        cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        cz = c * np.sqrt(1 - cos_beta**2 - ((cos_alpha - cos_beta * cos_gamma) / sin_gamma)**2)

        frac = np.asarray(frac)
        x = frac[0] * ax + frac[1] * bx + frac[2] * cx
        y = frac[0] * ay + frac[1] * by + frac[2] * cy
        z = frac[0] * az + frac[1] * bz + frac[2] * cz

        return np.array([x, y, z])

    def add_atom(self, element: str, x: float, y: float, z: float, occupancy: float = 1.0, b_iso: float = 1.0):
        """添加原子"""
        self.atoms.append(Atom(element, x, y, z, occupancy, b_iso))

    def __repr__(self):
        return (f"Crystal({self.name}, sg={self.space_group}, "
                f"a={self.a:.4f}, b={self.b:.4f}, c={self.c:.4f}, "
                f"alpha={self.alpha:.2f}, beta={self.beta:.2f}, gamma={self.gamma:.2f}, "
                f"atoms={len(self.atoms)})")

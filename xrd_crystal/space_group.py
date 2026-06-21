"""
空间群对称操作和系统消光规则
"""

import numpy as np
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass
class SymmetryOperation:
    """对称操作类，使用Seitz符号 (R, t)"""
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, coord: np.ndarray) -> np.ndarray:
        """对分数坐标应用对称操作"""
        result = self.rotation @ coord + self.translation
        result = result % 1.0
        return result

    def __repr__(self):
        return f"SymOp(R={self.rotation.tolist()}, t={self.translation.tolist()})"


def parse_symop_xyz(symop_str: str) -> SymmetryOperation:
    """解析x,y,z格式的对称操作字符串"""
    parts = [p.strip() for p in symop_str.split(',')]
    if len(parts) != 3:
        raise ValueError(f"无效的对称操作: {symop_str}")

    R = np.zeros((3, 3))
    t = np.zeros(3)

    for i, part in enumerate(parts):
        part = part.replace(' ', '')
        part = part.replace('+', ' +').replace('-', ' -')
        terms = part.split()

        for term in terms:
            if not term:
                continue

            if 'x' in term:
                coeff = 1.0
                if term == '-x':
                    coeff = -1.0
                elif term == '+x':
                    coeff = 1.0
                elif term == 'x':
                    coeff = 1.0
                else:
                    coeff = float(term.replace('x', ''))
                R[i, 0] = coeff
            elif 'y' in term:
                coeff = 1.0
                if term == '-y':
                    coeff = -1.0
                elif term == '+y':
                    coeff = 1.0
                elif term == 'y':
                    coeff = 1.0
                else:
                    coeff = float(term.replace('y', ''))
                R[i, 1] = coeff
            elif 'z' in term:
                coeff = 1.0
                if term == '-z':
                    coeff = -1.0
                elif term == '+z':
                    coeff = 1.0
                elif term == 'z':
                    coeff = 1.0
                else:
                    coeff = float(term.replace('z', ''))
                R[i, 2] = coeff
            else:
                try:
                    t[i] = float(term)
                except ValueError:
                    if '/' in term:
                        num, den = term.split('/')
                        t[i] = float(num) / float(den)

    return SymmetryOperation(R, t)


def generate_equivalent_positions(atom_coord: np.ndarray, symops: List[SymmetryOperation]) -> List[np.ndarray]:
    """生成所有等效原子位置"""
    positions = []
    seen = set()

    for symop in symops:
        pos = symop.apply(atom_coord)
        pos = np.round(pos, decimals=6) % 1.0
        pos_tuple = tuple(pos)

        if pos_tuple not in seen:
            seen.add(pos_tuple)
            positions.append(pos)

    return positions


class SpaceGroup:
    """空间群类"""

    def __init__(self, symbol: str = "P1", number: int = 1):
        self.symbol = symbol
        self.number = number
        self.symops = self._get_symops()

    def _get_symops(self) -> List[SymmetryOperation]:
        """获取空间群的对称操作"""
        symop_strs = SPACE_GROUP_SYMOPS.get(self.number, ["x,y,z"])
        return [parse_symop_xyz(s) for s in symop_strs]

    def is_extinct(self, h: int, k: int, l: int) -> bool:
        """判断(hkl)是否系统消光"""
        return check_systematic_extinction(h, k, l, self.number)

    @property
    def crystal_system(self) -> str:
        """获取晶系"""
        return get_crystal_system(self.number)

    @property
    def order(self) -> int:
        """空间群阶数（等效位置数）"""
        return len(self.symops)

    def __repr__(self):
        return f"SpaceGroup({self.symbol}, #{self.number}, order={self.order})"


def get_crystal_system(sg_number: int) -> str:
    """根据空间群编号获取晶系"""
    if 1 <= sg_number <= 2:
        return "triclinic"
    elif 3 <= sg_number <= 15:
        return "monoclinic"
    elif 16 <= sg_number <= 74:
        return "orthorhombic"
    elif 75 <= sg_number <= 142:
        return "tetragonal"
    elif 143 <= sg_number <= 167:
        return "trigonal"
    elif 168 <= sg_number <= 194:
        return "hexagonal"
    elif 195 <= sg_number <= 230:
        return "cubic"
    else:
        return "unknown"


def check_systematic_extinction(h: int, k: int, l: int, sg_number: int) -> bool:
    """
    检查系统消光规则
    返回True表示该反射被消光
    """
    if sg_number == 1:
        return False

    if sg_number in [3, 4]:
        if h != 0 and l != 0 and (h + l) % 2 != 0:
            pass
        return False

    if sg_number in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        if sg_number in [5, 6, 7, 8]:
            if l % 2 != 0:
                return True
        elif sg_number in [9, 10, 11, 12]:
            if (h + l) % 2 != 0:
                return True

    if sg_number in [16, 17, 18, 19, 20, 21, 22, 23, 24]:
        pass
    elif sg_number in [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]:
        if h + k + l % 2 != 0:
            if sg_number in [25, 26, 27, 28, 29, 30, 31, 32, 33, 34]:
                pass
            elif sg_number in [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]:
                if h % 2 != 0 or k % 2 != 0 or l % 2 != 0:
                    return True
    elif 47 <= sg_number <= 74:
        if (h + k + l) % 2 != 0:
            return True

    if 75 <= sg_number <= 142:
        if 75 <= sg_number <= 88:
            pass
        elif 89 <= sg_number <= 98:
            if h % 2 != 0 or k % 2 != 0:
                return True
        elif 99 <= sg_number <= 110:
            if (h + k) % 2 != 0:
                return True
        elif 111 <= sg_number <= 122:
            pass
        elif 123 <= sg_number <= 142:
            if (h + k + l) % 2 != 0:
                return True

    if 168 <= sg_number <= 194:
        if 168 <= sg_number <= 173:
            pass
        elif 174 <= sg_number <= 194:
            if (h + 2 * k) % 3 != 0 and l == 0:
                pass
            pass

    if 195 <= sg_number <= 230:
        if 195 <= sg_number <= 206:
            if (h + k + l) % 2 != 0:
                return True
        elif 207 <= sg_number <= 220:
            if h % 2 != 0 or k % 2 != 0 or l % 2 != 0:
                return True
        elif 221 <= sg_number <= 230:
            if not all_even_or_all_odd(h, k, l):
                return True

    return False


def all_even_or_all_odd(h, k, l):
    all_even = (h % 2 == 0 and k % 2 == 0 and l % 2 == 0)
    all_odd = (h % 2 != 0 and k % 2 != 0 and l % 2 != 0)
    return all_even or all_odd


def multiplicity(h: int, k: int, l: int, crystal_system: str) -> int:
    """计算多重性因子"""
    if crystal_system == "cubic":
        return _cubic_multiplicity(h, k, l)
    elif crystal_system == "tetragonal":
        return _tetragonal_multiplicity(h, k, l)
    elif crystal_system == "hexagonal" or crystal_system == "trigonal":
        return _hexagonal_multiplicity(h, k, l)
    elif crystal_system == "orthorhombic":
        return _orthorhombic_multiplicity(h, k, l)
    elif crystal_system == "monoclinic":
        return _monoclinic_multiplicity(h, k, l)
    else:
        return _triclinic_multiplicity(h, k, l)


def _cubic_multiplicity(h, k, l):
    h, k, l = abs(h), abs(k), abs(l)
    if h == 0 and k == 0 and l == 0:
        return 1
    if h == k == l:
        return 8
    if h == 0 and k == 0:
        return 6
    if h == 0 and k == l:
        return 12
    if h == k and l == 0:
        return 12
    if h == k:
        return 24
    if k == l or h == l:
        return 24
    if h == 0 or k == 0 or l == 0:
        return 24
    return 48


def _tetragonal_multiplicity(h, k, l):
    h, k, l = abs(h), abs(k), abs(l)
    if h == 0 and k == 0:
        return 2 if l != 0 else 1
    if h == k:
        return 8 if l != 0 else 4
    if h == 0 or k == 0:
        return 4 if l != 0 else 2
    return 8


def _hexagonal_multiplicity(h, k, l):
    h, k, l = abs(h), abs(k), abs(l)
    if h == 0 and k == 0:
        return 2 if l != 0 else 1
    if h == k:
        return 12 if l != 0 else 6
    if h == 0 or k == 0 or h + k == 0:
        return 6 if l != 0 else 3
    return 12


def _orthorhombic_multiplicity(h, k, l):
    h, k, l = abs(h), abs(k), abs(l)
    count = 1
    if h != 0:
        count *= 2
    if k != 0:
        count *= 2
    if l != 0:
        count *= 2
    return count


def _monoclinic_multiplicity(h, k, l):
    h, k, l = abs(h), abs(k), abs(l)
    count = 1
    if h != 0:
        count *= 2
    if l != 0:
        count *= 2
    return count


def _triclinic_multiplicity(h, k, l):
    if h == 0 and k == 0 and l == 0:
        return 1
    return 2


SPACE_GROUP_SYMOPS = {
    1: ["x,y,z"],
    2: ["x,y,z", "-x,-y,-z"],
    3: ["x,y,z", "-x,y,-z"],
    4: ["x,y,z", "x,-y,z"],
    5: ["x,y,z", "-x,y,-z", "x,y+1/2,z", "-x,y+1/2,-z"],
    14: ["x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z"],
    15: ["x,y,z", "-x,-y,-z", "-x,y,z", "x,-y,z", "x,y,-z", "-x,-y,z", "-x,y,-z", "x,-y,-z"],
    16: ["x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z"],
    19: ["x,y,z", "x+1/2,y+1/2,z", "x+1/2,y,z+1/2", "x,y+1/2,z+1/2",
         "-x,-y,z", "-x+1/2,-y+1/2,z", "-x+1/2,-y,z+1/2", "-x,-y+1/2,z+1/2",
         "-x,y,-z", "-x+1/2,y+1/2,-z", "-x+1/2,y,-z+1/2", "-x,y+1/2,-z+1/2",
         "x,-y,-z", "x+1/2,-y+1/2,-z", "x+1/2,-y,-z+1/2", "x,-y+1/2,-z+1/2"],
    47: ["x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z",
         "x+1/2,y+1/2,z+1/2", "-x+1/2,-y+1/2,z+1/2", "-x+1/2,y+1/2,-z+1/2", "x+1/2,-y+1/2,-z+1/2"],
    62: ["x,y,z", "y,z,x", "z,x,y",
         "-x,-y,z", "-y,-z,x", "-z,-x,y",
         "-x,y,-z", "-y,z,-x", "z,-x,-y",
         "x,-y,-z", "y,-z,-x", "-z,x,-y",
         "x+1/2,y+1/2,z+1/2", "y+1/2,z+1/2,x+1/2", "z+1/2,x+1/2,y+1/2",
         "-x+1/2,-y+1/2,z+1/2", "-y+1/2,-z+1/2,x+1/2", "-z+1/2,-x+1/2,y+1/2",
         "-x+1/2,y+1/2,-z+1/2", "-y+1/2,z+1/2,-x+1/2", "z+1/2,-x+1/2,-y+1/2",
         "x+1/2,-y+1/2,-z+1/2", "y+1/2,-z+1/2,-x+1/2", "-z+1/2,x+1/2,-y+1/2"],
    123: ["x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z",
          "y,x,-z", "-y,-x,-z", "-y,x,z", "y,-x,z",
          "x+1/2,y+1/2,z+1/2", "-x+1/2,-y+1/2,z+1/2", "-x+1/2,y+1/2,-z+1/2", "x+1/2,-y+1/2,-z+1/2",
          "y+1/2,x+1/2,-z+1/2", "-y+1/2,-x+1/2,-z+1/2", "-y+1/2,x+1/2,z+1/2", "y+1/2,-x+1/2,z+1/2"],
    129: ["x,y,z", "-x,-y,z", "x+1/2,-y+1/2,z+1/2", "-x+1/2,y+1/2,z+1/2",
          "x+1/2,y+1/2,-z+1/2", "-x+1/2,-y+1/2,-z+1/2", "x,-y,-z", "-x,y,-z"],
    141: ["x,y,z", "-y,x-y,z", "-x+y,-x,z",
          "y,-x+y,z+2/3", "x+y,-x,z+1/3", "-x,-y,z+2/3",
          "x+1/2,y+1/2,z+1/2", "-y+1/2,x-y+1/2,z+1/2", "-x+y+1/2,-x+1/2,z+1/2",
          "y+1/2,-x+y+1/2,z+1/6", "x+y+1/2,-x+1/2,z+5/6", "-x+1/2,-y+1/2,z+1/6",
          "-x,-y,-z", "y,-x+y,-z", "x-y,x,-z",
          "-y,x-y,-z+1/3", "-x-y,x,-z+2/3", "x,y,-z+1/3",
          "-x+1/2,-y+1/2,-z+1/2", "y+1/2,-x+y+1/2,-z+1/2", "x-y+1/2,x+1/2,-z+1/2",
          "-y+1/2,x-y+1/2,-z+5/6", "-x-y+1/2,x+1/2,-z+1/6", "x+1/2,y+1/2,-z+5/6"],
    194: ["x,y,z", "-y,x-y,z", "-x+y,-x,z",
          "-x,-y,z+1/2", "y,-x+y,z+1/2", "x-y,x,z+1/2",
          "y,x,-z", "x-y,-y,-z", "-x,x-y,-z",
          "-y,-x,-z+1/2", "-x+y,y,-z+1/2", "x,-x+y,-z+1/2"],
    221: ["x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z",
          "z,x,y", "-z,-x,y", "-z,x,-y", "z,-x,-y",
          "y,z,x", "-y,-z,x", "-y,z,-x", "y,-z,-x",
          "-x,-y,-z", "x,y,-z", "x,-y,z", "-x,y,z",
          "-z,-x,-y", "z,x,-y", "z,-x,y", "-z,x,y",
          "-y,-z,-x", "y,z,-x", "y,-z,x", "-y,z,x"],
    225: ["x,y,z", "x+1/2,y+1/2,z", "x+1/2,y,z+1/2", "x,y+1/2,z+1/2",
          "-x,-y,z", "-x+1/2,-y+1/2,z", "-x+1/2,-y,z+1/2", "-x,-y+1/2,z+1/2",
          "-x,y,-z", "-x+1/2,y+1/2,-z", "-x+1/2,y,-z+1/2", "-x,y+1/2,-z+1/2",
          "x,-y,-z", "x+1/2,-y+1/2,-z", "x+1/2,-y,-z+1/2", "x,-y+1/2,-z+1/2",
          "z,x,y", "z+1/2,x+1/2,y", "z+1/2,x,y+1/2", "z,x+1/2,y+1/2",
          "-z,-x,y", "-z+1/2,-x+1/2,y", "-z+1/2,-x,y+1/2", "-z,-x+1/2,y+1/2",
          "-z,x,-y", "-z+1/2,x+1/2,-y", "-z+1/2,x,-y+1/2", "-z,x+1/2,-y+1/2",
          "z,-x,-y", "z+1/2,-x+1/2,-y", "z+1/2,-x,-y+1/2", "z,-x+1/2,-y+1/2",
          "y,z,x", "y+1/2,z+1/2,x", "y+1/2,z,x+1/2", "y,z+1/2,x+1/2",
          "-y,-z,x", "-y+1/2,-z+1/2,x", "-y+1/2,-z,x+1/2", "-y,-z+1/2,x+1/2",
          "-y,z,-x", "-y+1/2,z+1/2,-x", "-y+1/2,z,-x+1/2", "-y,z+1/2,-x+1/2",
          "y,-z,-x", "y+1/2,-z+1/2,-x", "y+1/2,-z,-x+1/2", "y,-z+1/2,-x+1/2",
          "-x,-y,-z", "-x+1/2,-y+1/2,-z", "-x+1/2,-y,-z+1/2", "-x,-y+1/2,-z+1/2",
          "x,y,-z", "x+1/2,y+1/2,-z", "x+1/2,y,-z+1/2", "x,y+1/2,-z+1/2",
          "x,-y,z", "x+1/2,-y+1/2,z", "x+1/2,-y,z+1/2", "x,-y+1/2,z+1/2",
          "-x,y,z", "-x+1/2,y+1/2,z", "-x+1/2,y,z+1/2", "-x,y+1/2,z+1/2",
          "-z,-x,-y", "-z+1/2,-x+1/2,-y", "-z+1/2,-x,-y+1/2", "-z,-x+1/2,-y+1/2",
          "z,x,-y", "z+1/2,x+1/2,-y", "z+1/2,x,-y+1/2", "z,x+1/2,-y+1/2",
          "z,-x,y", "z+1/2,-x+1/2,y", "z+1/2,-x,y+1/2", "z,-x+1/2,y+1/2",
          "-z,x,y", "-z+1/2,x+1/2,y", "-z+1/2,x,y+1/2", "-z,x+1/2,y+1/2",
          "-y,-z,-x", "-y+1/2,-z+1/2,-x", "-y+1/2,-z,-x+1/2", "-y,-z+1/2,-x+1/2",
          "y,z,-x", "y+1/2,z+1/2,-x", "y+1/2,z,-x+1/2", "y,z+1/2,-x+1/2",
          "y,-z,x", "y+1/2,-z+1/2,x", "y+1/2,-z,x+1/2", "y,-z+1/2,x+1/2",
          "-y,z,x", "-y+1/2,z+1/2,x", "-y+1/2,z,x+1/2", "-y,z+1/2,x+1/2"],
}


def get_space_group_by_number(number: int) -> SpaceGroup:
    """根据编号获取空间群"""
    symbol = get_sg_symbol(number)
    return SpaceGroup(symbol, number)


def get_space_group_by_symbol(symbol: str) -> SpaceGroup:
    """根据符号获取空间群"""
    number = get_sg_number(symbol)
    return SpaceGroup(symbol, number)


def get_sg_symbol(number: int) -> str:
    """根据空间群编号获取H-M符号"""
    sg_symbols = {
        1: "P1", 2: "P-1",
        3: "P2", 4: "P2", 5: "C2",
        14: "P2/c", 15: "C2/c",
        16: "P222", 19: "P222",
        47: "Pmmm", 62: "Pnma",
        75: "P4", 81: "P4",
        89: "P422", 99: "P422",
        123: "P4/mmm", 129: "P4/nmm",
        141: "I41/amd",
        143: "P3", 147: "P3",
        150: "P321", 155: "R3",
        160: "R3c",
        168: "P6", 173: "P63",
        174: "P-6",
        194: "P63/mmc",
        195: "P23", 200: "Pm-3",
        207: "F23", 221: "Pm-3m",
        225: "Fm-3m", 229: "Im-3m",
    }
    return sg_symbols.get(number, f"SG_{number}")


def get_sg_number(symbol: str) -> int:
    """根据H-M符号获取空间群编号"""
    sg_numbers = {
        "P1": 1, "P-1": 2,
        "P2": 3, "P21": 4, "C2": 5,
        "P2/c": 14, "C2/c": 15,
        "P222": 16,
        "Pmmm": 47, "Pnma": 62,
        "P4/mmm": 123, "P4/nmm": 129,
        "I41/amd": 141,
        "P63/mmc": 194,
        "Pm-3m": 221, "Fm-3m": 225, "Im-3m": 229,
    }
    return sg_numbers.get(symbol, 1)

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
    
    基于晶格中心类型和滑移面/螺旋轴的消光条件判定
    """
    if sg_number == 1:
        return False

    if sg_number == 227:
        return _check_fd3m_extinction(h, k, l)
    elif sg_number == 225:
        return _check_fm3m_extinction(h, k, l)
    elif sg_number == 221:
        return _check_pm3m_extinction(h, k, l)
    elif sg_number == 229:
        return _check_im3m_extinction(h, k, l)
    elif sg_number == 194:
        return _check_p63mmc_extinction(h, k, l)
    
    lattice_type = _get_lattice_type(sg_number)
    
    if _check_lattice_extinction(h, k, l, lattice_type):
        return True
    
    if 75 <= sg_number <= 142:
        return _check_tetragonal_extinction(h, k, l, sg_number)
    
    if 16 <= sg_number <= 74:
        return _check_orthorhombic_extinction(h, k, l, sg_number)
    
    if 3 <= sg_number <= 15:
        return _check_monoclinic_extinction(h, k, l, sg_number)
    
    return False


def _get_lattice_type(sg_number: int) -> str:
    """根据空间群编号获取晶格中心类型"""
    if sg_number in _P_GROUPS:
        return "P"
    elif sg_number in _I_GROUPS:
        return "I"
    elif sg_number in _F_GROUPS:
        return "F"
    elif sg_number in _C_GROUPS:
        return "C"
    elif sg_number in _A_GROUPS:
        return "A"
    elif sg_number in _R_GROUPS:
        return "R"
    return "P"


_P_GROUPS = (
    set(range(1, 3))
    | set(range(3, 6))
    | {6, 7, 8}
    | set(range(16, 75))
    | set(range(75, 89))
    | set(range(143, 148))
    | set(range(168, 174))
    | set(range(195, 199))
)

_I_GROUPS = {
    14, 23, 24,
    44, 45, 46,
    71, 72, 73, 74,
    79, 80, 82, 87, 88,
    97, 98, 107, 108, 109, 110,
    121, 122,
    139, 140, 141, 142,
    148, 155, 160, 161, 166, 167,
    197, 199, 204, 206,
    211, 214, 217, 220,
    229, 230,
}

_F_GROUPS = {
    16, 17, 18, 19, 20, 21, 22, 23,
    42, 43,
    69, 70,
    225, 226, 227, 228,
    196, 202, 203, 209,
    216, 219,
}

_C_GROUPS = {
    5, 9, 10, 11, 12, 15,
    20, 21, 35, 36, 37, 38, 39, 40, 41,
    63, 64, 65, 66, 67, 68,
}

_A_GROUPS = {
    38, 39, 40, 41,
}

_R_GROUPS = {
    146, 148, 155, 160, 161, 166, 167,
}


def _check_lattice_extinction(h: int, k: int, l: int, lattice_type: str) -> bool:
    """
    检查晶格中心类型引起的系统消光
    
    P: 无消光
    I: h+k+l = 奇数时消光
    F: h,k,l 不全奇不全偶时消光
    C: h+k = 奇数时消光
    A: k+l = 奇数时消光
    R (六方): h-k = 3n外消光 (简化处理)
    """
    if lattice_type == "P":
        return False
    elif lattice_type == "I":
        if (h + k + l) % 2 != 0:
            return True
    elif lattice_type == "F":
        if not _all_even_or_all_odd(h, k, l):
            return True
    elif lattice_type == "C":
        if (h + k) % 2 != 0:
            return True
    elif lattice_type == "A":
        if (k + l) % 2 != 0:
            return True
    elif lattice_type == "R":
        if (-h + k + l) % 3 != 0:
            return True
    return False


def _all_even_or_all_odd(h: int, k: int, l: int) -> bool:
    """h,k,l全偶或全奇"""
    parities = (h % 2, k % 2, l % 2)
    return parities == (0, 0, 0) or parities == (1, 1, 1)


def _check_fd3m_extinction(h: int, k: int, l: int) -> bool:
    """
    Fd-3m (#227) 消光规则
    F面心: hkl不全奇不全偶 → 消光
    金刚石滑移d: h+k+l = 4n+2 (即不全奇时已消光，全奇时 h+k+l=3+6n=3(1+2n) 为奇也消光,
                全偶时 h+k+l=4n+2 消光)
    
    简化: F面心要求全奇或全偶;
          全偶时 h+k+l=4n 才允许, h+k+l=4n+2 消光(d滑移)
          全奇时全部允许
    """
    if not _all_even_or_all_odd(h, k, l):
        return True
    
    if h % 2 == 0 and k % 2 == 0 and l % 2 == 0:
        if (h + k + l) % 4 != 0:
            return True
    
    return False


def _check_fm3m_extinction(h: int, k: int, l: int) -> bool:
    """
    Fm-3m (#225) 消光规则
    F面心: hkl不全奇不全偶 → 消光
    """
    if not _all_even_or_all_odd(h, k, l):
        return True
    return False


def _check_pm3m_extinction(h: int, k: int, l: int) -> bool:
    """
    Pm-3m (#221) 消光规则
    P简单格子: 无晶格消光
    """
    return False


def _check_im3m_extinction(h: int, k: int, l: int) -> bool:
    """
    Im-3m (#229) 消光规则
    I体心: h+k+l = 奇数 → 消光
    """
    if (h + k + l) % 2 != 0:
        return True
    return False


def _check_p63mmc_extinction(h: int, k: int, l: int) -> bool:
    """
    P63/mmc (#194) 消光规则
    P简单格子: 无晶格消光
    63螺旋轴: l = 奇数时 h+2k=3n 的反射消光 (简化)
    """
    if l % 2 != 0:
        if (h + 2 * k) % 3 != 0:
            return True
    return False


def _check_tetragonal_extinction(h: int, k: int, l: int, sg_number: int) -> bool:
    """四方晶系消光规则"""
    if sg_number in {89, 90, 91, 92, 93, 94, 95, 96, 97, 98}:
        if h % 2 != 0 or k % 2 != 0:
            if l == 0:
                return True
    elif sg_number in {99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110}:
        if (h + k) % 2 != 0:
            if l == 0:
                return True
    elif sg_number in {123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142}:
        if (h + k + l) % 2 != 0:
            return True
    return False


def _check_orthorhombic_extinction(h: int, k: int, l: int, sg_number: int) -> bool:
    """正交晶系消光规则"""
    lattice = _get_lattice_type(sg_number)
    return _check_lattice_extinction(h, k, l, lattice)


def _check_monoclinic_extinction(h: int, k: int, l: int, sg_number: int) -> bool:
    """单斜晶系消光规则"""
    lattice = _get_lattice_type(sg_number)
    return _check_lattice_extinction(h, k, l, lattice)


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
    225: ["x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z",
          "z,x,y", "-z,-x,y", "-z,x,-y", "z,-x,-y",
          "y,z,x", "-y,-z,x", "-y,z,-x", "y,-z,-x",
          "-x,-y,-z", "x,y,-z", "x,-y,z", "-x,y,z",
          "-z,-x,-y", "z,x,-y", "z,-x,y", "-z,x,y",
          "-y,-z,-x", "y,z,-x", "y,-z,x", "-y,z,x",
          "x+1/2,y+1/2,z", "-x+1/2,-y+1/2,z", "-x+1/2,y+1/2,-z", "x+1/2,-y+1/2,-z",
          "z+1/2,x+1/2,y", "-z+1/2,-x+1/2,y", "-z+1/2,x+1/2,-y", "z+1/2,-x+1/2,-y",
          "y+1/2,z+1/2,x", "-y+1/2,-z+1/2,x", "-y+1/2,z+1/2,-x", "y+1/2,-z+1/2,-x",
          "-x+1/2,-y+1/2,-z", "x+1/2,y+1/2,-z", "x+1/2,-y+1/2,z", "-x+1/2,y+1/2,z",
          "-z+1/2,-x+1/2,-y", "z+1/2,x+1/2,-y", "z+1/2,-x+1/2,y", "-z+1/2,x+1/2,y",
          "-y+1/2,-z+1/2,-x", "y+1/2,z+1/2,-x", "y+1/2,-z+1/2,x", "-y+1/2,z+1/2,x",
          "x+1/2,y,z+1/2", "-x+1/2,-y,z+1/2", "-x+1/2,y,-z+1/2", "x+1/2,-y,-z+1/2",
          "z+1/2,x,y+1/2", "-z+1/2,-x,y+1/2", "-z+1/2,x,-y+1/2", "z+1/2,-x,-y+1/2",
          "y+1/2,z,x+1/2", "-y+1/2,-z,x+1/2", "-y+1/2,z,-x+1/2", "y+1/2,-z,-x+1/2",
          "-x+1/2,-y,-z+1/2", "x+1/2,y,-z+1/2", "x+1/2,-y,z+1/2", "-x+1/2,y,z+1/2",
          "-z+1/2,-x,-y+1/2", "z+1/2,x,-y+1/2", "z+1/2,-x,y+1/2", "-z+1/2,x,y+1/2",
          "-y+1/2,-z,-x+1/2", "y+1/2,z,-x+1/2", "y+1/2,-z,x+1/2", "-y+1/2,z,x+1/2",
          "x,y+1/2,z+1/2", "-x,-y+1/2,z+1/2", "-x,y+1/2,-z+1/2", "x,-y+1/2,-z+1/2",
          "z,x+1/2,y+1/2", "-z,-x+1/2,y+1/2", "-z,x+1/2,-y+1/2", "z,-x+1/2,-y+1/2",
          "y,z+1/2,x+1/2", "-y,-z+1/2,x+1/2", "-y,z+1/2,-x+1/2", "y,-z+1/2,-x+1/2",
          "-x,-y+1/2,-z+1/2", "x,y+1/2,-z+1/2", "x,-y+1/2,z+1/2", "-x,y+1/2,z+1/2",
          "-z,-x+1/2,-y+1/2", "z,x+1/2,-y+1/2", "z,-x+1/2,y+1/2", "-z,x+1/2,y+1/2",
          "-y,-z+1/2,-x+1/2", "y,z+1/2,-x+1/2", "y,-z+1/2,x+1/2", "-y,z+1/2,x+1/2"],
    227: ["x,y,z", "-x,-y,-z",
          "-y,z,-x", "y,-z,x",
          "-z,x,-y", "z,-x,y",
          "y,z,x", "-y,-z,-x",
          "z,x,y", "-z,-x,-y",
          "x+1/2,y+1/2,z", "-x+1/2,-y+1/2,-z",
          "-y+1/2,z+1/2,-x", "y+1/2,-z+1/2,x",
          "-z+1/2,x+1/2,-y", "z+1/2,-x+1/2,y",
          "y+1/2,z+1/2,x", "-y+1/2,-z+1/2,-x",
          "z+1/2,x+1/2,y", "-z+1/2,-x+1/2,-y",
          "x+1/2,y,z+1/2", "-x+1/2,-y,-z+1/2",
          "-y+1/2,z,-x+1/2", "y+1/2,-z,x+1/2",
          "-z+1/2,x,-y+1/2", "z+1/2,-x,y+1/2",
          "y+1/2,z,x+1/2", "-y+1/2,-z,-x+1/2",
          "z+1/2,x,y+1/2", "-z+1/2,-x,-y+1/2",
          "x,y+1/2,z+1/2", "-x,-y+1/2,-z+1/2",
          "-y,z+1/2,-x+1/2", "y,-z+1/2,x+1/2",
          "-z,x+1/2,-y+1/2", "z,-x+1/2,y+1/2",
          "y,z+1/2,x+1/2", "-y,-z+1/2,-x+1/2",
          "z,x+1/2,y+1/2", "-z,-x+1/2,-y+1/2",
          "x+3/4,y+1/4,z+3/4", "-x+3/4,-y+1/4,-z+3/4",
          "-y+3/4,z+1/4,-x+3/4", "y+3/4,-z+1/4,x+3/4",
          "-z+3/4,x+1/4,-y+3/4", "z+3/4,-x+1/4,y+3/4",
          "y+3/4,z+1/4,x+3/4", "-y+3/4,-z+1/4,-x+3/4",
          "z+3/4,x+1/4,y+3/4", "-z+3/4,-x+1/4,-y+3/4",
          "x+1/4,y+3/4,z+1/4", "-x+1/4,-y+3/4,-z+1/4",
          "-y+1/4,z+3/4,-x+1/4", "y+1/4,-z+3/4,x+1/4",
          "-z+1/4,x+3/4,-y+1/4", "z+1/4,-x+3/4,y+1/4",
          "y+1/4,z+3/4,x+1/4", "-y+1/4,-z+3/4,-x+1/4",
          "z+1/4,x+3/4,y+1/4", "-z+1/4,-x+3/4,-y+1/4",
          "x+1/4,y+1/4,z+1/4", "-x+1/4,-y+1/4,-z+1/4",
          "-y+1/4,z+1/4,-x+1/4", "y+1/4,-z+1/4,x+1/4",
          "-z+1/4,x+1/4,-y+1/4", "z+1/4,-x+1/4,y+1/4",
          "y+1/4,z+1/4,x+1/4", "-y+1/4,-z+1/4,-x+1/4",
          "z+1/4,x+1/4,y+1/4", "-z+1/4,-x+1/4,-y+1/4",
          "x+3/4,y+3/4,z+3/4", "-x+3/4,-y+3/4,-z+3/4",
          "-y+3/4,z+3/4,-x+3/4", "y+3/4,-z+3/4,x+3/4",
          "-z+3/4,x+3/4,-y+3/4", "z+3/4,-x+3/4,y+3/4",
          "y+3/4,z+3/4,x+3/4", "-y+3/4,-z+3/4,-x+3/4",
          "z+3/4,x+3/4,y+3/4", "-z+3/4,-x+3/4,-y+3/4"],
    229: ["x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z",
          "z,x,y", "-z,-x,y", "-z,x,-y", "z,-x,-y",
          "y,z,x", "-y,-z,x", "-y,z,-x", "y,-z,-x",
          "-x,-y,-z", "x,y,-z", "x,-y,z", "-x,y,z",
          "-z,-x,-y", "z,x,-y", "z,-x,y", "-z,x,y",
          "-y,-z,-x", "y,z,-x", "y,-z,x", "-y,z,x",
          "x+1/2,y+1/2,z+1/2", "-x+1/2,-y+1/2,z+1/2", "-x+1/2,y+1/2,-z+1/2", "x+1/2,-y+1/2,-z+1/2",
          "z+1/2,x+1/2,y+1/2", "-z+1/2,-x+1/2,y+1/2", "-z+1/2,x+1/2,-y+1/2", "z+1/2,-x+1/2,-y+1/2",
          "y+1/2,z+1/2,x+1/2", "-y+1/2,-z+1/2,x+1/2", "-y+1/2,z+1/2,-x+1/2", "y+1/2,-z+1/2,-x+1/2",
          "-x+1/2,-y+1/2,-z+1/2", "x+1/2,y+1/2,-z+1/2", "x+1/2,-y+1/2,z+1/2", "-x+1/2,y+1/2,z+1/2",
          "-z+1/2,-x+1/2,-y+1/2", "z+1/2,x+1/2,-y+1/2", "z+1/2,-x+1/2,y+1/2", "-z+1/2,x+1/2,y+1/2",
          "-y+1/2,-z+1/2,-x+1/2", "y+1/2,z+1/2,-x+1/2", "y+1/2,-z+1/2,x+1/2", "-y+1/2,z+1/2,x+1/2"],
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
        3: "P2", 4: "P21", 5: "C2",
        14: "P2/c", 15: "C2/c",
        16: "P222", 19: "F222",
        47: "Pmmm", 62: "Pnma",
        75: "P4", 81: "P41",
        89: "P422", 99: "P4122",
        123: "P4/mmm", 129: "P4/nmm",
        141: "I41/amd",
        143: "P3", 147: "P31",
        150: "P321", 155: "R3",
        160: "R3c",
        168: "P6", 173: "P63",
        174: "P-6",
        194: "P63/mmc",
        195: "P23", 200: "Pm-3",
        207: "F23", 221: "Pm-3m",
        225: "Fm-3m", 227: "Fd-3m", 229: "Im-3m",
    }
    return sg_symbols.get(number, f"SG_{number}")


def get_sg_number(symbol: str) -> int:
    """根据H-M符号获取空间群编号"""
    sg_numbers = {
        "P1": 1, "P-1": 2,
        "P2": 3, "P21": 4, "C2": 5,
        "P2/c": 14, "C2/c": 15,
        "P222": 16, "F222": 19,
        "Pmmm": 47, "Pnma": 62,
        "P4/mmm": 123, "P4/nmm": 129,
        "I41/amd": 141,
        "P63/mmc": 194,
        "Pm-3m": 221, "Fm-3m": 225, "Fd-3m": 227, "Im-3m": 229,
    }
    return sg_numbers.get(symbol, 1)

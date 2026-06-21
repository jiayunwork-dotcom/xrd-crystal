"""
X射线衍射谱计算
包含布拉格方程、结构因子、多重性因子、洛伦兹偏振因子等
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .crystal import Crystal, Atom
from .space_group import SpaceGroup, generate_equivalent_positions, multiplicity, get_space_group_by_number, _get_lattice_type, parse_symop_xyz, SymmetryOperation
from .scattering import structure_factor_from_positions


@dataclass
class DiffractionPeak:
    """衍射峰数据类"""
    h: int
    k: int
    l: int
    d: float
    two_theta: float
    intensity: float
    f_hkl: complex
    multiplicity: int
    lp_factor: float


def bragg_angle(d: float, wavelength: float) -> float:
    """
    布拉格方程: 2d sin(theta) = lambda
    返回 2theta (度)
    """
    sin_theta = wavelength / (2.0 * d)
    if sin_theta > 1.0:
        return np.nan
    if sin_theta < -1.0:
        return np.nan
    theta = np.arcsin(sin_theta)
    return 2 * np.rad2deg(theta)


def lorentz_polarization_factor(two_theta: float, wavelength: float = None) -> float:
    """
    洛伦兹偏振因子 (LP因子)
    对于粉末衍射: LP = (1 + cos^2(2theta)) / (sin^2(theta) * cos(theta))
    """
    if two_theta <= 0 or two_theta >= 180:
        return 0.0
    
    theta = np.deg2rad(two_theta / 2.0)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cos_2theta = np.cos(2 * theta)
    
    if sin_theta == 0 or cos_theta == 0:
        return 0.0
    
    lp = (1 + cos_2theta**2) / (sin_theta**2 * cos_theta)
    return lp


def generate_hkl_indices(max_h: int, max_k: int, max_l: int, 
                         crystal_system: str = "triclinic") -> List[Tuple[int, int, int]]:
    """
    生成所有可能的(hkl)指标
    
    参数:
        max_h, max_k, max_l: h,k,l的最大绝对值
        crystal_system: 晶系，用于确定唯一反射
    
    返回:
        (h, k, l) 列表
    """
    indices = []
    
    for h in range(-max_h, max_h + 1):
        for k in range(-max_k, max_k + 1):
            for l in range(-max_l, max_l + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                indices.append((h, k, l))
    
    return indices


def diffraction_simulation(crystal: Crystal, 
                           wavelength: float = 1.5406,
                           two_theta_min: float = 5.0,
                           two_theta_max: float = 80.0,
                           use_symmetry: bool = True) -> List[DiffractionPeak]:
    """
    计算理论衍射谱
    
    参数:
        crystal: Crystal对象
        wavelength: X射线波长 (埃)
        two_theta_min: 最小2theta (度)
        two_theta_max: 最大2theta (度)
        use_symmetry: 是否使用空间群对称性
    
    返回:
        衍射峰列表
    """
    peaks = []
    
    d_min = wavelength / (2.0 * np.sin(np.deg2rad(two_theta_max / 2.0)))
    d_max = wavelength / (2.0 * np.sin(np.deg2rad(two_theta_min / 2.0)))
    
    max_h = int(np.ceil(crystal.a / d_min)) + 1
    max_k = int(np.ceil(crystal.b / d_min)) + 1
    max_l = int(np.ceil(crystal.c / d_min)) + 1
    
    max_h = min(max_h, 20)
    max_k = min(max_k, 20)
    max_l = min(max_l, 20)
    
    sg = SpaceGroup(crystal.space_group, crystal.space_group_number)
    
    all_positions = []
    all_elements = []
    all_occupancies = []
    all_b_isos = []
    
    if use_symmetry and len(crystal.atoms) > 0:
        lattice_type = _get_lattice_type(crystal.space_group_number)
        lattice_translations = _get_lattice_translations(lattice_type)
        
        has_diamond_glide = crystal.space_group_number in {227, 228}
        diamond_glide_translations = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.25, 0.25, 0.25]),
        ] if has_diamond_glide else [np.array([0.0, 0.0, 0.0])]
        
        point_group_symops = _get_point_group_symops(crystal.crystal_system, crystal.space_group_number)
        
        for atom in crystal.atoms:
            full_positions = []
            for dg_trans in diamond_glide_translations:
                for pg_op in point_group_symops:
                    rotated = pg_op.apply(atom.frac_coord)
                    shifted = (rotated + dg_trans) % 1.0
                    for lt in lattice_translations:
                        pos = (shifted + lt) % 1.0
                        full_positions.append(pos)
            
            seen = set()
            for pos in full_positions:
                key = tuple(np.round(pos, 6))
                if key not in seen:
                    seen.add(key)
                    all_positions.append(pos)
                    all_elements.append(atom.element)
                    all_occupancies.append(atom.occupancy)
                    all_b_isos.append(atom.b_iso)
    else:
        for atom in crystal.atoms:
            all_positions.append(atom.frac_coord)
            all_elements.append(atom.element)
            all_occupancies.append(atom.occupancy)
            all_b_isos.append(atom.b_iso)
    
    unique_indices = set()
    
    for h in range(-max_h, max_h + 1):
        for k in range(-max_k, max_k + 1):
            for l in range(-max_l, max_l + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                
                if use_symmetry and sg.is_extinct(h, k, l):
                    continue
                
                d = crystal.d_spacing(h, k, l)
                
                if d < d_min or d > d_max:
                    continue
                
                two_theta = bragg_angle(d, wavelength)
                if np.isnan(two_theta):
                    continue
                if two_theta < two_theta_min or two_theta > two_theta_max:
                    continue
                
                F = structure_factor_from_positions(h, k, l, all_elements, all_positions, 
                                                    all_occupancies, all_b_isos, d)
                
                intensity = np.abs(F)**2
                
                if intensity < 1e-10:
                    continue
                
                ch, ck, cl = _canonicalize_hkl(h, k, l, crystal.crystal_system)
                
                key = (ch, ck, cl)
                if key in unique_indices:
                    continue
                unique_indices.add(key)
                
                mult = multiplicity(ch, ck, cl, crystal.crystal_system)
                
                lp = lorentz_polarization_factor(two_theta)
                
                intensity *= mult * lp
                
                if intensity < 1e-10:
                    continue
                
                peak = DiffractionPeak(
                    h=ch, k=ck, l=cl,
                    d=d,
                    two_theta=two_theta,
                    intensity=intensity,
                    f_hkl=F,
                    multiplicity=mult,
                    lp_factor=lp
                )
                peaks.append(peak)
    
    peaks.sort(key=lambda p: p.two_theta)
    
    if peaks:
        max_intensity = max(p.intensity for p in peaks)
        if max_intensity > 0:
            for peak in peaks:
                peak.intensity = peak.intensity / max_intensity * 100.0
    
    peaks = [p for p in peaks if p.intensity > 0.01]
    
    return peaks


def _get_point_group_symops(crystal_system: str, sg_number: int) -> List[SymmetryOperation]:
    """
    获取点群对称操作（不含晶格平移和滑移平移）
    这些操作只包含纯旋转/反演，平移分量为0
    """
    if crystal_system == "cubic":
        if sg_number in {195, 196, 197, 198, 199}:
            return [parse_symop_xyz(s) for s in [
                "x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z",
                "z,x,y", "-z,-x,y", "-z,x,-y", "z,-x,-y",
                "y,z,x", "-y,-z,x", "-y,z,-x", "y,-z,-x",
            ]]
        elif sg_number in {200, 201, 202, 203, 204, 205, 206}:
            return [parse_symop_xyz(s) for s in [
                "x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z",
                "z,x,y", "-z,-x,y", "-z,x,-y", "z,-x,-y",
                "y,z,x", "-y,-z,x", "-y,z,-x", "y,-z,-x",
                "-x,-y,-z", "x,y,-z", "x,-y,z", "-x,y,z",
                "-z,-x,-y", "z,x,-y", "z,-x,y", "-z,x,y",
                "-y,-z,-x", "y,z,-x", "y,-z,x", "-y,z,x",
            ]]
        elif sg_number in {207, 208, 209, 210, 211, 212, 213, 214}:
            return [parse_symop_xyz(s) for s in [
                "x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z",
                "z,x,y", "-z,-x,y", "-z,x,-y", "z,-x,-y",
                "y,z,x", "-y,-z,x", "-y,z,-x", "y,-z,-x",
                "-x,-y,-z", "x,y,-z", "x,-y,z", "-x,y,z",
                "-z,-x,-y", "z,x,-y", "z,-x,y", "-z,x,y",
                "-y,-z,-x", "y,z,-x", "y,-z,x", "-y,z,x",
            ]]
        else:
            return [parse_symop_xyz(s) for s in [
                "x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z",
                "z,x,y", "-z,-x,y", "-z,x,-y", "z,-x,-y",
                "y,z,x", "-y,-z,x", "-y,z,-x", "y,-z,-x",
                "-x,-y,-z", "x,y,-z", "x,-y,z", "-x,y,z",
                "-z,-x,-y", "z,x,-y", "z,-x,y", "-z,x,y",
                "-y,-z,-x", "y,z,-x", "y,-z,x", "-y,z,x",
            ]]
    elif crystal_system == "hexagonal":
        return [parse_symop_xyz(s) for s in [
            "x,y,z", "-y,x-y,z", "-x+y,-x,z", "-x,-y,z", "y,-x+y,z", "x-y,x,z",
        ]]
    elif crystal_system == "tetragonal":
        return [parse_symop_xyz(s) for s in [
            "x,y,z", "-x,-y,z", "-y,x,z", "y,-x,z",
        ]]
    elif crystal_system == "orthorhombic":
        return [parse_symop_xyz(s) for s in [
            "x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z",
        ]]
    elif crystal_system == "monoclinic":
        return [parse_symop_xyz(s) for s in [
            "x,y,z", "-x,y,-z",
        ]]
    elif crystal_system == "triclinic":
        return [parse_symop_xyz(s) for s in ["x,y,z"]]
    
    return [parse_symop_xyz("x,y,z")]


def _get_lattice_translations(lattice_type: str) -> List[np.ndarray]:
    """
    获取晶格中心平移向量
    
    P: 无额外平移
    I: + (1/2, 1/2, 1/2)
    F: + (1/2, 1/2, 0), (1/2, 0, 1/2), (0, 1/2, 1/2)
    C: + (1/2, 1/2, 0)
    A: + (0, 1/2, 1/2)
    R: + (2/3, 1/3, 1/3), (1/3, 2/3, 2/3) (六方设置)
    """
    base = [np.array([0.0, 0.0, 0.0])]
    
    if lattice_type == "P":
        return base
    elif lattice_type == "I":
        return base + [np.array([0.5, 0.5, 0.5])]
    elif lattice_type == "F":
        return base + [
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
        ]
    elif lattice_type == "C":
        return base + [np.array([0.5, 0.5, 0.0])]
    elif lattice_type == "A":
        return base + [np.array([0.0, 0.5, 0.5])]
    elif lattice_type == "R":
        return base + [
            np.array([2/3, 1/3, 1/3]),
            np.array([1/3, 2/3, 2/3]),
        ]
    return base


def _canonicalize_hkl(h: int, k: int, l: int, crystal_system: str) -> Tuple[int, int, int]:
    """
    将(hkl)规范化为晶体学惯例的正指标表示
    粉末衍射中习惯用正指标标注
    """
    ah, ak, al = abs(h), abs(k), abs(l)
    
    if crystal_system == "cubic":
        return tuple(sorted([ah, ak, al], reverse=True))
    elif crystal_system == "tetragonal":
        if ah >= ak:
            return (ah, ak, al)
        else:
            return (ak, ah, al)
    elif crystal_system in ("hexagonal", "trigonal"):
        return (max(ah, ak), min(ah, ak), al)
    elif crystal_system == "orthorhombic":
        return (ah, ak, al)
    elif crystal_system == "monoclinic":
        return (ah, ak, al)
    else:
        return (ah, ak, al)


def _get_unique_key(h: int, k: int, l: int, crystal_system: str) -> Tuple:
    """
    获取唯一反射的键，用于去重
    (只保留等效反射中的一个)
    """
    return _canonicalize_hkl(h, k, l, crystal_system)


def powder_pattern(peaks: List[DiffractionPeak],
                   two_theta_range: np.ndarray,
                   fwhm: float = 0.1,
                   peak_type: str = "pseudo_voigt",
                   eta: float = 0.5) -> np.ndarray:
    """
    根据衍射峰生成粉末衍射谱 (连续强度分布)
    
    参数:
        peaks: 衍射峰列表
        two_theta_range: 2theta范围数组
        fwhm: 半高宽 (度)
        peak_type: 峰形类型 ("gaussian", "lorentzian", "pseudo_voigt")
        eta: Pseudo-Voigt混合参数 (0=纯高斯, 1=纯洛伦兹)
    
    返回:
        强度数组
    """
    from .peak_fitting import pseudo_voigt, gaussian, lorentzian
    
    intensity = np.zeros_like(two_theta_range)
    
    for peak in peaks:
        if peak_type == "gaussian":
            profile = gaussian(two_theta_range, peak.two_theta, fwhm)
        elif peak_type == "lorentzian":
            profile = lorentzian(two_theta_range, peak.two_theta, fwhm)
        else:
            profile = pseudo_voigt(two_theta_range, peak.two_theta, fwhm, eta)
        
        intensity += peak.intensity * profile
    
    return intensity


def powder_pattern_caglioti(peaks: List[DiffractionPeak],
                            two_theta_range: np.ndarray,
                            U: float = 0.01,
                            V: float = 0.01,
                            W: float = 0.01,
                            eta: float = 0.5,
                            wavelength: float = 1.5406) -> np.ndarray:
    """
    使用Caglioti公式的粉末衍射谱
    
    参数:
        peaks: 衍射峰列表
        two_theta_range: 2theta范围数组
        U, V, W: Caglioti公式参数
        eta: Pseudo-Voigt混合参数
        wavelength: X射线波长
    
    返回:
        强度数组
    """
    from .peak_fitting import pseudo_voigt, caglioti_fwhm
    
    intensity = np.zeros_like(two_theta_range)
    
    for peak in peaks:
        fwhm = caglioti_fwhm(peak.two_theta, U, V, W, wavelength)
        profile = pseudo_voigt(two_theta_range, peak.two_theta, fwhm, eta)
        intensity += peak.intensity * profile
    
    return intensity

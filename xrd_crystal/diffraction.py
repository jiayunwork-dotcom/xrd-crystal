"""
X射线衍射谱计算
包含布拉格方程、结构因子、多重性因子、洛伦兹偏振因子等
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .crystal import Crystal, Atom
from .space_group import SpaceGroup, generate_equivalent_positions, multiplicity, get_space_group_by_number
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
        for atom in crystal.atoms:
            equiv_positions = generate_equivalent_positions(atom.frac_coord, sg.symops)
            for pos in equiv_positions:
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
                
                mult = multiplicity(h, k, l, crystal.crystal_system)
                
                lp = lorentz_polarization_factor(two_theta)
                
                intensity *= mult * lp
                
                key = _get_unique_key(h, k, l, crystal.crystal_system)
                if key in unique_indices:
                    continue
                unique_indices.add(key)
                
                peak = DiffractionPeak(
                    h=h, k=k, l=l,
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
    
    return peaks


def _get_unique_key(h: int, k: int, l: int, crystal_system: str) -> Tuple:
    """
    获取唯一反射的键，用于去重
    (只保留等效反射中的一个)
    """
    h, k, l = abs(h), abs(k), abs(l)
    
    if crystal_system == "cubic":
        return tuple(sorted([h, k, l], reverse=True))
    elif crystal_system == "tetragonal":
        if h >= k:
            return (h, k, l)
        else:
            return (k, h, l)
    elif crystal_system == "hexagonal" or crystal_system == "trigonal":
        return (max(h, k), min(h, k), l)
    elif crystal_system == "orthorhombic":
        return (h, k, l)
    elif crystal_system == "monoclinic":
        return (h, k, l)
    else:
        return (h, k, l)


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

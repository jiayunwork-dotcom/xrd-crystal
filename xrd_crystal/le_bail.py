"""
Le Bail全谱拟合
使用最小二乘法优化晶胞参数和峰形参数
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.optimize import leastsq, least_squares

from .crystal import Crystal
from .space_group import SpaceGroup, generate_equivalent_positions
from .diffraction import diffraction_simulation, bragg_angle, lorentz_polarization_factor
from .scattering import structure_factor_from_positions
from .peak_fitting import pseudo_voigt, caglioti_fwhm, caglioti_fwhm_array


@dataclass
class LeBailResult:
    """Le Bail拟合结果"""
    crystal: Crystal
    U: float
    V: float
    W: float
    eta: float
    zero_shift: float
    Rwp: float
    chi_squared: float
    peak_intensities: List[float]
    calculated_pattern: np.ndarray
    difference_pattern: np.ndarray


def le_bail_fit(two_theta: np.ndarray,
                intensity: np.ndarray,
                initial_crystal: Crystal,
                wavelength: float = 1.5406,
                initial_U: float = 0.02,
                initial_V: float = -0.01,
                initial_W: float = 0.02,
                initial_eta: float = 0.5,
                initial_zero_shift: float = 0.0,
                max_iterations: int = 50,
                tolerance: float = 1e-5) -> LeBailResult:
    """
    Le Bail全谱拟合
    
    参数:
        two_theta: 实验2theta数组
        intensity: 实验强度数组
        initial_crystal: 初始晶体结构
        wavelength: X射线波长
        initial_U, initial_V, initial_W: 初始Caglioti参数
        initial_eta: 初始Pseudo-Voigt混合参数
        initial_zero_shift: 初始零点偏移
        max_iterations: 最大迭代次数
        tolerance: 收敛阈值
    
    返回:
        LeBailResult对象
    """
    intensity_norm = intensity / np.max(intensity) if np.max(intensity) > 0 else intensity
    
    sg = SpaceGroup(initial_crystal.space_group, initial_crystal.space_group_number)
    
    all_positions = []
    all_elements = []
    all_occupancies = []
    all_b_isos = []
    
    for atom in initial_crystal.atoms:
        equiv_positions = generate_equivalent_positions(atom.frac_coord, sg.symops)
        for pos in equiv_positions:
            all_positions.append(pos)
            all_elements.append(atom.element)
            all_occupancies.append(atom.occupancy)
            all_b_isos.append(atom.b_iso)
    
    crystal_system = initial_crystal.crystal_system
    
    params = _params_to_array(initial_crystal, initial_U, initial_V, initial_W, 
                              initial_eta, initial_zero_shift, crystal_system)
    
    prev_Rwp = float('inf')
    
    for iteration in range(max_iterations):
        crystal, U, V, W, eta, zero_shift = _array_to_params(params, crystal_system)
        
        peaks = _generate_peaks(crystal, all_elements, all_positions, 
                               all_occupancies, all_b_isos, wavelength, sg)
        
        peak_intensities = _extract_peak_intensities(two_theta, intensity_norm, peaks, 
                                                     U, V, W, eta, wavelength, zero_shift)
        
        def residuals(p):
            cryst, u, v, w, e, zs = _array_to_params(p, crystal_system)
            calc_pattern = _calculate_pattern(two_theta, peaks, peak_intensities,
                                              u, v, w, e, wavelength, zs, cryst)
            weight = 1.0 / (intensity_norm + 1e-6)
            return (intensity_norm - calc_pattern) * np.sqrt(weight)
        
        try:
            result = least_squares(
                residuals,
                params,
                method='lm',
                ftol=1e-6,
                xtol=1e-6,
                max_nfev=100
            )
            params = result.x
        except:
            break
        
        crystal, U, V, W, eta, zero_shift = _array_to_params(params, crystal_system)
        calc_pattern = _calculate_pattern(two_theta, peaks, peak_intensities,
                                          U, V, W, eta, wavelength, zero_shift, crystal)
        
        Rwp = _calculate_Rwp(intensity_norm, calc_pattern)
        
        if abs(prev_Rwp - Rwp) < tolerance * prev_Rwp:
            break
        prev_Rwp = Rwp
    
    crystal, U, V, W, eta, zero_shift = _array_to_params(params, crystal_system)
    
    peaks = _generate_peaks(crystal, all_elements, all_positions, 
                           all_occupancies, all_b_isos, wavelength, sg)
    
    peak_intensities = _extract_peak_intensities(two_theta, intensity_norm, peaks,
                                                 U, V, W, eta, wavelength, zero_shift)
    
    calc_pattern = _calculate_pattern(two_theta, peaks, peak_intensities,
                                      U, V, W, eta, wavelength, zero_shift, crystal)
    
    diff_pattern = intensity_norm - calc_pattern
    
    Rwp = _calculate_Rwp(intensity_norm, calc_pattern)
    chi_sq = _calculate_chi_squared(intensity_norm, calc_pattern)
    
    return LeBailResult(
        crystal=crystal,
        U=U,
        V=V,
        W=W,
        eta=eta,
        zero_shift=zero_shift,
        Rwp=Rwp,
        chi_squared=chi_sq,
        peak_intensities=peak_intensities,
        calculated_pattern=calc_pattern,
        difference_pattern=diff_pattern
    )


def _params_to_array(crystal: Crystal, U: float, V: float, W: float, 
                     eta: float, zero_shift: float, crystal_system: str) -> np.ndarray:
    """将参数转换为数组"""
    params = []
    
    if crystal_system == "cubic":
        params.append(crystal.a)
    elif crystal_system == "tetragonal":
        params.append(crystal.a)
        params.append(crystal.c)
    elif crystal_system == "hexagonal":
        params.append(crystal.a)
        params.append(crystal.c)
    elif crystal_system == "orthorhombic":
        params.append(crystal.a)
        params.append(crystal.b)
        params.append(crystal.c)
    elif crystal_system == "monoclinic":
        params.append(crystal.a)
        params.append(crystal.b)
        params.append(crystal.c)
        params.append(crystal.beta)
    else:
        params.append(crystal.a)
        params.append(crystal.b)
        params.append(crystal.c)
        params.append(crystal.alpha)
        params.append(crystal.beta)
        params.append(crystal.gamma)
    
    params.extend([U, V, W, eta, zero_shift])
    
    return np.array(params)


def _array_to_params(params: np.ndarray, crystal_system: str) -> Tuple[Crystal, float, float, float, float, float]:
    """将数组转换为参数"""
    idx = 0
    
    if crystal_system == "cubic":
        a = params[idx]; idx += 1
        b = a; c = a
        alpha = beta = gamma = 90.0
    elif crystal_system == "tetragonal":
        a = params[idx]; idx += 1
        c = params[idx]; idx += 1
        b = a
        alpha = beta = gamma = 90.0
    elif crystal_system == "hexagonal":
        a = params[idx]; idx += 1
        c = params[idx]; idx += 1
        b = a
        alpha = beta = 90.0
        gamma = 120.0
    elif crystal_system == "orthorhombic":
        a = params[idx]; idx += 1
        b = params[idx]; idx += 1
        c = params[idx]; idx += 1
        alpha = beta = gamma = 90.0
    elif crystal_system == "monoclinic":
        a = params[idx]; idx += 1
        b = params[idx]; idx += 1
        c = params[idx]; idx += 1
        beta = params[idx]; idx += 1
        alpha = gamma = 90.0
    else:
        a = params[idx]; idx += 1
        b = params[idx]; idx += 1
        c = params[idx]; idx += 1
        alpha = params[idx]; idx += 1
        beta = params[idx]; idx += 1
        gamma = params[idx]; idx += 1
    
    U = params[idx]; idx += 1
    V = params[idx]; idx += 1
    W = params[idx]; idx += 1
    eta = params[idx]; idx += 1
    zero_shift = params[idx]; idx += 1
    
    crystal = Crystal(a, b, c, alpha, beta, gamma)
    
    return crystal, U, V, W, eta, zero_shift


def _generate_peaks(crystal: Crystal,
                    elements: List[str],
                    positions: List[np.ndarray],
                    occupancies: List[float],
                    b_isos: List[float],
                    wavelength: float,
                    sg: SpaceGroup) -> List:
    """生成衍射峰"""
    from .diffraction import DiffractionPeak
    from .space_group import multiplicity
    
    peaks = []
    
    max_h = int(np.ceil(crystal.a / 0.5)) + 1
    max_k = int(np.ceil(crystal.b / 0.5)) + 1
    max_l = int(np.ceil(crystal.c / 0.5)) + 1
    
    max_h = min(max_h, 15)
    max_k = min(max_k, 15)
    max_l = min(max_l, 15)
    
    seen = set()
    
    for h in range(-max_h, max_h + 1):
        for k in range(-max_k, max_k + 1):
            for l in range(-max_l, max_l + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                
                if sg.is_extinct(h, k, l):
                    continue
                
                d = crystal.d_spacing(h, k, l)
                if d < 0.5:
                    continue
                
                two_theta = bragg_angle(d, wavelength)
                if np.isnan(two_theta) or two_theta > 120:
                    continue
                
                F = structure_factor_from_positions(h, k, l, elements, positions,
                                                    occupancies, b_isos, d)
                
                intensity = np.abs(F)**2
                
                mult = multiplicity(h, k, l, crystal.crystal_system)
                lp = lorentz_polarization_factor(two_theta)
                
                intensity *= mult * lp
                
                key = (round(two_theta, 3),)
                if key in seen:
                    continue
                seen.add(key)
                
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
    
    return peaks


def _extract_peak_intensities(two_theta: np.ndarray,
                              intensity: np.ndarray,
                              peaks: List,
                              U: float, V: float, W: float,
                              eta: float,
                              wavelength: float,
                              zero_shift: float) -> List[float]:
    """提取峰强度 (Le Bail法的核心: 从实验谱中分解出各峰强度)"""
    peak_intensities = []
    
    for peak in peaks:
        tt = peak.two_theta + zero_shift
        fwhm = caglioti_fwhm(tt, U, V, W, wavelength)
        
        mask = np.abs(two_theta - tt) < 3 * fwhm
        
        if np.sum(mask) > 0:
            local_intensity = np.max(intensity[mask])
            peak_intensities.append(local_intensity)
        else:
            peak_intensities.append(0.0)
    
    if peak_intensities:
        max_i = max(peak_intensities)
        if max_i > 0:
            peak_intensities = [i / max_i * 100 for i in peak_intensities]
    
    return peak_intensities


def _calculate_pattern(two_theta: np.ndarray,
                       peaks: List,
                       peak_intensities: List[float],
                       U: float, V: float, W: float,
                       eta: float,
                       wavelength: float,
                       zero_shift: float,
                       crystal: Crystal = None) -> np.ndarray:
    """计算衍射谱"""
    pattern = np.zeros_like(two_theta)
    
    for peak, intensity in zip(peaks, peak_intensities):
        tt = peak.two_theta + zero_shift
        fwhm = caglioti_fwhm(tt, U, V, W, wavelength)
        
        if fwhm <= 0:
            continue
        
        profile = pseudo_voigt(two_theta, tt, fwhm, eta)
        pattern += intensity * profile
    
    if np.max(pattern) > 0:
        pattern = pattern / np.max(pattern) * 100
    
    return pattern


def _calculate_Rwp(observed: np.ndarray, calculated: np.ndarray) -> float:
    """计算加权R因子 Rwp"""
    weight = 1.0 / (observed + 1e-6)
    
    numerator = np.sum(weight * (observed - calculated)**2)
    denominator = np.sum(weight * observed**2)
    
    if denominator == 0:
        return 1.0
    
    return np.sqrt(numerator / denominator)


def _calculate_chi_squared(observed: np.ndarray, calculated: np.ndarray) -> float:
    """计算卡方值"""
    diff = observed - calculated
    variance = np.mean(observed**2) + 1e-6
    
    return np.sum(diff**2 / variance) / len(observed)

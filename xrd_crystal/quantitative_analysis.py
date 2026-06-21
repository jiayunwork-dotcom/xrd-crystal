"""
多相混合物定量分析模块
使用加权最小二乘法拟合各相权重因子
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import least_squares

from .crystal import Crystal
from .diffraction import diffraction_simulation, powder_pattern, powder_pattern_caglioti


@dataclass
class PhaseQuantResult:
    """单组相定量结果"""
    phase_name: str
    crystal_name: str
    space_group: str
    space_group_number: int
    weight: float
    weight_std: float
    mass_percent: float
    rwp_contribution: float


@dataclass
class QuantitativeResult:
    """定量分析总结果"""
    phase_results: List[PhaseQuantResult]
    background_coeffs: np.ndarray
    background_order: int
    Rwp: float
    chi_squared: float
    iterations: int
    converged: bool
    param_std_errors: np.ndarray
    calculated_pattern: np.ndarray
    background_pattern: np.ndarray
    phase_patterns: Dict[str, np.ndarray]
    two_theta: np.ndarray
    observed_intensity: np.ndarray
    difference_pattern: np.ndarray
    removed_phases: List[str] = field(default_factory=list)


def _polynomial_background(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    计算多项式背景
    coeffs为系数数组，从0阶到n阶: c0 + c1*x + c2*x^2 + ...
    """
    bg = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        bg += c * (x ** i)
    return bg


def _calculate_phase_patterns(
    crystals: Dict[str, Crystal],
    two_theta: np.ndarray,
    wavelength: float,
    use_caglioti: bool,
    U: float, V: float, W: float,
    fwhm: float,
    eta: float
) -> Dict[str, np.ndarray]:
    """
    计算各相的理论衍射谱
    """
    phase_patterns = {}
    
    for phase_name, crystal in crystals.items():
        peaks = diffraction_simulation(
            crystal,
            wavelength=wavelength,
            two_theta_min=float(np.min(two_theta)),
            two_theta_max=float(np.max(two_theta)),
            use_symmetry=True
        )
        
        if use_caglioti:
            pattern = powder_pattern_caglioti(
                peaks, two_theta,
                U=U, V=V, W=W,
                eta=eta,
                wavelength=wavelength
            )
        else:
            pattern = powder_pattern(
                peaks, two_theta,
                fwhm=fwhm,
                peak_type="pseudo_voigt",
                eta=eta
            )
        
        phase_patterns[phase_name] = pattern
    
    return phase_patterns


def _calculate_Rwp(
    observed: np.ndarray,
    calculated: np.ndarray
) -> float:
    """
    计算加权R因子 Rwp
    Rwp = sqrt( sum( w_i * (y_obs_i - y_calc_i)^2 ) / sum( w_i * y_obs_i^2 ) )
    其中 w_i = 1 / y_obs_i (y_obs_i > 0时)
    """
    valid = observed > 0
    if not np.any(valid):
        return 0.0
    
    y_obs = observed[valid]
    y_calc = calculated[valid]
    
    weights = 1.0 / np.maximum(y_obs, 1e-6)
    
    numerator = np.sum(weights * (y_obs - y_calc) ** 2)
    denominator = np.sum(weights * y_obs ** 2)
    
    if denominator <= 0:
        return 0.0
    
    return np.sqrt(numerator / denominator)


def _calculate_chi_squared(
    observed: np.ndarray,
    calculated: np.ndarray,
    n_params: int
) -> float:
    """
    计算卡方值 (自由度校正)
    """
    valid = observed > 0
    if not np.any(valid):
        return 0.0
    
    y_obs = observed[valid]
    y_calc = calculated[valid]
    
    weights = 1.0 / np.maximum(y_obs, 1e-6)
    n_points = len(y_obs)
    dof = max(n_points - n_params, 1)
    
    return np.sum(weights * (y_obs - y_calc) ** 2) / dof


def _residuals(
    params: np.ndarray,
    phase_patterns: List[np.ndarray],
    observed: np.ndarray,
    bg_order: int
) -> np.ndarray:
    """
    计算残差向量 (用于最小二乘优化)
    
    参数:
        params: [w1, w2, ..., wn, c0, c1, ..., cm]
                前n个是各相权重，后m+1个是多项式背景系数
        phase_patterns: 各相归一化理论谱列表
        observed: 实验强度
        bg_order: 背景多项式阶数
    """
    n_phases = len(phase_patterns)
    weights = params[:n_phases]
    bg_coeffs = params[n_phases:]
    
    calculated = np.zeros_like(observed)
    for w, pattern in zip(weights, phase_patterns):
        calculated += w * pattern
    
    calculated += _polynomial_background(observed if False else np.arange(len(observed)), bg_coeffs)
    
    valid = observed > 0
    weights_w = np.zeros_like(observed)
    weights_w[valid] = 1.0 / np.maximum(observed[valid], 1e-6)
    
    return (observed - calculated) * np.sqrt(weights_w)


def quantitative_analysis(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    selected_crystals: Dict[str, Crystal],
    wavelength: float = 1.5406,
    use_caglioti: bool = True,
    U: float = 0.02,
    V: float = -0.01,
    W: float = 0.015,
    fwhm: float = 0.15,
    eta: float = 0.5,
    background_order: int = 3,
    max_iterations: int = 200,
    tolerance: float = 1e-6
) -> QuantitativeResult:
    """
    多相混合物定量分析
    
    参数:
        two_theta: 实验2theta数组
        intensity: 实验强度数组
        selected_crystals: 选中的晶相字典 {phase_name: Crystal}
        wavelength: X射线波长 (Å)
        use_caglioti: 是否使用Caglioti公式
        U, V, W: Caglioti公式参数
        fwhm: 固定半高宽 (不使用Caglioti时)
        eta: Pseudo-Voigt混合参数
        background_order: 多项式背景阶数 (2-5)
        max_iterations: 最大迭代次数
        tolerance: 收敛阈值
    
    返回:
        QuantitativeResult对象
    """
    intensity_norm = intensity.copy()
    max_int = np.max(intensity_norm)
    if max_int > 0:
        intensity_norm = intensity_norm / max_int * 100.0
    
    phase_names = list(selected_crystals.keys())
    current_phases = phase_names.copy()
    removed_phases = []
    
    while True:
        current_crystals = {name: selected_crystals[name] for name in current_phases}
        
        phase_patterns_raw = _calculate_phase_patterns(
            current_crystals, two_theta, wavelength,
            use_caglioti, U, V, W, fwhm, eta
        )
        
        phase_patterns_list = []
        for name in current_phases:
            pattern = phase_patterns_raw[name]
            max_p = np.max(pattern)
            if max_p > 0:
                pattern = pattern / max_p * 100.0
            phase_patterns_list.append(pattern)
        
        n_phases = len(current_phases)
        n_bg_coeffs = background_order + 1
        
        init_weights = np.ones(n_phases) / n_phases * 50.0
        init_bg = np.zeros(n_bg_coeffs)
        init_bg[0] = np.min(intensity_norm) * 0.1
        params0 = np.concatenate([init_weights, init_bg])
        
        n_params_total = len(params0)
        lower_bounds = np.concatenate([np.zeros(n_phases), -np.inf * np.ones(n_bg_coeffs)])
        upper_bounds = np.inf * np.ones(n_params_total)
        
        def residuals_func(p):
            return _residuals(p, phase_patterns_list, intensity_norm, background_order)
        
        x_normalized = np.linspace(0, 1, len(two_theta))
        
        def residuals_func_v2(p):
            n_p = len(phase_patterns_list)
            weights = p[:n_p]
            bg_coeffs = p[n_p:]
            
            calculated = np.zeros_like(intensity_norm)
            for w, pattern in zip(weights, phase_patterns_list):
                calculated += w * pattern
            
            calculated += _polynomial_background(x_normalized, bg_coeffs)
            
            valid = intensity_norm > 0
            w_vec = np.zeros_like(intensity_norm)
            w_vec[valid] = 1.0 / np.maximum(intensity_norm[valid], 1e-6)
            
            return (intensity_norm - calculated) * np.sqrt(w_vec)
        
        try:
            result = least_squares(
                residuals_func_v2,
                params0,
                bounds=(lower_bounds, upper_bounds),
                method='trf',
                max_nfev=max_iterations,
                ftol=tolerance,
                xtol=tolerance,
                gtol=tolerance
            )
            
            opt_params = result.x
            n_iter = result.nfev
            converged = result.success
            
        except Exception:
            opt_params = params0
            n_iter = 0
            converged = False
        
        opt_weights = opt_params[:n_phases]
        opt_bg_coeffs = opt_params[n_phases:]
        
        min_weight_threshold = 0.01
        phases_to_remove = []
        for i, name in enumerate(current_phases):
            if opt_weights[i] < min_weight_threshold and n_phases > 2:
                phases_to_remove.append((i, name))
        
        if not phases_to_remove:
            break
        
        for _, name in reversed(phases_to_remove):
            removed_phases.append(name)
            current_phases.remove(name)
    
    n_phases = len(current_phases)
    final_crystals = {name: selected_crystals[name] for name in current_phases}
    
    final_patterns_raw = _calculate_phase_patterns(
        final_crystals, two_theta, wavelength,
        use_caglioti, U, V, W, fwhm, eta
    )
    
    final_patterns_list = []
    for name in current_phases:
        pattern = final_patterns_raw[name]
        max_p = np.max(pattern)
        if max_p > 0:
            pattern = pattern / max_p * 100.0
        final_patterns_list.append(pattern)
    
    x_normalized = np.linspace(0, 1, len(two_theta))
    opt_weights = opt_params[:n_phases]
    opt_bg_coeffs = opt_params[n_phases:]
    
    calculated = np.zeros_like(intensity_norm)
    phase_contributions = {}
    for i, name in enumerate(current_phases):
        contrib = opt_weights[i] * final_patterns_list[i]
        phase_contributions[name] = contrib
        calculated += contrib
    
    background = _polynomial_background(x_normalized, opt_bg_coeffs)
    calculated_total = calculated + background
    
    difference = intensity_norm - calculated_total
    
    Rwp = _calculate_Rwp(intensity_norm, calculated_total)
    
    n_params_total = len(opt_params)
    chi_sq = _calculate_chi_squared(intensity_norm, calculated_total, n_params_total)
    
    sum_weights = np.sum(opt_weights)
    if sum_weights > 0:
        mass_percents = opt_weights / sum_weights * 100.0
    else:
        mass_percents = np.ones(n_phases) / n_phases * 100.0
    
    try:
        J = result.jac
        if J is not None and J.shape[0] > J.shape[1]:
            JtJ = J.T @ J
            try:
                cov = np.linalg.inv(JtJ)
                s2 = np.sum(result.fun ** 2) / max(len(result.fun) - len(opt_params), 1)
                param_std = np.sqrt(np.maximum(np.diag(cov) * s2, 0))
            except np.linalg.LinAlgError:
                param_std = np.zeros_like(opt_params)
        else:
            param_std = np.zeros_like(opt_params)
    except Exception:
        param_std = np.zeros_like(opt_params)
    
    weight_std = param_std[:n_phases]
    
    rwp_contributions = {}
    for idx, name in enumerate(current_phases):
        reduced_patterns = [final_patterns_list[i] for i in range(n_phases) if i != idx]
        reduced_weights = np.array([opt_weights[i] for i in range(n_phases) if i != idx])
        
        if len(reduced_patterns) > 0:
            reduced_calc = np.zeros_like(intensity_norm)
            for w, pat in zip(reduced_weights, reduced_patterns):
                reduced_calc += w * pat
            reduced_calc += background
            rwp_without = _calculate_Rwp(intensity_norm, reduced_calc)
            rwp_contributions[name] = max(rwp_without - Rwp, 0.0)
        else:
            rwp_contributions[name] = Rwp
    
    phase_results = []
    for i, name in enumerate(current_phases):
        crystal = final_crystals[name]
        phase_results.append(PhaseQuantResult(
            phase_name=name,
            crystal_name=crystal.name,
            space_group=crystal.space_group,
            space_group_number=crystal.space_group_number,
            weight=float(opt_weights[i]),
            weight_std=float(weight_std[i]),
            mass_percent=float(mass_percents[i]),
            rwp_contribution=float(rwp_contributions.get(name, 0.0))
        ))
    
    return QuantitativeResult(
        phase_results=phase_results,
        background_coeffs=opt_bg_coeffs,
        background_order=background_order,
        Rwp=float(Rwp),
        chi_squared=float(chi_sq),
        iterations=int(n_iter),
        converged=bool(converged),
        param_std_errors=param_std,
        calculated_pattern=calculated_total,
        background_pattern=background,
        phase_patterns=phase_contributions,
        two_theta=two_theta,
        observed_intensity=intensity_norm,
        difference_pattern=difference,
        removed_phases=removed_phases
    )

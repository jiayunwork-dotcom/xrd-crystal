"""
寻峰算法和晶胞指标化
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.signal import savgol_filter, find_peaks


@dataclass
class Peak:
    """实验峰数据类"""
    two_theta: float
    intensity: float
    d_spacing: float
    fwhm: float = 0.0


@dataclass
class IndexingResult:
    """指标化结果"""
    crystal_system: str
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    M20: float
    F30: float
    n_peaks_indexed: int
    volume: float


def estimate_fwhm(two_theta: np.ndarray, 
                  intensity: np.ndarray, 
                  peak_idx: int,
                  window_points: int = 20) -> float:
    """
    估算峰的半高宽(FWHM)，带局部基线估计
    
    参数:
        two_theta: 2theta数组
        intensity: 强度数组（已扣除背景）
        peak_idx: 峰顶点的索引
        window_points: 估计局部基线的窗口点数（峰两侧各取多少点）
    
    返回:
        FWHM值 (度)，如果无法估算则返回0.0
    """
    n = len(two_theta)
    if peak_idx <= 0 or peak_idx >= n - 1:
        return 0.0
    
    peak_height = intensity[peak_idx]
    
    left_start = max(0, peak_idx - window_points)
    left_end = max(0, peak_idx - window_points // 3)
    right_start = min(n - 1, peak_idx + window_points // 3)
    right_end = min(n - 1, peak_idx + window_points)
    
    left_bkg = np.median(intensity[left_start:left_end + 1]) if left_end > left_start else 0.0
    right_bkg = np.median(intensity[right_start:right_end + 1]) if right_end > right_start else 0.0
    local_baseline = (left_bkg + right_bkg) / 2.0
    
    if local_baseline >= peak_height * 0.9:
        local_baseline = 0.0
    
    net_height = peak_height - local_baseline
    half_height = local_baseline + net_height / 2.0
    
    left_idx = peak_idx
    while left_idx > 0 and intensity[left_idx] > half_height:
        left_idx -= 1
    
    right_idx = peak_idx
    while right_idx < n - 1 and intensity[right_idx] > half_height:
        right_idx += 1
    
    if left_idx >= right_idx:
        return 0.0
    
    if intensity[left_idx] == intensity[left_idx + 1]:
        left_tt = two_theta[left_idx]
    else:
        frac_left = (half_height - intensity[left_idx]) / (intensity[left_idx + 1] - intensity[left_idx])
        left_tt = two_theta[left_idx] + frac_left * (two_theta[left_idx + 1] - two_theta[left_idx])
    
    if intensity[right_idx] == intensity[right_idx - 1]:
        right_tt = two_theta[right_idx]
    else:
        frac_right = (half_height - intensity[right_idx - 1]) / (intensity[right_idx] - intensity[right_idx - 1])
        right_tt = two_theta[right_idx - 1] + frac_right * (two_theta[right_idx] - two_theta[right_idx - 1])
    
    fwhm = right_tt - left_tt
    return max(fwhm, 0.0)


def find_peaks_derivative(two_theta: np.ndarray, 
                          intensity: np.ndarray,
                          threshold: float = 0.05,
                          min_distance: float = 0.5,
                          min_peak_height: float = 0.02,
                          wavelength: float = 1.5406) -> List[Peak]:
    """
    导数法寻峰
    
    参数:
        two_theta: 2theta数组
        intensity: 强度数组
        threshold: 相对阈值 (0-1)
        min_distance: 峰之间最小距离 (度)
        min_peak_height: 最小峰高 (相对最大强度)
        wavelength: X射线波长
    
    返回:
        峰列表
    """
    if len(two_theta) < 5:
        return []
    
    intensity_norm = intensity / np.max(intensity) if np.max(intensity) > 0 else intensity
    
    window_length = min(15, len(two_theta) // 4 * 2 + 1)
    if window_length < 5:
        window_length = 5
    if window_length % 2 == 0:
        window_length += 1
    
    polyorder = min(3, window_length - 2)
    if polyorder < 1:
        polyorder = 1
    
    try:
        intensity_smooth = savgol_filter(intensity_norm, window_length, polyorder)
    except:
        intensity_smooth = intensity_norm
    
    first_deriv = np.gradient(intensity_smooth, two_theta)
    second_deriv = np.gradient(first_deriv, two_theta)
    
    zero_crossings = np.where(np.diff(np.sign(first_deriv)))[0]
    
    peak_indices = []
    for idx in zero_crossings:
        if idx >= len(second_deriv):
            continue
        if second_deriv[idx] < 0:
            if intensity_smooth[idx] > min_peak_height:
                peak_indices.append(idx)
    
    min_dist_points = int(min_distance / np.mean(np.diff(two_theta)))
    
    peaks = []
    used_indices = set()
    
    peak_indices_sorted = sorted(peak_indices, key=lambda i: intensity_smooth[i], reverse=True)
    
    for idx in peak_indices_sorted:
        if any(abs(idx - used) < min_dist_points for used in used_indices):
            continue
        
        if idx > 0 and idx < len(two_theta) - 1:
            if intensity_smooth[idx] >= threshold * np.max(intensity_smooth):
                used_indices.add(idx)
                d = wavelength / (2 * np.sin(np.deg2rad(two_theta[idx] / 2)))
                fwhm = estimate_fwhm(two_theta, intensity_smooth, idx)
                peaks.append(Peak(
                    two_theta=two_theta[idx],
                    intensity=intensity_smooth[idx],
                    d_spacing=d,
                    fwhm=fwhm
                ))
    
    peaks.sort(key=lambda p: p.two_theta)
    return peaks


def find_peaks_scipy(two_theta: np.ndarray,
                     intensity: np.ndarray,
                     height: float = 0.05,
                     distance: float = 0.5,
                     prominence: float = 0.02,
                     wavelength: float = 1.5406) -> List[Peak]:
    """
    使用scipy的find_peaks寻峰
    
    参数:
        two_theta: 2theta数组
        intensity: 强度数组
        height: 最小峰高 (相对)
        distance: 峰之间最小距离 (度)
        prominence: 峰的突出度
        wavelength: X射线波长
    
    返回:
        峰列表
    """
    intensity_norm = intensity / np.max(intensity) if np.max(intensity) > 0 else intensity
    
    min_dist_points = max(1, int(distance / np.mean(np.diff(two_theta))))
    
    peak_indices, properties = find_peaks(
        intensity_norm,
        height=height,
        distance=min_dist_points,
        prominence=prominence
    )
    
    peaks = []
    for idx in peak_indices:
        d = wavelength / (2 * np.sin(np.deg2rad(two_theta[idx] / 2)))
        fwhm = estimate_fwhm(two_theta, intensity_norm, idx)
        peaks.append(Peak(
            two_theta=two_theta[idx],
            intensity=intensity_norm[idx],
            d_spacing=d,
            fwhm=fwhm
        ))
    
    return peaks


def d_from_twotheta(two_theta: float, wavelength: float = 1.5406) -> float:
    """根据2theta计算d间距"""
    return wavelength / (2 * np.sin(np.deg2rad(two_theta / 2)))


def twotheta_from_d(d: float, wavelength: float = 1.5406) -> float:
    """根据d间距计算2theta"""
    sin_theta = wavelength / (2 * d)
    if sin_theta > 1.0:
        return np.nan
    return 2 * np.rad2deg(np.arcsin(sin_theta))


def cubic_d(a: float, h: int, k: int, l: int) -> float:
    """立方晶系面间距"""
    return a / np.sqrt(h*h + k*k + l*l)


def tetragonal_d(a: float, c: float, h: int, k: int, l: int) -> float:
    """四方晶系面间距"""
    return 1.0 / np.sqrt((h*h + k*k) / (a*a) + l*l / (c*c))


def orthorhombic_d(a: float, b: float, c: float, h: int, k: int, l: int) -> float:
    """正交晶系面间距"""
    return 1.0 / np.sqrt(h*h/(a*a) + k*k/(b*b) + l*l/(c*c))


def hexagonal_d(a: float, c: float, h: int, k: int, l: int) -> float:
    """六方晶系面间距"""
    return 1.0 / np.sqrt((4/3.0) * (h*h + h*k + k*k) / (a*a) + l*l / (c*c))


def monoclinic_d(a: float, b: float, c: float, beta: float, h: int, k: int, l: int) -> float:
    """单斜晶系面间距"""
    beta_rad = np.deg2rad(beta)
    sin_beta = np.sin(beta_rad)
    cos_beta = np.cos(beta_rad)
    return 1.0 / np.sqrt(
        h*h/(a*a*sin_beta*sin_beta) + 
        k*k/(b*b) + 
        l*l/(c*c*sin_beta*sin_beta) +
        2*h*l*cos_beta/(a*c*sin_beta*sin_beta)
    )


def figure_of_merit_M(peaks: List[Peak], 
                      d_calculated: List[float],
                      N: int = 20) -> float:
    """
    计算品质因子M_N
    
    M_N = (N / (2 * |delta_2theta|_avg * N_observed)) * 100
    
    参数:
        peaks: 实验峰列表
        d_calculated: 计算的d值列表
        N: 前N个峰
    
    返回:
        M_N值
    """
    if len(peaks) < 2 or len(d_calculated) < 2:
        return 0.0
    
    n = min(N, len(peaks), len(d_calculated))
    
    d_obs = [p.d_spacing for p in peaks[:n]]
    d_calc_sorted = sorted(d_calculated)
    
    total_error = 0.0
    n_matched = 0
    
    for d_o in d_obs:
        min_diff = float('inf')
        for d_c in d_calc_sorted:
            diff = abs(d_o - d_c) / d_o
            if diff < min_diff:
                min_diff = diff
        if min_diff < 0.05:
            total_error += min_diff
            n_matched += 1
    
    if n_matched == 0:
        return 0.0
    
    avg_error = total_error / n_matched
    
    if avg_error == 0:
        return 1000.0
    
    M = (n_matched / (n * avg_error)) * 0.5
    return M


def figure_of_merit_F(peaks: List[Peak],
                      d_calculated: List[float],
                      N: int = 30,
                      tolerance: float = 0.02) -> float:
    """
    计算品质因子F_N (Smith-Snyder)
    
    F_N = (1 / (N * <delta_2theta>)) * 100
    
    参数:
        peaks: 实验峰列表
        d_calculated: 计算的d值列表
        N: 前N个峰
        tolerance: 容差
    
    返回:
        F_N值
    """
    if len(peaks) < 2 or len(d_calculated) < 2:
        return 0.0
    
    n = min(N, len(peaks))
    
    tt_obs = np.array([p.two_theta for p in peaks[:n]])
    d_calc_sorted = sorted(d_calculated)
    
    wavelength = 1.5406
    tt_calc = np.array([twotheta_from_d(d, wavelength) for d in d_calc_sorted])
    tt_calc = tt_calc[~np.isnan(tt_calc)]
    tt_calc.sort()
    
    total_error = 0.0
    n_matched = 0
    
    for tt_o in tt_obs:
        min_diff = float('inf')
        for tt_c in tt_calc:
            diff = abs(tt_o - tt_c)
            if diff < min_diff:
                min_diff = diff
        if min_diff < tolerance * tt_o:
            total_error += min_diff
            n_matched += 1
    
    if n_matched == 0:
        return 0.0
    
    avg_error = total_error / n_matched
    
    if avg_error == 0:
        return 1000.0
    
    F = 1.0 / (avg_error * n) * 10
    return F


def index_pattern(peaks: List[Peak], 
                  wavelength: float = 1.5406,
                  max_results: int = 5) -> List[IndexingResult]:
    """
    试错法指标化: 从高对称到低对称搜索晶胞参数
    
    参数:
        peaks: 实验峰列表
        wavelength: X射线波长
        max_results: 返回的候选结果数量
    
    返回:
        指标化结果列表 (按M20降序排列)
    """
    if len(peaks) < 8:
        return []
    
    results = []
    
    results.extend(_try_cubic(peaks, wavelength))
    
    results.extend(_try_tetragonal(peaks, wavelength))
    
    results.extend(_try_orthorhombic(peaks, wavelength))
    
    results.extend(_try_hexagonal(peaks, wavelength))
    
    results.extend(_try_monoclinic(peaks, wavelength))
    
    results.sort(key=lambda r: r.M20, reverse=True)
    
    return results[:max_results]


def _try_cubic(peaks: List[Peak], wavelength: float) -> List[IndexingResult]:
    """尝试立方晶系"""
    results = []
    
    d_first = peaks[0].d_spacing
    
    a_guesses = []
    for h, k, l in [(1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,1,0), (2,1,1), (2,2,0), (3,1,0), (2,2,2)]:
        a_guess = d_first * np.sqrt(h*h + k*k + l*l)
        if 2.0 < a_guess < 30.0:
            a_guesses.append(a_guess)
    
    for a in np.linspace(min(d_first * 0.8, 3.0), max(d_first * 3.0, 15.0), 30):
        d_calc = []
        for h in range(1, 8):
            for k in range(0, h+1):
                for l in range(0, k+1):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    d = cubic_d(a, h, k, l)
                    if d > 0.5:
                        d_calc.append(d)
        
        M20 = figure_of_merit_M(peaks, d_calc, 20)
        F30 = figure_of_merit_F(peaks, d_calc, 30)
        
        n_indexed = _count_indexed_peaks(peaks, d_calc)
        
        if M20 > 5.0:
            results.append(IndexingResult(
                crystal_system="cubic",
                a=a, b=a, c=a,
                alpha=90, beta=90, gamma=90,
                M20=M20, F30=F30,
                n_peaks_indexed=n_indexed,
                volume=a**3
            ))
    
    return results


def _try_tetragonal(peaks: List[Peak], wavelength: float) -> List[IndexingResult]:
    """尝试四方晶系"""
    results = []
    
    d_first = peaks[0].d_spacing
    
    a_range = np.linspace(max(d_first * 0.6, 2.0), d_first * 2.5, 15)
    c_range = np.linspace(d_first * 0.8, d_first * 4.0, 15)
    
    best_M = 0
    best_params = None
    
    for a in a_range:
        for c in c_range:
            d_calc = []
            for h in range(1, 7):
                for k in range(0, h+1):
                    for l in range(1, 7):
                        d = tetragonal_d(a, c, h, k, l)
                        if d > 0.5:
                            d_calc.append(d)
            
            M20 = figure_of_merit_M(peaks, d_calc, 20)
            
            if M20 > best_M:
                best_M = M20
                best_params = (a, c, d_calc)
    
    if best_params is not None and best_M > 3.0:
        a, c, d_calc = best_params
        M20 = figure_of_merit_M(peaks, d_calc, 20)
        F30 = figure_of_merit_F(peaks, d_calc, 30)
        n_indexed = _count_indexed_peaks(peaks, d_calc)
        
        results.append(IndexingResult(
            crystal_system="tetragonal",
            a=a, b=a, c=c,
            alpha=90, beta=90, gamma=90,
            M20=M20, F30=F30,
            n_peaks_indexed=n_indexed,
            volume=a*a*c
        ))
    
    return results


def _try_orthorhombic(peaks: List[Peak], wavelength: float) -> List[IndexingResult]:
    """尝试正交晶系"""
    results = []
    
    if len(peaks) < 5:
        return results
    
    d_values = [p.d_spacing for p in peaks[:5]]
    
    a_est = d_values[0] * 1.5
    b_est = d_values[1] * 1.5
    c_est = d_values[2] * 1.5
    
    best_M = 0
    best_params = None
    
    for a in np.linspace(a_est * 0.7, a_est * 1.3, 10):
        for b in np.linspace(b_est * 0.7, b_est * 1.3, 10):
            for c in np.linspace(c_est * 0.7, c_est * 1.3, 10):
                d_calc = []
                for h in range(1, 5):
                    for k in range(1, 5):
                        for l in range(1, 5):
                            d = orthorhombic_d(a, b, c, h, k, l)
                            if d > 0.5:
                                d_calc.append(d)
                
                M20 = figure_of_merit_M(peaks, d_calc, 15)
                
                if M20 > best_M:
                    best_M = M20
                    best_params = (a, b, c, d_calc)
    
    if best_params is not None and best_M > 2.0:
        a, b, c, d_calc = best_params
        M20 = figure_of_merit_M(peaks, d_calc, 20)
        F30 = figure_of_merit_F(peaks, d_calc, 30)
        n_indexed = _count_indexed_peaks(peaks, d_calc)
        
        results.append(IndexingResult(
            crystal_system="orthorhombic",
            a=a, b=b, c=c,
            alpha=90, beta=90, gamma=90,
            M20=M20, F30=F30,
            n_peaks_indexed=n_indexed,
            volume=a*b*c
        ))
    
    return results


def _try_hexagonal(peaks: List[Peak], wavelength: float) -> List[IndexingResult]:
    """尝试六方晶系"""
    results = []
    
    d_first = peaks[0].d_spacing
    
    best_M = 0
    best_params = None
    
    for a in np.linspace(max(d_first * 0.7, 2.0), d_first * 3.0, 15):
        for c in np.linspace(d_first * 0.8, d_first * 5.0, 15):
            d_calc = []
            for h in range(1, 5):
                for k in range(0, h+1):
                    for l in range(1, 7):
                        d = hexagonal_d(a, c, h, k, l)
                        if d > 0.5:
                            d_calc.append(d)
            
            M20 = figure_of_merit_M(peaks, d_calc, 15)
            
            if M20 > best_M:
                best_M = M20
                best_params = (a, c, d_calc)
    
    if best_params is not None and best_M > 2.0:
        a, c, d_calc = best_params
        M20 = figure_of_merit_M(peaks, d_calc, 20)
        F30 = figure_of_merit_F(peaks, d_calc, 30)
        n_indexed = _count_indexed_peaks(peaks, d_calc)
        
        results.append(IndexingResult(
            crystal_system="hexagonal",
            a=a, b=a, c=c,
            alpha=90, beta=90, gamma=120,
            M20=M20, F30=F30,
            n_peaks_indexed=n_indexed,
            volume = (np.sqrt(3) / 2) * a * a * c
        ))
    
    return results


def _try_monoclinic(peaks: List[Peak], wavelength: float) -> List[IndexingResult]:
    """尝试单斜晶系"""
    results = []
    
    if len(peaks) < 8:
        return results
    
    d_first = peaks[0].d_spacing
    
    best_M = 0
    best_params = None
    
    for a in np.linspace(max(d_first * 0.6, 2.0), d_first * 2.5, 8):
        for b in np.linspace(max(d_first * 0.6, 2.0), d_first * 2.5, 8):
            for c in np.linspace(d_first * 0.8, d_first * 3.0, 8):
                for beta in np.linspace(90, 120, 5):
                    d_calc = []
                    for h in range(1, 4):
                        for k in range(1, 4):
                            for l in range(1, 4):
                                d = monoclinic_d(a, b, c, beta, h, k, l)
                                if d > 0.5:
                                    d_calc.append(d)
                    
                    M20 = figure_of_merit_M(peaks, d_calc, 10)
                    
                    if M20 > best_M:
                        best_M = M20
                        best_params = (a, b, c, beta, d_calc)
    
    if best_params is not None and best_M > 1.5:
        a, b, c, beta, d_calc = best_params
        M20 = figure_of_merit_M(peaks, d_calc, 20)
        F30 = figure_of_merit_F(peaks, d_calc, 30)
        n_indexed = _count_indexed_peaks(peaks, d_calc)
        
        volume = a * b * c * np.sin(np.deg2rad(beta))
        
        results.append(IndexingResult(
            crystal_system="monoclinic",
            a=a, b=b, c=c,
            alpha=90, beta=beta, gamma=90,
            M20=M20, F30=F30,
            n_peaks_indexed=n_indexed,
            volume=volume
        ))
    
    return results


def _count_indexed_peaks(peaks: List[Peak], d_calc: List[float], tolerance: float = 0.02) -> int:
    """计算被指标化的峰的数量"""
    d_calc_sorted = sorted(d_calc)
    count = 0
    
    for peak in peaks:
        d_o = peak.d_spacing
        min_diff = float('inf')
        for d_c in d_calc_sorted:
            diff = abs(d_o - d_c) / d_o
            if diff < min_diff:
                min_diff = diff
        if min_diff < tolerance:
            count += 1
    
    return count

"""
峰形模拟模块
包含高斯、洛伦兹、Pseudo-Voigt峰形，以及Caglioti公式
"""

import numpy as np


def gaussian(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """
    高斯函数
    
    参数:
        x: 自变量数组
        center: 峰中心
        fwhm: 半高宽
    
    返回:
        归一化的高斯函数值 (最大值为1)
    """
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-(x - center)**2 / (2.0 * sigma**2))


def lorentzian(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """
    洛伦兹函数 (柯西分布)
    
    参数:
        x: 自变量数组
        center: 峰中心
        fwhm: 半高宽
    
    返回:
        归一化的洛伦兹函数值 (最大值为1)
    """
    gamma = fwhm / 2.0
    return 1.0 / (1.0 + ((x - center) / gamma)**2)


def pseudo_voigt(x: np.ndarray, center: float, fwhm: float, eta: float = 0.5) -> np.ndarray:
    """
    Pseudo-Voigt函数: 高斯与洛伦兹的线性混合
    
    参数:
        x: 自变量数组
        center: 峰中心
        fwhm: 半高宽
        eta: 混合参数 (0=纯高斯, 1=纯洛伦兹)
    
    返回:
        归一化的Pseudo-Voigt函数值
    """
    eta = np.clip(eta, 0.0, 1.0)
    g = gaussian(x, center, fwhm)
    l = lorentzian(x, center, fwhm)
    return eta * l + (1.0 - eta) * g


def caglioti_fwhm(two_theta: float, U: float, V: float, W: float, 
                  wavelength: float = 1.5406) -> float:
    """
    Caglioti公式: 半高宽随2theta变化
    
    FWHM^2 = U * tan^2(theta) + V * tan(theta) + W
    
    参数:
        two_theta: 衍射角2theta (度)
        U, V, W: Caglioti公式参数
        wavelength: X射线波长 (埃) (本公式中不直接使用，但保持接口一致)
    
    返回:
        半高宽FWHM (度)
    """
    theta = np.deg2rad(two_theta / 2.0)
    tan_theta = np.tan(theta)
    fwhm_sq = U * tan_theta**2 + V * tan_theta + W
    
    if fwhm_sq < 0:
        return 0.01
    
    return np.sqrt(fwhm_sq)


def caglioti_fwhm_array(two_theta: np.ndarray, U: float, V: float, W: float,
                        wavelength: float = 1.5406) -> np.ndarray:
    """
    Caglioti公式: 数组版本
    
    参数:
        two_theta: 衍射角2theta数组 (度)
        U, V, W: Caglioti公式参数
        wavelength: X射线波长 (埃)
    
    返回:
        FWHM数组 (度)
    """
    theta = np.deg2rad(two_theta / 2.0)
    tan_theta = np.tan(theta)
    fwhm_sq = U * tan_theta**2 + V * tan_theta + W
    fwhm_sq = np.maximum(fwhm_sq, 1e-6)
    return np.sqrt(fwhm_sq)


def debye_scherrer_fwhm(two_theta: float, K: float = 0.9, 
                        crystallite_size: float = 100.0, 
                        wavelength: float = 1.5406) -> float:
    """
    德拜-谢乐公式计算微晶尺寸引起的峰加宽
    
    参数:
        two_theta: 衍射角2theta (度)
        K: 形状因子 (通常取0.9)
        crystallite_size: 微晶尺寸 (埃)
        wavelength: X射线波长 (埃)
    
    返回:
        半高宽FWHM (度)
    """
    theta = np.deg2rad(two_theta / 2.0)
    beta_rad = K * wavelength / (crystallite_size * np.cos(theta))
    return np.rad2deg(beta_rad)


def voigt_fwhm(gaussian_fwhm: float, lorentzian_fwhm: float) -> float:
    """
    近似计算Voigt函数的FWHM
    
    参数:
        gaussian_fwhm: 高斯分量FWHM
        lorentzian_fwhm: 洛伦兹分量FWHM
    
    返回:
        Voigt函数的近似FWHM
    """
    fg = gaussian_fwhm
    fl = lorentzian_fwhm
    return 0.5346 * fl + np.sqrt(0.2166 * fl**2 + fg**2)

"""
Williamson-Hall分析功能测试脚本
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from xrd_crystal.indexing import Peak, estimate_fwhm
from app import williamson_hall_analysis, WHResult


def test_fwhm_estimation():
    """测试FWHM估算功能"""
    print("=" * 60)
    print("测试1: FWHM估算功能")
    print("=" * 60)
    
    from xrd_crystal.peak_fitting import gaussian
    
    true_fwhm = 0.2
    center = 30.0
    two_theta = np.linspace(29.0, 31.0, 1001)
    intensity = gaussian(two_theta, center, true_fwhm)
    
    peak_idx = np.argmax(intensity)
    estimated_fwhm = estimate_fwhm(two_theta, intensity, peak_idx)
    
    print(f"  真实FWHM: {true_fwhm:.4f}°")
    print(f"  估算FWHM: {estimated_fwhm:.4f}°")
    print(f"  误差: {abs(estimated_fwhm - true_fwhm):.6f}°")
    print(f"  结果: {'✓ 通过' if abs(estimated_fwhm - true_fwhm) < 0.01 else '✗ 失败'}")
    
    return abs(estimated_fwhm - true_fwhm) < 0.01


def test_wh_analysis_basic():
    """测试基本的W-H分析"""
    print("\n" + "=" * 60)
    print("测试2: Williamson-Hall分析核心逻辑")
    print("=" * 60)
    
    wavelength = 1.5406
    K = 0.9
    true_crystallite_size_nm = 50.0
    true_microstrain = 0.001
    
    crystallite_size_angstrom = true_crystallite_size_nm * 10
    intercept_true = K * wavelength * 0.1 / true_crystallite_size_nm
    slope_true = true_microstrain
    
    print(f"  模拟参数:")
    print(f"    晶粒尺寸: {true_crystallite_size_nm} nm")
    print(f"    微应变: {true_microstrain}")
    print(f"    理论截距: {intercept_true:.6f}")
    print(f"    理论斜率: {slope_true:.6f}")
    
    two_theta_values = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    peaks = []
    
    for i, tt in enumerate(two_theta_values):
        theta_rad = np.deg2rad(tt / 2.0)
        sin_theta = np.sin(theta_rad)
        cos_theta = np.cos(theta_rad)
        x_val = 4.0 * sin_theta
        
        y_value = intercept_true + slope_true * x_val
        beta_sample_rad = y_value / cos_theta
        beta_sample_deg = np.rad2deg(beta_sample_rad)
        
        instrument_fwhm = 0.05
        measured_fwhm = np.sqrt(beta_sample_deg**2 + instrument_fwhm**2)
        
        d = wavelength / (2 * sin_theta)
        
        peaks.append(Peak(
            two_theta=tt,
            intensity=1.0 / (i + 1) if i < 5 else 0.2,
            d_spacing=d,
            fwhm=measured_fwhm
        ))
    
    result = williamson_hall_analysis(
        peaks=peaks,
        wavelength=wavelength,
        use_caglioti=False,
        fwhm_fixed=instrument_fwhm,
        U=0, V=0, W=0,
        K=K
    )
    
    print(f"\n  分析结果:")
    print(f"    有效峰数: {result.valid_count}/{result.total_count}")
    print(f"    晶粒尺寸: {result.crystallite_size:.2f} nm (真值: {true_crystallite_size_nm})")
    print(f"    微应变: {result.microstrain:.6f} (真值: {true_microstrain})")
    print(f"    R²: {result.r_squared:.4f}")
    print(f"    斜率: {result.slope:.6f} (真值: {slope_true:.6f})")
    print(f"    截距: {result.intercept:.6f} (真值: {intercept_true:.6f})")
    
    size_ok = abs(result.crystallite_size - true_crystallite_size_nm) < 5.0
    strain_ok = abs(result.microstrain - true_microstrain) < 0.0002
    r2_ok = result.r_squared > 0.99
    
    print(f"\n  晶粒尺寸: {'✓ 通过' if size_ok else '✗ 失败'}")
    print(f"  微应变: {'✓ 通过' if strain_ok else '✗ 失败'}")
    print(f"  R²: {'✓ 通过' if r2_ok else '✗ 失败'}")
    
    return size_ok and strain_ok and r2_ok


def test_wh_peak_exclusion():
    """测试峰排除逻辑"""
    print("\n" + "=" * 60)
    print("测试3: 峰排除逻辑")
    print("=" * 60)
    
    wavelength = 1.5406
    
    peaks = []
    for i, tt in enumerate([20.0, 30.0, 40.0]):
        d = wavelength / (2 * np.sin(np.deg2rad(tt / 2.0)))
        peaks.append(Peak(
            two_theta=tt,
            intensity=1.0,
            d_spacing=d,
            fwhm=0.03
        ))
    
    instrument_fwhm = 0.05
    result = williamson_hall_analysis(
        peaks=peaks,
        wavelength=wavelength,
        use_caglioti=False,
        fwhm_fixed=instrument_fwhm,
        U=0, V=0, W=0
    )
    
    print(f"  总峰数: {result.total_count}")
    print(f"  有效峰数: {result.valid_count}")
    print(f"  预期有效峰数: 0 (所有峰FWHM < 仪器展宽)")
    
    excluded_all = result.valid_count == 0
    
    excluded_reasons_correct = all(
        p.exclude_reason == "实测FWHM≤仪器展宽" 
        for p in result.peaks_data if p.excluded
    )
    
    print(f"  峰排除逻辑: {'✓ 通过' if excluded_all else '✗ 失败'}")
    print(f"  排除原因正确: {'✓ 通过' if excluded_reasons_correct else '✗ 失败'}")
    
    return excluded_all and excluded_reasons_correct


def test_wh_min_peaks():
    """测试最小峰数要求"""
    print("\n" + "=" * 60)
    print("测试4: 最小有效峰数检查")
    print("=" * 60)
    
    wavelength = 1.5406
    
    peaks = []
    for tt in [20.0, 30.0]:
        d = wavelength / (2 * np.sin(np.deg2rad(tt / 2.0)))
        peaks.append(Peak(
            two_theta=tt,
            intensity=1.0,
            d_spacing=d,
            fwhm=0.2
        ))
    
    result = williamson_hall_analysis(
        peaks=peaks,
        wavelength=wavelength,
        use_caglioti=False,
        fwhm_fixed=0.01,
        U=0, V=0, W=0
    )
    
    print(f"  有效峰数: {result.valid_count}")
    print(f"  预期: 2 (不足3个)")
    print(f"  回归是否被禁止: {'✓ 通过' if result.valid_count < 3 and result.slope == 0 else '✗ 失败'}")
    
    return result.valid_count < 3 and result.slope == 0


if __name__ == "__main__":
    print("Williamson-Hall分析功能单元测试")
    print()
    
    tests = [
        ("FWHM估算", test_fwhm_estimation),
        ("W-H核心计算", test_wh_analysis_basic),
        ("峰排除逻辑", test_wh_peak_exclusion),
        ("最小峰数要求", test_wh_min_peaks),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  [异常] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n  🎉 所有测试通过!")
        sys.exit(0)
    else:
        print(f"\n  ⚠️  有 {total - passed} 个测试失败")
        sys.exit(1)

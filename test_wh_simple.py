"""
Williamson-Hall分析功能 - 简化测试脚本
"""
import numpy as np
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, '.')

from xrd_crystal.indexing import Peak, estimate_fwhm
from app import williamson_hall_analysis, WHResult


def test_fwhm_estimation():
    """测试FWHM估算功能"""
    from xrd_crystal.peak_fitting import gaussian
    
    true_fwhm = 0.2
    center = 30.0
    two_theta = np.linspace(29.0, 31.0, 1001)
    intensity = gaussian(two_theta, center, true_fwhm)
    
    peak_idx = np.argmax(intensity)
    estimated_fwhm = estimate_fwhm(two_theta, intensity, peak_idx)
    
    error = abs(estimated_fwhm - true_fwhm)
    passed = error < 0.01
    
    print(f"[Test 1] FWHM Estimation:")
    print(f"  True FWHM:      {true_fwhm:.4f} deg")
    print(f"  Estimated FWHM: {estimated_fwhm:.4f} deg")
    print(f"  Error:          {error:.6f} deg")
    print(f"  Result:         {'PASS' if passed else 'FAIL'}")
    
    return passed


def test_wh_analysis_basic():
    """测试基本的W-H分析"""
    wavelength = 1.5406
    K = 0.9
    true_crystallite_size_nm = 50.0
    true_microstrain = 0.001
    
    intercept_true = K * wavelength * 0.1 / true_crystallite_size_nm
    slope_true = true_microstrain
    
    print(f"\n[Test 2] Williamson-Hall Core Logic:")
    print(f"  Simulation params:")
    print(f"    Crystallite size: {true_crystallite_size_nm} nm")
    print(f"    Microstrain:      {true_microstrain}")
    print(f"    Expected intercept: {intercept_true:.6f}")
    print(f"    Expected slope:     {slope_true:.6f}")
    
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
    
    print(f"\n  Analysis results:")
    print(f"    Valid peaks:      {result.valid_count}/{result.total_count}")
    print(f"    Crystallite size: {result.crystallite_size:.2f} nm (true: {true_crystallite_size_nm})")
    print(f"    Microstrain:      {result.microstrain:.6f} (true: {true_microstrain})")
    print(f"    R-squared:        {result.r_squared:.4f}")
    print(f"    Slope:            {result.slope:.6f} (true: {slope_true:.6f})")
    print(f"    Intercept:        {result.intercept:.6f} (true: {intercept_true:.6f})")
    
    size_ok = abs(result.crystallite_size - true_crystallite_size_nm) < 5.0
    strain_ok = abs(result.microstrain - true_microstrain) < 0.0002
    r2_ok = result.r_squared > 0.99
    
    print(f"\n  Crystallite size check: {'PASS' if size_ok else 'FAIL'}")
    print(f"  Microstrain check:      {'PASS' if strain_ok else 'FAIL'}")
    print(f"  R-squared check:        {'PASS' if r2_ok else 'FAIL'}")
    
    return size_ok and strain_ok and r2_ok


def test_wh_peak_exclusion():
    """测试峰排除逻辑"""
    wavelength = 1.5406
    
    peaks = []
    for tt in [20.0, 30.0, 40.0]:
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
    
    print(f"\n[Test 3] Peak Exclusion Logic:")
    print(f"  Total peaks: {result.total_count}")
    print(f"  Valid peaks: {result.valid_count}")
    print(f"  Expected: 0 (all FWHM < instrument broadening)")
    
    excluded_all = result.valid_count == 0
    reasons_correct = all(
        p.exclude_reason == "实测FWHM≤仪器展宽" 
        for p in result.peaks_data if p.excluded
    )
    
    print(f"  Exclusion logic:    {'PASS' if excluded_all else 'FAIL'}")
    print(f"  Exclusion reasons:  {'PASS' if reasons_correct else 'FAIL'}")
    
    return excluded_all and reasons_correct


def test_wh_min_peaks():
    """测试最小峰数要求"""
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
    
    print(f"\n[Test 4] Minimum Valid Peaks Check:")
    print(f"  Valid peaks: {result.valid_count}")
    print(f"  Expected: 2 (insufficient, needs >= 3)")
    
    regression_blocked = result.valid_count < 3 and result.slope == 0
    print(f"  Regression blocked: {'PASS' if regression_blocked else 'FAIL'}")
    
    return regression_blocked


if __name__ == "__main__":
    print("=" * 60)
    print("Williamson-Hall Analysis - Unit Tests")
    print("=" * 60)
    
    tests = [
        ("FWHM Estimation", test_fwhm_estimation),
        ("W-H Core Calculation", test_wh_analysis_basic),
        ("Peak Exclusion Logic", test_wh_peak_exclusion),
        ("Minimum Peaks Requirement", test_wh_min_peaks),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[EXCEPTION in {name}]: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  ** ALL TESTS PASSED! **")
        sys.exit(0)
    else:
        print(f"\n  ** {total - passed} TEST(S) FAILED **")
        sys.exit(1)

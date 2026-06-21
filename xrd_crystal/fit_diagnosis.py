"""
拟合质量诊断模块
对定量分析拟合结果进行多项质量检查
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum


class DiagnosisStatus(Enum):
    """诊断状态枚举"""
    PASS = "pass"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class DiagnosisItem:
    """单项诊断结果"""
    name: str
    status: DiagnosisStatus
    message: str
    detail: str = ""


@dataclass
class FitDiagnosisResult:
    """拟合质量诊断总结果"""
    items: List[DiagnosisItem] = field(default_factory=list)
    summary: str = ""
    pass_count: int = 0
    warning_count: int = 0
    error_count: int = 0

    def add_item(self, item: DiagnosisItem):
        """添加诊断项"""
        self.items.append(item)
        if item.status == DiagnosisStatus.PASS:
            self.pass_count += 1
        elif item.status == DiagnosisStatus.WARNING:
            self.warning_count += 1
        elif item.status == DiagnosisStatus.ERROR:
            self.error_count += 1

    def generate_summary(self):
        """生成汇总结论"""
        total = len(self.items)
        if self.error_count > 0:
            self.summary = f"发现 {self.error_count} 项严重问题和 {self.warning_count} 项警告需关注"
        elif self.warning_count > 0:
            self.summary = f"发现 {self.warning_count} 项警告需关注，{self.pass_count} 项通过"
        else:
            self.summary = f"拟合质量良好，{self.pass_count} 项通过/0项警告/0项异常"
        return self.summary


def runs_test(residuals: np.ndarray) -> Tuple[float, int]:
    """
    游程检验 (Wald-Wolfowitz runs test)
    检验残差序列是否具有随机性

    参数:
        residuals: 残差数组

    返回:
        (p值, 游程数)
    """
    residuals = np.asarray(residuals).flatten()
    n = len(residuals)

    if n < 10:
        return 1.0, 0

    median = np.median(residuals)
    signs = residuals > median

    if np.all(signs) or not np.any(signs):
        return 0.0, 1

    n1 = np.sum(signs)
    n2 = n - n1

    runs = 1
    for i in range(1, n):
        if signs[i] != signs[i - 1]:
            runs += 1

    if n1 == 0 or n2 == 0:
        return 0.0, runs

    expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
    variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))

    if variance <= 0:
        return 1.0, runs

    z = (runs - expected_runs) / np.sqrt(variance)

    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return p_value, runs


def _integrate_curve(x: np.ndarray, y: np.ndarray) -> float:
    """
    数值积分（梯形法）
    """
    if len(x) < 2 or len(y) < 2:
        return 0.0
    y_pos = np.maximum(y, 0)
    return float(np.trapz(y_pos, x))


def diagnose_fit_quality(
    result,
    rwp_threshold_warning: float = 0.05
) -> FitDiagnosisResult:
    """
    对拟合结果进行质量诊断

    参数:
        result: QuantitativeResult对象
        rwp_threshold_warning: 单相贡献度警告阈值 (占总Rwp的比例)

    返回:
        FitDiagnosisResult对象
    """
    diagnosis = FitDiagnosisResult()

    # 1. 残差分布检验 - 游程检验
    try:
        residuals = result.difference_pattern
        p_value, n_runs = runs_test(residuals)

        if p_value < 0.05:
            diagnosis.add_item(DiagnosisItem(
                name="残差分布检验",
                status=DiagnosisStatus.WARNING,
                message="残差非随机，拟合可能遗漏了某个相或背景模型不足",
                detail=f"游程数: {n_runs}, p值: {p_value:.4f}"
            ))
        else:
            diagnosis.add_item(DiagnosisItem(
                name="残差分布检验",
                status=DiagnosisStatus.PASS,
                message="残差分布随机，拟合无明显系统偏差",
                detail=f"游程数: {n_runs}, p值: {p_value:.4f}"
            ))
    except Exception as e:
        diagnosis.add_item(DiagnosisItem(
            name="残差分布检验",
            status=DiagnosisStatus.WARNING,
            message=f"检验计算异常: {str(e)}",
            detail=""
        ))

    # 2. 单相贡献度警告
    try:
        total_rwp = result.Rwp
        low_contrib_phases = []

        for pr in result.phase_results:
            if total_rwp > 0 and pr.rwp_contribution / total_rwp <= rwp_threshold_warning:
                low_contrib_phases.append(pr.phase_name)

        if low_contrib_phases:
            phase_list = ", ".join(low_contrib_phases)
            diagnosis.add_item(DiagnosisItem(
                name="单相贡献度警告",
                status=DiagnosisStatus.WARNING,
                message=f"贡献度极低，可能不必要: {phase_list}",
                detail=f"警告阈值: Rwp变化 ≤ {rwp_threshold_warning*100:.1f}% 总Rwp"
            ))
        else:
            diagnosis.add_item(DiagnosisItem(
                name="单相贡献度警告",
                status=DiagnosisStatus.PASS,
                message="各相对拟合均有显著贡献",
                detail=f"共 {len(result.phase_results)} 个晶相"
            ))
    except Exception as e:
        diagnosis.add_item(DiagnosisItem(
            name="单相贡献度警告",
            status=DiagnosisStatus.WARNING,
            message=f"检验计算异常: {str(e)}",
            detail=""
        ))

    # 3. 背景占比异常检测
    try:
        two_theta = result.two_theta
        bg_pattern = result.background_pattern
        calc_pattern = result.calculated_pattern

        bg_area = _integrate_curve(two_theta, bg_pattern)
        total_area = _integrate_curve(two_theta, calc_pattern)

        if total_area > 0:
            bg_ratio = bg_area / total_area
        else:
            bg_ratio = 0.0

        if bg_ratio > 0.6:
            diagnosis.add_item(DiagnosisItem(
                name="背景占比检测",
                status=DiagnosisStatus.WARNING,
                message="背景占比过高，可能存在非晶相或背景阶数设置不当",
                detail=f"背景积分占比: {bg_ratio*100:.1f}%"
            ))
        elif bg_ratio < 0.05:
            diagnosis.add_item(DiagnosisItem(
                name="背景占比检测",
                status=DiagnosisStatus.WARNING,
                message="背景贡献极小，确认实验谱是否已扣除背景",
                detail=f"背景积分占比: {bg_ratio*100:.1f}%"
            ))
        else:
            diagnosis.add_item(DiagnosisItem(
                name="背景占比检测",
                status=DiagnosisStatus.PASS,
                message="背景占比在合理范围内",
                detail=f"背景积分占比: {bg_ratio*100:.1f}%"
            ))
    except Exception as e:
        diagnosis.add_item(DiagnosisItem(
            name="背景占比检测",
            status=DiagnosisStatus.WARNING,
            message=f"检测计算异常: {str(e)}",
            detail=""
        ))

    # 4. 权重因子合理性检查
    try:
        weights = np.array([pr.weight for pr in result.phase_results])
        weights = weights[weights > 0]

        if len(weights) >= 2:
            max_weight = np.max(weights)
            min_weight = np.min(weights)
            ratio = max_weight / min_weight if min_weight > 0 else float('inf')

            if ratio > 100:
                diagnosis.add_item(DiagnosisItem(
                    name="权重因子检查",
                    status=DiagnosisStatus.WARNING,
                    message="相间权重差异极大，可能存在初始相选择不当或某相衍射强度计算有误",
                    detail=f"最大/最小权重比: {ratio:.1f}"
                ))
            else:
                diagnosis.add_item(DiagnosisItem(
                    name="权重因子检查",
                    status=DiagnosisStatus.PASS,
                    message="相间权重差异在合理范围",
                    detail=f"最大/最小权重比: {ratio:.1f}"
                ))
        else:
            diagnosis.add_item(DiagnosisItem(
                name="权重因子检查",
                status=DiagnosisStatus.PASS,
                message="单相应力下不适用",
                detail="仅一个有效晶相"
            ))
    except Exception as e:
        diagnosis.add_item(DiagnosisItem(
            name="权重因子检查",
            status=DiagnosisStatus.WARNING,
            message=f"检查计算异常: {str(e)}",
            detail=""
        ))

    # 5. 收敛质量评估
    try:
        rwp_pct = result.Rwp * 100
        chi_sq = result.chi_squared

        if rwp_pct < 10 and chi_sq < 2:
            quality = "优"
            status = DiagnosisStatus.PASS
            message = "收敛质量优"
        elif rwp_pct < 20:
            quality = "良"
            status = DiagnosisStatus.PASS
            message = "收敛质量良"
        elif rwp_pct < 40:
            quality = "一般"
            status = DiagnosisStatus.WARNING
            message = "收敛质量一般"
        else:
            quality = "差"
            status = DiagnosisStatus.ERROR
            message = "收敛质量差，建议检查输入数据和相选择"

        diagnosis.add_item(DiagnosisItem(
            name="收敛质量评估",
            status=status,
            message=message,
            detail=f"Rwp = {rwp_pct:.2f}%, χ² = {chi_sq:.4f} (评级: {quality})"
        ))
    except Exception as e:
        diagnosis.add_item(DiagnosisItem(
            name="收敛质量评估",
            status=DiagnosisStatus.WARNING,
            message=f"评估计算异常: {str(e)}",
            detail=""
        ))

    diagnosis.generate_summary()
    return diagnosis

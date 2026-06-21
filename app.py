"""
XRD晶体衍射分析工具 - Streamlit主应用
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

import sys
sys.path.insert(0, '.')

from xrd_crystal.crystal import Crystal, Atom
from xrd_crystal.cif_parser import parse_cif
from xrd_crystal.space_group import SpaceGroup, get_space_group_by_number, get_space_group_by_symbol, get_sg_number, get_sg_symbol
from xrd_crystal.diffraction import diffraction_simulation, powder_pattern, powder_pattern_caglioti, DiffractionPeak
from xrd_crystal.peak_fitting import pseudo_voigt, caglioti_fwhm
from xrd_crystal.indexing import find_peaks_derivative, index_pattern, Peak, d_from_twotheta, twotheta_from_d
from xrd_crystal.le_bail import le_bail_fit
from xrd_crystal.visualization import plot_crystal_3d, get_atom_color
from xrd_crystal.quantitative_analysis import quantitative_analysis, QuantitativeResult, PhaseQuantResult


st.set_page_config(
    page_title="XRD晶体衍射分析工具",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 X射线粉末衍射谱模拟与指标化工具")
st.markdown("---")


def init_session_state():
    """初始化session state"""
    if 'crystals' not in st.session_state:
        st.session_state.crystals = {}
        st.session_state.crystals['default'] = create_default_crystal()
    
    if 'current_phase' not in st.session_state:
        st.session_state.current_phase = 'default'
    
    if 'phases' not in st.session_state:
        st.session_state.phases = ['default']
    
    if 'phase_ratios' not in st.session_state:
        st.session_state.phase_ratios = {'default': 1.0}
    
    if 'wavelength' not in st.session_state:
        st.session_state.wavelength = 1.5406
    
    if 'two_theta_min' not in st.session_state:
        st.session_state.two_theta_min = 5.0
    
    if 'two_theta_max' not in st.session_state:
        st.session_state.two_theta_max = 80.0
    
    if 'use_caglioti' not in st.session_state:
        st.session_state.use_caglioti = True
    
    if 'fwhm' not in st.session_state:
        st.session_state.fwhm = 0.15
    
    if 'U' not in st.session_state:
        st.session_state.U = 0.02
    
    if 'V' not in st.session_state:
        st.session_state.V = -0.01
    
    if 'W' not in st.session_state:
        st.session_state.W = 0.015
    
    if 'eta' not in st.session_state:
        st.session_state.eta = 0.5
    
    if 'peaks_data' not in st.session_state:
        st.session_state.peaks_data = None
    
    if 'exp_data' not in st.session_state:
        st.session_state.exp_data = None
    
    if 'exp_peaks' not in st.session_state:
        st.session_state.exp_peaks = None
    
    if 'indexing_results' not in st.session_state:
        st.session_state.indexing_results = None
    
    if 'lebail_result' not in st.session_state:
        st.session_state.lebail_result = None
    
    if 'compare_exp_data' not in st.session_state:
        st.session_state.compare_exp_data = None
    
    if 'compare_exp_peaks' not in st.session_state:
        st.session_state.compare_exp_peaks = None
    
    if 'compare_zero_shift' not in st.session_state:
        st.session_state.compare_zero_shift = 0.0
    
    if 'quant_selected_phases' not in st.session_state:
        st.session_state.quant_selected_phases = []
    
    if 'quant_exp_data' not in st.session_state:
        st.session_state.quant_exp_data = None
    
    if 'quant_result' not in st.session_state:
        st.session_state.quant_result = None
    
    if 'quant_bg_order' not in st.session_state:
        st.session_state.quant_bg_order = 3
    
    if 'quant_max_iter' not in st.session_state:
        st.session_state.quant_max_iter = 200
    
    if 'quant_tolerance' not in st.session_state:
        st.session_state.quant_tolerance = 1e-6


def create_default_crystal() -> Crystal:
    """创建默认晶体结构 (硅)"""
    crystal = Crystal(
        a=5.4309, b=5.4309, c=5.4309,
        alpha=90.0, beta=90.0, gamma=90.0,
        space_group="Fd-3m",
        space_group_number=227,
        name="硅 (Si)"
    )
    crystal.add_atom("Si", 0.0, 0.0, 0.0, occupancy=1.0, b_iso=0.5)
    return crystal


init_session_state()


with st.sidebar:
    st.header("📋 导航")
    page = st.radio(
        "选择功能",
        ["衍射谱模拟", "晶体结构", "指标化", "Le Bail拟合", "多相混合", "定量分析", "谱图对比"]
    )
    
    st.markdown("---")
    
    st.subheader("⚙️ 仪器参数")
    wavelength = st.number_input(
        "X射线波长 (Å)",
        value=st.session_state.wavelength,
        min_value=0.1,
        max_value=10.0,
        step=0.01,
        key="wavelength_input"
    )
    st.session_state.wavelength = wavelength
    
    col1, col2 = st.columns(2)
    with col1:
        two_theta_min = st.number_input(
            "2θ 最小 (°)",
            value=st.session_state.two_theta_min,
            min_value=0.0,
            max_value=180.0,
            step=1.0,
            key="tt_min"
        )
    with col2:
        two_theta_max = st.number_input(
            "2θ 最大 (°)",
            value=st.session_state.two_theta_max,
            min_value=0.0,
            max_value=180.0,
            step=1.0,
            key="tt_max"
        )
    st.session_state.two_theta_min = two_theta_min
    st.session_state.two_theta_max = two_theta_max
    
    st.markdown("---")
    
    st.subheader("📊 峰形参数")
    
    use_caglioti = st.checkbox(
        "使用Caglioti公式",
        value=st.session_state.use_caglioti,
        key="use_caglioti_cb"
    )
    st.session_state.use_caglioti = use_caglioti
    
    if use_caglioti:
        col1, col2, col3 = st.columns(3)
        with col1:
            U = st.number_input("U", value=st.session_state.U, step=0.001, format="%.4f", key="U_input")
        with col2:
            V = st.number_input("V", value=st.session_state.V, step=0.001, format="%.4f", key="V_input")
        with col3:
            W = st.number_input("W", value=st.session_state.W, step=0.001, format="%.4f", key="W_input")
        st.session_state.U = U
        st.session_state.V = V
        st.session_state.W = W
    else:
        fwhm = st.number_input(
            "FWHM (°)",
            value=st.session_state.fwhm,
            min_value=0.01,
            max_value=5.0,
            step=0.01,
            key="fwhm_input"
        )
        st.session_state.fwhm = fwhm
    
    eta = st.slider(
        "Pseudo-Voigt η (0=高斯, 1=洛伦兹)",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.eta,
        step=0.05,
        key="eta_slider"
    )
    st.session_state.eta = eta


def crystal_input_section():
    """晶体结构输入部分"""
    st.header("💎 晶体结构输入")
    
    input_method = st.radio(
        "输入方式",
        ["手动输入", "上传CIF文件"],
        horizontal=True
    )
    
    if input_method == "手动输入":
        manual_crystal_input()
    else:
        cif_upload_input()


def manual_crystal_input():
    """手动输入晶体结构"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("晶胞参数")
        
        sg_input = st.text_input(
            "空间群 (H-M符号或编号)",
            value="Fd-3m",
            help="例如: P1, Fm-3m, 225 等"
        )
        
        try:
            sg_num = int(sg_input)
            sg_symbol = get_sg_symbol(sg_num)
        except ValueError:
            sg_symbol = sg_input
            sg_num = get_sg_number(sg_input)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            a = st.number_input("a (Å)", value=5.4309, step=0.001, format="%.4f")
        with col_b:
            b = st.number_input("b (Å)", value=5.4309, step=0.001, format="%.4f")
        with col_c:
            c = st.number_input("c (Å)", value=5.4309, step=0.001, format="%.4f")
        
        col_alpha, col_beta, col_gamma = st.columns(3)
        with col_alpha:
            alpha = st.number_input("α (°)", value=90.0, step=0.1, format="%.2f")
        with col_beta:
            beta = st.number_input("β (°)", value=90.0, step=0.1, format="%.2f")
        with col_gamma:
            gamma = st.number_input("γ (°)", value=90.0, step=0.1, format="%.2f")
        
        crystal_name = st.text_input("晶体名称", value="自定义晶体")
    
    with col2:
        st.subheader("原子列表")
        
        if 'atom_list' not in st.session_state:
            st.session_state.atom_list = [
                {"element": "Si", "x": 0.0, "y": 0.0, "z": 0.0, "occupancy": 1.0, "b_iso": 0.5},
                {"element": "Si", "x": 0.25, "y": 0.25, "z": 0.25, "occupancy": 1.0, "b_iso": 0.5}
            ]
        
        for i, atom in enumerate(st.session_state.atom_list):
            with st.expander(f"原子 {i+1}: {atom['element']}", expanded=True):
                col_e, col_x, col_y = st.columns(3)
                with col_e:
                    atom['element'] = st.text_input(
                        "元素",
                        value=atom['element'],
                        key=f"elem_{i}"
                    )
                with col_x:
                    atom['x'] = st.number_input(
                        "x",
                        value=float(atom['x']),
                        step=0.01,
                        format="%.4f",
                        key=f"x_{i}"
                    )
                with col_y:
                    atom['y'] = st.number_input(
                        "y",
                        value=float(atom['y']),
                        step=0.01,
                        format="%.4f",
                        key=f"y_{i}"
                    )
                col_z, col_occ, col_b = st.columns(3)
                with col_z:
                    atom['z'] = st.number_input(
                        "z",
                        value=float(atom['z']),
                        step=0.01,
                        format="%.4f",
                        key=f"z_{i}"
                    )
                with col_occ:
                    atom['occupancy'] = st.number_input(
                        "占位",
                        value=float(atom['occupancy']),
                        min_value=0.0,
                        max_value=1.0,
                        step=0.1,
                        key=f"occ_{i}"
                    )
                with col_b:
                    atom['b_iso'] = st.number_input(
                        "B_iso",
                        value=float(atom['b_iso']),
                        step=0.1,
                        key=f"b_{i}"
                    )
                
                if st.button(f"删除原子 {i+1}", key=f"del_atom_{i}"):
                    st.session_state.atom_list.pop(i)
                    st.rerun()
        
        if st.button("➕ 添加原子"):
            st.session_state.atom_list.append({
                "element": "C",
                "x": 0.0, "y": 0.0, "z": 0.0,
                "occupancy": 1.0,
                "b_iso": 1.0
            })
            st.rerun()
    
    if st.button("✅ 确认晶体结构", type="primary"):
        crystal = Crystal(
            a=a, b=b, c=c,
            alpha=alpha, beta=beta, gamma=gamma,
            space_group=sg_symbol,
            space_group_number=sg_num,
            name=crystal_name
        )
        for atom_data in st.session_state.atom_list:
            crystal.add_atom(
                atom_data['element'],
                atom_data['x'], atom_data['y'], atom_data['z'],
                atom_data['occupancy'],
                atom_data['b_iso']
            )
        
        phase_name = f"相_{len(st.session_state.phases) + 1}"
        st.session_state.crystals[phase_name] = crystal
        st.session_state.current_phase = phase_name
        if phase_name not in st.session_state.phases:
            st.session_state.phases.append(phase_name)
            st.session_state.phase_ratios[phase_name] = 1.0
        
        st.success(f"晶体结构已保存为 {phase_name}")
        st.session_state.peaks_data = None


def cif_upload_input():
    """CIF文件上传输入"""
    
    uploaded_file = st.file_uploader(
        "上传CIF文件",
        type=['cif', 'CIF'],
        help="上传CIF格式的晶体结构文件"
    )
    
    if uploaded_file is not None:
        try:
            cif_text = uploaded_file.read().decode('utf-8')
            crystal = parse_cif(cif_text)
            
            st.success(f"成功解析CIF文件: {crystal.name}")
            
            phase_name = f"相_{len(st.session_state.phases) + 1}"
            st.session_state.crystals[phase_name] = crystal
            st.session_state.current_phase = phase_name
            if phase_name not in st.session_state.phases:
                st.session_state.phases.append(phase_name)
                st.session_state.phase_ratios[phase_name] = 1.0
            
            st.session_state.peaks_data = None
            
            with st.expander("📊 晶体结构摘要", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**空间群**: {crystal.space_group} (#{crystal.space_group_number})")
                    st.write(f"**晶系**: {crystal.crystal_system}")
                    st.write(f"**a =** {crystal.a:.4f} Å")
                    st.write(f"**b =** {crystal.b:.4f} Å")
                    st.write(f"**c =** {crystal.c:.4f} Å")
                with col2:
                    st.write(f"**α =** {crystal.alpha:.2f}°")
                    st.write(f"**β =** {crystal.beta:.2f}°")
                    st.write(f"**γ =** {crystal.gamma:.2f}°")
                    st.write(f"**体积 =** {crystal.volume:.4f} Å³")
                    st.write(f"**原子数 =** {len(crystal.atoms)}")
            
        except Exception as e:
            st.error(f"CIF文件解析失败: {e}")
            st.info("请检查CIF文件格式是否正确。")


def diffraction_simulation_section():
    """衍射谱模拟部分"""
    st.header("📈 衍射谱模拟")
    
    crystal = st.session_state.crystals.get(st.session_state.current_phase)
    
    if crystal is None:
        st.warning("请先定义晶体结构")
        return
    
    st.subheader(f"当前晶体: {crystal.name}")
    
    phase_selector = st.selectbox(
        "选择晶相",
        options=st.session_state.phases,
        index=st.session_state.phases.index(st.session_state.current_phase) if st.session_state.current_phase in st.session_state.phases else 0
    )
    st.session_state.current_phase = phase_selector
    crystal = st.session_state.crystals[phase_selector]
    
    if st.button("🔬 计算衍射谱", type="primary"):
        with st.spinner("正在计算衍射谱..."):
            peaks = diffraction_simulation(
                crystal,
                wavelength=st.session_state.wavelength,
                two_theta_min=st.session_state.two_theta_min,
                two_theta_max=st.session_state.two_theta_max,
                use_symmetry=True
            )
            st.session_state.peaks_data = peaks
            st.session_state.lebail_result = None
            st.rerun()
    
    if st.session_state.peaks_data is not None:
        peaks = st.session_state.peaks_data
        
        two_theta_range = np.linspace(
            st.session_state.two_theta_min,
            st.session_state.two_theta_max,
            1000
        )
        
        if st.session_state.use_caglioti:
            pattern = powder_pattern_caglioti(
                peaks, two_theta_range,
                U=st.session_state.U,
                V=st.session_state.V,
                W=st.session_state.W,
                eta=st.session_state.eta,
                wavelength=st.session_state.wavelength
            )
        else:
            pattern = powder_pattern(
                peaks, two_theta_range,
                fwhm=st.session_state.fwhm,
                peak_type="pseudo_voigt",
                eta=st.session_state.eta
            )
        
        tab1, tab2, tab3 = st.tabs(["衍射谱", "棒状图", "峰列表"])
        
        with tab1:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=two_theta_range,
                y=pattern,
                mode='lines',
                name='衍射谱',
                line=dict(color='#1f77b4', width=2)
            ))
            
            for i, peak in enumerate(peaks[:30]):
                if peak.intensity > 1:
                    fig.add_annotation(
                        x=peak.two_theta,
                        y=peak.intensity + 2,
                        text=f"({peak.h}{peak.k}{peak.l})<br>d={peak.d:.3f}",
                        showarrow=False,
                        font=dict(size=8),
                        textangle=-90,
                        yshift=10
                    )
            
            fig.update_layout(
                title=f"{crystal.name} - XRD衍射谱",
                xaxis_title="2θ (°)",
                yaxis_title="相对强度 (%)",
                xaxis=dict(range=[st.session_state.two_theta_min, st.session_state.two_theta_max]),
                yaxis=dict(range=[0, 110]),
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig2 = go.Figure()
            
            for peak in peaks:
                if peak.intensity <= 0.1:
                    continue
                fig2.add_trace(go.Scatter(
                    x=[peak.two_theta, peak.two_theta],
                    y=[0, peak.intensity],
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False,
                    hovertext=f"({peak.h}{peak.k}{peak.l})<br>2θ={peak.two_theta:.3f}°<br>d={peak.d:.4f} Å<br>I={peak.intensity:.2f}%",
                    hoverinfo='text'
                ))
            
            fig2.update_layout(
                title=f"{crystal.name} - 棒状图",
                xaxis_title="2θ (°)",
                yaxis_title="相对强度 (%)",
                xaxis=dict(range=[st.session_state.two_theta_min, st.session_state.two_theta_max]),
                yaxis=dict(range=[0, 110]),
                height=500
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            peak_data = []
            for peak in peaks:
                if peak.intensity > 0.1:
                    peak_data.append({
                        'h': peak.h,
                        'k': peak.k,
                        'l': peak.l,
                        'hkl': f"({peak.h}{peak.k}{peak.l})",
                        '2θ (°)': round(peak.two_theta, 4),
                        'd (Å)': round(peak.d, 4),
                        '强度 (%)': round(peak.intensity, 2),
                        '多重性': peak.multiplicity
                    })
            
            df = pd.DataFrame(peak_data)
            st.dataframe(df, use_container_width=True, height=400)
            
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 下载峰数据 (CSV)",
                csv,
                "xrd_peaks.csv",
                "text/csv",
                key='download-peaks'
            )


def crystal_structure_section():
    """晶体结构显示部分"""
    st.header("🔮 晶体结构")
    
    crystal = st.session_state.crystals.get(st.session_state.current_phase)
    
    if crystal is None:
        st.warning("请先定义晶体结构")
        return
    
    st.subheader(f"{crystal.name}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 晶胞参数")
        st.write(f"**空间群**: {crystal.space_group} (#{crystal.space_group_number})")
        st.write(f"**晶系**: {crystal.crystal_system}")
        st.write(f"**a =** {crystal.a:.4f} Å")
        st.write(f"**b =** {crystal.b:.4f} Å")
        st.write(f"**c =** {crystal.c:.4f} Å")
        st.write(f"**α =** {crystal.alpha:.2f}°")
        st.write(f"**β =** {crystal.beta:.2f}°")
        st.write(f"**γ =** {crystal.gamma:.2f}°")
        st.write(f"**体积 =** {crystal.volume:.4f} Å³")
        st.write(f"**独立原子数 =** {len(crystal.atoms)}")
        
        st.markdown("### 显示选项")
        show_symmetry = st.checkbox("显示对称等效原子", value=True)
        show_unit_cell = st.checkbox("显示晶胞边框", value=True)
        atom_scale = st.slider("原子大小", 0.2, 2.0, 0.8, 0.1)
        supercell_size = st.number_input("超胞大小", 1, 3, 1)
    
    with col2:
        with st.spinner("生成3D结构图..."):
            fig = plot_crystal_3d(
                crystal,
                use_symmetry=show_symmetry,
                show_unit_cell=show_unit_cell,
                show_bonds=False,
                supercell=(supercell_size, supercell_size, supercell_size),
                atom_scale=atom_scale
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("原子坐标")
    
    atom_data = []
    for i, atom in enumerate(crystal.atoms):
        atom_data.append({
            '序号': i + 1,
            '元素': atom.element,
            'x': round(atom.x, 5),
            'y': round(atom.y, 5),
            'z': round(atom.z, 5),
            '占位因子': round(atom.occupancy, 4),
            'B_iso': round(atom.b_iso, 3)
        })
    
    df = pd.DataFrame(atom_data)
    st.dataframe(df, use_container_width=True)


def indexing_section():
    """指标化部分"""
    st.header("🔍 谱图指标化")
    
    st.subheader("上传实验XRD数据")
    
    uploaded_file = st.file_uploader(
        "上传实验数据 (CSV或xy格式)",
        type=['csv', 'xy', 'txt', 'dat'],
        help="两列格式: 2theta 强度"
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, sep='\\s+', engine='python', header=None, comment='#')
            if data.shape[1] < 2:
                st.error("数据文件需要至少两列: 2theta 和 强度")
                return
            
            two_theta = data.iloc[:, 0].values.astype(float)
            intensity = data.iloc[:, 1].values.astype(float)
            
            valid_mask = ~np.isnan(two_theta) & ~np.isnan(intensity)
            two_theta = two_theta[valid_mask]
            intensity = intensity[valid_mask]
            
            if len(two_theta) == 0:
                st.error("有效数据点为0，请检查文件格式")
                return
            
            sort_idx = np.argsort(two_theta)
            two_theta = two_theta[sort_idx]
            intensity = intensity[sort_idx]
            
            st.session_state.exp_data = {
                'two_theta': two_theta,
                'intensity': intensity
            }
            st.session_state.exp_peaks = None
            st.session_state.indexing_results = None
            
            st.success(f"成功加载数据: {len(two_theta)} 个数据点")
            
        except Exception as e:
            st.error(f"数据加载失败: {e}")
            return
    
    if st.session_state.exp_data is not None:
        two_theta = st.session_state.exp_data['two_theta']
        intensity = st.session_state.exp_data['intensity']
        
        intensity_norm = intensity / np.max(intensity) * 100 if np.max(intensity) > 0 else intensity
        
        st.subheader("实验谱图")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=two_theta,
            y=intensity_norm,
            mode='lines',
            name='实验谱',
            line=dict(color='black', width=1)
        ))
        fig.update_layout(
            title="实验XRD谱图",
            xaxis_title="2θ (°)",
            yaxis_title="相对强度 (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("自动寻峰")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            threshold = st.slider("峰高阈值 (%)", 1, 50, 5, 1)
        with col2:
            min_distance = st.slider("最小峰间距 (°)", 0.1, 2.0, 0.3, 0.1)
        with col3:
            min_peak_height = st.slider("最小突出度 (%)", 0.5, 20, 2, 0.5)
        
        if st.button("🔎 寻峰", type="primary"):
            with st.spinner("正在寻峰..."):
                peaks = find_peaks_derivative(
                    two_theta,
                    intensity_norm,
                    threshold=threshold / 100.0,
                    min_distance=min_distance,
                    min_peak_height=min_peak_height / 100.0,
                    wavelength=st.session_state.wavelength
                )
                st.session_state.exp_peaks = peaks
                st.success(f"找到 {len(peaks)} 个峰")
        
        if st.session_state.exp_peaks is not None:
            peaks = st.session_state.exp_peaks
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=two_theta,
                y=intensity_norm,
                mode='lines',
                name='实验谱',
                line=dict(color='black', width=1)
            ))
            
            for peak in peaks:
                fig2.add_vline(x=peak.two_theta, line_dash="dash", line_color="red", opacity=0.5)
            
            peak_x = [p.two_theta for p in peaks]
            peak_y = [p.intensity * 100 for p in peaks]
            fig2.add_trace(go.Scatter(
                x=peak_x,
                y=peak_y,
                mode='markers',
                name='检出峰',
                marker=dict(color='red', size=8, symbol='triangle-down')
            ))
            
            fig2.update_layout(
                title=f"寻峰结果 (共 {len(peaks)} 个峰)",
                xaxis_title="2θ (°)",
                yaxis_title="相对强度 (%)",
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            with st.expander("📋 峰列表", expanded=True):
                peak_data = []
                for i, peak in enumerate(peaks):
                    peak_data.append({
                        '序号': i + 1,
                        '2θ (°)': round(peak.two_theta, 4),
                        'd (Å)': round(peak.d_spacing, 4),
                        '相对强度 (%)': round(peak.intensity * 100, 2)
                    })
                df = pd.DataFrame(peak_data)
                st.dataframe(df, use_container_width=True, height=300)
            
            if len(peaks) >= 8:
                st.markdown("---")
                st.subheader("晶胞参数反推 (指标化)")
                
                if st.button("🧪 开始指标化搜索", type="primary"):
                    with st.spinner("正在搜索晶胞参数... (可能需要几秒钟)"):
                        results = index_pattern(
                            peaks,
                            wavelength=st.session_state.wavelength,
                            max_results=5
                        )
                        st.session_state.indexing_results = results
                        
                        if results:
                            st.success(f"找到 {len(results)} 个候选晶胞")
                        else:
                            st.warning("未找到合适的晶胞参数，请检查数据质量或调整参数")
                
                if st.session_state.indexing_results is not None:
                    results = st.session_state.indexing_results
                    
                    if results:
                        st.subheader("候选晶胞 (按M20降序)")
                        
                        result_data = []
                        for i, result in enumerate(results):
                            result_data.append({
                                '排名': i + 1,
                                '晶系': result.crystal_system,
                                'a (Å)': round(result.a, 4),
                                'b (Å)': round(result.b, 4),
                                'c (Å)': round(result.c, 4),
                                'β (°)': round(result.beta, 2),
                                '体积 (Å³)': round(result.volume, 4),
                                'M20': round(result.M20, 2),
                                'F30': round(result.F30, 2),
                                '已指标化峰数': result.n_peaks_indexed
                            })
                        
                        df_results = pd.DataFrame(result_data)
                        st.dataframe(df_results, use_container_width=True)
                        
                        st.markdown("---")
                        st.subheader("选择候选晶胞进行后续分析")
                        
                        selected_idx = st.selectbox(
                            "选择候选晶胞",
                            options=range(len(results)),
                            format_func=lambda i: f"#{i+1} {results[i].crystal_system} - M20={results[i].M20:.2f}"
                        )
                        
                        if st.button("✅ 选择此晶胞并应用", type="primary"):
                            selected = results[selected_idx]
                            
                            new_crystal = Crystal(
                                a=selected.a,
                                b=selected.b,
                                c=selected.c,
                                alpha=selected.alpha,
                                beta=selected.beta,
                                gamma=selected.gamma,
                                space_group="P1",
                                space_group_number=1,
                                name=f"指标化_{selected.crystal_system}"
                            )
                            new_crystal.add_atom("X", 0.0, 0.0, 0.0, 1.0, 1.0)
                            
                            phase_name = f"指标化_{len(st.session_state.phases) + 1}"
                            st.session_state.crystals[phase_name] = new_crystal
                            st.session_state.current_phase = phase_name
                            if phase_name not in st.session_state.phases:
                                st.session_state.phases.append(phase_name)
                                st.session_state.phase_ratios[phase_name] = 1.0
                            
                            st.success(f"已创建晶相: {phase_name}")
                            st.info("可在'衍射谱模拟'或'Le Bail拟合'中继续分析")
            else:
                st.warning(f"检出峰数不足 (当前 {len(peaks)} 个)，至少需要8个峰才能进行指标化。请调整寻峰参数。")


def le_bail_section():
    """Le Bail拟合部分"""
    st.header("📊 Le Bail全谱拟合")
    
    crystal = st.session_state.crystals.get(st.session_state.current_phase)
    
    if crystal is None:
        st.warning("请先定义晶体结构")
        return
    
    if st.session_state.exp_data is None:
        st.warning("请先上传实验XRD数据 (在'指标化'页面)")
        return
    
    st.subheader(f"当前晶相: {crystal.name}")
    
    if len(crystal.atoms) == 0:
        st.warning("晶体结构中没有原子，请先添加原子")
        return
    
    st.markdown("### 拟合参数初始值")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        init_U = st.number_input("初始 U", value=st.session_state.U, step=0.001, format="%.4f")
    with col2:
        init_V = st.number_input("初始 V", value=st.session_state.V, step=0.001, format="%.4f")
    with col3:
        init_W = st.number_input("初始 W", value=st.session_state.W, step=0.001, format="%.4f")
    
    col4, col5 = st.columns(2)
    with col4:
        init_eta = st.number_input("初始 η", value=st.session_state.eta, step=0.05, format="%.2f")
    with col5:
        init_zero_shift = st.number_input("初始零点偏移 (°)", value=0.0, step=0.01, format="%.4f")
    
    max_iter = st.slider("最大迭代次数", 10, 100, 50, 5)
    
    if st.button("🧮 开始Le Bail拟合", type="primary"):
        with st.spinner("正在进行Le Bail拟合... (可能需要一些时间)"):
            try:
                two_theta = st.session_state.exp_data['two_theta']
                intensity = st.session_state.exp_data['intensity']
                
                intensity_norm = intensity / np.max(intensity) if np.max(intensity) > 0 else intensity
                
                result = le_bail_fit(
                    two_theta,
                    intensity_norm,
                    crystal,
                    wavelength=st.session_state.wavelength,
                    initial_U=init_U,
                    initial_V=init_V,
                    initial_W=init_W,
                    initial_eta=init_eta,
                    initial_zero_shift=init_zero_shift,
                    max_iterations=max_iter
                )
                
                st.session_state.lebail_result = result
                st.success("拟合完成!")
                
            except Exception as e:
                st.error(f"拟合失败: {e}")
                import traceback
                st.error(traceback.format_exc())
    
    if st.session_state.lebail_result is not None:
        result = st.session_state.lebail_result
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rwp", f"{result.Rwp*100:.2f}%")
        with col2:
            st.metric("χ²", f"{result.chi_squared:.4f}")
        
        with st.expander("📋 拟合参数详情", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**优化后 a**: {result.crystal.a:.5f} Å")
                st.write(f"**优化后 b**: {result.crystal.b:.5f} Å")
                st.write(f"**优化后 c**: {result.crystal.c:.5f} Å")
            with col2:
                st.write(f"**优化后 U**: {result.U:.6f}")
                st.write(f"**优化后 V**: {result.V:.6f}")
                st.write(f"**优化后 W**: {result.W:.6f}")
            with col3:
                st.write(f"**优化后 η**: {result.eta:.4f}")
                st.write(f"**零点偏移**: {result.zero_shift:.5f}°")
                st.write(f"**晶胞体积**: {result.crystal.volume:.4f} Å³")
        
        st.markdown("---")
        st.subheader("拟合图谱对比")
        
        two_theta = st.session_state.exp_data['two_theta']
        intensity = st.session_state.exp_data['intensity']
        intensity_norm = intensity / np.max(intensity) * 100 if np.max(intensity) > 0 else intensity
        calc_pattern = result.calculated_pattern * 100 if np.max(result.calculated_pattern) > 0 else result.calculated_pattern
        diff_pattern = result.difference_pattern * 100
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        fig.add_trace(go.Scatter(
            x=two_theta,
            y=intensity_norm,
            mode='markers',
            name='实验谱',
            marker=dict(color='black', size=3),
            legendgroup='exp'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=two_theta,
            y=calc_pattern,
            mode='lines',
            name='计算谱',
            line=dict(color='red', width=1.5),
            legendgroup='calc'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=two_theta,
            y=diff_pattern,
            mode='lines',
            name='差值',
            line=dict(color='blue', width=1),
            legendgroup='diff'
        ), row=2, col=1)
        
        fig.update_yaxes(title_text="强度 (%)", row=1, col=1)
        fig.update_yaxes(title_text="差值 (%)", row=2, col=1)
        fig.update_xaxes(title_text="2θ (°)", row=2, col=1)
        
        fig.update_layout(
            title=f"Le Bail拟合结果 - Rwp = {result.Rwp*100:.2f}%",
            height=600,
            legend=dict(orientation='h', y=1.02),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)


def multiphase_section():
    """多相混合部分"""
    st.header("🧪 多相混合模拟")
    
    st.subheader("晶相列表及比例")
    
    if len(st.session_state.phases) == 0:
        st.warning("请先定义至少一个晶体相")
        return
    
    phase_ratios = {}
    
    for i, phase_name in enumerate(st.session_state.phases):
        col1, col2 = st.columns([2, 3])
        with col1:
            st.write(f"**{phase_name}**: {st.session_state.crystals[phase_name].name}")
        with col2:
            ratio = st.slider(
                f"比例 - {phase_name}",
                0.0, 1.0,
                st.session_state.phase_ratios.get(phase_name, 0.5),
                0.01,
                key=f"ratio_{phase_name}"
            )
            phase_ratios[phase_name] = ratio
    
    st.session_state.phase_ratios = phase_ratios
    
    total_ratio = sum(phase_ratios.values())
    st.write(f"总比例: {total_ratio:.2f}")
    
    if total_ratio == 0:
        st.warning("总比例为0，请调整各相比例")
        return
    
    if st.button("🔬 计算混合衍射谱", type="primary"):
        with st.spinner("正在计算混合衍射谱..."):
            two_theta_range = np.linspace(
                st.session_state.two_theta_min,
                st.session_state.two_theta_max,
                1000
            )
            
            mixed_pattern = np.zeros_like(two_theta_range)
            phase_patterns = {}
            
            for phase_name, ratio in phase_ratios.items():
                if ratio <= 0:
                    continue
                
                crystal = st.session_state.crystals[phase_name]
                peaks = diffraction_simulation(
                    crystal,
                    wavelength=st.session_state.wavelength,
                    two_theta_min=st.session_state.two_theta_min,
                    two_theta_max=st.session_state.two_theta_max,
                    use_symmetry=True
                )
                
                if st.session_state.use_caglioti:
                    pattern = powder_pattern_caglioti(
                        peaks, two_theta_range,
                        U=st.session_state.U,
                        V=st.session_state.V,
                        W=st.session_state.W,
                        eta=st.session_state.eta,
                        wavelength=st.session_state.wavelength
                    )
                else:
                    pattern = powder_pattern(
                        peaks, two_theta_range,
                        fwhm=st.session_state.fwhm,
                        peak_type="pseudo_voigt",
                        eta=st.session_state.eta
                    )
                
                phase_patterns[phase_name] = pattern * ratio
                mixed_pattern += pattern * ratio
            
            if np.max(mixed_pattern) > 0:
                mixed_pattern = mixed_pattern / np.max(mixed_pattern) * 100
            
            st.session_state.mixed_pattern = mixed_pattern
            st.session_state.phase_patterns = phase_patterns
            st.session_state.mixed_twotheta = two_theta_range
            st.success("混合衍射谱计算完成!")
    
    if 'mixed_pattern' in st.session_state and st.session_state.mixed_pattern is not None:
        two_theta_range = st.session_state.mixed_twotheta
        mixed_pattern = st.session_state.mixed_pattern
        phase_patterns = st.session_state.phase_patterns
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (phase_name, pattern) in enumerate(phase_patterns.items()):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=two_theta_range,
                y=pattern / np.max(mixed_pattern) * 100 if np.max(mixed_pattern) > 0 else pattern,
                mode='lines',
                name=f'{phase_name} (比例={phase_ratios[phase_name]:.2f})',
                line=dict(color=color, width=1.5, dash='dash'),
                opacity=0.6
            ))
        
        fig.add_trace(go.Scatter(
            x=two_theta_range,
            y=mixed_pattern,
            mode='lines',
            name='混合谱',
            line=dict(color='black', width=2.5)
        ))
        
        fig.update_layout(
            title="多相混合衍射谱",
            xaxis_title="2θ (°)",
            yaxis_title="相对强度 (%)",
            xaxis=dict(range=[st.session_state.two_theta_min, st.session_state.two_theta_max]),
            yaxis=dict(range=[0, 110]),
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)


def _quant_manual_crystal_input():
    """定量分析页面内手动定义晶相"""
    st.markdown("#### ✏️ 手动定义新晶相")
    
    if 'qa_atom_list' not in st.session_state:
        st.session_state.qa_atom_list = [
            {"element": "Si", "x": 0.0, "y": 0.0, "z": 0.0, "occupancy": 1.0, "b_iso": 0.5},
        ]
    
    if 'qa_phase_count' not in st.session_state:
        st.session_state.qa_phase_count = 1
    
    col_add_btn, _ = st.columns([1, 3])
    with col_add_btn:
        if st.button("➕ 添加原子", key="qa_add_atom_btn", use_container_width=True):
            st.session_state.qa_atom_list.append({
                "element": "C", "x": 0.0, "y": 0.0, "z": 0.0,
                "occupancy": 1.0, "b_iso": 1.0
            })
            st.rerun()
    
    with st.form("quant_manual_crystal_form", clear_on_submit=False):
        sg_input = st.text_input(
            "空间群 (H-M符号或编号)",
            value="Fd-3m",
            help="例如: P1, Fm-3m, 225 等",
            key="qa_sg_input"
        )
        
        try:
            sg_num = int(sg_input)
            sg_symbol = get_sg_symbol(sg_num)
        except ValueError:
            sg_symbol = sg_input
            sg_num = get_sg_number(sg_input)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            qa = st.number_input("a (Å)", value=5.4309, step=0.001, format="%.4f", key="qa_a_in")
        with col_b:
            qb = st.number_input("b (Å)", value=5.4309, step=0.001, format="%.4f", key="qa_b_in")
        with col_c:
            qc = st.number_input("c (Å)", value=5.4309, step=0.001, format="%.4f", key="qa_c_in")
        
        col_alpha, col_beta, col_gamma = st.columns(3)
        with col_alpha:
            qalpha = st.number_input("α (°)", value=90.0, step=0.1, format="%.2f", key="qa_alpha_in")
        with col_beta:
            qbeta = st.number_input("β (°)", value=90.0, step=0.1, format="%.2f", key="qa_beta_in")
        with col_gamma:
            qgamma = st.number_input("γ (°)", value=90.0, step=0.1, format="%.2f", key="qa_gamma_in")
        
        crystal_name_q = st.text_input("晶体名称", value="自定义晶体", key="qa_crystal_name_in")
        
        st.markdown("**原子列表**")
        for i, atom in enumerate(st.session_state.qa_atom_list):
            with st.expander(f"原子 {i+1}: {atom['element']}", expanded=(i < 2)):
                col_e, col_x, col_y = st.columns(3)
                with col_e:
                    atom['element'] = st.text_input("元素", value=atom['element'], key=f"qa_elem_{i}_in")
                with col_x:
                    atom['x'] = st.number_input("x", value=float(atom['x']), step=0.01, format="%.4f", key=f"qa_x_{i}_in")
                with col_y:
                    atom['y'] = st.number_input("y", value=float(atom['y']), step=0.01, format="%.4f", key=f"qa_y_{i}_in")
                col_z, col_occ, col_b = st.columns(3)
                with col_z:
                    atom['z'] = st.number_input("z", value=float(atom['z']), step=0.01, format="%.4f", key=f"qa_z_{i}_in")
                with col_occ:
                    atom['occupancy'] = st.number_input("占位", value=float(atom['occupancy']), min_value=0.0, max_value=1.0, step=0.1, key=f"qa_occ_{i}_in")
                with col_b:
                    atom['b_iso'] = st.number_input("B_iso", value=float(atom['b_iso']), step=0.1, key=f"qa_b_{i}_in")
                
                if st.form_submit_button(f"🗑️ 删除原子 {i+1}", key=f"qa_del_atom_{i}_btn", use_container_width=True):
                    if len(st.session_state.qa_atom_list) > 1:
                        st.session_state.qa_atom_list.pop(i)
                        st.rerun()
                    else:
                        st.warning("至少需要保留一个原子")
        
        submitted = st.form_submit_button("✅ 添加到晶相列表", type="primary", use_container_width=True)
        if submitted:
            if len(st.session_state.qa_atom_list) == 0:
                st.warning("请至少添加一个原子")
            else:
                try:
                    crystal_q = Crystal(
                        a=qa, b=qb, c=qc,
                        alpha=qalpha, beta=qbeta, gamma=qgamma,
                        space_group=sg_symbol,
                        space_group_number=sg_num,
                        name=crystal_name_q
                    )
                    for atom_data in st.session_state.qa_atom_list:
                        crystal_q.add_atom(
                            atom_data['element'],
                            atom_data['x'], atom_data['y'], atom_data['z'],
                            atom_data['occupancy'],
                            atom_data['b_iso']
                        )
                    
                    while True:
                        phase_name_q = f"定量相_{st.session_state.qa_phase_count}"
                        st.session_state.qa_phase_count += 1
                        if phase_name_q not in st.session_state.crystals:
                            break
                    
                    st.session_state.crystals[phase_name_q] = crystal_q
                    if phase_name_q not in st.session_state.phases:
                        st.session_state.phases.append(phase_name_q)
                        st.session_state.phase_ratios[phase_name_q] = 1.0
                    
                    st.success(f"✅ 已保存: {phase_name_q} ({crystal_name_q})")
                    st.info(f"下方勾选列表已更新，可选中此晶相进行定量分析")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"保存失败: {e}")
                    import traceback
                    st.error(traceback.format_exc())


def _format_weight_std(weight_std_value):
    """格式化权重标准误差，处理NaN和极小值"""
    is_invalid = (weight_std_value is None 
                  or (isinstance(weight_std_value, float) and not np.isfinite(weight_std_value)))
    if is_invalid:
        return "⚠ 无法计算"
    if isinstance(weight_std_value, float) and weight_std_value < 1e-6:
        return f"< 1e-6 (极小)"
    return f"{weight_std_value:.6f}"


def _is_std_invalid(weight_std_value):
    """检查标准误差是否无效（无法计算或极小）"""
    is_invalid = (weight_std_value is None 
                  or (isinstance(weight_std_value, float) and not np.isfinite(weight_std_value)))
    is_tiny = isinstance(weight_std_value, float) and weight_std_value < 1e-6
    return is_invalid or is_tiny


def quantitative_analysis_section():
    """定量分析部分"""
    st.header("🔬 多相定量分析")
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown("### 📂 实验数据")
        uploaded_file = st.file_uploader(
            "上传实验XRD数据",
            type=['csv', 'xy', 'txt', 'dat'],
            help="两列格式: 2theta 强度",
            key="quant_upload"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file, sep='\\s+', engine='python', header=None, comment='#')
                if data.shape[1] < 2:
                    st.error("数据文件需要至少两列: 2theta 和 强度")
                else:
                    two_theta = data.iloc[:, 0].values.astype(float)
                    intensity = data.iloc[:, 1].values.astype(float)
                    
                    valid_mask = ~np.isnan(two_theta) & ~np.isnan(intensity)
                    two_theta = two_theta[valid_mask]
                    intensity = intensity[valid_mask]
                    
                    if len(two_theta) > 0:
                        sort_idx = np.argsort(two_theta)
                        two_theta = two_theta[sort_idx]
                        intensity = intensity[sort_idx]
                        
                        st.session_state.quant_exp_data = {
                            'two_theta': two_theta,
                            'intensity': intensity
                        }
                        st.session_state.quant_result = None
                        st.success(f"成功加载数据: {len(two_theta)} 个数据点")
                    else:
                        st.error("有效数据点为0")
            except Exception as e:
                st.error(f"数据加载失败: {e}")
        
        if st.session_state.quant_exp_data is not None:
            exp_tt = st.session_state.quant_exp_data['two_theta']
            exp_int = st.session_state.quant_exp_data['intensity']
            exp_int_norm = exp_int / np.max(exp_int) * 100 if np.max(exp_int) > 0 else exp_int
            
            with st.expander("📈 查看实验谱预览", expanded=False):
                fig_preview = go.Figure()
                fig_preview.add_trace(go.Scatter(
                    x=exp_tt, y=exp_int_norm,
                    mode='markers', name='实验谱',
                    marker=dict(color='black', size=3)
                ))
                fig_preview.update_layout(
                    title="实验XRD谱",
                    xaxis_title="2θ (°)",
                    yaxis_title="相对强度 (%)",
                    height=250
                )
                st.plotly_chart(fig_preview, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 💎 晶相选择")
        st.caption("从已有晶相库中选取，或手动定义新晶相 (需2-4个)")
        
        with st.expander("➕ 手动定义新晶相", expanded=False):
            _quant_manual_crystal_input()
        
        st.markdown("#### 已有晶相选择")
        available_phases = list(st.session_state.crystals.keys())
        if len(available_phases) == 0:
            st.warning("暂无晶体相，请在上方手动定义或切换到其他页面创建")
        else:
            for phase_name in available_phases:
                crystal = st.session_state.crystals[phase_name]
                is_selected = phase_name in st.session_state.quant_selected_phases
                checked = st.checkbox(
                    f"{phase_name}: {crystal.name} ({crystal.space_group})",
                    value=is_selected,
                    key=f"quant_cb_{phase_name}"
                )
                if checked and phase_name not in st.session_state.quant_selected_phases:
                    if len(st.session_state.quant_selected_phases) < 4:
                        st.session_state.quant_selected_phases.append(phase_name)
                    else:
                        st.warning("最多只能选择4个晶相")
                elif not checked and phase_name in st.session_state.quant_selected_phases:
                    st.session_state.quant_selected_phases.remove(phase_name)
        
        n_selected = len(st.session_state.quant_selected_phases)
        if n_selected > 0:
            st.info(f"✓ 已选择 {n_selected} 个晶相: {', '.join(st.session_state.quant_selected_phases)} (需2-4个)")
        else:
            st.info("目前未选择晶相 (需2-4个)")
        
        st.markdown("---")
        st.markdown("### ⚙️ 拟合参数")
        
        bg_order = st.slider(
            "多项式背景阶数",
            min_value=2, max_value=5,
            value=st.session_state.quant_bg_order,
            step=1,
            key="quant_bg_order_slider"
        )
        st.session_state.quant_bg_order = bg_order
        
        max_iter = st.slider(
            "最大迭代次数",
            min_value=50, max_value=1000,
            value=st.session_state.quant_max_iter,
            step=50,
            key="quant_max_iter_slider"
        )
        st.session_state.quant_max_iter = max_iter
        
        tol_options = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        tol_labels = ["1e-4", "1e-5", "1e-6", "1e-7", "1e-8"]
        tol_idx = tol_options.index(st.session_state.quant_tolerance) if st.session_state.quant_tolerance in tol_options else 2
        tolerance = float(st.selectbox(
            "收敛阈值",
            options=tol_options,
            format_func=lambda x: tol_labels[tol_options.index(x)],
            index=tol_idx,
            key="quant_tolerance_sb"
        ))
        st.session_state.quant_tolerance = tolerance
        
        can_run = (st.session_state.quant_exp_data is not None 
                   and 2 <= n_selected <= 4)
        
        if st.button("🧮 开始定量分析", type="primary", disabled=not can_run):
            if not can_run:
                if st.session_state.quant_exp_data is None:
                    st.warning("请先上传实验XRD数据")
                else:
                    st.warning("请选择2-4个晶相")
            else:
                selected_crystals = {}
                for phase_name in st.session_state.quant_selected_phases:
                    selected_crystals[phase_name] = st.session_state.crystals[phase_name]
                
                progress_bar = st.progress(0, text="初始化中...")
                rwp_display = st.empty()
                last_progress = [0]
                
                def on_progress(current_iter, max_iter, current_rwp):
                    pct = max(min(int(current_iter / max(max_iter, 1) * 100), 100), 0)
                    pct = max(pct, last_progress[0])
                    last_progress[0] = pct
                    progress_bar.progress(pct, text=f"迭代中... ({current_iter}/{max_iter})")
                    rwp_display.caption(f"当前 Rwp = {current_rwp*100:.3f}%")
                
                try:
                    exp_data = st.session_state.quant_exp_data
                    two_theta = exp_data['two_theta']
                    intensity = exp_data['intensity']
                    
                    progress_bar.progress(5, text="正在计算各相理论谱...")
                    rwp_display.caption("预处理中...")
                    
                    result = quantitative_analysis(
                        two_theta=two_theta,
                        intensity=intensity,
                        selected_crystals=selected_crystals,
                        wavelength=st.session_state.wavelength,
                        use_caglioti=st.session_state.use_caglioti,
                        U=st.session_state.U,
                        V=st.session_state.V,
                        W=st.session_state.W,
                        fwhm=st.session_state.fwhm,
                        eta=st.session_state.eta,
                        background_order=bg_order,
                        max_iterations=max_iter,
                        tolerance=tolerance,
                        progress_callback=on_progress
                    )
                    
                    progress_bar.progress(100, text="✅ 计算完成!")
                    rwp_display.caption(f"最终 Rwp = {result.Rwp*100:.3f}% | 迭代 {result.iterations} 次")
                    st.session_state.quant_result = result
                    
                    from time import sleep
                    sleep(0.5)
                    progress_bar.empty()
                    rwp_display.empty()
                    
                except Exception as e:
                    st.error(f"定量分析失败: {e}")
                    import traceback
                    st.error(traceback.format_exc())
                    progress_bar.empty()
                    rwp_display.empty()
    
    with col_right:
        if st.session_state.quant_result is None:
            st.info("请在左侧上传实验数据、选择晶相并点击\"开始定量分析\"")
            st.markdown("""
            ### 使用说明
            1. **上传实验数据**: 导入两列格式(2θ, 强度)的XRD谱数据文件
            2. **选择晶相**: 从已有晶体库中选择2-4个已知晶相
            3. **设置参数**: 调整背景阶数、迭代次数和收敛阈值
            4. **开始计算**: 点击按钮执行加权最小二乘拟合
            
            ### 算法说明
            - 对每个候选相计算理论衍射谱(含峰形展宽)
            - 各相理论谱按权重线性叠加 + 多项式背景
            - 优化目标: 加权残差平方和(权重取1/y_obs)
            - 约束: 各相权重非负; 含量<1%自动剔除
            - 最终结果归一化到质量百分比
            """)
        else:
            result = st.session_state.quant_result
            
            st.subheader("📊 定量结果")
            
            has_std_warning = False
            result_data = []
            for pr in result.phase_results:
                std_str = _format_weight_std(pr.weight_std)
                has_std_warning = has_std_warning or _is_std_invalid(pr.weight_std)
                weight_display = f"{pr.weight:.4f} ± {std_str}"
                result_data.append({
                    '相名称': pr.phase_name,
                    '晶体名称': pr.crystal_name,
                    '空间群': f"{pr.space_group} (#{pr.space_group_number})",
                    '权重因子': weight_display,
                    '质量百分比 (%)': f"{pr.mass_percent:.2f}",
                    'Rwp贡献度': f"{pr.rwp_contribution*100:.3f}%"
                })
            
            df_result = pd.DataFrame(result_data)
            st.dataframe(df_result, use_container_width=True, hide_index=True)
            
            if has_std_warning:
                st.caption("⚠️ 注意：部分权重标准误差标记为\"无法计算\"或\"极小\"的原因可能是：该相对拟合影响极小或参数空间高度相关；数值近似计算困难。可尝试增加迭代次数或调整收敛阈值以改善。")
            
            csv_buffer = io.StringIO()
            df_result.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                "📥 导出定量结果 (CSV)",
                csv_data,
                "quantitative_result.csv",
                "text/csv",
                key='download-quant-csv'
            )
            
            if result.removed_phases:
                st.warning(f"以下晶相因含量过低(<1%)已自动剔除: {', '.join(result.removed_phases)}")
            
            rwp_hist = getattr(result, 'rwp_history', [])
            if len(rwp_hist) > 1:
                with st.expander("📉 Rwp收敛曲线", expanded=False):
                    fig_rwp = go.Figure()
                    iterations_x = list(range(1, len(rwp_hist) + 1))
                    fig_rwp.add_trace(go.Scatter(
                        x=iterations_x,
                        y=[r*100 for r in rwp_hist],
                        mode='lines+markers',
                        name='Rwp',
                        line=dict(color='#d62728', width=2),
                        marker=dict(size=4)
                    ))
                    fig_rwp.update_layout(
                        title="Rwp随迭代变化曲线",
                        xaxis_title="迭代次数",
                        yaxis_title="Rwp (%)",
                        height=300,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_rwp, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("📈 拟合谱图")
            
            phase_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
            
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.75, 0.25],
                shared_xaxes=True,
                vertical_spacing=0.05
            )
            
            fig.add_trace(go.Scatter(
                x=result.two_theta,
                y=result.observed_intensity,
                mode='markers',
                name='实验谱',
                marker=dict(color='black', size=3.5, opacity=0.7),
                legendgroup='exp'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=result.two_theta,
                y=result.calculated_pattern,
                mode='lines',
                name='总计算谱',
                line=dict(color='red', width=2),
                legendgroup='calc_total'
            ), row=1, col=1)
            
            for i, pr in enumerate(result.phase_results):
                color = phase_colors[i % len(phase_colors)]
                phase_pat = result.phase_patterns.get(pr.phase_name, np.zeros_like(result.two_theta))
                fig.add_trace(go.Scatter(
                    x=result.two_theta,
                    y=phase_pat,
                    mode='lines',
                    name=f"{pr.phase_name}分谱",
                    line=dict(color=color, width=1.5, dash='dash'),
                    opacity=0.7,
                    legendgroup=f'phase_{i}'
                ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=result.two_theta,
                y=result.background_pattern,
                mode='lines',
                name='背景',
                line=dict(color='gray', width=1.5),
                opacity=0.6,
                legendgroup='bg'
            ), row=1, col=1)
            
            for i, pr in enumerate(result.phase_results):
                crystal = st.session_state.crystals.get(pr.phase_name)
                if crystal is not None:
                    try:
                        peaks = diffraction_simulation(
                            crystal,
                            wavelength=st.session_state.wavelength,
                            two_theta_min=float(np.min(result.two_theta)),
                            two_theta_max=float(np.max(result.two_theta)),
                            use_symmetry=True
                        )
                        if peaks:
                            max_peak = max(peaks, key=lambda p: p.intensity)
                            y_max = np.max(result.calculated_pattern)
                            color = phase_colors[i % len(phase_colors)]
                            for peak in peaks[:5]:
                                if peak.intensity > max_peak.intensity * 0.1:
                                    fig.add_annotation(
                                        x=peak.two_theta,
                                        y=y_max * 0.95,
                                        text=f"({peak.h}{peak.k}{peak.l})",
                                        showarrow=False,
                                        font=dict(size=8, color=color),
                                        textangle=-90,
                                        xanchor='center',
                                        yanchor='top',
                                        row=1, col=1
                                    )
                    except Exception:
                        pass
            
            fig.add_trace(go.Scatter(
                x=result.two_theta,
                y=result.difference_pattern,
                mode='lines',
                name='差值 (实验-计算)',
                line=dict(color='#2ca02c', width=1.2),
                legendgroup='diff'
            ), row=2, col=1)
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
            
            fig.update_yaxes(title_text="相对强度 (%)", row=1, col=1)
            fig.update_yaxes(title_text="差值", row=2, col=1)
            fig.update_xaxes(title_text="2θ (°)", row=2, col=1)
            
            fig.update_layout(
                title=f"定量分析拟合结果 - Rwp = {result.Rwp*100:.2f}%",
                height=650,
                legend=dict(orientation='h', y=1.02),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("📋 收敛信息")
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric(
                    "迭代次数",
                    f"{result.iterations}",
                    help="实际执行的迭代次数"
                )
            
            with col_stat2:
                st.metric(
                    "收敛状态",
                    "✓ 已收敛" if result.converged else "⚠ 未收敛",
                    help="优化算法是否达到收敛条件"
                )
            
            with col_stat3:
                st.metric(
                    "最终 Rwp",
                    f"{result.Rwp*100:.3f}%",
                    help="加权残差因子，越小拟合越好"
                )
            
            with col_stat4:
                st.metric(
                    "χ² (卡方)",
                    f"{result.chi_squared:.4f}",
                    help="自由度校正后的卡方值"
                )
            
            with st.expander("📊 参数标准误差详情", expanded=False):
                n_phases = len(result.phase_results)
                bg_coeffs = result.background_coeffs
                bg_std = result.param_std_errors[n_phases:] if len(result.param_std_errors) == len(result.background_coeffs) + n_phases else []
                
                st.markdown("**各相权重标准误差:**")
                has_bg_std_warning = False
                for i, pr in enumerate(result.phase_results):
                    fmt_std = _format_weight_std(pr.weight_std)
                    st.write(f"- {pr.phase_name}: σ = {fmt_std}")
                
                st.markdown("**背景多项式系数 (c₀ + c₁x + c₂x² + ...):**")
                bg_data = []
                for i in range(len(bg_coeffs)):
                    coeff = bg_coeffs[i]
                    std_val = bg_std[i] if i < len(bg_std) else float('nan')
                    fmt_std = _format_weight_std(std_val)
                    has_bg_std_warning = has_bg_std_warning or _is_std_invalid(std_val)
                    bg_data.append({
                        '阶数': i,
                        '系数': f"{coeff:.6f}",
                        '标准误差': fmt_std
                    })
                df_bg = pd.DataFrame(bg_data)
                st.dataframe(df_bg, use_container_width=True, hide_index=True)
                
                if has_bg_std_warning or any(_is_std_invalid(pr.weight_std) for pr in result.phase_results):
                    st.caption("💡 提示：标准误差无法计算或极小时，通常表示：(1) 该参数对拟合质量影响较小；(2) 多个参数存在强共线性（冗余）；(3) 算法未收敛至参数空间的稳定区域。建议检查实验数据质量或调整晶相选择。")


def remove_linear_baseline(two_theta: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """
    线性背景扣除: 使用谱图两端点的线性插值作为基线
    """
    if len(two_theta) < 2:
        return intensity
    
    x1, y1 = two_theta[0], intensity[0]
    x2, y2 = two_theta[-1], intensity[-1]
    
    if x2 == x1:
        return intensity - y1
    
    slope = (y2 - y1) / (x2 - x1)
    baseline = y1 + slope * (two_theta - x1)
    corrected = intensity - baseline
    corrected = np.maximum(corrected, 0)
    
    return corrected


def normalize_intensity(intensity: np.ndarray) -> np.ndarray:
    """
    归一化: 最高峰设为100
    """
    max_val = np.max(intensity)
    if max_val > 0:
        return intensity / max_val * 100.0
    return intensity


def load_compare_experimental_data(uploaded_file):
    """
    加载并预处理实验谱数据 (用于谱图对比)
    返回: two_theta, intensity (已做背景扣除和归一化)
    """
    try:
        data = pd.read_csv(uploaded_file, sep='\\s+', engine='python', header=None, comment='#')
        if data.shape[1] < 2:
            return None, None, "数据文件需要至少两列: 2theta 和 强度"
        
        two_theta = data.iloc[:, 0].values.astype(float)
        intensity = data.iloc[:, 1].values.astype(float)
        
        valid_mask = ~np.isnan(two_theta) & ~np.isnan(intensity)
        two_theta = two_theta[valid_mask]
        intensity = intensity[valid_mask]
        
        if len(two_theta) == 0:
            return None, None, "有效数据点为0，请检查文件格式"
        
        sort_idx = np.argsort(two_theta)
        two_theta = two_theta[sort_idx]
        intensity = intensity[sort_idx]
        
        intensity_bkg = remove_linear_baseline(two_theta, intensity)
        intensity_norm = normalize_intensity(intensity_bkg)
        
        return two_theta, intensity_norm, None
    except Exception as e:
        return None, None, f"数据加载失败: {str(e)}"


def compute_rwp(exp_intensity: np.ndarray, calc_intensity: np.ndarray) -> float:
    """
    计算Rwp (Weighted Profile R-factor)
    Rwp = sqrt( sum( w_i * (y_obs_i - y_calc_i)^2 ) / sum( w_i * y_obs_i^2 ) )
    其中 w_i = 1 / y_obs_i (y_obs_i > 0时)
    """
    valid = exp_intensity > 0
    if not np.any(valid):
        return 0.0
    
    y_obs = exp_intensity[valid]
    y_calc = calc_intensity[valid]
    
    weights = 1.0 / np.maximum(y_obs, 1e-6)
    
    numerator = np.sum(weights * (y_obs - y_calc) ** 2)
    denominator = np.sum(weights * y_obs ** 2)
    
    if denominator <= 0:
        return 0.0
    
    return np.sqrt(numerator / denominator)


def match_peaks(exp_peaks, theo_peaks, tolerance: float = 0.3):
    """
    匹配实验峰和理论峰
    参数:
        exp_peaks: 实验峰列表 (Peak对象)
        theo_peaks: 理论峰列表 (DiffractionPeak对象)
        tolerance: 容差 (度)
    返回:
        matched_exp: 匹配的实验峰列表
        unmatched_exp: 未匹配的实验峰列表
        deviations: 匹配峰的2theta偏差列表
    """
    matched_exp = []
    unmatched_exp = []
    deviations = []
    
    used_theo = set()
    
    for exp_peak in exp_peaks:
        best_diff = float('inf')
        best_theo_idx = -1
        
        for i, theo_peak in enumerate(theo_peaks):
            if i in used_theo:
                continue
            diff = abs(exp_peak.two_theta - theo_peak.two_theta)
            if diff < best_diff:
                best_diff = diff
                best_theo_idx = i
        
        if best_diff <= tolerance and best_theo_idx >= 0:
            matched_exp.append(exp_peak)
            deviations.append(best_diff)
            used_theo.add(best_theo_idx)
        else:
            unmatched_exp.append(exp_peak)
    
    return matched_exp, unmatched_exp, deviations


def compare_section():
    """谱图对比部分"""
    st.header("📊 谱图对比")
    
    crystal = st.session_state.crystals.get(st.session_state.current_phase)
    
    if crystal is None:
        st.warning("请先定义晶体结构")
        return
    
    st.subheader(f"当前晶相: {crystal.name}")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### 实验数据")
        uploaded_file = st.file_uploader(
            "上传实验数据 (CSV/xy)",
            type=['csv', 'xy', 'txt', 'dat'],
            help="两列格式: 2theta 强度",
            key="compare_upload"
        )
        
        if uploaded_file is not None:
            two_theta, intensity_norm, error = load_compare_experimental_data(uploaded_file)
            if error is not None:
                st.error(error)
            else:
                st.session_state.compare_exp_data = {
                    'two_theta': two_theta,
                    'intensity': intensity_norm
                }
                st.session_state.compare_exp_peaks = None
                st.success(f"成功加载: {len(two_theta)} 个数据点")
        
        st.markdown("---")
        st.markdown("### 寻峰参数")
        exp_threshold = st.slider("峰高阈值 (%)", 1, 50, 3, 1, key="compare_threshold")
        exp_min_distance = st.slider("最小峰间距 (°)", 0.1, 2.0, 0.3, 0.1, key="compare_mindist")
        exp_min_peak_height = st.slider("最小突出度 (%)", 0.5, 20.0, 1.5, 0.5, key="compare_minheight")
        
        if st.button("🔎 寻峰", type="primary", key="compare_findpeaks"):
            if st.session_state.compare_exp_data is None:
                st.warning("请先上传实验数据")
            else:
                with st.spinner("正在寻峰..."):
                    exp_tt = st.session_state.compare_exp_data['two_theta']
                    exp_int = st.session_state.compare_exp_data['intensity']
                    peaks = find_peaks_derivative(
                        exp_tt,
                        exp_int,
                        threshold=exp_threshold / 100.0,
                        min_distance=exp_min_distance,
                        min_peak_height=exp_min_peak_height / 100.0,
                        wavelength=st.session_state.wavelength
                    )
                    st.session_state.compare_exp_peaks = peaks
                    st.success(f"找到 {len(peaks)} 个实验峰")
        
        st.markdown("---")
        st.markdown("### 零点校正")
        zero_shift = st.slider(
            "零点偏移 (°)",
            min_value=-1.0,
            max_value=1.0,
            value=st.session_state.compare_zero_shift,
            step=0.01,
            format="%.3f",
            key="compare_zeroshift"
        )
        st.session_state.compare_zero_shift = zero_shift
        st.caption(f"实验谱整体平移: {zero_shift:+.3f}°")
    
    with col2:
        if st.session_state.compare_exp_data is None:
            st.info("请在左侧上传实验XRD数据")
            return
        
        exp_tt_orig = st.session_state.compare_exp_data['two_theta']
        exp_int = st.session_state.compare_exp_data['intensity']
        
        exp_tt = exp_tt_orig + zero_shift
        
        if st.session_state.peaks_data is None:
            with st.spinner("正在计算理论衍射谱..."):
                peaks = diffraction_simulation(
                    crystal,
                    wavelength=st.session_state.wavelength,
                    two_theta_min=st.session_state.two_theta_min,
                    two_theta_max=st.session_state.two_theta_max,
                    use_symmetry=True
                )
                st.session_state.peaks_data = peaks
        
        theo_peaks = st.session_state.peaks_data
        
        tt_min = max(st.session_state.two_theta_min, np.min(exp_tt))
        tt_max = min(st.session_state.two_theta_max, np.max(exp_tt))
        
        two_theta_range = np.linspace(tt_min, tt_max, 1000)
        
        if st.session_state.use_caglioti:
            theo_pattern = powder_pattern_caglioti(
                theo_peaks, two_theta_range,
                U=st.session_state.U,
                V=st.session_state.V,
                W=st.session_state.W,
                eta=st.session_state.eta,
                wavelength=st.session_state.wavelength
            )
        else:
            theo_pattern = powder_pattern(
                theo_peaks, two_theta_range,
                fwhm=st.session_state.fwhm,
                peak_type="pseudo_voigt",
                eta=st.session_state.eta
            )
        
        theo_pattern_norm = normalize_intensity(theo_pattern)
        
        exp_interp = np.interp(two_theta_range, exp_tt, exp_int, left=0, right=0)
        difference = exp_interp - theo_pattern_norm
        
        exp_peaks = st.session_state.compare_exp_peaks
        
        matched_exp = []
        unmatched_exp = []
        deviations = []
        if exp_peaks is not None and len(theo_peaks) > 0:
            shifted_exp_peaks = []
            for p in exp_peaks:
                shifted_exp_peaks.append(Peak(
                    two_theta=p.two_theta + zero_shift,
                    intensity=p.intensity,
                    d_spacing=p.d_spacing
                ))
            matched_exp, unmatched_exp, deviations = match_peaks(shifted_exp_peaks, theo_peaks, tolerance=0.3)
        
        rwp = compute_rwp(exp_interp, theo_pattern_norm)
        mean_dev = np.mean(deviations) if deviations else 0.0
        std_dev = np.std(deviations) if deviations else 0.0
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        fig.add_trace(go.Scatter(
            x=exp_tt,
            y=exp_int,
            mode='markers',
            name='实验谱',
            marker=dict(color='#1f77b4', size=3.5, opacity=0.7),
            legendgroup='exp'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=two_theta_range,
            y=theo_pattern_norm,
            mode='lines',
            name='理论谱',
            line=dict(color='#d62728', width=2),
            legendgroup='theo'
        ), row=1, col=1)
        
        if exp_peaks is not None:
            shifted_peak_x = [p.two_theta + zero_shift for p in exp_peaks]
            shifted_peak_y = [p.intensity * 100 if p.intensity <= 1 else p.intensity for p in exp_peaks]
            fig.add_trace(go.Scatter(
                x=shifted_peak_x,
                y=shifted_peak_y,
                mode='markers',
                name='实验峰位',
                marker=dict(
                    color='#1f77b4',
                    size=10,
                    symbol='triangle-down',
                    line=dict(width=1, color='black')
                ),
                legendgroup='exp_peaks'
            ), row=1, col=1)
        
        if len(theo_peaks) > 0:
            theo_peak_x = [p.two_theta for p in theo_peaks if p.intensity > 0.5]
            theo_peak_y = [p.intensity for p in theo_peaks if p.intensity > 0.5]
            fig.add_trace(go.Scatter(
                x=theo_peak_x,
                y=theo_peak_y,
                mode='markers',
                name='理论峰位',
                marker=dict(
                    color='#d62728',
                    size=10,
                    symbol='triangle-up',
                    line=dict(width=1, color='black')
                ),
                legendgroup='theo_peaks'
            ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=two_theta_range,
            y=difference,
            mode='lines',
            name='差值 (实验-理论)',
            line=dict(color='#2ca02c', width=1.2),
            legendgroup='diff'
        ), row=2, col=1)
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
        
        fig.update_yaxes(title_text="相对强度 (%)", row=1, col=1)
        fig.update_yaxes(title_text="差值 (%)", row=2, col=1)
        fig.update_xaxes(title_text="2θ (°)", row=2, col=1)
        fig.update_xaxes(range=[tt_min, tt_max])
        
        fig.update_layout(
            title="实验谱与理论谱对比",
            height=650,
            legend=dict(orientation='h', y=1.02),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
        
        with stat_col1:
            st.metric(
                "匹配峰数",
                f"{len(matched_exp)}",
                help="实验峰与最近理论峰2θ差值≤0.3°"
            )
        
        with stat_col2:
            st.metric(
                "未匹配实验峰",
                f"{len(unmatched_exp)}",
                help="未找到对应理论峰的实验峰数"
            )
        
        with stat_col3:
            st.metric(
                "Rwp",
                f"{rwp*100:.2f}%",
                help="加权残差因子，越小拟合越好"
            )
        
        with stat_col4:
            st.metric(
                "峰位偏差均值",
                f"{mean_dev:.3f}°",
                help="匹配峰的2θ偏差绝对值的平均值"
            )
        
        with stat_col5:
            st.metric(
                "峰位偏差标准差",
                f"{std_dev:.3f}°",
                help="匹配峰的2θ偏差的标准差"
            )
        
        with st.expander("📋 峰匹配详情", expanded=False):
            if len(matched_exp) > 0:
                match_data = []
                for i, ep in enumerate(matched_exp):
                    nearest_theo = None
                    min_diff = float('inf')
                    for tp in theo_peaks:
                        diff = abs(ep.two_theta - tp.two_theta)
                        if diff < min_diff:
                            min_diff = diff
                            nearest_theo = tp
                    
                    match_data.append({
                        '序号': i + 1,
                        '实验峰 2θ (°)': round(ep.two_theta, 4),
                        '理论峰 2θ (°)': round(nearest_theo.two_theta, 4) if nearest_theo else '-',
                        '偏差 (°)': round(min_diff, 4),
                        '理论 hkl': f"({nearest_theo.h}{nearest_theo.k}{nearest_theo.l})" if nearest_theo else '-',
                        '实验强度 (%)': round(ep.intensity * 100 if ep.intensity <= 1 else ep.intensity, 2),
                    })
                df_match = pd.DataFrame(match_data)
                st.dataframe(df_match, use_container_width=True, height=300)
            else:
                st.info("暂无匹配峰数据，请先上传实验数据并寻峰")


if page == "衍射谱模拟":
    crystal_input_section()
    st.markdown("---")
    diffraction_simulation_section()

elif page == "晶体结构":
    crystal_input_section()
    st.markdown("---")
    crystal_structure_section()

elif page == "指标化":
    indexing_section()

elif page == "Le Bail拟合":
    le_bail_section()

elif page == "多相混合":
    multiphase_section()

elif page == "定量分析":
    quantitative_analysis_section()

elif page == "谱图对比":
    compare_section()


with st.sidebar:
    st.markdown("---")
    st.subheader("📊 当前晶相摘要")
    crystal = st.session_state.crystals.get(st.session_state.current_phase)
    if crystal is not None:
        st.write(f"**名称**: {crystal.name}")
        st.write(f"**空间群**: {crystal.space_group}")
        st.write(f"**晶系**: {crystal.crystal_system}")
        st.write(f"**a =** {crystal.a:.3f} Å")
        st.write(f"**b =** {crystal.b:.3f} Å")
        st.write(f"**c =** {crystal.c:.3f} Å")
        st.write(f"**α =** {crystal.alpha:.1f}°")
        st.write(f"**β =** {crystal.beta:.1f}°")
        st.write(f"**γ =** {crystal.gamma:.1f}°")
        st.write(f"**原子数**: {len(crystal.atoms)}")
        st.write(f"**体积**: {crystal.volume:.3f} Å³")
    else:
        st.info("暂无晶相数据")

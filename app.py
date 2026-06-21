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
        ["衍射谱模拟", "晶体结构", "指标化", "Le Bail拟合", "多相混合"]
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
            data = pd.read_csv(uploaded_file, sep=None, engine='python', header=None, comment='#')
            if data.shape[1] < 2:
                st.error("数据文件需要至少两列: 2theta 和 强度")
                return
            
            two_theta = data.iloc[:, 0].values.astype(float)
            intensity = data.iloc[:, 1].values.astype(float)
            
            valid_mask = ~np.isnan(two_theta) & ~np.isnan(intensity)
            two_theta = two_theta[valid_mask]
            intensity = intensity[valid_mask]
            
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

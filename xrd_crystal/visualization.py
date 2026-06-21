"""
3D晶体结构可视化
使用plotly 3D散点图
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple

from .crystal import Crystal, Atom
from .space_group import SpaceGroup, generate_equivalent_positions


ATOM_COLORS = {
    "H": "#FFFFFF",
    "He": "#D9FFFF",
    "Li": "#CC80FF",
    "Be": "#C2FF00",
    "B": "#FFB5B5",
    "C": "#909090",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "Ne": "#B3E3F5",
    "Na": "#AB5CF2",
    "Mg": "#8AFF00",
    "Al": "#BFA6A6",
    "Si": "#F0C8A0",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "Ar": "#80D1E3",
    "K": "#8F40D4",
    "Ca": "#3DFF00",
    "Sc": "#E6E6E6",
    "Ti": "#BFC2C7",
    "V": "#A6A6AB",
    "Cr": "#8A99C7",
    "Mn": "#9C7AC7",
    "Fe": "#E06633",
    "Co": "#F090A0",
    "Ni": "#50D050",
    "Cu": "#C88033",
    "Zn": "#7D80B0",
    "Ga": "#C28F8F",
    "Ge": "#668F8F",
    "As": "#BD80E3",
    "Se": "#FFA100",
    "Br": "#A62929",
    "Kr": "#5CB8D1",
    "Rb": "#702EB0",
    "Sr": "#00FF00",
    "Y": "#94FFFF",
    "Zr": "#94E0E0",
    "Nb": "#73C2C9",
    "Mo": "#54B5B5",
    "Tc": "#3B9E9E",
    "Ru": "#248F8F",
    "Rh": "#0A7D8C",
    "Pd": "#006985",
    "Ag": "#C0C0C0",
    "Cd": "#FFD98F",
    "In": "#A67573",
    "Sn": "#668080",
    "Sb": "#9E63B5",
    "Te": "#D47A00",
    "I": "#940094",
    "Xe": "#429EB0",
    "Cs": "#57178F",
    "Ba": "#00C900",
    "La": "#70D4FF",
    "Ce": "#FFFFC7",
    "Pr": "#D9FFC7",
    "Nd": "#C7FFC7",
    "Pm": "#A3FFC7",
    "Sm": "#8FFFC7",
    "Eu": "#61FFC7",
    "Gd": "#45FFC7",
    "Tb": "#30FFC7",
    "Dy": "#1FFFC7",
    "Ho": "#00FF9C",
    "Er": "#00E675",
    "Tm": "#00D452",
    "Yb": "#00BF38",
    "Lu": "#00AB24",
    "Hf": "#4DC2FF",
    "Ta": "#4DA6FF",
    "W": "#2194D6",
    "Re": "#267DAB",
    "Os": "#266696",
    "Ir": "#175487",
    "Pt": "#D0D0E0",
    "Au": "#FFD123",
    "Hg": "#B8B8D0",
    "Tl": "#A6544D",
    "Pb": "#575961",
    "Bi": "#9E4FB5",
    "Th": "#00BAFF",
    "Pa": "#00A1FF",
    "U": "#008FFF",
    "Np": "#0080FF",
    "Pu": "#006BFF",
    "Am": "#545CF2",
    "Cm": "#785CE3",
    "Bk": "#8A4FE3",
    "Cf": "#A136D4",
    "Es": "#B31FD4",
    "Fm": "#B31FBA",
    "Md": "#B30DA6",
    "No": "#BD0D87",
    "Lr": "#C70066",
}


ATOMIC_RADII = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.61,
    "Fe": 1.52,
    "Co": 1.50,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Ga": 1.36,
    "Ge": 1.25,
    "As": 1.14,
    "Se": 1.17,
    "Br": 1.14,
    "Kr": 1.17,
    "Rb": 2.20,
    "Sr": 1.95,
    "Y": 1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "In": 1.42,
    "Sn": 1.39,
    "Sb": 1.40,
    "Te": 1.37,
    "I": 1.36,
    "Xe": 1.40,
    "Cs": 2.44,
    "Ba": 2.15,
    "La": 2.07,
    "Ce": 2.04,
    "Pr": 2.03,
    "Nd": 2.01,
    "Pm": 1.99,
    "Sm": 1.98,
    "Eu": 1.98,
    "Gd": 1.96,
    "Tb": 1.94,
    "Dy": 1.92,
    "Ho": 1.92,
    "Er": 1.89,
    "Tm": 1.90,
    "Yb": 1.87,
    "Lu": 1.87,
    "Hf": 1.75,
    "Ta": 1.70,
    "W": 1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
    "Tl": 1.45,
    "Pb": 1.46,
    "Bi": 1.48,
    "Th": 1.79,
    "Pa": 1.63,
    "U": 1.56,
}


def get_atom_color(element: str) -> str:
    """获取元素的颜色"""
    base_element = ''.join(c for c in element if c.isalpha())
    base_element = base_element.capitalize()
    return ATOM_COLORS.get(base_element, "#808080")


def get_atom_radius(element: str) -> float:
    """获取元素的原子半径"""
    base_element = ''.join(c for c in element if c.isalpha())
    base_element = base_element.capitalize()
    return ATOMIC_RADII.get(base_element, 1.0)


def plot_crystal_3d(crystal: Crystal,
                    use_symmetry: bool = True,
                    show_unit_cell: bool = True,
                    show_bonds: bool = False,
                    supercell: Tuple[int, int, int] = (1, 1, 1),
                    atom_scale: float = 0.5) -> go.Figure:
    """
    绘制3D晶体结构图
    
    参数:
        crystal: Crystal对象
        use_symmetry: 是否使用空间群对称性
        show_unit_cell: 是否显示晶胞边框
        show_bonds: 是否显示化学键
        supercell: 超胞大小 (nx, ny, nz)
        atom_scale: 原子大小缩放因子
    
    返回:
        plotly Figure对象
    """
    fig = go.Figure()
    
    sg = SpaceGroup(crystal.space_group, crystal.space_group_number)
    
    all_positions = []
    all_elements = []
    
    if use_symmetry and len(crystal.atoms) > 0:
        for atom in crystal.atoms:
            equiv_positions = generate_equivalent_positions(atom.frac_coord, sg.symops)
            for pos in equiv_positions:
                all_positions.append(pos)
                all_elements.append(atom.element)
    else:
        for atom in crystal.atoms:
            all_positions.append(atom.frac_coord)
            all_elements.append(atom.element)
    
    nx, ny, nz = supercell
    
    supercell_positions = []
    supercell_elements = []
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for pos, elem in zip(all_positions, all_elements):
                    new_pos = pos + np.array([i, j, k])
                    supercell_positions.append(new_pos)
                    supercell_elements.append(elem)
    
    cart_positions = []
    for pos in supercell_positions:
        cart = crystal.frac_to_cart(pos)
        cart_positions.append(cart)
    
    cart_positions = np.array(cart_positions)
    
    element_groups = {}
    for i, elem in enumerate(supercell_elements):
        if elem not in element_groups:
            element_groups[elem] = []
        element_groups[elem].append(i)
    
    for elem, indices in element_groups.items():
        color = get_atom_color(elem)
        radius = get_atom_radius(elem) * atom_scale
        
        x = cart_positions[indices, 0]
        y = cart_positions[indices, 1]
        z = cart_positions[indices, 2]
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=radius * 20,
                color=color,
                opacity=0.8,
                line=dict(width=1, color='black')
            ),
            name=elem,
            hovertext=[elem] * len(indices),
            hoverinfo='text'
        ))
    
    if show_unit_cell:
        unit_cell_lines = _get_unit_cell_lines(crystal, supercell)
        for line in unit_cell_lines:
            fig.add_trace(go.Scatter3d(
                x=line[:, 0],
                y=line[:, 1],
                z=line[:, 2],
                mode='lines',
                line=dict(color='black', width=2),
                name='晶胞',
                showlegend=False,
                hoverinfo='none'
            ))
    
    if show_bonds:
        bond_lines = _get_bond_lines(cart_positions, supercell_elements, crystal)
        for line in bond_lines:
            fig.add_trace(go.Scatter3d(
                x=line[:, 0],
                y=line[:, 1],
                z=line[:, 2],
                mode='lines',
                line=dict(color='gray', width=2),
                name='化学键',
                showlegend=False,
                hoverinfo='none'
            ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            aspectmode='data'
        ),
        title=f"{crystal.name} - 晶体结构",
        showlegend=True,
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def _get_unit_cell_lines(crystal: Crystal, supercell: Tuple[int, int, int]) -> List[np.ndarray]:
    """获取晶胞边框线段"""
    nx, ny, nz = supercell
    
    corners = []
    for i in [0, nx]:
        for j in [0, ny]:
            for k in [0, nz]:
                frac = np.array([i, j, k])
                cart = crystal.frac_to_cart(frac)
                corners.append(cart)
    
    corners = np.array(corners)
    
    edges = [
        (0, 1), (0, 2), (0, 4),
        (3, 1), (3, 2), (3, 7),
        (5, 1), (5, 4), (5, 7),
        (6, 2), (6, 4), (6, 7),
    ]
    
    lines = []
    for start, end in edges:
        line = np.array([corners[start], corners[end]])
        lines.append(line)
    
    return lines


def _get_bond_lines(positions: np.ndarray, elements: List[str], 
                    crystal: Crystal, max_bond_length: float = 2.5) -> List[np.ndarray]:
    """获取化学键线段"""
    lines = []
    
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < max_bond_length and dist > 0.5:
                line = np.array([positions[i], positions[j]])
                lines.append(line)
    
    return lines


def generate_py3dmol_html(crystal: Crystal,
                          use_symmetry: bool = True,
                          width: int = 600,
                          height: int = 400,
                          style: str = "ball and stick") -> str:
    """
    生成py3Dmol的HTML代码
    
    参数:
        crystal: Crystal对象
        use_symmetry: 是否使用空间群对称性
        width: 图像宽度
        height: 图像高度
        style: 显示样式 ("ball and stick", "stick", "spacefill")
    
    返回:
        HTML字符串
    """
    sg = SpaceGroup(crystal.space_group, crystal.space_group_number)
    
    atoms_data = []
    
    if use_symmetry and len(crystal.atoms) > 0:
        for atom in crystal.atoms:
            equiv_positions = generate_equivalent_positions(atom.frac_coord, sg.symops)
            for pos in equiv_positions:
                cart = crystal.frac_to_cart(pos)
                atoms_data.append({
                    'element': atom.element,
                    'x': cart[0],
                    'y': cart[1],
                    'z': cart[2]
                })
    else:
        for atom in crystal.atoms:
            cart = crystal.frac_to_cart(atom.frac_coord)
            atoms_data.append({
                'element': atom.element,
                'x': cart[0],
                'y': cart[1],
                'z': cart[2]
            })
    
    atoms_str = ""
    for atom in atoms_data:
        atoms_str += f"{atom['element']} {atom['x']:.4f} {atom['y']:.4f} {atom['z']:.4f}\n"
    
    style_config = ""
    if style == "ball and stick":
        style_config = """
        viewer.setStyle({stick: {radius: 0.2}, sphere: {radius: 0.3}});
        """
    elif style == "stick":
        style_config = """
        viewer.setStyle({stick: {radius: 0.2}});
        """
    elif style == "spacefill":
        style_config = """
        viewer.setStyle({sphere: {scale: 0.8}});
        """
    
    html = f"""
    <div style="width: {width}px; height: {height}px;">
        <script src="https://3dmol.org/build/3Dmol-min.js"></script>
        <div id="viewer" style="width: 100%; height: 100%;"></div>
        <script>
            var viewer = $3Dmol.createViewer("viewer");
            var xyzData = `{len(atoms_data)}
            
{atoms_str}`;
            
            viewer.addModel(xyzData, "xyz");
            {style_config}
            viewer.zoomTo();
            viewer.render();
        </script>
    </div>
    """
    
    return html

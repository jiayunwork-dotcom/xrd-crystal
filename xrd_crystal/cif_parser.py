"""
CIF文件解析器
"""

import re
from typing import List, Optional
from .crystal import Crystal, Atom


def parse_cif(cif_text: str) -> Crystal:
    """解析CIF文件内容为Crystal对象"""
    lines = cif_text.split('\n')
    
    cell_a = cell_b = cell_c = None
    cell_alpha = cell_beta = cell_gamma = 90.0
    space_group = None
    space_group_number = None
    atoms = []
    
    in_loop = False
    loop_keys = []
    loop_values = []
    data_name = "Untitled"
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line or line.startswith('#'):
            i += 1
            continue
        
        if line.startswith('data_'):
            data_name = line[5:]
            i += 1
            continue
        
        if line.startswith('_cell_length_a'):
            cell_a = _parse_cif_value(line)
            i += 1
            continue
        if line.startswith('_cell_length_b'):
            cell_b = _parse_cif_value(line)
            i += 1
            continue
        if line.startswith('_cell_length_c'):
            cell_c = _parse_cif_value(line)
            i += 1
            continue
        if line.startswith('_cell_angle_alpha'):
            cell_alpha = _parse_cif_value(line)
            i += 1
            continue
        if line.startswith('_cell_angle_beta'):
            cell_beta = _parse_cif_value(line)
            i += 1
            continue
        if line.startswith('_cell_angle_gamma'):
            cell_gamma = _parse_cif_value(line)
            i += 1
            continue
        
        if line.startswith('_space_group_name_H-M_alt') or line.startswith('_symmetry_space_group_name_H-M'):
            space_group = _parse_cif_value(line).strip("'\"")
            i += 1
            continue
        if line.startswith('_space_group_IT_number') or line.startswith('_symmetry_Int_Tables_number'):
            try:
                space_group_number = int(_parse_cif_value(line))
            except ValueError:
                pass
            i += 1
            continue
        
        if line.startswith('loop_'):
            in_loop = True
            loop_keys = []
            loop_values = []
            i += 1
            
            while i < len(lines):
                loop_line = lines[i].strip()
                if loop_line.startswith('_'):
                    loop_keys.append(loop_line)
                    i += 1
                else:
                    break
            
            values_buffer = []
            while i < len(lines):
                val_line = lines[i].strip()
                if not val_line or val_line.startswith('#'):
                    i += 1
                    continue
                if val_line.startswith('_') or val_line.startswith('loop_') or val_line.startswith('data_'):
                    break
                
                if val_line.startswith("'") or val_line.startswith('"'):
                    values_buffer.append(val_line.strip("'\""))
                else:
                    parts = val_line.split()
                    values_buffer.extend(parts)
                
                if len(values_buffer) >= len(loop_keys):
                    row_values = values_buffer[:len(loop_keys)]
                    loop_values.append(row_values)
                    values_buffer = values_buffer[len(loop_keys):]
                
                i += 1
            
            if _is_atom_loop(loop_keys):
                atoms.extend(_parse_atom_loop(loop_keys, loop_values))
            
            in_loop = False
            continue
        
        i += 1
    
    if cell_a is None or cell_b is None or cell_c is None:
        raise ValueError("CIF文件缺少晶胞参数")
    
    crystal = Crystal(
        a=cell_a,
        b=cell_b,
        c=cell_c,
        alpha=cell_alpha,
        beta=cell_beta,
        gamma=cell_gamma,
        space_group=space_group or "P1",
        space_group_number=space_group_number or 1,
        atoms=atoms,
        name=data_name
    )
    
    return crystal


def _parse_cif_value(line: str) -> float:
    """解析CIF行中的数值"""
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"无法解析CIF行: {line}")
    
    value_str = parts[1]
    value_str = re.sub(r'\(.*?\)', '', value_str)
    value_str = value_str.strip("'\"")
    
    try:
        return float(value_str)
    except ValueError:
        return value_str


def _is_atom_loop(keys: List[str]) -> bool:
    """判断loop是否为原子坐标loop"""
    atom_keywords = ['_atom_site_label', '_atom_site_type_symbol', '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z']
    has_atom = any(any(kw in k for kw in atom_keywords) for k in keys)
    return has_atom and any('fract' in k for k in keys)


def _parse_atom_loop(keys: List[str], values: List[List[str]]) -> List[Atom]:
    """解析原子坐标loop"""
    atoms = []
    
    label_idx = _find_key_index(keys, ['_atom_site_label', '_atom_site_type_symbol'])
    element_idx = _find_key_index(keys, ['_atom_site_type_symbol', '_atom_site_label'])
    x_idx = _find_key_index(keys, ['_atom_site_fract_x'])
    y_idx = _find_key_index(keys, ['_atom_site_fract_y'])
    z_idx = _find_key_index(keys, ['_atom_site_fract_z'])
    occ_idx = _find_key_index(keys, ['_atom_site_occupancy', '_atom_site_occup'])
    b_idx = _find_key_index(keys, ['_atom_site_B_iso_or_equiv', '_atom_site_B_iso', '_atom_site_U_iso_or_equiv'])
    
    if x_idx is None or y_idx is None or z_idx is None:
        return atoms
    
    for row in values:
        if len(row) <= max(x_idx, y_idx, z_idx):
            continue
        
        try:
            x = float(re.sub(r'\(.*?\)', '', row[x_idx]))
            y = float(re.sub(r'\(.*?\)', '', row[y_idx]))
            z = float(re.sub(r'\(.*?\)', '', row[z_idx]))
        except ValueError:
            continue
        
        element = "Unknown"
        if element_idx is not None and element_idx < len(row):
            element = row[element_idx].strip("'\"")
        elif label_idx is not None and label_idx < len(row):
            label = row[label_idx].strip("'\"")
            element = re.match(r'([A-Za-z]+)', label).group(1) if re.match(r'([A-Za-z]+)', label) else label
        
        occupancy = 1.0
        if occ_idx is not None and occ_idx < len(row):
            try:
                occupancy = float(re.sub(r'\(.*?\)', '', row[occ_idx]))
            except ValueError:
                pass
        
        b_iso = 1.0
        if b_idx is not None and b_idx < len(row):
            try:
                b_val = float(re.sub(r'\(.*?\)', '', row[b_idx]))
                if 'U_iso' in keys[b_idx]:
                    b_iso = b_val * 8 * np.pi**2
                else:
                    b_iso = b_val
            except ValueError:
                pass
        
        atoms.append(Atom(element, x, y, z, occupancy, b_iso))
    
    return atoms


def _find_key_index(keys: List[str], patterns: List[str]) -> Optional[int]:
    """在key列表中查找匹配pattern的索引"""
    for pattern in patterns:
        for i, key in enumerate(keys):
            if pattern in key:
                return i
    return None


import numpy as np

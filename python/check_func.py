import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
import sys
from fiberassign.hardware import load_hardware
from fiberassign.assign import read_assignment_fits_tile
from astropy.table import Table, vstack
from shapely.geometry import Polygon, MultiPolygon, Point
from plyfile import PlyData
from shapely.prepared import prep
import subprocess
import os


# =========================
# 工具函数 - run fba
# =========================
def run_cmd(cmd):
    print("\n[RUN]", cmd)
    subprocess.run(cmd, shell=True, check=True)


def run_fba(target_file, footprint, output_dir, runtime, prefix):
    cmd = (
        f"fba_run "
        f"--targets {target_file} "
        f"--rundate {runtime} "
        f"--footprint {footprint} "
        f"--dir {output_dir} "
        f"--overwrite "
        f"--sky_per_module 1 "
        f"--standards_per_module 1 "
        f"--prefix {prefix}"
    )
    print(cmd)
    run_cmd(cmd)


def find_fba_file(prefix, output_dir):
    """
    自动找到 fba 输出文件（因为 tileid 可能不止一个）
    """
    files = os.listdir(output_dir)
    matches = [f for f in files if f.startswith(prefix) and f.endswith(".fits")]
    
    if len(matches) == 0:
        raise FileNotFoundError(f"No fba output for prefix {prefix}")
    
    # 如果只有一个 tile，直接用第一个
    return os.path.join(output_dir, matches[0])


def get_assigned_ids(fba_file):
    tab = Table.read(fba_file)

    # 关键：FA_TYPE > 0 表示分配成功
    mask = tab["FA_TYPE"] > 0

    assigned_ids = np.unique(tab["TARGETID"][mask])

    print(f"  Assigned this round: {len(assigned_ids)}")
    return assigned_ids


def update_target(old_target, assigned_ids, new_target):
    t = Table.read(old_target)

    mask = np.isin(t["TARGETID"], assigned_ids)

    # print(f"  Update PRIORITY=0 for {mask.sum()} targets")

    # t["PRIORITY"][mask] = 0
    t = t[~mask]  # 直接删除已分配的目标
    print(f"  Remaining targets for next round: {len(t)}")   

    t.write(new_target, overwrite=True)


def read_ply(filename):
    """
    读取 PLY 文件
    
    Returns:
        vertices: (N, 3) 数组，顶点坐标
        faces: list of arrays，每个面的顶点索引
    """
    vertices = []
    faces = []
    
    with open(filename, 'r') as f:
        # 读取 header
        line = f.readline()
        assert line.strip() == 'ply', "不是有效的 PLY 文件"
        
        num_vertices = 0
        num_faces = 0
        header_done = False
        
        while not header_done:
            line = f.readline().strip()
            
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('element face'):
                num_faces = int(line.split()[-1])
            elif line == 'end_header':
                header_done = True
        
        # 读取顶点
        for _ in range(num_vertices):
            line = f.readline().strip()
            x, y, z = map(float, line.split())
            vertices.append([x, y, z])
        
        # 读取面
        for _ in range(num_faces):
            line = f.readline().strip()
            parts = list(map(int, line.split()))
            num_verts = parts[0]
            indices = parts[1:num_verts+1]
            faces.append(indices)
    
    return np.array(vertices), faces

def plot_ply_with_fill(ply_file, output_file=None):
    """
    读取 PLY 文件并绘制，内部灰色，外部白色
    使用 shapely 来处理多边形填充
    """
    vertices, faces = read_ply(ply_file)
    
    print(f"顶点数: {len(vertices)}")
    print(f"面数: {len(faces)}")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 12), dpi=150)
    
    # 转换为 shapely 多边形用于并集运算
    polygons = []
    for face in faces:
        face_vertices = vertices[face][:, :2]  # 只取 x, y
        if len(face_vertices) >= 3:
            try:
                poly = ShapelyPolygon(face_vertices)
                if poly.is_valid:
                    polygons.append(poly)
            except:
                pass
    
    # 合并所有多边形为并集
    if polygons:
        merged_poly = unary_union(polygons)
    else:
        merged_poly = None
    
    # 获取坐标范围
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]
    x_min, x_max = x_coords.min() - 10, x_coords.max() + 10
    y_min, y_max = y_coords.min() - 10, y_coords.max() + 10
    
    # 创建背景（白色）
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_facecolor('white')
    
    # 绘制 PLY 区域（灰色填充）
    if merged_poly:
        if merged_poly.geom_type == 'Polygon':
            polygons_to_plot = [merged_poly]
        else:  # MultiPolygon
            polygons_to_plot = list(merged_poly.geoms)
        
        for poly in polygons_to_plot:
            x, y = poly.exterior.xy
            ax.fill(x, y, color='lightgray', edgecolor='black', linewidth=0.5, alpha=0.7)
    
    # 绘制边缘线
    for face in faces:
        face_vertices = vertices[face]
        x = np.append(face_vertices[:, 0], face_vertices[0, 0])
        y = np.append(face_vertices[:, 1], face_vertices[0, 1])
        ax.plot(x, y, 'b-', linewidth=0.3, alpha=0.5)
    
    # 设置坐标轴
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_title('Module Edges - Gray Inside, White Outside', fontsize=14)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')
    
    # 保存或显示
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"已保存图片: {output_file}")
    else:
        plt.show()
    
    return fig, ax
# 绘制PLY文件的边缘
def plot_ply_edges_on_same_plot(ply_file_1, ply_file_2, output_file=None):
    """
    在同一张图上绘制两个PLY文件的边缘进行比较
    """
    # 读取第一个 PLY 文件
    vertices1, faces1 = read_ply(ply_file_1)
    
    # 读取第二个 PLY 文件
    vertices2, faces2 = read_ply(ply_file_2)

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 绘制第一个 PLY 文件的边缘（原始模块边缘）
    for face in faces1:
        face_vertices = vertices1[face]
        x = np.append(face_vertices[:, 0], face_vertices[0, 0])
        y = np.append(face_vertices[:, 1], face_vertices[0, 1])
        ax.plot(x, y, 'b-', linewidth=0.7, alpha=0.7, label='Module Edges (Original)' if not 'Module Edges (Original)' in [label.get_text() for label in ax.get_legend_handles_labels()[1]] else "")
    
    # 绘制第二个 PLY 文件的边缘（最外轮廓）
    for face in faces2:
        face_vertices = vertices2[face]
        x = np.append(face_vertices[:, 0], face_vertices[0, 0])
        y = np.append(face_vertices[:, 1], face_vertices[0, 1])
        ax.plot(x, y, 'r-', linewidth=2, alpha=0.7, label='Outer Contour' if not 'Outer Contour' in [label.get_text() for label in ax.get_legend_handles_labels()[1]] else "")

    # 设置坐标轴和图形
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_title('Comparison of Module Edges and Outer Contour', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 显示图例
    ax.legend()

    # 保存或显示图像
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"已保存图像: {output_file}")
    else:
        plt.show()

    return fig, ax


def read_ply_as_multipolygon(filename):
    """
    读取 PLY 文件并返回一个 MultiPolygon 对象，包含所有面的多边形。
    
    Args:
        filename: PLY 文件路径
    
    Returns:
        MultiPolygon 对象，包含所有面的多边形。
    """
    # 读取 PLY 文件
    ply_data = PlyData.read(filename)
    
    # 获取顶点和面数据
    vertices = np.array([list(vertex) for vertex in ply_data['vertex']])
    faces = ply_data['face'].data['vertex_indices']
    
    # 将所有面的顶点坐标转换为多边形，并将它们加入一个列表
    polygons = []
    for face in faces:
        # 获取该面顶点的坐标
        face_vertices = vertices[face]
        
        # 创建多边形
        polygon = Polygon(face_vertices[:, :2])  # 只取 x, y 坐标
        if polygon.is_valid:
            polygons.append(polygon)
    
    # 如果没有有效的多边形，返回空的 MultiPolygon
    if polygons:
        return MultiPolygon(polygons)
    else:
        return MultiPolygon()
def check_points_in_multipolygon(points, multipolygon):
    """
    检查多个点是否在MultiPolygon内部
    
    参数:
    points: numpy数组，形状为(n, 2)，每行是一个点的(x,y)坐标
    multipolygon: shapely的MultiPolygon对象
    
    返回:
    numpy数组，布尔值，表示每个点是否在MultiPolygon内部
    """
    # 准备MultiPolygon（优化性能）
    prepared_multipolygon = prep(multipolygon)
    
    # 判断每个点是否在MultiPolygon内部
    is_inside = np.array([prepared_multipolygon.contains(Point(x, y)) for x, y in points])
    
    return is_inside

def compute_completeness(T_x, T_y, A_x, A_y, input_ply):
    multipolygon = read_ply_as_multipolygon(input_ply)

    # 打印 MultiPolygon 信息
    print(multipolygon)

    points_t = np.column_stack((T_x, T_y))
    points_a = np.column_stack((A_x, A_y))

    is_inside_t = check_points_in_multipolygon(points_t, multipolygon)
    is_inside_a = check_points_in_multipolygon(points_a, multipolygon)


    completeness = np.sum(is_inside_a) / np.sum(is_inside_t)

    return completeness,  np.sum(is_inside_t)
def completeness_ply(T_x, T_y, fa_files, hw, tile_ra, tile_dec, tile_obstime, tile_theta, tile_obsha):
    # 这里实现你的 completeness 计算逻辑
    # 例如，读取 PLY 文件，构建 MultiPolygon，然后检查哪些点在内

    fa_list = []
    for f in fa_files:
        tab = Table.read(f)
        tab = tab[tab['FA_TYPE'] > 0]  # 只保留成功分配的目标
        fa_list.append(tab['TARGETID', 'TARGET_RA', 'TARGET_DEC'])

    # =========================
    # 2. 合并
    # =========================
    fa_all = vstack(fa_list)

    print(f"Total assigned targets: {len(np.unique(fa_all['TARGETID']))/len(fa_all)}")

    # =========================
    # 4. 去重（关键步骤）
    # =========================
    # 方法1（推荐）：按 TARGETID 去重，保留第一次出现
    _, unique_idx = np.unique(fa_all['TARGETID'], return_index=True)

    fa_unique = fa_all[unique_idx]
    print(len(fa_unique), len(fa_all))



    # 假设字段名如下（根据你的文件改）
    ra_f, dec_f = fa_unique['TARGET_RA'], fa_unique['TARGET_DEC']
    A_x, A_y = radec2xy(hw, tile_ra, tile_dec, tile_obstime, tile_theta, tile_obsha,
                ra_f, dec_f, use_cs5=True)
    

    input_ply = '/home/hmf/work/FA_test/data/focalplane_ply/2025-10-27T17:01:02+00:00_module_edges.ply'

    completeness, all_tars = compute_completeness(T_x, T_y, A_x, A_y, input_ply)

    

    # =========================
    # 所有区域的completeness
    # =========================

    ra_bins  = np.arange(-560, 580, 20)
    dec_bins = np.arange(-560, 580, 20)

    H_target, _, _ = np.histogram2d(T_x, T_y, bins=[ra_bins, dec_bins])
    H_assigned, _, _ = np.histogram2d(A_x, A_y, bins=[ra_bins, dec_bins])

    comple_all = np.zeros_like(H_target)
    mask = H_assigned > 0
    comple_r = np.sum(H_assigned[mask]) / np.sum(H_target[mask])
    print('completeness ply:', completeness, 'num:', all_tars)
    print('completeness all:', comple_r, np.sum(H_target[mask]))


    return completeness, comple_r

# =========================
# 2. 读取 npz（支持 subset）
# =========================

def load_dat_as_table(filename, N=None):
    # 读取（自动跳过 # 注释行）
    tab = Table.read(filename, format='ascii')

    # 重命名列（非常重要！）
    tab.rename_columns(tab.colnames, ['RA', 'DEC', 'Z', 'ZERR'])

    # subset（用于测试）
    if N is not None:
        tab = tab[:N]

    return tab

def load_npz_as_table(filename, N=None, random=False):
    data = np.load(filename)
    keys = data.files

    n_total = len(data[keys[0]])

    if N is not None:
        if random:
            idx = np.random.choice(n_total, N, replace=False)
            subset = {k: data[k][idx] for k in keys}
        else:
            subset = {k: data[k][:N] for k in keys}
    else:
        subset = {k: data[k] for k in keys}

    return Table(subset)
def load_npy_as_table(filename, N=None, random=False):
    # mmap：不会一次性读入全部（大数据很重要）
    data = np.load(filename, mmap_mode='r')

    n_total = data.shape[0]

    # subset
    if N is not None:
        if random:
            idx = np.random.choice(n_total, N, replace=False)
            data = data[idx]
        else:
            data = data[:N]

    # ⚠️ 根据你的数据结构定义列名
    # 假设 npy = [RA, DEC, Z, ZERR]
    tab = Table()
    tab['RA'] = data[:, 0] #(((360. - data[:, 0] )+ 180) % 360) - 180 #
    tab['DEC'] = data[:, 1]
    tab['Z'] = data[:, 2]
    tab['Z_cosmo'] = data[:, 3]

    return tab

# =========================
# 3. 添加 galaxy 信息（通用版本）
# =========================
def add_galaxy_info(table, gal_type, config):
    n = len(table)

    table['GAL_TYPE'] = np.array([gal_type] * n, dtype='U8')

    # ---- 第一轮：常数列 ----
    for key, value in config[gal_type].items():
        if not callable(value):
            colname = key.upper()

            if isinstance(value, int):
                dtype = np.int32
            elif isinstance(value, float):
                dtype = np.float32
            else:
                dtype = None

            table[colname] = np.full(n, value, dtype=dtype)

    # ---- 第二轮：函数列 ----
    for key, value in config[gal_type].items():
        if callable(value):
            colname = key.upper()
            result = value(table)

            if isinstance(result, np.ndarray):
                table[colname] = result
            else:
                table[colname] = np.array(result)

    return table



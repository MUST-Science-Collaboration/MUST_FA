import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.patches import Circle, Ellipse, Polygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
import sys
from fiberassign.hardware import load_hardware, radec2xy
from fiberassign.assign import read_assignment_fits_tile
from astropy.io import fits
from astropy.table import Table, vstack
from shapely.geometry import Polygon, MultiPolygon, Point
from plyfile import PlyData
from shapely.prepared import prep
import subprocess
import os
import sys
sys.path.append('/home/hmf/work/software/must_fbautil-main/python/')
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, vstack
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Ellipse
import ezdxf
import glob
from shapely.ops import unary_union, polygonize
from shapely.geometry import LineString, Polygon, MultiLineString, MultiPoint
import numpy as np


# =========================
# 工具函数 - run fba
# =========================

def get_hw(assigned_file):
    header,_, _, _, _ = \
            read_assignment_fits_tile(assigned_file)
    run_date = header["FA_RUN"]
    margins = {}
    for key in ['pos', 'gfa', 'module']:
        hdrkey = "FA_M_%s" % key[:3]
        if hdrkey in header:
            margins[key] = header[hdrkey]
    # load hardware
    hw = load_hardware(rundate=run_date, add_margins=margins)
    return hw


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
    ofname = new_target.format(len(t))
    t.write(ofname, overwrite=True)
    return ofname


def wrap_ra(ra, center=180):
    ra = np.asarray(ra, dtype=float)
    return (ra + center) % 360 - center


def select_targets_near_tile(target, ra_center, dec_center, ra_half_width=2.3, dec_half_width=1.3):
    mask = (
        (target['RA'] > ra_center - ra_half_width)
        & (target['RA'] < ra_center + ra_half_width)
        & (target['DEC'] > dec_center - dec_half_width)
        & (target['DEC'] < dec_center + dec_half_width)
    )
    return target[mask]


def read_tile_center_from_fits(fits_file, ext_preferences=(1, 0)):
    for ext in ext_preferences:
        try:
            header = fits.getheader(fits_file, ext=ext)
        except Exception:
            continue
        if header.get('TILERA') is not None and header.get('TILEDEC') is not None:
            return header['TILERA'], header['TILEDEC']
    raise KeyError(f'Cannot find TILERA/TILEDEC in {fits_file}')


def collect_unique_assignments(files, keep_assigned_only=True, columns=('TARGETID', 'TARGET_RA', 'TARGET_DEC')):
    files = sorted(files)
    fa_list = []
    for file in files:
        tab = Table.read(file)
        if keep_assigned_only and 'FA_TYPE' in tab.colnames:
            tab = tab[tab['FA_TYPE'] > 0]
        fa_list.append(tab[list(columns)])

    if not fa_list:
        raise ValueError('No fiberassign files found.')

    fa_all = vstack(fa_list)
    unique_ids, unique_idx = np.unique(fa_all['TARGETID'], return_index=True)
    fa_unique = fa_all[np.sort(unique_idx)]

    stats = {
        'n_rows': len(fa_all),
        'n_unique': len(unique_ids),
        'unique_fraction': len(unique_ids) / len(fa_all) if len(fa_all) else 0.0,
    }
    return fa_all, fa_unique, stats


def compute_completeness_grid(target_xy, assigned_xy, x_bins, y_bins):
    target_x, target_y = target_xy
    assigned_x, assigned_y = assigned_xy

    H_target, _, _ = np.histogram2d(target_x, target_y, bins=[x_bins, y_bins])
    H_assigned, _, _ = np.histogram2d(assigned_x, assigned_y, bins=[x_bins, y_bins])

    completeness = np.zeros_like(H_target, dtype=float)
    mask = H_target > 0
    completeness[mask] = H_assigned[mask] / H_target[mask]

    return {
        'H_target': H_target,
        'H_assigned': H_assigned,
        'completeness': completeness,
        'x_bins': np.asarray(x_bins),
        'y_bins': np.asarray(y_bins),
    }


def plot_targets_vs_tiles(target, tiles, sample_step=1000, figsize=(8, 6), title='Targets vs Tiles', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sub = target[::sample_step]
    ax.scatter(sub['RA'], sub['DEC'], s=1, alpha=0.5, label='targets')
    ax.scatter(tiles['RA'], tiles['DEC'], c='r', marker='*', s=20, label='tiles')
    ax.set_xlabel('RA [deg]')
    ax.set_ylabel('Dec [deg]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_fba_tile_view(
    fba_res,
    tile_ra,
    tile_dec,
    background_target=None,
    wrap_center=180,
    background_window=None,
    ellipse_diameter=2.56,
    figsize=(12, 10),
    xlim=None,
    ylim=None,
    invert_xaxis=False,
    title=None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    mask = fba_res['FA_TYPE'] == 1
    target_ra = wrap_ra(fba_res['TARGET_RA'], center=wrap_center)
    target_dec = np.asarray(fba_res['TARGET_DEC'], dtype=float)
    tile_ra = np.atleast_1d(np.asarray(tile_ra, dtype=float))
    tile_dec = np.atleast_1d(np.asarray(tile_dec, dtype=float))
    tile_ra_wrapped = wrap_ra(tile_ra, center=wrap_center)
    delta_ra = ellipse_diameter / np.cos(np.radians(tile_dec))

    # ax.scatter(target_ra[~mask], target_dec[~mask], s=2, alpha=0.5, c='tomato', label='non-targets')
    ax.scatter(target_ra[mask], target_dec[mask], s=2, alpha=0.5,  c='tomato', label='targets')

    if background_target is not None:
        if background_window is None:
            window_target = background_target
        else:
            window_target = select_targets_near_tile(
                background_target,
                tile_ra[0],
                tile_dec[0],
                ra_half_width=background_window[0],
                dec_half_width=background_window[1],
            )
        ax.scatter(window_target['RA'], window_target['DEC'], s=2, alpha=0.1, c='green', label='targets_all')

    ellipses = [
        Ellipse((ra, dec), dra, ellipse_diameter)
        for ra, dec, dra in zip(tile_ra_wrapped, tile_dec, delta_ra)
    ]
    collection = PatchCollection(
        ellipses,
        facecolor='none',
        edgecolor='red',
        lw=2,
        alpha=0.8,
        zorder=2,
    )
    ax.add_collection(collection)
    ax.scatter(tile_ra_wrapped, tile_dec, color='red', s=50, zorder=3, label='tile center')

    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if invert_xaxis:
        ax.invert_xaxis()
    return fig, ax


def plot_multiple_fba_tiles(files,background_target=None, wrap_center=180, ellipse_diameter=2.56, figsize=(24, 20), ax=None, background_window=[1.5, 1.5]):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for file in sorted(files):
        fba_res = Table.read(file)
        tile_ra, tile_dec = read_tile_center_from_fits(file)
        plot_fba_tile_view(
            fba_res,
            tile_ra,
            tile_dec,
            wrap_center=wrap_center,
            ellipse_diameter=ellipse_diameter,
            ax=ax,
        )
    if background_target is not None:
        if background_window is None:
            window_target = background_target
        else:
            window_target = select_targets_near_tile(
                background_target,
                tile_ra,
                tile_dec,
                ra_half_width= background_window[0],
                dec_half_width=background_window[1],
            )
        ax.scatter(window_target['RA'], window_target['DEC'], s=2, alpha=0.1, c='green', label='targets_all')

    return fig, ax


def plot_completeness_map(
    grid,
    tile_centers=None,
    tile_radius=1.28,
    figsize=(20, 10),
    dpi=150,
    min_target_count=10,
    scatter_points=None,
    scatter_label='assigned targets',
    circle_kwargs=None,
    title='Completeness Map with Tile Footprints',
    xlabel='RA [deg]',
    ylabel='DEC [deg]',
    xlim=None,
    ylim=None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    mesh = ax.pcolormesh(grid['x_bins'], grid['y_bins'], grid['completeness'].T, shading='auto')
    plt.colorbar(mesh, ax=ax, label='Completeness')

    if scatter_points is not None:
        scatter_x, scatter_y = scatter_points
        ax.scatter(scatter_x, scatter_y, s=1, alpha=0.5, label=scatter_label)

    for i in range(len(grid['x_bins']) - 1):
        for j in range(len(grid['y_bins']) - 1):
            if grid['H_target'][i, j] < min_target_count:
                continue
            x_center = 0.5 * (grid['x_bins'][i] + grid['x_bins'][i + 1])
            y_center = 0.5 * (grid['y_bins'][j] + grid['y_bins'][j + 1])
            value = grid['completeness'][i, j]
            color = 'white' if value < 0.5 else 'black'
            ax.text(x_center, y_center, f'{value:.2f}', ha='center', va='center', fontsize=7, color=color)

    if tile_centers is not None:
        circle_kwargs = circle_kwargs or {}
        base_circle_kwargs = {
            'edgecolor': 'red',
            'facecolor': 'none',
            'linewidth': 0.5,
            'alpha': 0.5,
        }
        base_circle_kwargs.update(circle_kwargs)
        for x_center, y_center in tile_centers:
            ax.add_patch(Circle((x_center, y_center), tile_radius, **base_circle_kwargs))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(grid['x_bins'][0], grid['x_bins'][-1])
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(grid['y_bins'][0], grid['y_bins'][-1])
    if scatter_points is not None or tile_centers is not None:
        ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_completeness_curve(x_values, completeness_list, labels=('covered', 'all'), xlabel='N_pass', ylabel='completeness', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for idx, label in enumerate(labels):
        ax.plot(x_values, [values[idx] for values in completeness_list], marker='o', label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig, ax


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

def plot_ply_with_fill(ply_file, output_file=None, ax = None):
    """
    读取 PLY 文件并绘制，内部灰色，外部白色
    使用 shapely 来处理多边形填充
    """
    vertices, faces = read_ply(ply_file)
    
    print(f"顶点数: {len(vertices)}")
    print(f"面数: {len(faces)}")
    
    # 创建图形
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 12), dpi=150)
    else:
        fig = ax.figure
    
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
        if ax is None:  # 只有在创建新图时才显示
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

# def read_ply_as_multipolygon(filename, buffer_distance=2.0):
#     """
#     读取 PLY 文件并返回 MultiPolygon，用于点包含测试
    
#     Args:
#         filename: PLY 文件路径（顶点+边格式）
#         buffer_distance: 缓冲距离，用于闭合微小的间隙
    
#     Returns:
#         MultiPolygon 对象
#     """
#     vertices, edges = read_ply(filename)
    
#     if len(vertices) == 0 or len(edges) == 0:
#         return MultiPolygon()
    
#     # 从边重建线段
#     lines = []
#     for edge in edges:
#         v1 = vertices[edge[0], :2]
#         v2 = vertices[edge[1], :2]
#         if np.linalg.norm(v2 - v1) > 1e-6:  # 过滤零长度边
#             lines.append(LineString([v1, v2]))
    
#     # 使用 polygonize 自动识别闭合区域
#     polygons = list(polygonize(lines))
    
#     # 如果 polygonize 失败，使用 buffer 方法
#     if not polygons:
#         multi_line = MultiLineString(lines)
#         merged = unary_union(multi_line)
#         buffered = merged.buffer(buffer_distance, resolution=8)
        
#         if isinstance(buffered, Polygon):
#             polygons = [buffered]
#         elif isinstance(buffered, MultiPolygon):
#             polygons = list(buffered.geoms)
    
#     # 创建 MultiPolygon
#     if polygons:
#         # 合并重叠区域
#         merged = unary_union(polygons)
#         if isinstance(merged, Polygon):
#             return MultiPolygon([merged])
#         elif isinstance(merged, MultiPolygon):
#             return merged
    
#     return MultiPolygon()





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

    points_t = np.column_stack((T_x, T_y))
    points_a = np.column_stack((A_x, A_y))

    is_inside_t = check_points_in_multipolygon(points_t, multipolygon)
    is_inside_a = check_points_in_multipolygon(points_a, multipolygon)

    n_inside_t = np.sum(is_inside_t)
    completeness = np.sum(is_inside_a) / n_inside_t if n_inside_t > 0 else 0.0
    return completeness, n_inside_t



def completeness_ply(
    T_x,
    T_y,
    A_x=None, 
    A_y = None,
    fa_files=None,
    hw = None,
    module_ply='/home/hmf/work/FA_test/data/focalplane_ply/2025-10-27T17:01:02+00:00_module_edges.ply',
    outboun_ply='/home/hmf/work/FA_test/data/focalplane_ply/2025-10-27T17:01:02+00:00_module_edges.ply',
):
    header, _, _, _, _ = read_assignment_fits_tile(fa_files[0])
    tile_ra = header['TILERA']
    tile_dec = header['TILEDEC']
    tile_theta = header['FIELDROT']
    tile_obstime = header['FA_PLAN']
    tile_obsha = header['FA_HA']

    if hw is None:
        hw = get_hw(fa_files[0])
        
    
    _, fa_unique, stats = collect_unique_assignments(fa_files, keep_assigned_only=True)
    print(f"Total assigned targets: {stats['unique_fraction']}")
    print(stats)

    ra_f, dec_f = fa_unique['TARGET_RA'], fa_unique['TARGET_DEC']
    A_x, A_y = radec2xy(
        hw,
        tile_ra,
        tile_dec,
        tile_obstime,
        tile_theta,
        tile_obsha,
        ra_f,
        dec_f,
        use_cs5=True,
    )
    
    completeness, all_tars = compute_completeness(T_x, T_y, A_x, A_y, module_ply)
    completeness_out, all_tars = compute_completeness(T_x, T_y, A_x, A_y, outboun_ply)

    return completeness, completeness_out

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


# check fba results with focal plane and completeness calculation
def plot_fa_on_focalplane_auto(
    target_file,
    fa_pattern,
    ply_file= None,
    hw=None,   # ⭐ 新增
    only_inside = True,
    keep_assigned_only=True,
    figsize=(20, 20),
    Tar_r = None,
    s_target=1,
    s_assigned=1,
    plot_Tars = False,
    plot_FP = False,
    output=None,
    return_cat = False
):
    """
    自动从 FA 文件读取 tile 信息；支持传入 hw（避免重复加载）
    """

    import glob
    import matplotlib.pyplot as plt
    from astropy.table import Table

    # =========================
    # 1. 找 FA 文件
    # =========================
    files = sorted(glob.glob(fa_pattern))
    if len(files) == 0:
        raise ValueError("No FA files found!")

    infile = files[0]

    # =========================
    # 2. 读取 tile 信息
    # =========================
    header, fiber_data, targets_data, avail_data, gfa_data = \
        read_assignment_fits_tile(infile)

    tile_ra = float(header["TILERA"])
    tile_dec = float(header["TILEDEC"])
    tile_theta = float(header["FIELDROT"])
    tile_obstime = header["FA_PLAN"]
    tile_obsha = float(header["FA_HA"])
    run_date = header["FA_RUN"]

    # =========================
    # 3. hardware（可选加载）
    # =========================
    if hw is None:
        margins = {}
        for key in ['pos', 'gfa', 'module']:
            hdrkey = "FA_M_%s" % key[:3]
            if hdrkey in header:
                margins[key] = header[hdrkey]

        print("Loading hardware (slow)...")
        hw = load_hardware(rundate=run_date, add_margins=margins)
    else:
        print("Using provided hardware")

    # =========================
    # 4. 读取数据
    # =========================
    target = Table.read(target_file)
    if Tar_r is None:
        Tar_r = [[tile_ra-1.5, tile_ra+1.5],[tile_dec-1.5, tile_dec+1.5]]
    target = target[(target['RA']>=Tar_r[0][0])&(target['RA']<=Tar_r[0][1])&(target['DEC']>=Tar_r[1][0])&(target['DEC']<=Tar_r[1][1])]

    _, fa_unique, stats = collect_unique_assignments(
        files, keep_assigned_only=keep_assigned_only
    )

    print(f"Total assigned targets: {stats['unique_fraction']}")
    print(stats)

    

    # =========================
    # 5. 坐标投影
    # =========================
    ra_t_xy, dec_t_xy = radec2xy(
        hw, tile_ra, tile_dec, tile_obstime, tile_theta, tile_obsha,
        target['RA'], target['DEC'], use_cs5=True,
    )

    ra_f_xy, dec_f_xy = radec2xy(
        hw, tile_ra, tile_dec, tile_obstime, tile_theta, tile_obsha,
        fa_unique['TARGET_RA'], fa_unique['TARGET_DEC'], use_cs5=True,
    )
    # =========================
    # 筛选 inside（用你的函数）
    # =========================
    if only_inside:
        print("Filtering targets inside focal plane (multipolygon)...")

        multipolygon = read_ply_as_multipolygon(ply_file)

        points_t = np.column_stack((ra_t_xy, dec_t_xy))
        mask_t = check_points_in_multipolygon(points_t, multipolygon)

        print(f"Targets inside: {mask_t.sum()} / {len(mask_t)}")

        ra_t_xy = ra_t_xy[mask_t]
        dec_t_xy = dec_t_xy[mask_t]
    print(f"Final target count: {len(ra_t_xy)} {len(ra_f_xy)}")
    # =========================
    # 6. 画图
    # ========================
    if plot_Tars:
        fig, ax = plt.subplots(figsize=figsize)

        # 

        ax.scatter(ra_t_xy, dec_t_xy,
                s=s_target, alpha=0.2,
                label='Targets', zorder=2)

        ax.scatter(ra_f_xy, dec_f_xy,
                s=s_assigned, alpha=0.9,
                label='Assigned', zorder=3)
        if plot_FP:
            plot_ply_with_fill(ply_file, ax=ax)

        ax.legend()
        ax.set_title("FA + Focal Plane")
        ax.set_aspect('equal')

        if output is not None:
            plt.savefig(output, dpi=200, bbox_inches='tight')

        plt.show()
    if return_cat and plot_Tars:
        return fig, ax, stats, hw, ra_t_xy, dec_t_xy, ra_f_xy, dec_f_xy
    elif return_cat:
        return stats, hw, ra_t_xy, dec_t_xy, ra_f_xy, dec_f_xy
    else:
        return fig, ax, stats, hw
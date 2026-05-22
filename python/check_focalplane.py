'''
conda activate DESI_FA
python /home/hmf/work/FA_test/nb/check_focalplane.py /home/hmf/work/FA_test/output/MUST_FP_test/iter00000001.fits
'''
import os
os.environ['DESIMODEL'] = os.path.expanduser('~/work/FA_test/desi')
print(os.environ.get('DESIMODEL'))
import sys
from fiberassign.hardware import load_hardware
from fiberassign.assign import read_assignment_fits_tile
import numpy as np
import json
from astropy.table import Table

# 假设你已经加载了硬件对象 hw
# hw = load_hardware(rundate=run_date, add_margins=margins)

def extract_module_polygons(hw, modules=None):
    """
    从硬件对象中提取模块边缘的多边形数据
    
    Args:
        hw: 硬件对象
        modules: 要提取的模块列表，如果为None则提取所有模块
    
    Returns:
        dict: {module_id: {'gfa': [polygon_list], 'module': [polygon_list]}}
    """
    polygons = {}
    
    if modules is None:
        modules = list(hw.mod_gfa_excl.keys())
    
    for mod in modules:
        polygons[mod] = {'gfa': [], 'module': []}
        
        # 提取 GFA 边缘
        if mod in hw.mod_gfa_excl:
            gfa_shape = hw.mod_gfa_excl[mod]
            polygons[mod]['gfa'] = shape_to_polygons(gfa_shape)
        
        # 提取 Module 边缘  
        if mod in hw.mod_module_excl:
            module_shape = hw.mod_module_excl[mod]
            polygons[mod]['module'] = shape_to_polygons(module_shape)
    
    return polygons

def shape_to_polygons(shape):
    """
    将 Shape 对象转换为多边形列表
    每个多边形是 (x, y) 坐标点的列表
    """
    polygons = []
    
    for segment in shape.segments:
        # 提取 segment 中的所有点
        points = [(p[0], p[1]) for p in segment.points]
        if len(points) > 2:  # 至少需要3个点形成多边形
            polygons.append(points)
    
    return polygons

def save_polygons_to_geojson(polygons, filename):
    """
    将多边形数据保存为 GeoJSON 格式
    注意：这里使用的是焦平面坐标，不是地理坐标
    """
    features = []
    
    for mod_id, mod_data in polygons.items():
        for poly_type, poly_list in mod_data.items():
            for i, polygon in enumerate(poly_list):
                # 转换为 GeoJSON Polygon 格式
                coordinates = [polygon + [polygon[0]]]  # 闭合多边形
                
                feature = {
                    "type": "Feature",
                    "properties": {
                        "module": mod_id,
                        "type": poly_type,
                        "polygon_id": i
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": coordinates
                    }
                }
                features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(filename, 'w') as f:
        json.dump(geojson, f, indent=2)

def save_polygons_to_text(polygons, filename):
    """
    将多边形数据保存为简单文本格式
    """
    with open(filename, 'w') as f:
        f.write("# Module edge polygons\n")
        f.write("# Format: module_id type polygon_id x1,y1 x2,y2 ...\n")
        
        for mod_id, mod_data in polygons.items():
            for poly_type, poly_list in mod_data.items():
                for i, polygon in enumerate(poly_list):
                    coords_str = ' '.join([f"{x},{y}" for x, y in polygon])
                    f.write(f"{mod_id} {poly_type} {i} {coords_str}\n")
def save_polygons_to_ply(polygons, filename):
    """
    将多边形数据保存为 PLY 格式
    PLY 是一种常见的 3D 几何格式，这里用于保存 2D 焦平面坐标
    """
    vertices = []
    faces = []
    vertex_count = 0
    
    for mod_id, mod_data in polygons.items():
        for poly_type, poly_list in mod_data.items():
            for polygon in poly_list:
                # 每个多边形的顶点数
                num_verts = len(polygon)
                
                # 添加顶点
                for x, y in polygon:
                    vertices.append(f"{x} {y} 0")
                
                # 添加面（多边形）
                vert_indices = ' '.join(str(vertex_count + i) for i in range(num_verts))
                faces.append(f"{num_verts} {vert_indices}")
                
                vertex_count += num_verts
    
    # 写入 PLY 文件
    with open(filename, 'w') as f:
        # 写入 header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # 写入顶点数据
        for vertex in vertices:
            f.write(vertex + "\n")
        
        # 写入面数据
        for face in faces:
            f.write(face + "\n")
    
    print(f"已保存为 PLY 文件: {filename}")
    print(f"顶点数: {len(vertices)}, 面数: {len(faces)}")


# 使用示例
# 1. 加载硬件


infile = '/home/hmf/work/FA_test/output/tiles_box10_1visit_10000010.fits'

infile = sys.argv[1] if len(sys.argv) > 1 else infile

header, fiber_data, targets_data, avail_data, gfa_data = \
        read_assignment_fits_tile(infile)
    # tile_id = int(header["TILEID"])
    # tile_ra = float(header["TILERA"])
    # tile_dec = float(header["TILEDEC"])
    # tile_theta = float(header["FIELDROT"])
    # tile_obstime = header["FA_PLAN"]
    # tile_obsha = float(header["FA_HA"])
run_date = header["FA_RUN"]
margins = {}
for key in ['pos', 'gfa', 'module']:
    hdrkey = "FA_M_%s" % key[:3]
    if hdrkey in header:
        margins[key] = header[hdrkey]
# load hardware
hw = load_hardware(rundate=run_date, add_margins=margins)



# 2. 提取多边形
polygons = extract_module_polygons(hw)

# 3. 保存为 PLY 文件
save_polygons_to_ply(polygons, f'/home/hmf/work/FA_test/data/focalplane_ply/{run_date.replace(" ", "_")}_module_edges.ply')

# save_polygons_to_geojson(polygons, 'module_edges.geojson')
# save_polygons_to_text(polygons, 'module_edges.txt')

print("多边形文件已生成")


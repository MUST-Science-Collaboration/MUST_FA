
from fiberassign.hardware import load_hardware
from fiberassign.assign import read_assignment_fits_tile
from test_MUST_fp import *

import os
os.environ["DESIMODEL"] = os.path.expanduser("~/work/FA_test/desi")
print(os.environ["DESIMODEL"])
from fiberassign.hardware import radec2xy
import sys
from fiberassign.hardware import load_hardware
from fiberassign.assign import read_assignment_fits_tile
import time
from astropy.table import Table
import matplotlib.pyplot as plt

import pickle
import json
import numpy as np
from datetime import datetime
import glob

import ezdxf
import glob
from shapely.ops import unary_union, polygonize
from shapely.geometry import LineString, Polygon, MultiLineString, MultiPoint
import numpy as np

# =========================
# 保存 all_results
# =========================
def save_all_results(all_results, save_dir="/home/hmf/work/FA_test/results/completeness_curves/"):
    """
    保存 all_results 字典到多种格式
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 方法1: 保存为 pickle 文件（推荐，能保存完整 Python 对象）
    pickle_file = os.path.join(save_dir, f"all_results_{timestamp}.pkl")
    with open(pickle_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"✅ Pickle 保存到: {pickle_file}")
    
    # 方法2: 保存为 JSON 文件（需要转换 numpy 数组）
    json_file = os.path.join(save_dir, f"all_results_{timestamp}.json")
    try:
        # 转换 all_results 为可 JSON 序列化的格式
        json_compatible = convert_to_json_serializable(all_results)
        with open(json_file, 'w') as f:
            json.dump(json_compatible, f, indent=2)
        print(f"✅ JSON 保存到: {json_file}")
    except Exception as e:
        print(f"⚠️ JSON 保存失败: {e}")
    
    # 方法3: 保存为简化的文本摘要
    summary_file = os.path.join(save_dir, f"completeness_summary_{timestamp}.txt")
    try:
        with open(summary_file, 'w') as f:
            f.write("Completeness Results Summary\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for density in sorted(all_results.keys()):
                f.write(f"Density: {density}\n")
                f.write("-"*30 + "\n")
                
                for run_time in all_results[density].keys():
                    result = all_results[density][run_time]
                    com_list = result.get('completeness_module', [])
                    
                    f.write(f"  Run: {run_time[:16]}\n")
                    f.write(f"    Config: {result.get('plybase', 'N/A')}\n")
                    f.write(f"    FA Files: {result.get('num_fa_files', 0)}\n")
                    
                    if com_list:
                        final_com = com_list[-1]
                        if isinstance(final_com, (list, tuple)):
                            f.write(f"    Final Inside: {final_com[0]:.4f}\n")
                            f.write(f"    Final Overall: {final_com[1]:.4f}\n")
                        else:
                            f.write(f"    Final Completeness: {final_com:.4f}\n")
                        
                        # 添加更多统计信息
                        if all(isinstance(c, (list, tuple)) for c in com_list):
                            overall_vals = [c[1] for c in com_list]
                            inside_vals = [c[0] for c in com_list]
                            f.write(f"    Max Overall: {max(overall_vals):.4f}\n")
                            f.write(f"    Max Inside: {max(inside_vals):.4f}\n")
                    
                    f.write("\n")
        
        print(f"✅ 文本摘要保存到: {summary_file}")
    except Exception as e:
        print(f"⚠️ 文本摘要保存失败: {e}")
    
    return pickle_file  # 返回主文件路径


def convert_to_json_serializable(obj):
    """
    递归转换对象为 JSON 可序列化的格式
    """
    if isinstance(obj, dict):
        return {str(key): convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, '__dict__'):
        return str(obj)  # 对于复杂对象，转为字符串
    else:
        return obj


def load_all_results(pickle_file):
    """
    加载保存的 all_results
    """
    with open(pickle_file, 'rb') as f:
        all_results = pickle.load(f)
    print(f"✅ 加载结果从: {pickle_file}")
    return all_results

# =========================
# 1. 配置参数
# =========================
density_values = [4000, 6000, 8000, 10000]
run_times = [
    '2026-04-13T13:23:00+00:00',
    '2026-04-15T16:37:00+00:00', 
    '2026-04-15T17:10:00+00:00',
    '2026-04-15T17:15:00+00:00'
]
ply_bases = [
    'Framed-GlobalGap4.4mm-ModuleWalls',
    'Framed-GlobalGap4.4mm-ModuleWalls',
    'SemiFrameless-GlobalGap4.4mm-InnerGap0.5mm-ModuleWalls', 
    'SemiFrameless-GlobalGap4.4mm-InnerGap0.5mm-NoModuleWalls'
]

# 存储所有结果的字典
# 结构: {density: {run_time: completeness_list}}
all_results = {}

# =========================
# 2. 外层循环：遍历不同密度
# =========================
for density in density_values:
    print("\n" + "="*80)
    print(f"处理密度: {density}")
    print("="*80)
    
    # 构建对应的 target 文件路径
    target_file = f"/home/hmf/work/FA_test/data/random/targets_exdense_box10.000_00_{density}.0.fits"
    
    # 检查文件是否存在
    if not os.path.exists(target_file):
        print(f"⚠️ Target 文件不存在: {target_file}")
        continue
    
    # 存储当前密度的所有运行结果
    density_results = {}
    
    # =========================
    # 3. 内层循环：遍历不同的运行配置
    # =========================
    for run_time, plybase in zip(run_times, ply_bases):
        print(f"\n{'='*60}")
        print(f"处理运行: {run_time}")
        print(f"配置: {plybase}")
        print(f"{'='*60}")
        
        # 构建路径
        fa_pattern = f"/home/hmf/work/FA_test/output/MUST_FP_test/{run_time}/{os.path.basename(target_file).replace('.fits', '')}/iter*.fits"
        ply_file = f'/home/hmf/work/FA_test/MUST_FP/2026.04.15/{plybase}/'
        
        # 检查 FA 文件是否存在
        fa_files_check = glob.glob(fa_pattern)
        if not fa_files_check:
            print(f"⚠️ 没有找到 FA 文件: {fa_pattern}")
            continue
        
        try:
            # =========================
            # 4. 单次运行（自动加载 hw）
            # =========================
            print("\n----- 运行 plot_fa_on_focalplane_auto -----")
            t0 = time.time()
            
            stats, hw, ra_t_xy, dec_t_xy, ra_f_xy, dec_f_xy = plot_fa_on_focalplane_auto(
                target_file=target_file,
                fa_pattern=fa_pattern,
                ply_file=ply_file,
                only_inside=False,
                return_cat=True,

            )
            
            t1 = time.time()
            print(f"运行时间: {t1 - t0:.2f} 秒")
            
            # 关闭图形以节省内存
            # plt.close(fig)
            
            # =========================
            # 5. 计算 completeness
            # =========================
            print("\n----- 计算 Completeness -----")
            
            # 获取 FA 文件列表
            fba_files = sorted(glob.glob(fa_pattern))
            
            if not fba_files:
                print(f"⚠️ 没有找到 FA 文件")
                continue
            
            # 获取首个文件的 header 信息
            # single_fba_file = fba_files[0]
            # header, _, _, _, _ = read_assignment_fits_tile(str(single_fba_file))
            # hw = get_hw(str(single_fba_file))
            
            # 提取坐标
            T_x, T_y = ra_t_xy, dec_t_xy
            A_x, A_y = ra_f_xy, dec_f_xy
            
            # 计算 completeness
            com_list = []
            for i in range(len(fba_files)):
                com = completeness_ply(
                    T_x,
                    T_y,
                    fa_files=fba_files[:i + 1],
                    hw=hw,
                    module_ply = f'{ply_file}/segments.ply',
                    outboun_ply = f'{ply_file}/outer_boundary.ply'
                )
                com_list.append(com)
            
            # 存储结果
            density_results[run_time] = {
                'completeness_module': com_list,
                'completeness_outer': com_list,
                'stats': stats,
                'num_fa_files': len(fba_files),
                'plybase': plybase
            }
            
            print(f"Completeness 计算完成，共 {len(com_list)} 个迭代")
            
            # # =========================
            # # 6. 绘制单个配置的 completeness 曲线
            # # =========================
            # print("\n----- 绘制 Completeness 曲线 -----")
            # fig, ax = plt.subplots(figsize=(10, 6))
            
            # iterations = range(1, len(com_list) + 1)
            
            # # 检查 com_list 的结构并相应绘制
            # if com_list and isinstance(com_list[0], (list, tuple)):
            #     # 如果是 (inside_footprint, overall) 的形式
            #     inside_vals = [c[0] for c in com_list]
            #     overall_vals = [c[1] for c in com_list]
                
            #     ax.plot(iterations, inside_vals, 'b-', linewidth=2, label='Inside Footprint')
            #     ax.plot(iterations, overall_vals, 'r--', linewidth=2, label='Overall')
            # else:
            #     # 如果是单一值
            #     ax.plot(iterations, com_list, 'b-', linewidth=2, label='Completeness')
            
            # ax.set_xlabel('Number of FA Iterations', fontsize=12)
            # ax.set_ylabel('Completeness', fontsize=12)
            # ax.set_title(f'Completeness Curve - Density: {density}\n{plybase}\n{run_time}', fontsize=11)
            # ax.grid(True, alpha=0.3)
            # ax.legend()
            # ax.set_ylim([0, 1.05])
            
            # plt.tight_layout()
            
            # # 保存图片
            # save_dir = f"/home/hmf/work/FA_test/results/completeness_curves/"
            # os.makedirs(save_dir, exist_ok=True)
            # save_path = os.path.join(save_dir, f"completeness_density{density}_{run_time[:16].replace(':', '')}.png")
            # plt.savefig(save_path, dpi=150, bbox_inches='tight')
            # print(f"图片保存到: {save_path}")
            # plt.show()
            
        except Exception as e:
            print(f"❌ 处理运行 {run_time} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 存储当前密度的结果
    if density_results:
        all_results[density] = density_results




if all_results:
    saved_file = save_all_results(all_results)
    print(saved_file)

# =========================
# 7. 汇总比较所有密度的 Completeness
# =========================
print("\n" + "="*80)
print("汇总比较所有密度")
print("="*80)

if all_results:
    # 创建汇总图
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, run_time in enumerate(run_times):
        ax = axes[idx]
        
        for density in density_values:
            if density in all_results and run_time in all_results[density]:
                com_list = all_results[density][run_time]['completeness']
                iterations = range(1, len(com_list) + 1)
                
                # 取 overall completeness（如果是元组的话）
                if com_list and isinstance(com_list[0], (list, tuple)):
                    overall_vals = [c[1] if isinstance(c, (list, tuple)) else c for c in com_list]
                else:
                    overall_vals = com_list
                
                ax.plot(iterations, overall_vals, '-o', linewidth=2, 
                       label=f'Density {density}', markersize=4)
        
        ax.set_xlabel('Number of FA Iterations', fontsize=11)
        ax.set_ylabel('Overall Completeness', fontsize=11)
        ax.set_title(f'Run: {run_time[:16]}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim([0, 1.05])
    
    plt.suptitle('Completeness Comparison Across Different Target Densities', fontsize=14)
    plt.tight_layout()
    
    # 保存汇总图
    summary_path = os.path.join(save_dir, "completeness_summary_all_densities.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"汇总图保存到: {summary_path}")
    plt.show()
    
    # 打印统计摘要
    print("\n" + "="*80)
    print("最终 Completeness 统计")
    print("="*80)
    
    for density in density_values:
        if density in all_results:
            print(f"\n密度 {density}:")
            for run_time in run_times:
                if run_time in all_results[density]:
                    com_list = all_results[density][run_time]['completeness_module']
                    final_com = com_list[-1]
                    if isinstance(final_com, (list, tuple)):
                        print(f"  {run_time[:16]}: inside={final_com[0]:.4f}, overall={final_com[1]:.4f}")
                    else:
                        print(f"  {run_time[:16]}: {final_com:.4f}")
else:
    print("❌ 没有成功计算任何结果")

print("\n✅ 所有任务完成！")
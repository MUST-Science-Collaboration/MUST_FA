import numpy as np
from astropy.table import Table, vstack
import glob

# =========================
# 1. Galaxy 配置（核心）
# =========================
GALAXY_CONFIG = {
    'BGS': {
        'PRIORITY_INIT': 1,
        'type_id': 0,
        'time': 0.04,
        'NUMOBS_INIT': int(np.ceil(0.04 * 3.0)),
        'OBSCONDITIONS': 2,
        'DESI_TARGET': np.int64(2**0),
        'BGS_TARGET': np.int64(2**0),
    },
    'LRG': {
        'PRIORITY_INIT': 3,
        'type_id': 1,
        'time': 1.15,
        'NUMOBS_INIT': int(np.ceil(1.15 * 3.0)),
        'OBSCONDITIONS': 2,
        'DESI_TARGET': np.int64(2**0)
    },
    'ELG': {
        'PRIORITY_INIT': 5,
        'type_id': 2,
        'time': 0.32,
        'NUMOBS_INIT': int(np.ceil(0.32 * 3.0)),
        'OBSCONDITIONS': 2,
        'DESI_TARGET': np.int64(2**1)
    },
    'LBG': {
        'PRIORITY_INIT': 9,
        'type_id': 3,
        'time': lambda tab: np.where(tab['Z'] > 3.5, 5.8, 2.5),
        'NUMOBS_INIT': lambda tab: np.int32(np.ceil(tab['TIME'] * 3.0)),
        'OBSCONDITIONS': 1,
        'DESI_TARGET': np.int64(2**2)
    }  
    # 以后可以直接扩展
}

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

# =========================
# 4. 主流程
# =========================
def main():

    # ===== 参数 =====
    N = None
    RANDOM = False
    base_dir = "/home/hmf/work/FA_test/MUST_mock/"
    # ===== 读取（改为 npy）=====
    cats=[]
    gts = []
    for gt in GALAXY_CONFIG.keys():
        file = f"{base_dir}/MUST_{gt}_*.npy"
        file = glob.glob(file)[0]
        print(f"Found file: {file}")
        cat = load_npy_as_table(file, N=N, random=RANDOM)
        # gt = file.split('/')[-1].split('_')[1]
        print(gt)
        gts.append(gt)

        # ===== 添加信息 =====
        cat_o = add_galaxy_info(cat, gt, GALAXY_CONFIG)
        cats.append(cat_o)

    # ===== 合并 =====
    merged = vstack(cats, join_type='outer')
    npoints = len(merged)
    merged.add_column(np.arange(npoints).astype('int64') + 1, name='TARGETID')
    merged.add_column(np.random.uniform(low=0,high=1,size=npoints).astype('float64'), name='SUBPRIORITY')# add_column(np.array([0.]*npoints).astype('float64'), name='SUBPRIORITY')
    # # ===== 输出 =====
    output_file = f'{base_dir}/MUST_merged_{"_".join(gts)}.fits'
    merged.write(output_file, overwrite=True)

    

    print(f"Saved to {output_file}")
    print(merged.keys())
    return merged


# =========================
# 4. 主流程
# =========================
def main2():

    # ===== 参数 =====
    N = None
    RANDOM = False
    base_dir = "/home/hmf/work/FA_test/MUST_mock/"
    # ===== 读取（改为 npy）=====

    points = 0
    for gt in ['BGS']:# GALAXY_CONFIG.keys():
        file = f"{base_dir}/MUST_{gt}_*.npy"
        file = glob.glob(file)[0]
        print(f"Found file: {file}")
        cat = load_npy_as_table(file, N=N, random=RANDOM)
        # gt = file.split('/')[-1].split('_')[1]
        print(gt)

        # ===== 添加信息 =====
        cat_o = add_galaxy_info(cat, gt, GALAXY_CONFIG)
        # ===== 合并 =====

        npoints = len(cat_o)
        cat_o.add_column(np.arange(points, points+npoints).astype('int64') + 1, name='TARGETID')
        cat_o.add_column(np.random.uniform(low=0,high=1,size=npoints).astype('float64'), name='SUBPRIORITY')
        points+= len(cat_o)

        # # ===== 输出 =====
        output_file = f'{base_dir}/MUST_{gt}_0-360.fits'
        cat_o.write(output_file, overwrite=True)

    

        print(f"Saved to {output_file}")


# =========================
# 5. 执行
# =========================
if __name__ == "__main__":
    main2()
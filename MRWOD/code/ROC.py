import os
import numpy as np
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def evaluation_outlier(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return fpr * 100, tpr * 100

# ======================= 方法定义 ============================
outlier_method = [
    'MIX', 'LPOD', 'ITB', 'VARE', 'ODGrCR', 'FGAS', 'VOS', 'WNINOD', 'WFRDA', 'MRWOD'
]

outlier_method_name = outlier_method.copy()

line_styles = ['r-', 'g-', 'c-', 'm-', 'b-', 'y-',
                'r-', 'g-', 'm-', 'b-', 'k-', 'y-']
marker_list = ["*", "x", "d", "+", "p", "^", "o", "o", ">", "s", "", ">", "h", "D", "v"]

# ======================= 数据集列表 ============================
datasets = [
        'annthyroid', 'heart270_2_16_variant1','hepatitis_2_9_variant1',
        'horse_1_12_variant1','vote_republican_29_variant1',
        'yeast_ERL_5_variant1'
]

result_root_dir = r'D:\Experimental_results'

# 创建保存路径
save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', '_figures', 'ROC')
os.makedirs(save_dir, exist_ok=True)

# 数据集所在目录
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# ======================= 主循环 ============================
for data_nameori in datasets:
    data_arr_name = data_nameori

    # 加载原始数据（含标签）
    data_mat_path = os.path.join(data_dir, f'{data_nameori}.mat')
    if not os.path.exists(data_mat_path):
        print(f"❌ 未找到原始数据: {data_mat_path}")
        continue

    try:
        data_mat = scio.loadmat(data_mat_path)

        # Determine how ground truth labels are stored based on memory
        if 'trandata' in data_mat:
            labels = data_mat['trandata'][:, -1].flatten()
        elif 'y' in data_mat:
            labels = data_mat['y'].flatten()
        else:
            print(f"❌ 数据 {data_nameori} 缺少标签变量")
            continue
    except Exception as e:
        print(f"❌ 加载数据崩溃 {data_mat_path}: {e}")
        continue

    # 开始绘图
    plt.figure(figsize=(6, 5))

    for i, method in enumerate(outlier_method):
        method_file_path = None

        # 特殊处理MRWOD，直接从results里读取
        if method == 'MRWOD':
            nanread_path_mat = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', data_nameori, f'{data_nameori}.mat')
            nanread_path_xls = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', data_nameori, f'{data_nameori}.xls')
            if os.path.exists(nanread_path_mat):
                method_file_path = nanread_path_mat
            elif os.path.exists(nanread_path_xls):
                method_file_path = nanread_path_xls
            else:
                print(f"⚠️ 缺少方法结果文件:{data_nameori}_{method}")
                continue
        else:
            # 遍历目录查找匹配文件
            for root, dirs, files in os.walk(result_root_dir):
                for file in files:
                    if (file.lower() == f'{data_nameori.lower()}_{method.lower()}.xlsx'
                    or file.lower() == f'{data_nameori.lower()}_{method.lower()}.mat'
                    or file.lower() == f'{data_nameori.lower()}_{method.lower()}.xls'):
                        method_file_path = os.path.join(root, file)
                        break
                if method_file_path:
                    break

        if not method_file_path or not os.path.isfile(method_file_path):
            print(f"⚠️ 缺少方法结果文件:{data_nameori}_{method}")
            continue

        try:
            if method_file_path.endswith('.mat'):
                result_mat = scio.loadmat(method_file_path)
                found_key = None
                for key in result_mat.keys():
                    if not key.startswith('__'):
                        found_key = key
                        break
                scores = result_mat[found_key][:, 0]
            elif method_file_path.endswith('.xlsx') or method_file_path.endswith('.xls'):
                # 读取Excel
                result_df = pd.read_excel(method_file_path)
                scores = result_df.iloc[:, 0].values
        except Exception as e:
            print(f"❌ 读取文件崩溃 {method_file_path}: {e}")
            continue

        fpr, tpr = evaluation_outlier(scores, labels)

        plt.plot(fpr, tpr, line_styles[i % len(line_styles)],
                 marker=marker_list[i % len(marker_list)],
                 markevery=0.1,
                 markersize=5,
                 linewidth=1.2,
                 label=outlier_method_name[i])

    plt.xlabel('FPR (%)', fontsize=12)
    plt.ylabel('TPR (%)', fontsize=12)
    plt.xlim([1, 100])
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(False)

    # 保存图像
    filename = os.path.join(save_dir, f'{data_arr_name}_ROC')
    plt.savefig(f'{filename}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)
    plt.savefig(f'{filename}.svg', format='svg', bbox_inches='tight', pad_inches=0.01)
    plt.savefig(f'{filename}.eps', format='eps', bbox_inches='tight', pad_inches=0.01)
    plt.close()

print("✅ ROC画图执行完毕")

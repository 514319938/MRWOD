import os
import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn.metrics import precision_recall_curve

def evaluation_best_f1(scores, labels):
    """
    计算最佳 F1 分数 (F1-Max)。
    通过 precision_recall_curve 遍历所有得分作为阈值，找到能使 F1 最大的点。
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores, pos_label=1)
    
    # 避免分母为 0 的情况
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(numerator), where=(denom != 0))
    
    return np.max(f1_scores)

def main():
    # ======================= 方法与数据集定义 ============================
    outlier_method = [
        'MIX', 'LPOD', 'ITB', 'VARE', 'ODGrCR', 'FGAS', 'VOS', 'WNINOD', 'WFRDA', 'MRWOD'
    ]
    
    datasets = [
        'annthyroid', 'heart270_2_16_variant1','hepatitis_2_9_variant1',
        'horse_1_12_variant1','vote_republican_29_variant1',
        'yeast_ERL_5_variant1'
    ]

    # ======================= 路径配置 ============================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    data_dir = os.path.join(project_root, 'data')
    result_root_dir = r'D:\Experimental_results' # 沿用你 ROC.py 中的基线数据路径
    save_dir = os.path.join(project_root, 'results')
    os.makedirs(save_dir, exist_ok=True)

    # 结果存储字典，格式: {dataset_name: {method_name: f1_score}}
    all_f1_results = []

    # ======================= 主循环 ============================
    for data_nameori in datasets:
        print(f"正在评估数据集: {data_nameori}...")
        
        # 加载原始数据获取真实标签
        data_mat_path = os.path.join(data_dir, f'{data_nameori}.mat')
        if not os.path.exists(data_mat_path):
            print(f"   未找到原始数据: {data_mat_path}")
            continue

        try:
            data_mat = scio.loadmat(data_mat_path)
            if 'trandata' in data_mat:
                labels = data_mat['trandata'][:, -1].flatten()
            elif 'y' in data_mat:
                labels = data_mat['y'].flatten()
            else:
                print(f"   数据 {data_nameori} 缺少标签变量")
                continue
        except Exception as e:
            print(f"   加载数据崩溃 {data_mat_path}: {e}")
            continue

        row_data = {'dataset': data_nameori}

        # 遍历算法计算 F1
        for method in outlier_method:
            method_file_path = None

            # 特殊处理 NaNREAD 的路径
            if method == 'NaNREAD':
                nanread_path_mat = os.path.join(save_dir, data_nameori, f'{data_nameori}.mat')
                nanread_path_xls = os.path.join(save_dir, data_nameori, f'{data_nameori}.xls')
                if os.path.exists(nanread_path_mat):
                    method_file_path = nanread_path_mat
                elif os.path.exists(nanread_path_xls):
                    method_file_path = nanread_path_xls
            else:
                # 遍历目录查找基线算法文件
                for root, dirs, files in os.walk(result_root_dir):
                    for file in files:
                        if file.lower() in [f'{data_nameori.lower()}_{method.lower()}.xlsx',
                                            f'{data_nameori.lower()}_{method.lower()}.mat',
                                            f'{data_nameori.lower()}_{method.lower()}.xls']:
                            method_file_path = os.path.join(root, file)
                            break
                    if method_file_path:
                        break

            if not method_file_path or not os.path.isfile(method_file_path):
                row_data[method] = np.nan
                continue

            # 读取得分
            try:
                if method_file_path.endswith('.mat'):
                    result_mat = scio.loadmat(method_file_path)
                    found_key = next((key for key in result_mat.keys() if not key.startswith('__')), None)
                    scores = result_mat[found_key][:, 0]
                elif method_file_path.endswith(('.xlsx', '.xls')):
                    result_df = pd.read_excel(method_file_path)
                    scores = result_df.iloc[:, 0].values
                
                # 计算 F1-Max
                best_f1 = evaluation_best_f1(scores, labels)
                row_data[method] = best_f1
                
            except Exception as e:
                print(f"   计算 {method} 崩溃: {e}")
                row_data[method] = np.nan

        all_f1_results.append(row_data)

    # ======================= 保存结果 ============================
    results_df = pd.DataFrame(all_f1_results)
    output_path = os.path.join(save_dir, 'all_F1_results.xlsx')
    
    # 按照与其他表格类似的格式保存
    results_df.to_excel(output_path, index=False)
    print(f"\n F1 分数计算完毕，汇总结果已保存至: {output_path}")

if __name__ == "__main__":
    main()
import os
import pandas as pd
import numpy as np
import xlwt

def main():
    # 1. 固定你论文的16个数据集
    target_datasets = [
        'annthyroid',
        'bands_band_42_variant1',
        'cardio',
        'chess_nowin_145_variant1',
        'chess_nowin_87_variant1',
        'creditA_plus_42_variant1',
        'german_1_14_variant1',
        'heart270_2_16_variant1',
        'hepatitis_2_9_variant1',
        'horse_1_12_variant1',
        'ionosphere_b_24_variant1',
        'mushroom_p_365_variant1',
        'pageblocks_1_258_variant1',
        'sick_sick_35_variant1',
        'vote_republican_29_variant1',
        'yeast_ERL_5_variant1'
    ]
    # 2. 需要保留的算法列
    target_algorithms = [
        'dataset',
        'MIX',
        'LPOD',
        'ITB',
        'VarE',
        'ODGrCR',
        'FGAS',
        'VOS',
        'WNINOD',
        'WFRDA',
        'MRWOD'
    ]

    base_results_path = 'results/all_outlier_result-20251025.xls'
    if not os.path.exists(base_results_path):
        print(f"Error: {base_results_path} not found.")
        return
        
    df = pd.read_excel(base_results_path)
    # 数据集名称清洗
    df['dataset'] = df['dataset'].astype(str).str.strip()
    # 只保留指定16个数据集，并强制按论文顺序排序
    df = df[df['dataset'].isin(target_datasets)]
    df['dataset'] = pd.Categorical(df['dataset'], categories=target_datasets, ordered=True)
    df = df.sort_values('dataset').reset_index(drop=True)
    print(f"已筛选数据集：共{len(df)}个")
    
    # 读取MRWOD的AUC结果【增加详细调试打印】
    mrwod_aucs = []
    for _, row in df.iterrows():
        dataset_name = row['dataset']
        res_file = os.path.join('results', dataset_name, f"{dataset_name}.xls")
        print(f"\n===== 数据集：{dataset_name} =====")
        print(f"尝试读取路径：{res_file}")
        file_exists = os.path.exists(res_file)
        print(f"文件是否存在：{file_exists}")

        auc_val = np.nan
        if file_exists:
            try:
                temp_df = pd.read_excel(res_file)
                print(f"该表格所有列名：{list(temp_df.columns)}")
                if 'opt_ROC_AUC' in temp_df.columns:
                    auc_val = temp_df['opt_ROC_AUC'].iloc[0]
                    print(f"成功读取AUC：{auc_val}")
                else:
                    print("错误：表格中没有 opt_ROC_AUC 这一列")
            except Exception as e:
                print(f"读取文件异常：{str(e)}")
        else:
            print("错误：文件不存在，AUC为空")
        mrwod_aucs.append(auc_val)

    # 写入MRWOD列
    df['MRWOD'] = mrwod_aucs

    # 只保留你指定的列，丢弃其他所有算法
    exist_cols = [col for col in target_algorithms if col in df.columns]
    df = df[exist_cols]
    print(f"\n最终保留算法列：{exist_cols}")

    # 导出xls文件
    out_path = 'results/all_outlier_result_filtered.xls'
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Sheet1')
    cols = list(df.columns)

    # 写入表头
    for col_idx, col_name in enumerate(cols):
        ws.write(0, col_idx, str(col_name))

    # 写入表格数据
    for row_idx, row in df.iterrows():
        for col_idx, col_name in enumerate(cols):
            val = row[col_name]
            if pd.isna(val):
                ws.write(row_idx + 1, col_idx, "")
            elif isinstance(val, (int, float, np.integer, np.floating)):
                ws.write(row_idx + 1, col_idx, float(val))
            else:
                ws.write(row_idx + 1, col_idx, str(val))

    wb.save(out_path)
    print(f"\n筛选完成！结果保存至：{out_path}")
    print(f"保留数据集：{len(df)} 个 | 保留算法：{len(cols)-1} 个")

if __name__ == '__main__':
    main()
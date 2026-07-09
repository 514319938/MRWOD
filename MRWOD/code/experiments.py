import os
import time
import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn.metrics import roc_auc_score
import warnings

# Suppress warnings that might clutter the output
warnings.filterwarnings("ignore")

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')

    os.makedirs(results_dir, exist_ok=True)

    # Dynamically find all datasets in the data folder
    target_datasets = []
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith('.mat'):
                target_datasets.append(f[:-4])

    # Import mrwod_anomaly_detection from MRWOD.py
    from MRWOD import mrwod_anomaly_detection

    # Grid search for lmbda from 0.1 to 1.5, step 0.05
    # Due to floating point precision issues with np.arange, we use linspace or multiply
    lmbda_grid = np.arange(0.1, 1.5 + 0.001, 0.05)

    print(f"Starting experiments on {len(target_datasets)} datasets...")

    for dataset_name in target_datasets:
        mat_path = os.path.join(data_dir, f"{dataset_name}.mat")

        print(f"\nProcessing dataset: {dataset_name}")

        try:
            # Load data
            mat = scio.loadmat(mat_path)

            if 'trandata' in mat:
                data = mat['trandata']
                X = data[:, :-1]
                y = data[:, -1]
            elif 'y' in mat and 'X' in mat:
                X = mat['X']
                y = mat['y'].flatten()
            else:
                print(f"  Cannot find data keys in {dataset_name}.mat. Skipping...")
                continue

            nominal_indices = [] # Assuming no nominal attributes pre-defined, or already processed

            best_auc = -1
            best_lmbda = -1
            best_scores = None
            best_time = -1

            print(f"  Shape: {X.shape}, Searching optimal lmbda...")

            for lmbda in lmbda_grid:
                lmbda = round(lmbda, 2)

                start_time = time.time()
                try:
                    outliers, scores = mrwod_anomaly_detection(
                        X=X,
                        nominal_indices=nominal_indices,
                        lmbda=lmbda,
                        mu=0.8,
                        sorting_mode='desc'
                    )

                    elapsed_time = time.time() - start_time

                    # Calculate ROC AUC
                    auc = roc_auc_score(y, scores)

                    if auc > best_auc:
                        best_auc = auc
                        best_lmbda = lmbda
                        best_scores = scores
                        best_time = elapsed_time

                except Exception as e:
                    # Catch errors like division by zero for certain lambda values
                    # print(f"  Error with lmbda={lmbda}: {e}")
                    pass

            if best_scores is not None:
                print(f"  Best lmbda: {best_lmbda}, Best AUC: {best_auc:.4f}, Time: {best_time:.4f}s")

                # Format scores to 4 decimal places without scientific notation
                formatted_scores = [float(f"{s:.4f}") for s in best_scores]
                formatted_auc = float(f"{best_auc:.4f}")
                formatted_time = float(f"{best_time:.4f}")

                # Create DataFrame for results
                # opt_out_scores is a column of length n
                # opt_ROC_AUC and opt_time are columns of length n, but only the first row has values

                n = len(formatted_scores)

                df = pd.DataFrame({
                    'opt_out_scores': formatted_scores,
                    'opt_ROC_AUC': [formatted_auc] + [np.nan] * (n - 1),
                    'opt_time': [formatted_time] + [np.nan] * (n - 1)
                })

                # Save to results/dataset_name/dataset_name.xls
                ds_result_dir = os.path.join(results_dir, dataset_name)
                os.makedirs(ds_result_dir, exist_ok=True)

                out_path = os.path.join(ds_result_dir, f"{dataset_name}.xls")

                import xlwt
                wb = xlwt.Workbook()
                ws = wb.add_sheet('Sheet1')

                cols = list(df.columns)
                for col_idx, col_name in enumerate(cols):
                    ws.write(0, col_idx, str(col_name))

                for row_idx, row in df.iterrows():
                    for col_idx, col_name in enumerate(cols):
                        val = row[col_name]
                        if pd.isna(val):
                            ws.write(row_idx + 1, col_idx, "")
                        else:
                            ws.write(row_idx + 1, col_idx, float(val))

                wb.save(out_path)
                print(f"  Results saved to {out_path}")
            else:
                print(f"  Failed to process {dataset_name} for all lmbda values.")

        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")

if __name__ == '__main__':
    main()

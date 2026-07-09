import numpy as np
import pandas as pd


def mrwod_anomaly_detection(X, nominal_indices, lmbda=1.0, mu=0.8, d=0.1, max_iter=1000, tol=1e-7, sorting_mode='desc'):
    """
    基于马尔可夫随机游走并融合多属性邻域粒信息的异常检测(MRWOD)算法
    严格对齐论文【3.1 融合多属性邻域粒信息的马尔可夫随机游走】行归一化标准版

    修复说明：
    1. 修复 numpy.int64 与 list.remove 类型不兼容报错
    2. 新增参数合法性校验，避免静默错误
    3. 拆分防零常量语义，避免混用混乱
    4. 移除ANODM循环冗余变量，降低错位风险
    5. 补充阻尼项公式注释，明确d为重启概率
    """
    X = np.array(X, dtype=object)
    n, m = X.shape
    numerical_indices = [i for i in range(m) if i not in nominal_indices]

    # ========== 新增：入口参数合法性校验 ==========
    # 校验标称属性索引合法性
    nominal_indices = list(set(nominal_indices))
    for idx in nominal_indices:
        if idx < 0 or idx >= m:
            raise ValueError(f"标称属性索引 {idx} 超出数据列范围 [0, {m-1}]")
    # 校验排序模式
    if sorting_mode not in ('asc', 'desc'):
        raise ValueError("sorting_mode 只能取值 'asc' 或 'desc'")
    # 校验阻尼参数范围
    if not (0 <= d <= 1):
        raise ValueError("阻尼/重启参数 d 必须在 [0, 1] 区间内")
    if abs(lmbda) < 1e-12:
        raise ValueError("lmbda 不能为0，否则邻域半径无意义")

    # 常量语义拆分
    eps_log = 1e-12    # 专门用于对数防零
    eps_zero = 1e-12   # 专门用于除零保护、判零阈值

    # ================= 步骤一：数据标准化处理 (对应论文公式 1) =================
    X_norm = X.copy()
    for j in numerical_indices:
        col = X[:, j].astype(float)
        col_min = np.min(col)
        col_max = np.max(col)
        if (col_max - col_min) > eps_zero:
            X_norm[:, j] = (col - col_min) / (col_max - col_min)
        else:
            X_norm[:, j] = np.zeros(n)

    # 预计算单属性的距离矩阵与邻域半径 (对应论文公式 2, 3)
    d_single = np.zeros((m, n, n))
    eps_single = np.zeros(m)

    for j in range(m):
        if j in nominal_indices:
            col = X_norm[:, j]
            d_single[j] = (col[:, None] != col[None, :]).astype(float)
            eps_single[j] = 0.0
        else:
            col = X_norm[:, j].astype(float)
            d_single[j] = np.abs(col[:, None] - col[None, :])
            std_j = np.std(col)
            eps_single[j] = std_j / lmbda

    # ================= 步骤二：计算单属性邻域密度与邻域熵 (对应论文公式 7, 8) =================
    ND_single = np.zeros((m, n))
    NE_single = np.zeros(m)

    for j in range(m):
        delta = (d_single[j] <= eps_single[j]).astype(float)
        ND_single[j] = np.sum(delta, axis=1) / n
        ND_single[j] = np.clip(ND_single[j], eps_log, 1.0)
        NE_single[j] = -np.mean(np.log2(ND_single[j]))

    # ================= 步骤三：构建属性序列及属性集序列 (对应论文公式 9, 10) =================
    # 修复：转成Python原生int列表，避免list.remove因np.int64类型不匹配报错
    if sorting_mode == 'asc':
        sorted_features = np.argsort(NE_single).tolist()
    else:
        sorted_features = np.argsort(NE_single)[::-1].tolist()

    QS = []
    current_set = list(range(m))
    QS.append(list(current_set))
    for k in range(m - 1):
        current_set.remove(sorted_features[k])
        QS.append(list(current_set))

    # =========================================================================
    # 【论文 3.1 节核心：融合多属性邻域粒信息的马尔可夫随机游走】
    # =========================================================================

    # 1. 单属性邻域对象差异度量 (SNODM) —— 对应论文公式 (13)
    sum_NE_single = np.sum(NE_single)
    if sum_NE_single > eps_zero:
        SW = NE_single / sum_NE_single
    else:
        SW = np.ones(m) / m

    SNODM = np.zeros((n, n))
    for j in range(m):
        nd_diff = np.abs(ND_single[j][:, None] - ND_single[j][None, :])
        SNODM += SW[j] * nd_diff

    # 2. 属性集邻域对象差异度量 (ANODM) —— 对应论文公式 (14)
    ANODM = np.zeros((n, n))
    NE_QS = np.zeros(len(QS))
    ND_QS = []

    for t, Q_t in enumerate(QS):
        MHMOM_Qt = np.mean(d_single[Q_t], axis=0)
        eps_Qt = np.mean(eps_single[Q_t])

        delta_Qt = (MHMOM_Qt <= eps_Qt).astype(float)
        ND_Qt = np.sum(delta_Qt, axis=1) / n
        ND_Qt = np.clip(ND_Qt, eps_log, 1.0)
        ND_QS.append(ND_Qt)
        NE_QS[t] = -np.mean(np.log2(ND_Qt))

    sum_NE_QS = np.sum(NE_QS)
    if sum_NE_QS > eps_zero:
        AW = NE_QS / sum_NE_QS
    else:
        AW = np.ones(len(QS)) / len(QS)

    # 修复：移除冗余的Q_t变量，仅按索引遍历，避免顺序错位
    for t in range(len(QS)):
        nd_diff_Qt = np.abs(ND_QS[t][:, None] - ND_QS[t][None, :])
        ANODM += AW[t] * nd_diff_Qt

    # 3. 终端邻域对象差异度量矩阵 (NODM) 与 规范化状态转移矩阵 P —— 对应论文公式 (15)
    NODM = SNODM + ANODM

    # 行归一化得到转移概率矩阵
    row_sums = np.sum(NODM, axis=1, keepdims=True)
    row_sums[row_sums < eps_zero] = 1.0
    P = NODM / row_sums

    # ================= 迭代计算马尔可夫平稳分布 (公式18) =================
    # 注意：此处 d 为【重启概率】，即随机游走有d的概率跳回均匀分布
    # 若你的论文中d为【阻尼因子(继续游走概率)】，请调换为：(1-d)*I_vec + d*np.dot(nu, P)
    nu = np.ones(n) / n
    I_vec = np.ones(n) / n
    converge_flag = False

    for iteration in range(max_iter):
        nu_next = d * I_vec + (1 - d) * np.dot(nu, P)
        diff = np.sum(np.abs(nu_next - nu))
        if diff < tol:
            nu = nu_next
            converge_flag = True
            break
        nu = nu_next

    # ================= 计算异常得分 MRWOF (公式19) =================
    nu_min = np.min(nu)
    nu_max = np.max(nu)
    if (nu_max - nu_min) > eps_zero:
        MRWOF = (nu - nu_min) / (nu_max - nu_min)
    else:
        MRWOF = np.zeros(n)

    outliers = np.where(MRWOF > mu)[0]

    return outliers, MRWOF


if __name__ == "__main__":
    # 标准测试用例
    raw_data = [
        ["c", 10, 0.7],
        ["b", 6, 0.3],
        ["d", 2, 0.5],
        ["b", 3, 0.3],
        ["b", 7, 0.4],
        ["d", 3, 0.6]
    ]
    data_mat = np.array(raw_data, dtype=object)
    nominal_cols = [0]

    outliers, scores = mrwod_anomaly_detection(
        data_mat,
        nominal_indices=nominal_cols,
        lmbda=1.0,
        mu=0.8,
        tol=1e-7,
        sorting_mode='desc'
    )

    print("===== MRWOD 异常检测结果 =====")
    for i, score in enumerate(scores):
        is_outlier = "【异常】" if i in outliers else "正常"
        print(f"样本 x{i + 1}: 得分 = {score:.4f} -> 状态: {is_outlier}")
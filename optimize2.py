import basico
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import basico.task_parameterestimation as pe
from utils import HC_state, IgG_state, ALL_PARAMS, STATE_NAMES, rhs, ALIASES  # 直接从utils导入

print("正在初始化 COPASI 模型...")
basico.new_model(name='IgG4_Model_Full')
basico.set_model_unit(time_unit='h')  # 只设置时间单位，因为物种单位无法分开设置

# ============================================================================
# PART A: 参数定义 (直接使用 utils.py 中的 param)
# ============================================================================
print("正在添加参数...")
rng = np.random.default_rng(seed=42)
p_random = {name: float(rng.uniform(0.0, 100.0)) for name in ALL_PARAMS}

# 添加所有参数到模型
for p_name, p_val in p_random.items():
    basico.add_parameter(p_name, initial_value=p_val)

# ============================================================================
# PART B: 状态变量与初值 (直接使用 utils.py 中的 HC_state)
# ============================================================================
print("正在添加状态变量...")
y0 = HC_state() # 获取 utils.py 中的初始状态

for s_name, s_val in y0.items():
    basico.add_species(s_name, initial_concentration=s_val)

# ============================================================================
# PART C: 定义微分方程
# ============================================================================
print("正在添加微分方程...")

# 注意：这里需要保持原有的字符串形式方程，因为 utils.py 里是 Python 函数 rhs()
# 为了脚本能运行，这里还是必须显式写出这些方程的字符串形式。
# (这部分保持上一版代码不变，因为必须把 Python 逻辑翻译成 COPASI 表达式)

# 1. Antigen
Ag_Max = 100.0
Ag_Center = 125.0
Ag_Width = 5.0
sigmoid_formula = f"{Ag_Max} / (1 + exp(-(Time - {Ag_Center}) / {Ag_Width}))"
basico.set_species('Antigen', status='assignment', expression=sigmoid_formula)
test_sim = basico.run_time_course(duration=300, intervals=300, method='LSODA')
print(test_sim[['Antigen']].iloc[100:150:10])

# 2. nDC
# 分离生成和消耗反应

# nDC 生成反应 (Production)
expr_nDC_prod = "k_nDC_f * nDC * (1 - nDC / k_nDC_m)"
basico.add_reaction('nDC_production', '-> nDC', function=expr_nDC_prod)

# nDC 消耗反应 (Consumption) - 转化为mDC
expr_nDC_to_mDC_1 = "k_mDC_Antigen_f * Antigen * nDC * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10))"
basico.add_reaction('nDC_to_mDC_Antigen', 'nDC ->', function=expr_nDC_to_mDC_1)

expr_nDC_to_mDC_2 = "k_mDC_GMCSF_f * Antigen * nDC * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10))"
basico.add_reaction('nDC_to_mDC_GMCSF', 'nDC ->', function=expr_nDC_to_mDC_2)

# nDC 消耗反应 (Consumption) - 转化为pDC
expr_nDC_to_pDC = "k_pDC_Antigen_f * nDC * Antigen"
basico.add_reaction('nDC_to_pDC', 'nDC ->', function=expr_nDC_to_pDC)

# nDC 降解反应 (Decay)
expr_nDC_decay = "k_nDC_d * nDC"
basico.add_reaction('nDC_decay', 'nDC ->', function=expr_nDC_decay)

# 3. mDC
# 分离生成和降解反应，避免速率函数出现负值

# mDC 生成反应 (Production)
expr_mDC_prod = "k_mDC_Antigen_f * Antigen * nDC * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)) + k_mDC_GMCSF_f * Antigen * nDC * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)) + k_mDC_f * mDC * (1 - mDC / k_mDC_m)"
basico.add_reaction('mDC_production', '-> mDC', function=expr_mDC_prod)

# mDC 降解反应 (Decay)
expr_mDC_decay = "k_mDC_d * mDC"
basico.add_reaction('mDC_decay', 'mDC ->', function=expr_mDC_decay)

# 4. GMCSF
# 分离生成和消耗反应

# GMCSF 生成反应 (Production)
expr_GMCSF_prod_1 = "k_GMCSF_Th2_f * Th2"
basico.add_reaction('GMCSF_production_Th2', '-> GMCSF', function=expr_GMCSF_prod_1)

expr_GMCSF_prod_2 = "k_GMCSF_Th2_Antigen_f * Th2 * Antigen"
basico.add_reaction('GMCSF_production_Th2_Antigen', '-> GMCSF', function=expr_GMCSF_prod_2)

expr_GMCSF_prod_3 = "k_GMCSF_act_NK_f * act_NK"
basico.add_reaction('GMCSF_production_act_NK', '-> GMCSF', function=expr_GMCSF_prod_3)

# GMCSF 消耗反应 (Consumption) - 被mDC利用
expr_GMCSF_consumption = "k_mDC_GMCSF_d * Antigen * nDC * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10))"
basico.add_reaction('GMCSF_consumption_by_mDC', 'GMCSF ->', function=expr_GMCSF_consumption)

# GMCSF 降解反应 (Decay)
expr_GMCSF_decay = "k_GMCSF_d * GMCSF"
basico.add_reaction('GMCSF_decay', 'GMCSF ->', function=expr_GMCSF_decay)

# 5. pDC
# 分离生成和消耗反应

# pDC 生成反应 (Production) - 从nDC转化
expr_pDC_from_nDC = "k_pDC_Antigen_f * nDC * Antigen"
basico.add_reaction('pDC_from_nDC', '-> pDC', function=expr_pDC_from_nDC)

# pDC 增殖反应 (Proliferation)
expr_pDC_proliferation = "k_pDC_f * pDC * (1 - pDC / k_pDC_m)"
basico.add_reaction('pDC_proliferation', '-> pDC', function=expr_pDC_proliferation)

# pDC 降解反应 (Decay)
expr_pDC_decay = "k_pDC_d * pDC"
basico.add_reaction('pDC_decay', 'pDC ->', function=expr_pDC_decay)

# 6. IL_33
# 分离生成和消耗反应

# IL_33 生成反应 (Production)
expr_IL_33_prod = "k_IL_33_pDC_f * pDC"
basico.add_reaction('IL_33_production', '-> IL_33', function=expr_IL_33_prod)

# IL_33 消耗反应 (Consumption)
expr_IL_33_consumption = "k_act_CD4_IL_33_d * act_CD4 * IL_33 / (k_Th2_IL_33_m + IL_33)"
basico.add_reaction('IL_33_consumption_by_act_CD4', 'IL_33 ->', function=expr_IL_33_consumption)

# IL_33 降解反应 (Decay)
expr_IL_33_decay = "k_IL_33_d * IL_33"
basico.add_reaction('IL_33_decay', 'IL_33 ->', function=expr_IL_33_decay)

# 7. IL_6
# 分离生成和消耗反应

# IL_6 生成反应 (Production)
expr_IL_6_prod_1 = "k_IL_6_pDC_f * pDC"
basico.add_reaction('IL_6_production_pDC', '-> IL_6', function=expr_IL_6_prod_1)

expr_IL_6_prod_2 = "k_IL_6_mDC_f * mDC"
basico.add_reaction('IL_6_production_mDC', '-> IL_6', function=expr_IL_6_prod_2)

expr_IL_6_prod_3 = "k_IL_6_TFH_f * TFH * (k_TFH_nTreg_m / (nTreg + k_TFH_nTreg_m))"
basico.add_reaction('IL_6_production_TFH', '-> IL_6', function=expr_IL_6_prod_3)

# IL_6 消耗反应 (Consumption)
expr_IL_6_consumption = "k_TFH_IL_6_d * act_CD4 * IL_6 / (k_TFH_IL_6_m + IL_6)"
basico.add_reaction('IL_6_consumption_by_act_CD4', 'IL_6 ->', function=expr_IL_6_consumption)

# IL_6 降解反应 (Decay)
expr_IL_6_decay = "k_IL_6_d * IL_6"
basico.add_reaction('IL_6_decay', 'IL_6 ->', function=expr_IL_6_decay)

# 8. IL_12
# 分离生成和消耗反应

# IL_12 生成反应 (Production)
expr_IL_12_prod = "k_IL_12_mDC_f * mDC"
basico.add_reaction('IL_12_production', '-> IL_12', function=expr_IL_12_prod)

# IL_12 消耗反应 (Consumption)
expr_IL_12_consumption = "k_act_NK_IL_12_d * NK * IL_12 / (IL_12 + k_act_NK_IL_12_m)"
basico.add_reaction('IL_12_consumption_by_NK', 'IL_12 ->', function=expr_IL_12_consumption)

# IL_12 降解反应 (Decay)
expr_IL_12_decay = "k_IL_12_d * IL_12"
basico.add_reaction('IL_12_decay', 'IL_12 ->', function=expr_IL_12_decay)

# 9. IL_15
# 分离生成和消耗反应

# IL_15 生成反应 (Production)
expr_IL_15_prod_1 = "k_IL_15_f"
basico.add_reaction('IL_15_production_basal', '-> IL_15', function=expr_IL_15_prod_1)

expr_IL_15_prod_2 = "k_IL_15_Antigen_f * Antigen"
basico.add_reaction('IL_15_production_Antigen', '-> IL_15', function=expr_IL_15_prod_2)

# IL_15 消耗反应 (Consumption)
expr_IL_15_consumption_1 = "k_naive_CD4_IL_15_d * naive_CD4 * IL_15 / (k_naive_CD4_IL_15_m + IL_15)"
basico.add_reaction('IL_15_consumption_by_naive_CD4', 'IL_15 ->', function=expr_IL_15_consumption_1)

expr_IL_15_consumption_2 = "k_act_CD4_IL_15_d * act_CD4 * IL_15 / (k_act_CD4_IL_15_m + IL_15)"
basico.add_reaction('IL_15_consumption_by_act_CD4', 'IL_15 ->', function=expr_IL_15_consumption_2)

# IL_15 降解反应 (Decay)
expr_IL_15_decay = "k_IL_15_d * IL_15"
basico.add_reaction('IL_15_decay', 'IL_15 ->', function=expr_IL_15_decay)

# 10. IL_7
# 分离生成和消耗反应

# IL_7 生成反应 (Production)
expr_IL_7_prod = "k_IL_7_f"
basico.add_reaction('IL_7_production', '-> IL_7', function=expr_IL_7_prod)

# IL_7 消耗反应 (Consumption)
expr_IL_7_consumption_1 = "k_naive_CD4_IL_7_d * naive_CD4 * IL_7 / (k_naive_CD4_IL_7_m + IL_7)"
basico.add_reaction('IL_7_consumption_by_naive_CD4', 'IL_7 ->', function=expr_IL_7_consumption_1)

expr_IL_7_consumption_2 = "k_act_CD4_IL_7_d * act_CD4 * IL_7 / (k_act_CD4_IL_7_m + IL_7)"
basico.add_reaction('IL_7_consumption_by_act_CD4', 'IL_7 ->', function=expr_IL_7_consumption_2)

# IL_7 降解反应 (Decay)
expr_IL_7_decay = "k_IL_7_d * IL_7"
basico.add_reaction('IL_7_decay', 'IL_7 ->', function=expr_IL_7_decay)

# 11. IFN1
# 分离生成和消耗反应

# IFN1 生成反应 (Production)
expr_IFN1_prod = "k_IFN1_pDC_f * pDC"
basico.add_reaction('IFN1_production', '-> IFN1', function=expr_IFN1_prod)

# IFN1 消耗反应 (Consumption)
expr_IFN1_consumption_1 = "k_act_CD4_IFN1_d * act_CD4 * IFN1 / (k_IFN1_CD4_CTL_m + IFN1)"
basico.add_reaction('IFN1_consumption_by_act_CD4', 'IFN1 ->', function=expr_IFN1_consumption_1)

expr_IFN1_consumption_2 = "k_act_NK_IFN1_d * NK * IFN1 / (IFN1 + k_act_NK_IFN1_m)"
basico.add_reaction('IFN1_consumption_by_NK', 'IFN1 ->', function=expr_IFN1_consumption_2)

# IFN1 降解反应 (Decay)
expr_IFN1_decay = "k_IFN1_d * IFN1"
basico.add_reaction('IFN1_decay', 'IFN1 ->', function=expr_IFN1_decay)

# 12. IL_1
# 分离生成和消耗反应

# IL_1 生成反应 (Production)
expr_IL_1_prod = "k_IL_1_mDC_f * mDC"
basico.add_reaction('IL_1_production', '-> IL_1', function=expr_IL_1_prod)

# IL_1 降解反应 (Decay)
expr_IL_1_decay = "k_IL_1_d * IL_1"
basico.add_reaction('IL_1_decay', 'IL_1 ->', function=expr_IL_1_decay)

# 13. IL_2
# 分离生成和消耗反应

# IL_2 生成反应 (Production)
expr_IL_2_prod_1 = "k_IL_2_act_CD4_f * act_CD4"
basico.add_reaction('IL_2_production_act_CD4', '-> IL_2', function=expr_IL_2_prod_1)

expr_IL_2_prod_2 = "k_IL_2_act_CD4_Antigen_f * act_CD4 * Antigen"
basico.add_reaction('IL_2_production_act_CD4_Antigen', '-> IL_2', function=expr_IL_2_prod_2)

# IL_2 消耗反应 (Consumption)
expr_IL_2_consumption_1 = "k_act_CD4_IL_2_d * naive_CD4 * IL_2 / (k_act_CD4_IL_2_m + IL_2)"
basico.add_reaction('IL_2_consumption_by_naive_CD4', 'IL_2 ->', function=expr_IL_2_consumption_1)

expr_IL_2_consumption_2 = "k_act_NK_IL_2_d * NK * IL_2 / (IL_2 + k_act_NK_IL_2_m)"
basico.add_reaction('IL_2_consumption_by_NK', 'IL_2 ->', function=expr_IL_2_consumption_2)

# IL_2 降解反应 (Decay)
expr_IL_2_decay = "k_IL_2_d * IL_2"
basico.add_reaction('IL_2_decay', 'IL_2 ->', function=expr_IL_2_decay)

# 14. IL_4
# 分离生成和消耗反应

# IL_4 生成反应 (Production)
expr_IL_4_prod_1 = "k_IL_4_Th2_f * Th2"
basico.add_reaction('IL_4_production_Th2', '-> IL_4', function=expr_IL_4_prod_1)

expr_IL_4_prod_2 = "k_IL_4_Th2_Antigen_f * Th2 * Antigen"
basico.add_reaction('IL_4_production_Th2_Antigen', '-> IL_4', function=expr_IL_4_prod_2)

# IL_4 消耗反应 (Consumption)
expr_IL_4_consumption = "k_act_CD4_IL_4_d * act_CD4 * IL_4 / (k_Th2_IL_4_m + IL_4)"
basico.add_reaction('IL_4_consumption_by_act_CD4', 'IL_4 ->', function=expr_IL_4_consumption)

# IL_4 降解反应 (Decay)
expr_IL_4_decay = "k_IL_4_d * IL_4"
basico.add_reaction('IL_4_decay', 'IL_4 ->', function=expr_IL_4_decay)

# 15. IL_10
# 分离生成和消耗反应

# IL_10 生成反应 (Production)
expr_IL_10_prod_1 = "k_IL_10_iTreg_f * iTreg"
basico.add_reaction('IL_10_production_iTreg', '-> IL_10', function=expr_IL_10_prod_1)

expr_IL_10_prod_2 = "k_IL_10_nTreg_f * nTreg * mDC / (k_IL_10_nTreg_mDC_m + mDC)"
basico.add_reaction('IL_10_production_nTreg', '-> IL_10', function=expr_IL_10_prod_2)

# IL_10 消耗反应 (Consumption)
expr_IL_10_consumption = "k_iTreg_mDC_d * act_CD4 * IL_10 / (k_iTreg_IL_10_m + IL_10)"
basico.add_reaction('IL_10_consumption_by_act_CD4', 'IL_10 ->', function=expr_IL_10_consumption)

# IL_10 降解反应 (Decay)
expr_IL_10_decay = "k_IL_10_d * IL_10"
basico.add_reaction('IL_10_decay', 'IL_10 ->', function=expr_IL_10_decay)

# 16. TGFbeta
# 分离生成和消耗反应

# TGFbeta 生成反应 (Production)
expr_TGFbeta_prod_1 = "k_TGFbeta_iTreg_f * iTreg"
basico.add_reaction('TGFbeta_production_iTreg', '-> TGFbeta', function=expr_TGFbeta_prod_1)

expr_TGFbeta_prod_2 = "k_TGFbeta_CD4_CTL_f * CD4_CTL"
basico.add_reaction('TGFbeta_production_CD4_CTL', '-> TGFbeta', function=expr_TGFbeta_prod_2)

expr_TGFbeta_prod_3 = "k_TGFbeta_nTreg_f * nTreg * mDC / (k_TGFbeta_nTreg_mDC_m + mDC)"
basico.add_reaction('TGFbeta_production_nTreg', '-> TGFbeta', function=expr_TGFbeta_prod_3)

# TGFbeta 消耗反应 (Consumption)
expr_TGFbeta_consumption = "k_iTreg_mDC_d * act_CD4 * TGFbeta / (k_iTreg_TGFbeta_m + TGFbeta)"
basico.add_reaction('TGFbeta_consumption_by_act_CD4', 'TGFbeta ->', function=expr_TGFbeta_consumption)

# TGFbeta 降解反应 (Decay)
expr_TGFbeta_decay = "k_TGFbeta_d * TGFbeta"
basico.add_reaction('TGFbeta_decay', 'TGFbeta ->', function=expr_TGFbeta_decay)

# 17. IFN_g
# 分离生成和消耗反应

# IFN_g 生成反应 (Production)
expr_IFN_g_prod_1 = "k_IFN_g_CD4_CTL_f * CD4_CTL"
basico.add_reaction('IFN_g_production_CD4_CTL', '-> IFN_g', function=expr_IFN_g_prod_1)

expr_IFN_g_prod_2 = "k_IFN_g_act_NK_f * act_NK"
basico.add_reaction('IFN_g_production_act_NK', '-> IFN_g', function=expr_IFN_g_prod_2)

# IFN_g 消耗反应 (Consumption)
expr_IFN_g_consumption = "k_act_NK_IFN_g_d * NK * IFN_g / (IFN_g + k_act_NK_IFN_g_m)"
basico.add_reaction('IFN_g_consumption_by_NK', 'IFN_g ->', function=expr_IFN_g_consumption)

# IFN_g 降解反应 (Decay)
expr_IFN_g_decay = "k_IFN_g_d * IFN_g"
basico.add_reaction('IFN_g_decay', 'IFN_g ->', function=expr_IFN_g_decay)

# 18. naive_CD4
# 分离生成和消耗反应

# naive_CD4 生成反应 (Production)
expr_naive_CD4_prod_1 = "k_CD4_f * naive_CD4 * (1 - naive_CD4 / k_CD4_m)"
basico.add_reaction('naive_CD4_proliferation_basal', '-> naive_CD4', function=expr_naive_CD4_prod_1)

expr_naive_CD4_prod_2 = "k_naive_CD4_IL_15_f * naive_CD4 * (1 - naive_CD4 / k_CD4_m) * IL_15 / (k_naive_CD4_IL_15_m + IL_15)"
basico.add_reaction('naive_CD4_proliferation_IL_15', '-> naive_CD4', function=expr_naive_CD4_prod_2)

expr_naive_CD4_prod_3 = "k_naive_CD4_IL_7_f * naive_CD4 * (1 - naive_CD4 / k_CD4_m) * IL_7 / (k_naive_CD4_IL_7_m + IL_7)"
basico.add_reaction('naive_CD4_proliferation_IL_7', '-> naive_CD4', function=expr_naive_CD4_prod_3)

# naive_CD4 消耗反应 (Consumption) - 转化为act_CD4
expr_naive_CD4_to_act_CD4_1 = "k_act_CD4_mDC_f * naive_CD4 * mDC / (k_act_CD4_mDC_m + mDC)"
basico.add_reaction('naive_CD4_to_act_CD4_mDC', 'naive_CD4 ->', function=expr_naive_CD4_to_act_CD4_1)

expr_naive_CD4_to_act_CD4_2 = "k_act_CD4_IL_2_f * naive_CD4 * IL_2 / (k_act_CD4_IL_2_m + IL_2)"
basico.add_reaction('naive_CD4_to_act_CD4_IL_2', 'naive_CD4 ->', function=expr_naive_CD4_to_act_CD4_2)

# naive_CD4 降解反应 (Decay)
expr_naive_CD4_decay = "k_naive_CD4_d * naive_CD4"
basico.add_reaction('naive_CD4_decay', 'naive_CD4 ->', function=expr_naive_CD4_decay)

# 19. act_CD4
# 分离生成和消耗反应

# act_CD4 生成反应 (Production) - 从naive_CD4转化
expr_act_CD4_from_naive_1 = "k_act_CD4_mDC_f * naive_CD4 * mDC / (k_act_CD4_mDC_m + mDC)"
basico.add_reaction('act_CD4_from_naive_mDC', '-> act_CD4', function=expr_act_CD4_from_naive_1)

expr_act_CD4_from_naive_2 = "k_act_CD4_IL_2_f * naive_CD4 * IL_2 / (k_act_CD4_IL_2_m + IL_2)"
basico.add_reaction('act_CD4_from_naive_IL_2', '-> act_CD4', function=expr_act_CD4_from_naive_2)

# act_CD4 增殖反应 (Proliferation)
expr_act_CD4_proliferation_1 = "k_act_CD4_f * act_CD4 * (1 - act_CD4 / k_act_CD4_m)"
basico.add_reaction('act_CD4_proliferation_basal', '-> act_CD4', function=expr_act_CD4_proliferation_1)

expr_act_CD4_proliferation_2 = "k_act_CD4_IL_15_f * act_CD4 * (1 - act_CD4 / k_act_CD4_m) * IL_15 / (k_act_CD4_IL_15_m + IL_15)"
basico.add_reaction('act_CD4_proliferation_IL_15', '-> act_CD4', function=expr_act_CD4_proliferation_2)

expr_act_CD4_proliferation_3 = "k_act_CD4_IL_7_f * act_CD4 * (1 - act_CD4 / k_act_CD4_m) * IL_7 / (k_act_CD4_IL_7_m + IL_7)"
basico.add_reaction('act_CD4_proliferation_IL_7', '-> act_CD4', function=expr_act_CD4_proliferation_3)

# act_CD4 消耗反应 (Consumption) - 转化为Th2
expr_act_CD4_to_Th2_1 = "act_CD4 * k_Th2_f * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))"
basico.add_reaction('act_CD4_to_Th2_basal', 'act_CD4 ->', function=expr_act_CD4_to_Th2_1)

expr_act_CD4_to_Th2_2 = "act_CD4 * k_Th2_IL_4_f * IL_4 / (k_Th2_IL_4_m + IL_4) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))"
basico.add_reaction('act_CD4_to_Th2_IL_4', 'act_CD4 ->', function=expr_act_CD4_to_Th2_2)

expr_act_CD4_to_Th2_3 = "act_CD4 * k_Th2_IL_33_f * IL_33 / (k_Th2_IL_33_m + IL_33) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))"
basico.add_reaction('act_CD4_to_Th2_IL_33', 'act_CD4 ->', function=expr_act_CD4_to_Th2_3)

# act_CD4 消耗反应 (Consumption) - 转化为iTreg
expr_act_CD4_to_iTreg_1 = "act_CD4 * k_iTreg_mDC_f * k_iTreg_TGFbeta_f * TGFbeta / (k_iTreg_TGFbeta_m + TGFbeta) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1))"
basico.add_reaction('act_CD4_to_iTreg_TGFbeta', 'act_CD4 ->', function=expr_act_CD4_to_iTreg_1)

expr_act_CD4_to_iTreg_2 = "act_CD4 * k_iTreg_mDC_f * k_iTreg_IL_10_f * IL_10 / (k_iTreg_IL_10_m + IL_10) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1))"
basico.add_reaction('act_CD4_to_iTreg_IL_10', 'act_CD4 ->', function=expr_act_CD4_to_iTreg_2)

# act_CD4 消耗反应 (Consumption) - 转化为CD4_CTL
expr_act_CD4_to_CD4_CTL_1 = "k_act_CD4_CTL_basal_f * act_CD4"
basico.add_reaction('act_CD4_to_CD4_CTL_basal', 'act_CD4 ->', function=expr_act_CD4_to_CD4_CTL_1)

expr_act_CD4_to_CD4_CTL_2 = "k_act_CD4_CTL_antigen_f * act_CD4 * Antigen"
basico.add_reaction('act_CD4_to_CD4_CTL_antigen', 'act_CD4 ->', function=expr_act_CD4_to_CD4_CTL_2)

expr_act_CD4_to_CD4_CTL_3 = "k_act_CD4_IFN1_f * act_CD4 * IFN1 / (k_IFN1_CD4_CTL_m + IFN1)"
basico.add_reaction('act_CD4_to_CD4_CTL_IFN1', 'act_CD4 ->', function=expr_act_CD4_to_CD4_CTL_3)

# act_CD4 消耗反应 (Consumption) - 转化为TFH
expr_act_CD4_to_TFH_1 = "k_TFH_mDC_f * act_CD4"
basico.add_reaction('act_CD4_to_TFH_mDC', 'act_CD4 ->', function=expr_act_CD4_to_TFH_1)

expr_act_CD4_to_TFH_2 = "k_TFH_mDC_Antigen_f * act_CD4 * Antigen"
basico.add_reaction('act_CD4_to_TFH_antigen', 'act_CD4 ->', function=expr_act_CD4_to_TFH_2)

expr_act_CD4_to_TFH_3 = "k_TFH_IFN1_f * act_CD4 * IFN1 / (k_TFH_IFN1_m + IFN1)"
basico.add_reaction('act_CD4_to_TFH_IFN1', 'act_CD4 ->', function=expr_act_CD4_to_TFH_3)

expr_act_CD4_to_TFH_4 = "k_TFH_IL_6_f * act_CD4 * IL_6 / (k_TFH_IL_6_m + IL_6)"
basico.add_reaction('act_CD4_to_TFH_IL_6', 'act_CD4 ->', function=expr_act_CD4_to_TFH_4)

# act_CD4 降解反应 (Decay)
expr_act_CD4_decay = "k_act_CD4_d * act_CD4"
basico.add_reaction('act_CD4_decay', 'act_CD4 ->', function=expr_act_CD4_decay)

# 20. Th2
# 分离生成和消耗反应

# Th2 生成反应 (Production) - 从act_CD4转化
expr_Th2_from_act_CD4_1 = "act_CD4 * k_Th2_f * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))"
basico.add_reaction('Th2_from_act_CD4_basal', '-> Th2', function=expr_Th2_from_act_CD4_1)

expr_Th2_from_act_CD4_2 = "act_CD4 * k_Th2_IL_4_f * IL_4 / (k_Th2_IL_4_m + IL_4) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))"
basico.add_reaction('Th2_from_act_CD4_IL_4', '-> Th2', function=expr_Th2_from_act_CD4_2)

expr_Th2_from_act_CD4_3 = "act_CD4 * k_Th2_IL_33_f * IL_33 / (k_Th2_IL_33_m + IL_33) * (k_Th2_TGFbeta_m / (k_Th2_TGFbeta_m + TGFbeta)) * (k_Th2_IL_10_m / (k_Th2_IL_10_m + IL_10)) * (k_Th2_IL_12_m / (k_Th2_IL_12_m + IL_12))"
basico.add_reaction('Th2_from_act_CD4_IL_33', '-> Th2', function=expr_Th2_from_act_CD4_3)

# Th2 增殖反应 (Proliferation)
expr_Th2_proliferation = "k_Th2_f * Th2 * (1 - Th2 / k_Th2_m)"
basico.add_reaction('Th2_proliferation', '-> Th2', function=expr_Th2_proliferation)

# Th2 降解反应 (Decay)
expr_Th2_decay = "k_Th2_d * Th2"
basico.add_reaction('Th2_decay', 'Th2 ->', function=expr_Th2_decay)

# 21. iTreg
# 分离生成和消耗反应

# iTreg 生成反应 (Production) - 从act_CD4转化
expr_iTreg_from_act_CD4_1 = "act_CD4 * k_iTreg_mDC_f * k_iTreg_TGFbeta_f * TGFbeta / (k_iTreg_TGFbeta_m + TGFbeta) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1))"
basico.add_reaction('iTreg_from_act_CD4_TGFbeta', '-> iTreg', function=expr_iTreg_from_act_CD4_1)

expr_iTreg_from_act_CD4_2 = "act_CD4 * k_iTreg_mDC_f * k_iTreg_IL_10_f * IL_10 / (k_iTreg_IL_10_m + IL_10) * (k_iTreg_IL_1_m / (k_iTreg_IL_1_m + IL_1))"
basico.add_reaction('iTreg_from_act_CD4_IL_10', '-> iTreg', function=expr_iTreg_from_act_CD4_2)

# iTreg 增殖反应 (Proliferation)
expr_iTreg_proliferation = "k_iTreg_f * iTreg * (1 - iTreg / k_iTreg_m)"
basico.add_reaction('iTreg_proliferation', '-> iTreg', function=expr_iTreg_proliferation)

# iTreg 降解反应 (Decay)
expr_iTreg_decay = "k_iTreg_d * iTreg"
basico.add_reaction('iTreg_decay', 'iTreg ->', function=expr_iTreg_decay)

# 22. CD4_CTL
# 分离生成和消耗反应

# CD4_CTL 生成反应 (Production) - 从act_CD4转化
expr_CD4_CTL_from_act_CD4_1 = "k_act_CD4_CTL_basal_f * act_CD4"
basico.add_reaction('CD4_CTL_from_act_CD4_basal', '-> CD4_CTL', function=expr_CD4_CTL_from_act_CD4_1)

expr_CD4_CTL_from_act_CD4_2 = "k_act_CD4_CTL_antigen_f * act_CD4 * Antigen"
basico.add_reaction('CD4_CTL_from_act_CD4_antigen', '-> CD4_CTL', function=expr_CD4_CTL_from_act_CD4_2)

expr_CD4_CTL_from_act_CD4_3 = "k_act_CD4_IFN1_f * act_CD4 * IFN1 / (k_IFN1_CD4_CTL_m + IFN1)"
basico.add_reaction('CD4_CTL_from_act_CD4_IFN1', '-> CD4_CTL', function=expr_CD4_CTL_from_act_CD4_3)

# CD4_CTL 增殖反应 (Proliferation)
expr_CD4_CTL_proliferation = "k_CD4_CTL_f * CD4_CTL * (1 - CD4_CTL / k_CD4_CTL_m)"
basico.add_reaction('CD4_CTL_proliferation', '-> CD4_CTL', function=expr_CD4_CTL_proliferation)

# CD4_CTL 降解反应 (Decay)
expr_CD4_CTL_decay = "k_CD4_CTL_d * CD4_CTL"
basico.add_reaction('CD4_CTL_decay', 'CD4_CTL ->', function=expr_CD4_CTL_decay)

# 23. nTreg
# 分离生成和消耗反应

# nTreg 增殖反应 (Proliferation)
expr_nTreg_proliferation = "k_nTreg_mDC_f * nTreg * (1 - nTreg / k_nTreg_m) * mDC / (k_nTreg_mDC_m + mDC)"
basico.add_reaction('nTreg_proliferation', '-> nTreg', function=expr_nTreg_proliferation)

# nTreg 降解反应 (Decay)
expr_nTreg_decay = "k_nTreg_d * nTreg"
basico.add_reaction('nTreg_decay', 'nTreg ->', function=expr_nTreg_decay)

# 24. TFH
# 分离生成和消耗反应

# TFH 生成反应 (Production) - 从act_CD4转化
expr_TFH_from_act_CD4_1 = "k_TFH_mDC_f * act_CD4"
basico.add_reaction('TFH_from_act_CD4_mDC', '-> TFH', function=expr_TFH_from_act_CD4_1)

expr_TFH_from_act_CD4_2 = "k_TFH_mDC_Antigen_f * act_CD4 * Antigen"
basico.add_reaction('TFH_from_act_CD4_antigen', '-> TFH', function=expr_TFH_from_act_CD4_2)

expr_TFH_from_act_CD4_3 = "k_TFH_IFN1_f * act_CD4 * IFN1 / (k_TFH_IFN1_m + IFN1)"
basico.add_reaction('TFH_from_act_CD4_IFN1', '-> TFH', function=expr_TFH_from_act_CD4_3)

expr_TFH_from_act_CD4_4 = "k_TFH_IL_6_f * act_CD4 * IL_6 / (k_TFH_IL_6_m + IL_6)"
basico.add_reaction('TFH_from_act_CD4_IL_6', '-> TFH', function=expr_TFH_from_act_CD4_4)

# TFH 增殖反应 (Proliferation)
expr_TFH_proliferation = "k_TFH_f * TFH * (1 - TFH / k_TFH_m)"
basico.add_reaction('TFH_proliferation', '-> TFH', function=expr_TFH_proliferation)

# TFH 降解反应 (Decay)
expr_TFH_decay = "k_TFH_d * TFH"
basico.add_reaction('TFH_decay', 'TFH ->', function=expr_TFH_decay)

# 25. NK
# 分离生成和消耗反应

# NK 增殖反应 (Proliferation)
expr_NK_proliferation = "k_NK_f * NK * (1 - NK / k_NK_m)"
basico.add_reaction('NK_proliferation', '-> NK', function=expr_NK_proliferation)

# NK 消耗反应 (Consumption) - 转化为act_NK
expr_NK_to_act_NK_1 = "k_act_NK_base_f * NK"
basico.add_reaction('NK_to_act_NK_basal', 'NK ->', function=expr_NK_to_act_NK_1)

expr_NK_to_act_NK_2 = "k_act_NK_IL_12_f * NK * IL_12 / (IL_12 + k_act_NK_IL_12_m)"
basico.add_reaction('NK_to_act_NK_IL_12', 'NK ->', function=expr_NK_to_act_NK_2)

expr_NK_to_act_NK_3 = "k_act_NK_IL_2_f * NK * IL_2 / (IL_2 + k_act_NK_IL_2_m)"
basico.add_reaction('NK_to_act_NK_IL_2', 'NK ->', function=expr_NK_to_act_NK_3)

expr_NK_to_act_NK_4 = "k_act_NK_IFN1_f * NK * IFN1 / (IFN1 + k_act_NK_IFN1_m)"
basico.add_reaction('NK_to_act_NK_IFN1', 'NK ->', function=expr_NK_to_act_NK_4)

expr_NK_to_act_NK_5 = "k_act_NK_IFN_g_f * NK * IFN_g / (IFN_g + k_act_NK_IFN_g_m)"
basico.add_reaction('NK_to_act_NK_IFN_g', 'NK ->', function=expr_NK_to_act_NK_5)

# NK 降解反应 (Decay)
expr_NK_decay = "k_NK_d * NK"
basico.add_reaction('NK_decay', 'NK ->', function=expr_NK_decay)

# 26. act_NK
# 分离生成和消耗反应

# act_NK 生成反应 (Production) - 从NK转化
expr_act_NK_from_NK_1 = "k_act_NK_base_f * NK"
basico.add_reaction('act_NK_from_NK_basal', '-> act_NK', function=expr_act_NK_from_NK_1)

expr_act_NK_from_NK_2 = "k_act_NK_IL_12_f * NK * IL_12 / (IL_12 + k_act_NK_IL_12_m)"
basico.add_reaction('act_NK_from_NK_IL_12', '-> act_NK', function=expr_act_NK_from_NK_2)

expr_act_NK_from_NK_3 = "k_act_NK_IL_2_f * NK * IL_2 / (IL_2 + k_act_NK_IL_2_m)"
basico.add_reaction('act_NK_from_NK_IL_2', '-> act_NK', function=expr_act_NK_from_NK_3)

expr_act_NK_from_NK_4 = "k_act_NK_IFN1_f * NK * IFN1 / (IFN1 + k_act_NK_IFN1_m)"
basico.add_reaction('act_NK_from_NK_IFN1', '-> act_NK', function=expr_act_NK_from_NK_4)

expr_act_NK_from_NK_5 = "k_act_NK_IFN_g_f * NK * IFN_g / (IFN_g + k_act_NK_IFN_g_m)"
basico.add_reaction('act_NK_from_NK_IFN_g', '-> act_NK', function=expr_act_NK_from_NK_5)

# act_NK 增殖反应 (Proliferation)
expr_act_NK_proliferation = "k_act_NK_f * act_NK * (1 - act_NK / k_act_NK_m)"
basico.add_reaction('act_NK_proliferation', '-> act_NK', function=expr_act_NK_proliferation)

# act_NK 降解反应 (Decay)
expr_act_NK_decay = "k_act_NK_d * act_NK"
basico.add_reaction('act_NK_decay', 'act_NK ->', function=expr_act_NK_decay)

# 27. Naive_B
# 分离生成和消耗反应

# Naive_B 增殖反应 (Proliferation)
expr_Naive_B_proliferation_1 = "k_Naive_B_f * Naive_B * (1 - Naive_B / k_Naive_B_m)"
basico.add_reaction('Naive_B_proliferation_basal', '-> Naive_B', function=expr_Naive_B_proliferation_1)

expr_Naive_B_proliferation_2 = "k_Naive_B_Antigen_f * Naive_B * Antigen * (1 - Naive_B / k_Naive_B_m)"
basico.add_reaction('Naive_B_proliferation_Antigen', '-> Naive_B', function=expr_Naive_B_proliferation_2)

# Naive_B 消耗反应 (Consumption) - 转化为Act_B
expr_Naive_B_to_Act_B_1 = "k_Act_B_basal_f * Naive_B"
basico.add_reaction('Naive_B_to_Act_B_basal', 'Naive_B ->', function=expr_Naive_B_to_Act_B_1)

expr_Naive_B_to_Act_B_2 = "k_Act_B_Antigen_f * Naive_B * Antigen"
basico.add_reaction('Naive_B_to_Act_B_antigen', 'Naive_B ->', function=expr_Naive_B_to_Act_B_2)

# Naive_B 降解反应 (Decay)
expr_Naive_B_decay = "k_Naive_B_d * Naive_B"
basico.add_reaction('Naive_B_decay', 'Naive_B ->', function=expr_Naive_B_decay)

# 28. Act_B
# 分离生成和消耗反应

# Act_B 生成反应 (Production) - 从Naive_B转化
expr_Act_B_from_Naive_B_1 = "k_Act_B_basal_f * Naive_B"
basico.add_reaction('Act_B_from_Naive_B_basal', '-> Act_B', function=expr_Act_B_from_Naive_B_1)

expr_Act_B_from_Naive_B_2 = "k_Act_B_Antigen_f * Naive_B * Antigen"
basico.add_reaction('Act_B_from_Naive_B_antigen', '-> Act_B', function=expr_Act_B_from_Naive_B_2)

# Act_B 增殖反应 (Proliferation)
expr_Act_B_proliferation_1 = "k_Act_B_f * Act_B * (1 - Act_B / k_Act_B_m)"
basico.add_reaction('Act_B_proliferation_basal', '-> Act_B', function=expr_Act_B_proliferation_1)

expr_Act_B_proliferation_2 = "k_Act_B_Antigen_pro_f * Act_B * Antigen * (1 - Act_B / k_Act_B_m)"
basico.add_reaction('Act_B_proliferation_Antigen', '-> Act_B', function=expr_Act_B_proliferation_2)

# Act_B 降解反应 (Decay)
expr_Act_B_decay = "k_Act_B_d * Act_B"
basico.add_reaction('Act_B_decay', 'Act_B ->', function=expr_Act_B_decay)

# 29. TD_IS_B
# 分离生成和消耗反应

# TD_IS_B 生成反应 (Production) - 从Act_B转化
expr_TD_IS_B_from_Act_B_1 = "k_TD_base_f * Act_B"
basico.add_reaction('TD_IS_B_from_Act_B_basal', '-> TD_IS_B', function=expr_TD_IS_B_from_Act_B_1)

expr_TD_IS_B_from_Act_B_2 = "k_TD_IL_4_f * Act_B * IL_4"
basico.add_reaction('TD_IS_B_from_Act_B_IL_4', '-> TD_IS_B', function=expr_TD_IS_B_from_Act_B_2)

# TD_IS_B 增殖反应 (Proliferation)
expr_TD_IS_B_proliferation = "k_TD_f * TD_IS_B * (1 - TD_IS_B / k_TD_m)"
basico.add_reaction('TD_IS_B_proliferation', '-> TD_IS_B', function=expr_TD_IS_B_proliferation)

# TD_IS_B 降解反应 (Decay)
expr_TD_IS_B_decay = "k_TD_d * TD_IS_B"
basico.add_reaction('TD_IS_B_decay', 'TD_IS_B ->', function=expr_TD_IS_B_decay)

# 30. TI_IS_B
# 分离生成和消耗反应

# TI_IS_B 生成反应 (Production) - 从Act_B转化
expr_TI_IS_B_from_Act_B_1 = "k_TI_base_f * Act_B"
basico.add_reaction('TI_IS_B_from_Act_B_basal', '-> TI_IS_B', function=expr_TI_IS_B_from_Act_B_1)

expr_TI_IS_B_from_Act_B_2 = "k_TI_IFN_g_f * Act_B * IFN_g"
basico.add_reaction('TI_IS_B_from_Act_B_IFN_g', '-> TI_IS_B', function=expr_TI_IS_B_from_Act_B_2)

expr_TI_IS_B_from_Act_B_3 = "k_TI_IL_10_f * Act_B * IL_10"
basico.add_reaction('TI_IS_B_from_Act_B_IL_10', '-> TI_IS_B', function=expr_TI_IS_B_from_Act_B_3)

# TI_IS_B 增殖反应 (Proliferation)
expr_TI_IS_B_proliferation = "k_TI_f * TI_IS_B * (1 - TI_IS_B / k_TI_m)"
basico.add_reaction('TI_IS_B_proliferation', '-> TI_IS_B', function=expr_TI_IS_B_proliferation)

# TI_IS_B 降解反应 (Decay)
expr_TI_IS_B_decay = "k_TI_d * TI_IS_B"
basico.add_reaction('TI_IS_B_decay', 'TI_IS_B ->', function=expr_TI_IS_B_decay)

# 31. IgG4
# 分离生成和消耗反应

# IgG4 生成反应 (Production) - 从TI_IS_B和TD_IS_B转化
expr_IgG4_from_TI_IS_B = "k_IgG4_TI_f * TI_IS_B"
basico.add_reaction('IgG4_from_TI_IS_B', '-> IgG4', function=expr_IgG4_from_TI_IS_B)

expr_IgG4_from_TD_IS_B = "k_IgG4_TD_f * TD_IS_B"
basico.add_reaction('IgG4_from_TD_IS_B', '-> IgG4', function=expr_IgG4_from_TD_IS_B)

# IgG4 降解反应 (Decay)
expr_IgG4_decay = "k_IgG4_d * IgG4"
basico.add_reaction('IgG4_decay', 'IgG4 ->', function=expr_IgG4_decay)

# ============================================================================
# PART D: 仿真与敏感度分析
# ============================================================================
print("模型构建完成。正在运行初始状态仿真...")

# 强制设置初始 Antigen 为 0
basico.set_species('Antigen', initial_concentration=0.0)

initial_results = basico.run_time_course(duration=300, intervals=300, method='LSODA')

# ============================================================================
# PART E: 参数优化流程 (修复版)
# ============================================================================
print("\n" + "="*50)
print("开始参数优化流程")
print("="*50)

# 1. 生成目标数据文件（使用正确的 basico 格式）
# 生成两个时间段的网格：t<100用HC初值，t>=200用IgG4目标值
time_grid_hc = np.linspace(0.0, 100.0, 11)   # HC稳态区间
time_grid_igg = np.linspace(200.0, 300.0, 11)  # IgG4稳态区间

hc_dict = HC_state()      # HC稳态初值
target_dict = IgG_state() # IgG4稳态目标值

# 创建符合 basico 格式的 DataFrame
data_rows = []

# t<100: 使用HC初值（系统已达稳态）
for t in time_grid_hc:
    row = {'Time': t}
    row.update(hc_dict)
    data_rows.append(row)

# t>=200: 使用IgG4目标值（抗原刺激后的稳态）
for t in time_grid_igg:
    row = {'Time': t}
    row.update(target_dict)
    data_rows.append(row)

target_df = pd.DataFrame(data_rows)

# 确保 Time 列在第一列
columns = ['Time'] + [col for col in target_df.columns if col != 'Time']
target_df = target_df[columns]

# 保存为 CSV
target_csv_name = 'target_data.csv'
target_df.to_csv(target_csv_name, sep='\t', index=False, float_format='%.6f')
print(f"已生成目标数据文件: {target_csv_name}")
print(f"  - 数据形状: {target_df.shape} (时间点 × 变量)")
print(f"  - 时间范围: 0-100 (HC稳态) + 200-300 (IgG4稳态)")
print(f"  - 变量数量: {len(target_dict)}")

# 2. 加载实验数据
print("\n正在加载实验数据到 basico...")
basico.add_experiment(
    name="IgG4_target",
    data=target_df,
    file_name=target_csv_name,
)
print(f"实验数据已加载: IgG4_target")

# 3. 设置待优化参数
print("正在配置优化参数...")
fit_items = []
for p_name in ALL_PARAMS:
    curr = basico.get_parameters(p_name).iloc[0]['initial_value']
    if curr == 0:
        lower, upper = 0.0, 1.0
    else:
        lower, upper = curr * 1e-4, curr * 1e4
    fit_items.append({
        'name': f'Values[{p_name}]',
        'lower': lower,
        'upper': upper,
    })

pe.set_fit_parameters(fit_items)

# 4. 阶段一：全局优化 (Particle Swarm)
print("运行阶段一：粒子群优化 (Particle Swarm)...")
# 增加迭代次数以获得更好效果
stat_global = basico.run_parameter_estimation(
    method='Particle Swarm',
    method_parameters={'Iteration Limit': 200, 'Swarm Size': 200}
)
print("全局优化完成。")

# 5. 阶段二：局部精修 (Levenberg-Marquardt)
print("运行阶段二：Levenberg-Marquardt 精修...")
stat_local = basico.run_parameter_estimation(
    method='Levenberg-Marquardt',
    update_model=True # 优化结束后更新模型参数到当前模型
)

print("\n" + "="*50)
print("优化完成")

optimized_params = {
    p: float(basico.get_parameters(p).iloc[0]['initial_value'])
    for p in ALL_PARAMS
}

# 运行优化后的仿真
print("运行优化后的仿真...")
final_sim = basico.run_time_course(duration=300, intervals=300, method='LSODA')

time_values: np.ndarray
if 'Time' in final_sim.columns:
    time_values = final_sim['Time'].to_numpy(dtype=float)
elif 'time' in final_sim.columns:
    time_values = final_sim['time'].to_numpy(dtype=float)
else:
    time_values = final_sim.index.to_numpy(dtype=float)

alias_reverse = {new: old for old, new in ALIASES.items()}
canonical_data = {}
for name in STATE_NAMES:
    if name in final_sim.columns:
        canonical_data[name] = final_sim[name].to_numpy(dtype=float)
    else:
        alias_name = alias_reverse.get(name)
        if alias_name and alias_name in final_sim.columns:
            canonical_data[name] = final_sim[alias_name].to_numpy(dtype=float)
        else:
            canonical_data[name] = np.zeros(len(final_sim), dtype=float)

canonical_df = pd.DataFrame(canonical_data, index=final_sim.index)[STATE_NAMES]

# ============================================================================
# 评价指标计算
# ============================================================================
print("\n计算评价指标...")
pre_mask = time_values < 100.0
post_mask = time_values >= 200.0

pre_window = canonical_df[pre_mask]
post_window = canonical_df[post_mask]

if pre_window.empty or post_window.empty:
    raise RuntimeError("时间采样不足，无法计算评价指标")

pre_rhs_norms = []
post_rhs_norms = []

for t_val, state_vec in zip(time_values[pre_mask], pre_window.to_numpy(dtype=float)):
    rhs_vec = rhs(float(t_val), state_vec, optimized_params)
    pre_rhs_norms.append(np.linalg.norm(rhs_vec))

for t_val, state_vec in zip(time_values[post_mask], post_window.to_numpy(dtype=float)):
    rhs_vec = rhs(float(t_val), state_vec, optimized_params)
    post_rhs_norms.append(np.linalg.norm(rhs_vec))

pre_rhs_mean = float(np.mean(pre_rhs_norms))
post_rhs_mean = float(np.mean(post_rhs_norms))

print(f"t<100 区间 RHS 均值范数: {pre_rhs_mean:.4e}")
print(f"t>200 区间 RHS 均值范数: {post_rhs_mean:.4e}")

post_means = post_window.mean(axis=0)
non_igg4_targets = {
    name: target_dict[name]
    for name in target_dict
    if name in post_means.index and name != 'IgG4'
}

if non_igg4_targets:
    diff_map = {
        name: float(post_means[name] - target_val)
        for name, target_val in non_igg4_targets.items()
    }
    avg_abs_gap = float(np.mean([abs(delta) for delta in diff_map.values()]))
    print(f"非 IgG4 变量平均绝对偏差: {avg_abs_gap:.4e}")
    top_misses = sorted(diff_map.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
    for var_name, delta in top_misses:
        print(f"  {var_name}: 偏差 {delta:.4e}")
else:
    print("未找到除 IgG4 外可对比的目标变量。")

if 'IgG4' in post_means.index and 'IgG4' in pre_window.columns:
    igg4_growth = float(post_means['IgG4'] - pre_window['IgG4'].mean())
    print(f"IgG4 在 t>200 区间的平均增量: {igg4_growth:.4e}")


# ============================================================================
# PART F: 全状态可视化
# ============================================================================
print("\n绘制所有状态的时间轨迹...")
species_names = STATE_NAMES
n_series = len(species_names)
if n_series == 0:
    raise RuntimeError("未找到可绘制的状态变量列")
n_cols = 4
n_rows = math.ceil(n_series / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.0 * n_rows))
axes = axes.flatten()

time_final = time_values

for idx, name in enumerate(species_names):
    ax = axes[idx]
    ax.plot(time_final, canonical_df[name], color='forestgreen', linewidth=1.3, label='Optimized')
    if name in target_dict:
        ax.axhline(target_dict[name], color='indianred', linestyle='--', linewidth=1.0, alpha=0.7, label='Target')
    ax.set_title(name, fontsize=9, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=8)
    ax.set_ylabel('Value', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)

for idx in range(n_series, len(axes)):
    axes[idx].axis('off')

axes[0].legend(fontsize=7, loc='upper right')
plt.suptitle('IgG4 COPASI Simulation (Optimized)', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=(0, 0, 1, 0.97))
plot_path = 'optimize2_timeseries.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"已保存轨迹图: {plot_path}")
plt.show()

print("\n程序执行完毕。")
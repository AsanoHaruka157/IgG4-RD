## =========================================================
## main.R  (IgG4-RD ODE, direct integration with deSolve)
## 使用R实装ODE的废案
## =========================================================
cat("=== running main.R (autoRN) ===\n")

suppressPackageStartupMessages({
  library(deSolve)
})

## -----------------------------
## 0) 31 个 RHS Equation  
## 使用 list() 而不是 c() 以避免字符串编码问题
## 然后使用 unlist() 转换回向量
## 最后对每个元素应用强力修复
## -----------------------------
## 0) 31 个 RHS Equation  
## 使用 list() 而不是 c() 以避免字符串编码问题
## ----------
## 警告: 由于 R 在处理多行字符串时可能出现编码问题，
## 我们使用 paste() 显式构造每个表达式，避免隐式的字符串连接
rhs_vec <- list(

# =====================================================
# Antigen
# =====================================================
Antigen = "0",

# =====================================================
# Dendritic cells
# =====================================================
nDC = "k_nDC_f * nDC * (1 - nDC / k_nDC_m) - k_mDC_Antigen_f * Antigen * nDC * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)) - k_mDC_GMCSF_f * Antigen * nDC * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)) - k_pDC_Antigen_f * nDC * Antigen - k_nDC_d * nDC",

mDC = "k_mDC_Antigen_f * Antigen * nDC * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)) + k_mDC_GMCSF_f * Antigen * nDC * (GMCSF / (GMCSF + k_mDC_GMCSF_m)) * (k_mDC_IL_10_m / (k_mDC_IL_10_m + IL_10)) + k_mDC_f * mDC * (1 - mDC / k_mDC_m) - k_mDC_d * mDC",

pDC = "k_pDC_Antigen_f * nDC * Antigen + k_pDC_f * pDC * (1 - pDC / k_pDC_m) - k_pDC_d * pDC",

# =====================================================
# CD4 T cells
# =====================================================
naive_CD4 = "k_CD4_f * naive_CD4 * (1 - naive_CD4 / k_CD4_m) + k_naive_CD4_IL_15_f * naive_CD4 * (1 - naive_CD4 / k_CD4_m) * IL_15 / (k_naive_CD4_IL_15_m + IL_15) + k_naive_CD4_IL_7_f  * naive_CD4 * (1 - naive_CD4 / k_CD4_m) * IL_7  / (k_naive_CD4_IL_7_m  + IL_7) - k_act_CD4_mDC_f * naive_CD4 * mDC / (k_act_CD4_mDC_m + mDC) - k_act_CD4_IL_2_f * naive_CD4 * IL_2 / (k_act_CD4_IL_2_m + IL_2) - k_naive_CD4_d * naive_CD4",

act_CD4 = "k_act_CD4_mDC_f * naive_CD4 * mDC / (k_act_CD4_mDC_m + mDC) + k_act_CD4_IL_2_f * naive_CD4 * IL_2 / (k_act_CD4_IL_2_m + IL_2) + k_act_CD4_f * act_CD4 * (1 - act_CD4 / k_act_CD4_m) + k_act_CD4_IL_15_f * act_CD4 * (1 - act_CD4 / k_act_CD4_m) * IL_15 / (k_act_CD4_IL_15_m + IL_15) + k_act_CD4_IL_7_f  * act_CD4 * (1 - act_CD4 / k_act_CD4_m) * IL_7  / (k_act_CD4_IL_7_m  + IL_7) - act_CD4 * k_Th2_f * k_Th2_TGFbeta_m/(k_Th2_TGFbeta_m + TGFbeta) * k_Th2_IL_10_m/(k_Th2_IL_10_m + IL_10) * k_Th2_IL_12_m/(k_Th2_IL_12_m + IL_12) - act_CD4 * k_Th2_IL_4_f * IL_4/(k_Th2_IL_4_m + IL_4) * k_Th2_TGFbeta_m/(k_Th2_TGFbeta_m + TGFbeta) * k_Th2_IL_10_m/(k_Th2_IL_10_m + IL_10) * k_Th2_IL_12_m/(k_Th2_IL_12_m + IL_12) - act_CD4 * k_Th2_IL_33_f * IL_33/(k_Th2_IL_33_m + IL_33) * k_Th2_TGFbeta_m/(k_Th2_TGFbeta_m + TGFbeta) * k_Th2_IL_10_m/(k_Th2_IL_10_m + IL_10) * k_Th2_IL_12_m/(k_Th2_IL_12_m + IL_12) - act_CD4 * k_iTreg_mDC_f * k_iTreg_TGFbeta_f * TGFbeta/(k_iTreg_TGFbeta_m + TGFbeta) * k_iTreg_IL_1_m/(k_iTreg_IL_1_m + IL_1) - act_CD4 * k_iTreg_mDC_f * k_iTreg_IL_10_f * IL_10/(k_iTreg_IL_10_m + IL_10) * k_iTreg_IL_1_m/(k_iTreg_IL_1_m + IL_1) - k_act_CD4_CTL_basal_f * act_CD4 - k_act_CD4_CTL_antigen_f * act_CD4 * Antigen - k_act_CD4_IFN1_f * act_CD4 * IFN1/(k_IFN1_CD4_CTL_m + IFN1) - k_TFH_mDC_f * act_CD4 - k_TFH_mDC_Antigen_f * act_CD4 * Antigen - k_TFH_IFN1_f * act_CD4 * IFN1/(k_TFH_IFN1_m + IFN1) - k_TFH_IL_6_f * act_CD4 * IL_6/(k_TFH_IL_6_m + IL_6) - k_act_CD4_d * act_CD4",

Th2 = "k_Th2_f * Th2 * (1 - Th2 / k_Th2_m) + act_CD4 * k_Th2_f * k_Th2_TGFbeta_m/(k_Th2_TGFbeta_m + TGFbeta) * k_Th2_IL_10_m/(k_Th2_IL_10_m + IL_10) * k_Th2_IL_12_m/(k_Th2_IL_12_m + IL_12) + act_CD4 * k_Th2_IL_4_f * IL_4/(k_Th2_IL_4_m + IL_4) * k_Th2_TGFbeta_m/(k_Th2_TGFbeta_m + TGFbeta) * k_Th2_IL_10_m/(k_Th2_IL_10_m + IL_10) * k_Th2_IL_12_m/(k_Th2_IL_12_m + IL_12) + act_CD4 * k_Th2_IL_33_f * IL_33/(k_Th2_IL_33_m + IL_33) * k_Th2_TGFbeta_m/(k_Th2_TGFbeta_m + TGFbeta) * k_Th2_IL_10_m/(k_Th2_IL_10_m + IL_10) * k_Th2_IL_12_m/(k_Th2_IL_12_m + IL_12) - k_Th2_d * Th2",

iTreg = "act_CD4 * k_iTreg_mDC_f * k_iTreg_TGFbeta_f * TGFbeta/(k_iTreg_TGFbeta_m + TGFbeta) * k_iTreg_IL_1_m/(k_iTreg_IL_1_m + IL_1) + act_CD4 * k_iTreg_mDC_f * k_iTreg_IL_10_f * IL_10/(k_iTreg_IL_10_m + IL_10) * k_iTreg_IL_1_m/(k_iTreg_IL_1_m + IL_1) + k_iTreg_f * iTreg * (1 - iTreg / k_iTreg_m) - k_iTreg_d * iTreg",

CD4_CTL = "k_act_CD4_CTL_basal_f * act_CD4 + k_act_CD4_CTL_antigen_f * act_CD4 * Antigen + k_act_CD4_IFN1_f * act_CD4 * IFN1/(k_IFN1_CD4_CTL_m + IFN1) + k_CD4_CTL_f * CD4_CTL * (1 - CD4_CTL / k_CD4_CTL_m) - k_CD4_CTL_d * CD4_CTL",

nTreg = "k_nTreg_mDC_f * nTreg * (1 - nTreg / k_nTreg_m) * mDC/(k_nTreg_mDC_m + mDC) - k_nTreg_d * nTreg",

TFH = "k_TFH_mDC_f * act_CD4 + k_TFH_mDC_Antigen_f * act_CD4 * Antigen + k_TFH_IFN1_f * act_CD4 * IFN1/(k_TFH_IFN1_m + IFN1) + k_TFH_IL_6_f * act_CD4 * IL_6/(k_TFH_IL_6_m + IL_6) + k_TFH_f * TFH * (1 - TFH / k_TFH_m) - k_TFH_d * TFH",

# =====================================================
# NK cells
# =====================================================
NK = "k_NK_f * NK * (1 - NK / k_NK_m) - k_act_NK_base_f * NK - k_act_NK_IL_12_f * NK * IL_12/(IL_12 + k_act_NK_IL_12_m) - k_act_NK_IL_2_f  * NK * IL_2 /(IL_2  + k_act_NK_IL_2_m) - k_act_NK_IFN1_f * NK * IFN1/(IFN1 + k_act_NK_IFN1_m) - k_act_NK_IFN_g_f * NK * IFN_g/(IFN_g + k_act_NK_IFN_g_m) - k_NK_d * NK",

act_NK = "k_act_NK_base_f * NK + k_act_NK_IL_12_f * NK * IL_12/(IL_12 + k_act_NK_IL_12_m) + k_act_NK_IL_2_f  * NK * IL_2 /(IL_2  + k_act_NK_IL_2_m) + k_act_NK_IFN1_f * NK * IFN1/(IFN1 + k_act_NK_IFN1_m) + k_act_NK_IFN_g_f * NK * IFN_g/(IFN_g + k_act_NK_IFN_g_m) + k_act_NK_f * act_NK * (1 - act_NK / k_act_NK_m) - k_act_NK_d * act_NK",

# =====================================================
# B cells & plasma
# =====================================================
Naive_B = "k_Naive_B_f * Naive_B * (1 - Naive_B / k_Naive_B_m) + k_Naive_B_Antigen_f * Naive_B * Antigen * (1 - Naive_B / k_Naive_B_m) - k_Act_B_basal_f * Naive_B - k_Act_B_Antigen_f * Naive_B * Antigen - k_Naive_B_d * Naive_B",

Act_B = "k_Act_B_basal_f * Naive_B + k_Act_B_Antigen_f * Naive_B * Antigen + k_Act_B_f * Act_B * (1 - Act_B / k_Act_B_m) + k_Act_B_Antigen_pro_f * Act_B * Antigen * (1 - Act_B / k_Act_B_m) - k_Act_B_d * Act_B",

TD_IS_B = "k_TD_base_f * Act_B + k_TD_IL_4_f * Act_B * IL_4 + k_TD_f * TD_IS_B * (1 - TD_IS_B / k_TD_m) - k_TD_d * TD_IS_B",

TI_IS_B = "k_TI_base_f * Act_B + k_TI_IFN_g_f * Act_B * IFN_g + k_TI_IL_10_f * Act_B * IL_10 + k_TI_f * TI_IS_B * (1 - TI_IS_B / k_TI_m) - k_TI_d * TI_IS_B",

# =====================================================
# Antibody
# =====================================================
IgG4 = "k_IgG4_TI_f * 1e8 * TI_IS_B + k_IgG4_TD_f * 1e8 * TD_IS_B - k_IgG4_d * IgG4",

# =====================================================
# Cytokines
# =====================================================
GMCSF = "- k_mDC_GMCSF_d * Antigen * nDC * GMCSF/(GMCSF + k_mDC_GMCSF_m) * k_mDC_IL_10_m/(k_mDC_IL_10_m + IL_10) + k_GMCSF_Th2_f * Th2 + k_GMCSF_Th2_Antigen_f * Th2 * Antigen + k_GMCSF_act_NK_f * act_NK - k_GMCSF_d * GMCSF",

IL_33 = "k_IL_33_pDC_f * pDC - k_act_CD4_IL_33_d * act_CD4 * IL_33/(k_Th2_IL_33_m + IL_33) - k_IL_33_d * IL_33",

IL_6 = "k_IL_6_pDC_f * pDC + k_IL_6_mDC_f * mDC + k_IL_6_TFH_f * TFH * k_TFH_nTreg_m/(nTreg + k_TFH_nTreg_m) - k_TFH_IL_6_d * act_CD4 * IL_6/(k_TFH_IL_6_m + IL_6) - k_IL_6_d * IL_6",

IL_12 = "k_IL_12_mDC_f * mDC - k_act_NK_IL_12_d * NK * IL_12/(IL_12 + k_act_NK_IL_12_m) - k_IL_12_d * IL_12",

IL_15 = "k_IL_15_f + k_IL_15_Antigen_f * Antigen - k_naive_CD4_IL_15_d * naive_CD4 * IL_15/(k_naive_CD4_IL_15_m + IL_15) - k_act_CD4_IL_15_d * act_CD4 * IL_15/(k_act_CD4_IL_15_m + IL_15) - k_IL_15_d * IL_15",

IL_7 = "k_IL_7_f - k_naive_CD4_IL_7_d * naive_CD4 * IL_7/(k_naive_CD4_IL_7_m + IL_7) - k_act_CD4_IL_7_d * act_CD4 * IL_7/(k_act_CD4_IL_7_m + IL_7) - k_IL_7_d * IL_7",

IFN1 = "k_IFN1_pDC_f * pDC - k_act_CD4_IFN1_d * act_CD4 * IFN1/(k_IFN1_CD4_CTL_m + IFN1) - k_act_NK_IFN1_d * NK * IFN1/(IFN1 + k_act_NK_IFN1_m) - k_IFN1_d * IFN1",

IL_1 = "k_IL_1_mDC_f * mDC - k_IL_1_d * IL_1",

IL_2 = "k_IL_2_act_CD4_f * act_CD4 + k_IL_2_act_CD4_Antigen_f * act_CD4 * Antigen - k_act_CD4_IL_2_d * naive_CD4 * IL_2/(k_act_CD4_IL_2_m + IL_2) - k_act_NK_IL_2_d * NK * IL_2/(IL_2 + k_act_NK_IL_2_m) - k_IL_2_d * IL_2",

IL_4 = "k_IL_4_Th2_f * Th2 + k_IL_4_Th2_Antigen_f * Th2 * Antigen - k_act_CD4_IL_4_d * act_CD4 * IL_4/(k_Th2_IL_4_m + IL_4) - k_IL_4_d * IL_4",

IL_10 = "k_IL_10_iTreg_f * iTreg + k_IL_10_nTreg_f * nTreg * mDC/(k_IL_10_nTreg_mDC_m + mDC) - k_iTreg_mDC_d * act_CD4 * IL_10/(k_iTreg_IL_10_m + IL_10) - k_IL_10_d * IL_10",

TGFbeta = "k_TGFbeta_iTreg_f * iTreg + k_TGFbeta_CD4_CTL_f * CD4_CTL + k_TGFbeta_nTreg_f * nTreg * mDC/(k_TGFbeta_nTreg_mDC_m + mDC) - k_iTreg_mDC_d * act_CD4 * TGFbeta/(k_iTreg_TGFbeta_m + TGFbeta) - k_TGFbeta_d * TGFbeta",

IFN_g = "k_IFN_g_CD4_CTL_f * CD4_CTL + k_IFN_g_act_NK_f * act_NK - k_act_NK_IFN_g_d * NK * IFN_g/(IFN_g + k_act_NK_IFN_g_m) - k_IFN_g_d * IFN_g"
)

# 将 list 转换回向量，并赋予名称
rhs_vec <- unlist(rhs_vec)
names(rhs_vec) <- c(
  "Antigen", "nDC", "mDC", "pDC",
  "naive_CD4", "act_CD4", "Th2", "iTreg", "CD4_CTL", "nTreg", "TFH",
  "NK", "act_NK",
  "Naive_B", "Act_B", "TD_IS_B", "TI_IS_B",
  "IgG4",
  "GMCSF", "IL_33", "IL_6", "IL_12", "IL_15", "IL_7", "IFN1", "IL_1", "IL_2", "IL_4", "IL_10", "TGFbeta", "IFN_g"
)

# 修复 rhs_vec 中所有被换行或空格拆开的参数名
for(i in seq_along(rhs_vec)) {
  expr <- rhs_vec[[i]]
  # 先统一处理换行符，把它们都替换成空格
  expr <- gsub("[\\r\\n]+", " ", expr)
  # 先压缩多个空格，确保后续修复规则能正确匹配
  expr <- gsub("\\s+", " ", expr)
  expr <- trimws(expr)
  
  # ========== 修复被拆开的参数和变量名 ==========
  # 最重要的：修复任何 "A tige" （不管前面是什么）→ "Antigen"
  expr <- gsub("A\\s+tige\\b", "Antigen", expr)
  # 修复可能出现在末尾的孤立下划线（由于 Antigen 被拆分）
  expr <- gsub("Antigen\\s+_\\b", "Antigen", expr)
  expr <- gsub("Antigen_$", "Antigen", expr)
  
  # 修复 "Antigen tige" → "Antigen"
  expr <- gsub("Antigen\\s+tige\\b", "Antigen", expr)
  
  # 修复 Antigenf f 为 Antigen（最高优先级）
  expr <- gsub("Antigenf\\s+f\\b", "Antigen", expr)
  # 修复任何 Antigenf 为 Antigen
  expr <- gsub("Antigenf\\b", "Antigen", expr)
  
  # 修复 k_ 被拆开的情况（k_ + 空格 + 任意字母/数字/下划线）
  expr <- gsub("k_\\s+([a-zA-Z0-9_]+)", "k_\\1", expr)
  
  # 修复 IL_ 被拆开的情况（IL_ + 空格 + 数字）
  expr <- gsub("IL_\\s+([0-9]+)", "IL_\\1", expr)
  
  # 修复其他参数前缀被拆开的情况（参数通常以 k_ 开头，但也可能有其他模式）
  expr <- gsub("k([a-zA-Z])\\s+([a-zA-Z])", "k_\\1_\\2", expr)
  
  # 修复 DC（独立的变量）被拆开为 nDC
  expr <- gsub("\\bDC\\b", "nDC", expr)
  
  # 修复变量名被拆开的情况（如被中断的 "Antigen"）
  expr <- gsub("Antigen\\s+f\\b", "Antigen", expr)  # Antigen f → Antigen
  
  # ===== 新增：修复被拆分的参数名末尾 =====
  # 修复 "_Antigen" 末尾缺失的参数（被拆为 "_A" 和 "ntigen"、"tige" 等）
  expr <- gsub("_A\\b", "_Antigen", expr)
  
  # 修复末尾孤立的 "_"
  expr <- gsub("\\s+_\\s*$", "", expr)
  expr <- gsub("\\s+_\\s+", " ", expr)
  
  # 最终清理
  expr <- gsub("\\s+", " ", expr)
  expr <- trimws(expr)
  
  rhs_vec[[i]] <- expr
}

stopifnot(exists("rhs_vec"))
stopifnot(is.character(rhs_vec), !is.null(names(rhs_vec)))
states <- names(rhs_vec)

## ----------------------------------------------------------
## 2) 参数赋值（保留原始 pars_raw，不改含义）
## ----------------------------------------------------------
pars_raw <- c(
  k_IFN_g_CD4_CTL_f = 22978.01,  # 1/time, sd=17070.54
  k_GMCSF_Th2_Antigen_f = 3053.59,  # 1/time, sd=2477.62
  k_IL_10_nTreg_f = 2774.48,  # 1/time, sd=5306.99
  k_IL_4_Th2_Antigen_f = 1721.09,  # 1/time, sd=756.54
  k_TI_IS_B_cells_d = 860.77,  # 1/time, sd=2271.49
  k_Naive_B_cells_Antigen_f = 814.14,  # 1/time, sd=1291.61
  k_IL_6_mDC_f = 654.92,  # 1/time, sd=884.03
  k_IL_6_TFH_f = 490.47,  # 1/time, sd=612.03
  k_IL_12_mDC_f = 478.06,  # 1/time, sd=1224.31
  k_IL_2_act_CD4_Antigen_f = 443.38,  # 1/time, sd=479.62
  k_IL_33_pDC_f = 416.88,  # 1/time, sd=614.37
  k_IL_6_pDC_f = 270.62,  # 1/time, sd=351.16
  k_TI_IS_B_cells_TI_IS_B_cells_f = 266.02,  # 1/time, sd=364.51
  k_Act_B_Act_B_Antigen_f = 149.55,  # 1/time, sd=314.95
  k_mDC_GMCSF_f = 94.6,  # 1/time, sd=278.73
  k_nDC_f = 86.9,  # 1/time, sd=274.42
  k_act_CD4_IL_7_d = 55.56,  # 1/time, sd=37.12
  k_act_CD4_mDC_f = 39.54,  # 1/time, sd=99.5
  k_IFN_g_act_NK_f = 34.31,  # 1/time, sd=27.11
  k_naive_CD4_IL_7_f = 22.35,  # 1/time, sd=59.09
  k_IL_2_d = 16.72,  # 1/time, sd=18.1
  k_TD_IS_B_cells_d = 15.46,  # 1/time, sd=17.38
  k_IL_10_iTreg_f = 14.1,  # 1/time, sd=22.05
  k_pro_act_NK_IL_12_f = 12,  # 1/time, sd=30.27
  k_GMCSF_act_NK_f = 11.37,  # 1/time, sd=12.6
  k_mDC_Antigen_f = 8.5,  # 1/time, sd=13.73
  k_pDC_Antigen_f = 7.92,  # 1/time, sd=12.12
  k_pDC_f = 1.0,  # 1/time, missing in original, added to match RHS
  k_mDC_d = 3.17,  # 1/time, sd=10.01
  k_mDC_f = 1.0,  # 1/time, missing in original, added to match RHS
  # 注意：原始表里写的是 k_IL1_mDC_f，这里改成与 RHS 一致的 k_IL_1_mDC_f
  k_IL_1_mDC_f = 3.03,  # 1/time, sd=2.13
  k_CD4_f = 2.75,  # 1/time, sd=6.69
  k_naive_CD4_IL_15_f = 2.68,  # 1/time, sd=4.41
  k_pDC_d = 2,  # 1/time, sd=6.52
  k_Act_B_cells_Antigen_f = 1.69,  # 1/time, sd=1.23
  k_iTreg_mDC_f = 1.21,  # 1/time, sd=2.17
  k_NK_f = 1.2,  # 1/time, sd=2.28
  k_IgG4_TD_IS_B_cells_f = 1,  # 1/time, sd=0
  k_IgG4_TI_IS_B_cells_f = 1,  # 1/time, sd=0
  k_TFH_f = 1,  # 1/time, sd=0
  k_act_NK_IL_12_f = 0.96,  # 1/time, sd=1.76
  k_GMCSF_d = 0.83,  # 1/time, sd=0.94
  k_iTreg_d = 0.57,  # 1/time, sd=1.33
  k_nTreg_d = 0.51,  # 1/time, sd=0.69
  k_TFH_mDC_Antigen_f = 0.3,  # 1/time, sd=0.37
  k_nTreg_mDC_f = 0.3,  # 1/time, sd=0.58
  k_IL_6_d = 0.28,  # 1/time, sd=0.33
  k_act_CD4_CTL_antigen_f = 0.25,  # 1/time, sd=0.57
  k_IL_10_d = 0.23,  # 1/time, sd=0.39
  k_act_NK_d = 0.23,  # 1/time, sd=0.58
  k_IL_4_d = 0.19,  # 1/time, sd=0.2
  k_CD4_CTL_CD4_CTL_f = 0.13,  # 1/time, sd=0.2
  k_act_NK_IFN1_f = 0.13,  # 1/time, sd=0.14
  k_IFN1_pDC_f = 0.12,  # 1/time, sd=0.15
  k_Th2_f = 0.12,  # 1/time, sd=0.09
  k_act_CD4_IL_4_d = 0.12,  # 1/time, sd=0.09
  k_TD_IS_B_cells_Act_B_f = 0.1,  # 1/time, sd=0.1
  k_iTreg_f = 0.1,  # 1/time, sd=0.11
  k_naive_CD4_d = 0.09,  # 1/time, sd=0.07
  k_IL_12_d = 0.09,  # 1/time, sd=0.1
  k_IFN1_d = 0.08,  # 1/time, sd=0.08
  k_TGFbeta_d = 0.08,  # 1/time, sd=0.06
  k_IL_33_d = 0.08,  # 1/time, sd=0.07
  k_IL_15_d = 0.08,  # 1/time, sd=0.08
  k_Naive_B_cells_d = 0.08,  # 1/time, sd=0.06
  k_act_NK_base_f = 0.08,  # 1/time, sd=0.09
  k_act_CD4_d = 0.07,  # 1/time, sd=0.06
  k_IL_7_d = 0.07,  # 1/time, sd=0.07
  k_iTreg_mDC_d = 0.06,  # 1/time, sd=0.05
  k_CD4_CTL_d = 0.06,  # 1/time, sd=0.06
  k_Act_B_Act_B_f = 0.06,  # 1/time, sd=0.07
  k_nTreg_f = 0.06,  # 1/time, sd=0.06
  k_IL_15_f = 0.06,  # 1/time, sd=0.05
  k_TGFbeta_nTreg_f = 0.06,  # 1/time, sd=0.07
  k_IL_7_f = 0.06,  # 1/time, sd=0.05
  k_naive_CD4_IL_7_d = 0.06,  # 1/time, sd=0.06
  k_Act_B_Act_B_d = 0.05,  # 1/time, sd=0.06
  k_NK_d = 0.05,  # 1/time, sd=0.06
  k_nDC_d = 0.05,  # 1/time, sd=0.05
  k_act_CD4_mDC_m = 0.05,  # 1/time, sd=0.05
  k_IL_2_act_CD4_f = 0.05,  # 1/time, sd=0.05
  k_act_NK_IL_12_m = 0.05,  # 1/time, sd=0.05
  k_naive_CD4_IL_15_d = 0.05,  # 1/time, sd=0.05
  k_act_CD4_IL_15_d = 0.05,  # 1/time, sd=0.05
  k_IL_15_Antigen_f = 0.05,  # 1/time, sd=0.05
  k_IL_2_act_CD4_Antigen_f2 = 0.05,  # 1/time, sd=0.05
  k_act_CD4_IL_15_f = 0.05,  # 1/time, sd=0.05
  k_act_CD4_f = 0.05,  # 1/time, sd=0.05
  k_NK_m = 1.45,  # 1/µl, sd=1.46
  k_CD4_m = 1.44,  # 1/µl, sd=1.43
  k_nDC_m = 1.17,  # 1/µl, sd=1.31
  k_TFH_m = 1.12,  # 1/µl, sd=1.31
  k_act_CD4_m = 1.11,  # 1/µl, sd=1.37
  k_nTreg_m = 1.09,  # 1/µl, sd=1.38
  k_TD_IS_B_cells_TD_IS_B_cells_m = 1.09,  # 1/µl, sd=0.45
  k_mDC_m = 1.08,  # 1/µl, sd=1.37
  k_act_NK_m = 1.07,  # 1/µl, sd=1.35
  k_iTreg_m = 1.06,  # 1/µl, sd=1.33
  k_CD4_CTL_CD4_CTL_m = 0.03,  # 1/µl, sd=0.02
  k_Th2_TGFbeta_m = 0.04,  # 1/µl, sd=0.02
  k_iTreg_TGFbeta_m = 0.06,  # 1/µl, sd=0.06
  k_act_NK_IFN1_m = 0.06,  # 1/µl, sd=0.06
  k_TGFbeta_nTreg_mDC_m = 0.06,  # 1/µl, sd=0.06
  k_TFH_IFN1_m = 0.07,  # 1/µl, sd=0.06
  k_mDC_IL_10_m = 0.08,  # 1/µl, sd=0.08
  k_mDC_GMCSF_m = 0.09,  # 1/µl, sd=0.08
  k_iTreg_IL_10_m = 0.09,  # 1/µl, sd=0.08
  k_act_CD4_IL_2_m = 0.09,  # 1/µl, sd=0.08
  k_act_CD4_IFN1_m = 0.09,  # 1/µl, sd=0.08
  k_act_NK_IL_2_m = 0.09,  # 1/µl, sd=0.08
  k_act_NK_IFN_g_m = 0.09,  # 1/µl, sd=0.08
  k_IL_10_nTreg_mDC_m = 0.1,  # 1/µl, sd=0.08
  k_act_CD4_IL_15_m = 0.1,  # 1/µl, sd=0.08
  k_naive_CD4_IL_7_m = 0.1,  # 1/µl, sd=0.08
  k_naive_CD4_IL_15_m = 0.1,  # 1/µl, sd=0.08
  k_Th2_IL_33_f = 0.11,  # 1/µl, sd=0.08
  k_Th2_IL_10_m = 0.12,  # 1/µl, sd=0.08
  k_iTreg_IL_1_m = 0.12,  # 1/µl, sd=0.08
  k_act_CD4_CTL_basal_f = 0.12,  # 1/µl, sd=0.08
  k_act_CD4_IL_7_m = 0.12,  # 1/µl, sd=0.08
  k_act_CD4_IL_33_d = 0.12,  # 1/µl, sd=0.08
  k_act_CD4_IL_33_f = 0.12,  # 1/µl, sd=0.08
  k_TGFbeta_nTreg_mDC_f = 0.12,  # 1/µl, sd=0.08
  k_Th2_IL_4_f = 0.12,  # 1/µl, sd=0.08
  k_TFH_mDC_f = 0.13,  # 1/µl, sd=0.08
  k_Th2_IL_10_f = 0.13,  # 1/µl, sd=0.08
  k_iTreg_IL_10_f = 0.13,  # 1/µl, sd=0.08
  k_Th2_TGFbeta_f = 0.13,  # 1/µl, sd=0.08
  k_act_CD4_CTL_antigen_f2 = 0.13,  # 1/µl, sd=0.08
  k_act_CD4_IL_2_f = 0.13,  # 1/µl, sd=0.08
  k_act_NK_IL_2_f = 0.13,  # 1/µl, sd=0.08
  k_act_NK_IL_12_d = 0.13,  # 1/µl, sd=0.08
  k_act_NK_base_d = 0.13,  # 1/µl, sd=0.08
  k_act_NK_IFN1_d = 0.13,  # 1/µl, sd=0.08
  k_act_NK_IFN_g_f2 = 0.13,  # 1/µl, sd=0.08
  k_Th2_d2 = 0.13,  # 1/µl, sd=0.08
  k_TFH_IL_6_d = 0.13,  # 1/µl, sd=0.08
  k_IL_12_mDC_f2 = 0.13,  # 1/µl, sd=0.08
  k_mDC_f2 = 0.13,  # 1/µl, sd=0.08
  k_IL_7_f2 = 0.13,  # 1/µl, sd=0.08
  k_naive_CD4_IL_7_f2 = 0.13,  # 1/µl, sd=0.08
  k_naive_CD4_IL_15_f2 = 0.13,  # 1/µl, sd=0.08
  k_TFH_mDC_Antigen_f2 = 0.13,  # 1/µl, sd=0.08
  k_act_CD4_mDC_f2 = 0.13,  # 1/µl, sd=0.08
  k_TFH_f2 = 0.13,  # 1/µl, sd=0.08
  k_TFH_m2 = 0.13,  # 1/µl, sd=0.08
  k_act_CD4_CTL_antigen_f3 = 0.13,  # 1/µl, sd=0.08
  k_act_CD4_mDC_m2 = 0.13,  # 1/µl, sd=0.08
  k_mDC_d2 = 0.13,  # 1/µl, sd=0.08
  k_mDC_m2 = 0.13,  # 1/µl, sd=0.08
  k_nDC_f2 = 0.13,  # 1/µl, sd=0.08
  k_act_NK_IL_12_f2 = 0.13,  # 1/µl, sd=0.08
  k_pro_act_NK_IL_12_f2 = 0.13,  # 1/µl, sd=0.08
  k_Th2_IL_4_f2 = 0.13,  # 1/µl, sd=0.08
  k_Th2_IL_33_f2 = 0.13,  # 1/µl, sd=0.08
  k_Th2_IL_10_m2 = 0.13,  # 1/µl, sd=0.08
  k_TFH_IFN1_m2 = 0.13,  # 1/µl, sd=0.08
  k_Th2_TGFbeta_m2 = 0.13,  # 1/µl, sd=0.08
  k_IL_12_d2 = 0.13,  # 1/µl, sd=0.08
  k_NK_d2 = 0.13,  # 1/µl, sd=0.08
  k_act_NK_IL_2_f2 = 0.13,  # 1/µl, sd=0.08
  k_act_NK_base_f2 = 0.13,  # 1/µl, sd=0.08
  k_act_NK_IFN_g_f3 = 0.13,  # 1/µl, sd=0.08
  k_act_NK_d2 = 0.13,  # 1/µl, sd=0.08
  k_act_NK_IL_12_m2 = 0.13,  # 1/µl, sd=0.08
  k_act_NK_IL_12_f3 = 0.13,  # 1/µl, sd=0.08
  k_IL_12_mDC_f3 = 0.13,  # 1/µl, sd=0.08
  k_mDC_m3 = 0.13,  # 1/µl, sd=0.08
  k_nDC_f3 = 0.13,  # 1/µl, sd=0.08
  k_mDC_f3 = 0.13,  # 1/µl, sd=0.08
  k_IL_7_f3 = 0.13,  # 1/µl, sd=0.08
  k_naive_CD4_IL_7_f3 = 0.13,  # 1/µl, sd=0.08
  k_naive_CD4_IL_15_f3 = 0.13,  # 1/µl, sd=0.08
  k_TFH_mDC_f2 = 0.13,  # 1/µl, sd=0.08
  k_TFH_IL_6_f2 = 0.13,  # 1/µl, sd=0.08
  k_TFH_mDC_Antigen_f3 = 0.13,  # 1/µl, sd=0.08
  k_act_CD4_mDC_f3 = 0.13,  # 1/µl, sd=0.08
  k_TFH_f3 = 0.13,  # 1/µl, sd=0.08
  k_TFH_m3 = 0.13,  # 1/µl, sd=0.08
  k_act_CD4_CTL_antigen_f4 = 0.13,  # 1/µl, sd=0.08
  k_act_CD4_mDC_m3 = 0.13,  # 1/µl, sd=0.08
  k_mDC_d3 = 0.13,  # 1/µl, sd=0.08
  k_TGFbeta_d2 = 0.13,  # 1/µl, sd=0.08
  k_Th2_TGFbeta_m3 = 0.13,  # 1/µl, sd=0.08
  k_CD4_CTL_CD4_CTL_m2 = 0.03  # 1/µl, sd=0.02
)


stopifnot(exists("pars_raw"))
stopifnot(is.numeric(pars_raw), !is.null(names(pars_raw)))

normalize_par_names <- function(nm) {
  nm2 <- nm
  # 常见：_cells 去掉（避免 k_Naive_B_cells_* vs k_Naive_B_*）
  nm2 <- gsub("_cells", "", nm2)

  # 常见：把 “Naive_B_cells” / “Act_B_cells” / “TD_IS_B_cells” / “TI_IS_B_cells”
  # 归一到你的 state 命名风格（你 RHS 里用的是 Naive_B / Act_B / TD_IS_B / TI_IS_B）
  nm2 <- gsub("Naive_B_", "Naive_B_", nm2) # 占位（不改）
  nm2 <- gsub("Naive_B", "Naive_B", nm2)
  nm2 <- gsub("Act_B", "Act_B", nm2)
  nm2 <- gsub("TD_IS_B", "TD_IS_B", nm2)
  nm2 <- gsub("TI_IS_B", "TI_IS_B", nm2)

  nm2
}

pars <- pars_raw
names(pars) <- normalize_par_names(names(pars))

## ----------------------------------------------------------
## 3) 自动从 RHS 抽取参数名，检查缺参/多余参
## ----------------------------------------------------------
rhs_text <- paste(rhs_vec, collapse = " ")
pars_in_rhs <- unique(unlist(regmatches(rhs_text, gregexpr("\\bk_[A-Za-z0-9_]+\\b", rhs_text))))

missing_pars <- setdiff(pars_in_rhs, names(pars))
extra_pars   <- setdiff(names(pars), pars_in_rhs)

cat("Parameters in RHS:", length(pars_in_rhs), "\n")
cat("Provided pars:", length(names(pars)), "\n")
cat("Missing pars:", length(missing_pars), "\n")
if (length(missing_pars) > 0) {
  cat("---- Missing parameter names (need to be added / renamed) ----\n")
  print(missing_pars)
}
cat("Extra pars (not used in RHS):", length(extra_pars), "\n")
if (length(extra_pars) > 0) {
  cat("---- Extra parameter names (harmless, but unused) ----\n")
  print(extra_pars)
}

## 关键：为了“先跑通”，把缺的参数先用 1 补齐
if (length(missing_pars) > 0) {
  pars[missing_pars] <- 1
  cat("Filled missing pars with 1 to allow compilation/integration.\n")
}

## ----------------------------------------------------------
## 4) 直接用 deSolve 构建并积分 ODE
## ----------------------------------------------------------

ode_rhs <- function(t, y, parms) {
  # 在独立环境中评估，避免污染全局
  env <- new.env(parent = baseenv())

  # 状态变量
  for (nm in names(y)) {
    assign(nm, y[[nm]], envir = env)
  }

  # 参数
  for (nm in names(parms)) {
    assign(nm, parms[[nm]], envir = env)
  }

  dydt <- numeric(length(y))
  names(dydt) <- names(y)

  for (st in names(y)) {
    expr <- rhs_vec[[st]]
    if (is.null(expr) || expr == "" || expr == "0") {
      # 常量 0 的方程
      dydt[st] <- 0
    } else {
      # 只做最基础的换行与空白规范化
      expr_clean <- gsub("[\\r\\n]+", " ", expr)
      expr_clean <- gsub("\\s+", " ", expr_clean)
      expr_clean <- trimws(expr_clean)
      
      # ========== 第一阶段：修复被拆开的参数和变量名 ==========
      # 最重要的：修复任何 "A tige" （不管前面是什么）→ "Antigen"
      expr_clean <- gsub("A\\s+tige\\b", "Antigen", expr_clean)
      # 修复可能出现在末尾的孤立下划线（由于 Antigen 被拆分）
      expr_clean <- gsub("Antigen\\s+_\\b", "Antigen", expr_clean)
      expr_clean <- gsub("Antigen_$", "Antigen", expr_clean)
      
      # 修复 "Antigen tige" → "Antigen"
      expr_clean <- gsub("Antigen\\s+tige\\b", "Antigen", expr_clean)
      
      # 修复 Antigenf f 为 Antigen（最高优先级）
      expr_clean <- gsub("Antigenf\\s+f\\b", "Antigen", expr_clean)
      # 修复任何 Antigenf 为 Antigen
      expr_clean <- gsub("Antigenf\\b", "Antigen", expr_clean)
      
      # 修复 k_ 被拆开的情况（k_ + 空格 + 任意字母/数字/下划线）
      expr_clean <- gsub("k_\\s+([a-zA-Z0-9_]+)", "k_\\1", expr_clean)
      
      # 修复 IL_ 被拆开的情况（IL_ + 空格 + 数字）
      expr_clean <- gsub("IL_\\s+([0-9]+)", "IL_\\1", expr_clean)
      
      # 修复其他参数前缀被拆开的情况
      expr_clean <- gsub("k([a-zA-Z])\\s+([a-zA-Z])", "k_\\1_\\2", expr_clean)
      
      # ========== 第二阶段：修复其他常见的拆分 ==========
      # 修复 DC（独立的变量）被拆开为 nDC
      expr_clean <- gsub("\\bDC\\b", "nDC", expr_clean)
      
      # 修复变量名被拆开的情况
      expr_clean <- gsub("Antigen\\s+f\\b", "Antigen", expr_clean)  # Antigen f → Antigen
      
      # ===== 新增：修复被拆分的参数名末尾 =====
      # 修复 "_Antigen" 末尾缺失的参数（被拆为 "_A" 和 "ntigen"、"tige" 等）
      expr_clean <- gsub("_A\\b", "_Antigen", expr_clean)
      
      # 修复末尾孤立的 "_"
      expr_clean <- gsub("\\s+_\\s*$", "", expr_clean)
      expr_clean <- gsub("\\s+_\\s+", " ", expr_clean)
      
      # 最终清理：确保没有多余的空格
      expr_clean <- gsub("\\s+", " ", expr_clean)
      expr_clean <- trimws(expr_clean)
      
      # 执行表达式
      dydt[st] <- eval(parse(text = expr_clean), envir = env)
    }
  }

  list(as.numeric(dydt))
}

## 初始条件 & 时间轴
x0 <- setNames(rep(1, length(states)), states)
times <- seq(0, 10, by = 0.1)

## 数值积分
out <- deSolve::ode(y = x0, times = times, func = ode_rhs, parms = pars)

print(head(out))

cat("=== DONE ===\n")

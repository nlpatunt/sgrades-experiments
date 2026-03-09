import pandas as pd
import numpy as np

# ── FILE PATHS — update folders to match your machine ────────────────────────
gpt_files = {
    "Abductive": "gpt4o/stability_summary_abductive_3call_predictions_1773036604.xlsx",
    "Deductive": "gpt4o/stability_summary_deductive_3call_predictions_1773036605.xlsx",
    "Inductive": "gpt4o/stability_summary_inductive_3call_predictions_1773036606.xlsx",
    "Ded-Abd":   "gpt4o/stability_summary_deductive_abductive_3call_predictions_1773036607.xlsx",
    "Ind-Ded":   "gpt4o/stability_summary_inductive_deductive_3call_predictions_1773036608.xlsx",
    "Ind-Abd":   "gpt4o/stability_summary_inductive_abductive_3call_predictions_1773036609.xlsx",
}
gem_files = {
    "Abductive": "gemini/stability_summary_abductive_3call_predictions_1773036597.xlsx",
    "Deductive": "gemini/stability_summary_deductive_3call_predictions_1773036598.xlsx",
    "Inductive": "gemini/stability_summary_inductive_3call_predictions_1773036599.xlsx",
    "Ded-Abd":   "gemini/stability_summary_deductive_abductive_3call_predictions_1773036600.xlsx",
    "Ind-Ded":   "gemini/stability_summary_inductive_deductive_3call_predictions_1773036601.xlsx",
    "Ind-Abd":   "gemini/stability_summary_inductive_abductive_3call_predictions_1773036602.xlsx",
}
llm_files = {
    "Abductive": "llama/stability_summary_abductive_3call_predictions_1773036611.xlsx",
    "Deductive": "llama/stability_summary_deductive_3call_predictions_1773036612.xlsx",
    "Inductive": "llama/stability_summary_inductive_3call_predictions_1773036613.xlsx",
    "Ded-Abd":   "llama/stability_summary_deductive_abductive_3call_predictions_1773036614.xlsx",
    "Ind-Ded":   "llama/stability_summary_inductive_deductive_3call_predictions_1773036615.xlsx",
    "Ind-Abd":   "llama/stability_summary_inductive_abductive_3call_predictions_1773036616.xlsx",
}

AES_DATASETS  = ["D_ASAP-AES", "D_ASAP_plus_plus", "D_ASAP2", "D_persuade_2",
                 "D_Ielts_Writing_Dataset", "D_Ielts_Writing_Task_2_Dataset"]
ASAG_DATASETS = ["D_ASAP-SAS", "D_Regrading_Dataset_J2C", "D_CSEE",
                 "D_Mohlar", "D_OS_Dataset", "D_Rice_Chem"]

def build_num_df(files):
    data = {}
    for strat, fpath in files.items():
        df = pd.read_excel(fpath, sheet_name="Main")
        data[strat] = df[df["type"] == "numeric"].set_index("dataset")["mean_std"]
    df = pd.DataFrame(data)
    return df[~df.index.str.startswith("POOLED")]

def check_claims(name, files):
    num = build_num_df(files)
    aes  = num[num.index.isin(AES_DATASETS)]
    asag = num[num.index.isin(ASAG_DATASETS)]
    asag_no_os = asag[~asag.index.str.contains("OS_Dataset")]

    aes_mean = np.nanmean(aes.values)
    asag_mean = np.nanmean(asag.values)
    asag_no_os_mean = np.nanmean(asag_no_os.values)
    ratio         = asag_mean / aes_mean
    ratio_no_os   = asag_no_os_mean / aes_mean

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  AES  mean std  : {aes_mean:.3f}")
    print(f"  ASAG mean std  : {asag_mean:.3f}")
    print(f"  Ratio ASAG/AES : {ratio:.2f}x")
    print(f"  ASAG (excl. OS_Dataset) mean std : {asag_no_os_mean:.3f}")
    print(f"  Ratio (excl. OS) / AES           : {ratio_no_os:.2f}x")
    print(f"\n  Paper claims:")
    print(f"    ASAG/AES ratio  → paper says {'1.6x' if name != 'LLaMA-4-Scout' else '2.0x'}")
    print(f"    Actual ratio    → {ratio:.1f}x  {'✅' if abs(ratio - (2.0 if name == 'LLaMA-4-Scout' else 1.6)) < 0.05 else '❌ MISMATCH'}")
    if name != "Gemini-2.5-Flash":
        print(f"    excl OS ratio   → paper says 1.3x")
        print(f"    Actual          → {ratio_no_os:.1f}x  {'✅' if abs(ratio_no_os - 1.3) < 0.05 else '❌ MISMATCH'}")

for name, files in [("GPT-4o-mini", gpt_files),
                    ("Gemini-2.5-Flash", gem_files),
                    ("LLaMA-4-Scout", llm_files)]:
    check_claims(name, files)
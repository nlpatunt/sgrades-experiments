
#!/usr/bin/env python3
from huggingface_hub import login
from datasets import load_dataset
import json, warnings
warnings.filterwarnings("ignore")

HF_TOKEN = os.getenv("HF_TOKEN", "")
login(token=HF_TOKEN)

results = {}

standard = [
    ("nlpatunt/D_ASAP-AES",      None,  "ASAP-AES"),
    ("nlpatunt/D_ASAP_plus_plus",     None,  "ASAP++"),
    ("nlpatunt/D_ASAP2",         None,  "ASAP2.0"),
    ("nlpatunt/D_persuade_2",    None,  "Persuade_2"),
    ("nlpatunt/D_Ielts_Writing_Dataset", None,  "IELTS_General"),
    ("nlpatunt/D_Ielts_Writing_Task_2_Dataset",   None,  "IELTS_Task2"),
    ("nlpatunt/D_ASAP-SAS",      None,  "ASAP-SAS"),
    ("nlpatunt/D_Regrading_Dataset_J2C",     None,  "ReGrading"),
    ("nlpatunt/D_CSEE",          None,  "CSEE"),
    ("nlpatunt/D_Mohlar",        None,  "Mohlar"),
    ("nlpatunt/D_BEEtlE",        "2way","BEEtlE_2way"),
    ("nlpatunt/D_BEEtlE",        "3way","BEEtlE_3way"),
    ("nlpatunt/D_SciEntSBank",   "2way","SciEntSBank_2way"),
    ("nlpatunt/D_SciEntSBank",   "3way","SciEntSBank_3way"),
    ("nlpatunt/D_Rice_Chem",     "Q1",  "Rice_Chem_Q1"),
    ("nlpatunt/D_Rice_Chem",     "Q2",  "Rice_Chem_Q2"),
    ("nlpatunt/D_Rice_Chem",     "Q3",  "Rice_Chem_Q3"),
    ("nlpatunt/D_Rice_Chem",     "Q4",  "Rice_Chem_Q4")
]
for hf_name, config, label in standard:
    try:
        ds = load_dataset(hf_name, name=config, trust_remote_code=True) if config \
             else load_dataset(hf_name, trust_remote_code=True)
        test_size = len(ds["test"]) if "test" in ds else None
        results[label] = {"test_size": test_size, "splits": list(ds.keys())}
        print(f"  ✓ {label}: test={test_size}")
    except Exception as e:
        results[label] = {"error": str(e)[:100]}
        print(f"  ✗ {label}: {str(e)[:80]}")

# ── OS_Dataset: folder-based (q1-q5) ────────────────────────────────────────
for q in ["q1", "q2", "q3", "q4", "q5"]:
    label = f"OS_Dataset_{q.upper()}"
    try:
        ds = load_dataset(
            "nlpatunt/D_OS_Dataset",
            data_files={
                "train":      f"{q}/train.csv",
                "validation": f"{q}/validation.csv",
                "test":       f"{q}/test.csv",
            },
            trust_remote_code=True
        )
        test_size = len(ds["test"])
        results[label] = {"test_size": test_size, "splits": list(ds.keys())}
        print(f"  ✓ {label}: test={test_size}")
    except Exception as e:
        results[label] = {"error": str(e)[:100]}
        print(f"  ✗ {label}: {str(e)[:80]}")

with open("test_sizes.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n✓ Saved to test_sizes.json")
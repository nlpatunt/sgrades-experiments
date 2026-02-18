#!/bin/bash
# reorganize.sh
# Run this once from inside sgrades-experiments/
# It will reorganize files into folders and clean up git

echo "Starting reorganization..."

# ─────────────────────────────────────────────
# 1. CREATE FOLDERS
# ─────────────────────────────────────────────
mkdir -p experiments
mkdir -p generalization
mkdir -p randomization
mkdir -p visualization/figures
mkdir -p utils
mkdir -p results

# ─────────────────────────────────────────────
# 2. MOVE EXPERIMENT SCRIPTS
# ─────────────────────────────────────────────
mv run_all_models_sequential.py experiments/ 2>/dev/null
mv run_models.py experiments/ 2>/dev/null
mv run_evaluation.py experiments/ 2>/dev/null
mv test_zero_shot_all.py experiments/ 2>/dev/null
mv single_model_test_few_shot.py experiments/ 2>/dev/null
mv test_few_shot_single.py experiments/ 2>/dev/null
mv batch_evaluate_datasets.py experiments/ 2>/dev/null
mv submit_zero_shot_benchmark.py experiments/ 2>/dev/null
mv sample_test_datasets.py experiments/ 2>/dev/null
mv import_json_results.py experiments/ 2>/dev/null

# ─────────────────────────────────────────────
# 3. MOVE GENERALIZATION EXPERIMENTS
# ─────────────────────────────────────────────
mv lama_exp/ generalization/ 2>/dev/null

# ─────────────────────────────────────────────
# 4. MOVE RANDOMIZATION EXPERIMENTS
# ─────────────────────────────────────────────
mv calculate_SD.py randomization/ 2>/dev/null
mv calculate_SD_2.py randomization/ 2>/dev/null
mv vizualize_random_sampling.py randomization/ 2>/dev/null

# ─────────────────────────────────────────────
# 5. MOVE VISUALIZATION CODE AND FIGURES
# ─────────────────────────────────────────────
mv visualization_code_aes.py visualization/ 2>/dev/null
mv visualization_code_asag.py visualization/ 2>/dev/null
mv visualize_classification.py visualization/ 2>/dev/null

# Move all figures into visualization/figures/
mv *.png visualization/figures/ 2>/dev/null
mv *.pdf visualization/figures/ 2>/dev/null

# ─────────────────────────────────────────────
# 6. MOVE UTILS
# ─────────────────────────────────────────────
mv evaluation_engine.py utils/ 2>/dev/null
mv evaluation_avg.py utils/ 2>/dev/null
mv dataset_ranges.py utils/ 2>/dev/null
mv mllm_models.py utils/ 2>/dev/null

# ─────────────────────────────────────────────
# 7. MOVE RESULTS
# ─────────────────────────────────────────────
mv dataset_counts_results.json results/ 2>/dev/null
mv stored_submissions/ results/ 2>/dev/null

# ─────────────────────────────────────────────
# 8. DELETE JUNK FILES
# ─────────────────────────────────────────────
rm -f check.py
rm -f test_setup.py
rm -f test_individual_datasets.py
rm -f sequential_test.log
rm -f besesr_integration.py
rm -f setup.sh
rm -rf __pycache__/
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

echo "Files reorganized."

# ─────────────────────────────────────────────
# 9. UPDATE .gitignore
# ─────────────────────────────────────────────
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/

# Environment
.env
.venv
venv/
env/

# Secrets
*.key
hf_token.txt

# Logs
*.log

# OS
.DS_Store
Thumbs.db

# Results (optional - remove these lines if you want to track results)
results/stored_submissions/

# Jupyter
.ipynb_checkpoints/
EOF

echo ".gitignore updated."

# ─────────────────────────────────────────────
# 10. COMMIT THE REORGANIZATION
# ─────────────────────────────────────────────
git rm -r --cached . 2>/dev/null
git add .
git commit -m "refactor: reorganize repo into experiments, generalization, randomization, visualization, utils"

echo ""
echo "Done! Final structure:"
find . -not -path './.git/*' -not -name '.gitignore' | sort | head -60
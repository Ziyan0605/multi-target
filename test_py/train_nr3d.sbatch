#!/bin/bash
#SBATCH --job-name=train_nr3d
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=output-%j.log
#SBATCH --error=error-%j.log

source ~/.bashrc

# 1. 加载必要模块
module load CUDA/11.3.1
module load Miniforge3

# 2. 正确激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate 3dvista

# 3. 切换到代码目录（保持与提交作业时的路径一致）
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# 4. 确认 torch 已安装（可选，排查用）
python3 -c "import torch; print('✅ PyTorch version:', torch.__version__)" || exit 1

# 5. 启动训练
python3 run.py --config project/vista/nr3d_config.yml


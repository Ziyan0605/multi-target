#!/bin/bash
#SBATCH --job-name=sr3d_eval
#SBATCH --time=04:00:00             # 根据实际情况调整
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=output-%j.log
#SBATCH --error=error-%j.log

# 1. 加载必要模块
module load CUDA/11.3.1
module load Miniforge3

# 2. 设置 Java 路径，供 Meteor 使用
export JAVA_HOME=/home/zhao0546/jdk-17.0.12
export PATH=$JAVA_HOME/bin:$PATH

# 3. 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate 3dvista

# 4. 切换到代码目录（保持与提交作业时的路径一致）
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# 5. 可选：确认 PyTorch 和 Java
python3 -c "import torch; print('✅ PyTorch version:', torch.__version__)" || exit 1
java -version || exit 1

# 6. 设置 METEOR 路径（如果你有 meteor jar 文件）
# export METEOR_PATH="${SLURM_SUBMIT_DIR:-$PWD}/utils/meteor/meteor-1.5.jar"

# 7. 运行 pipeline
python3 run.py --config project/vista/scanqa_config.yml

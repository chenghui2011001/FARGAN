# 安装指南

## 🐍 虚拟环境设置（推荐）

### 方案一：conda 环境（推荐）

```bash
# 创建新环境
conda create -n fargan-rvq python=3.9
conda activate fargan-rvq

# 安装 PyTorch（根据 CUDA 版本选择）
# CUDA 11.8
conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 12.1  
conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# CPU 版本
conda install pytorch torchaudio cpuonly -c pytorch

# 安装其他依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 方案二：venv 虚拟环境

```bash
# 创建虚拟环境
python -m venv fargan-rvq-env

# 激活环境
# Linux/Mac
source fargan-rvq-env/bin/activate
# Windows
fargan-rvq-env\Scripts\activate

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 方案三：Poetry 环境（开发推荐）

```bash
# 安装 Poetry（如果未安装）
curl -sSL https://install.python-poetry.org | python3 -

# 初始化项目（在项目根目录）
poetry init

# 安装依赖
poetry install

# 激活环境
poetry shell

# 或直接运行
poetry run python training/enhanced_adv_train.py --help
```

## 🔧 环境配置文件

### pyproject.toml（Poetry）
```toml
[tool.poetry]
name = "fargan-rvq"
version = "0.1.0"
description = "Enhanced FarGan with RVQ & JSCC"
authors = ["Your Name <email@example.com>"]

[tool.poetry.dependencies]
python = "^3.9"
torch = "^2.0.0"
torchaudio = "^2.0.0"
numpy = "^1.21.0"
scipy = "^1.7.0"
soundfile = "^0.12.1"
librosa = "^0.10.0"
matplotlib = "^3.5.0"
scikit-learn = "^1.1.0"
pandas = "^1.5.0"
pyyaml = "^6.0"
tqdm = "^4.64.0"
tensorboard = "^2.10.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0.0"
black = "^22.0.0"
isort = "^5.10.0"
wandb = "^0.13.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### environment.yml（conda）
```yaml
name: fargan-rvq
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch>=2.0.0
  - torchaudio>=2.0.0
  - pytorch-cuda=11.8  # or 12.1
  - numpy>=1.21.0
  - scipy>=1.7.0
  - matplotlib>=3.5.0
  - pandas>=1.5.0
  - pyyaml>=6.0
  - tqdm>=4.64.0
  - pip
  - pip:
    - soundfile>=0.12.1
    - librosa>=0.10.0
    - scikit-learn>=1.1.0
    - tensorboard>=2.10.0
    - wandb>=0.13.0
    - pytest>=7.0.0
    - black>=22.0.0
    - isort>=5.10.0
```

```bash
# 使用 environment.yml 创建环境
conda env create -f environment.yml
conda activate fargan-rvq
```

## 🚀 快速安装脚本

### setup.sh
```bash
#!/bin/bash
set -e

echo "🔧 Setting up FarGanRVQ environment..."

# 检查 conda 是否可用
if command -v conda &> /dev/null; then
    echo "📦 Using conda for environment setup"
    
    # 创建环境
    conda create -n fargan-rvq python=3.9 -y
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate fargan-rvq
    
    # 安装 PyTorch
    if nvidia-smi &> /dev/null; then
        echo "🚀 NVIDIA GPU detected, installing CUDA version"
        conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        echo "💻 No GPU detected, installing CPU version"
        conda install pytorch torchaudio cpuonly -c pytorch -y
    fi
    
    # 安装其他依赖
    pip install -r requirements.txt
    
else
    echo "📦 Using pip venv for environment setup"
    
    # 创建虚拟环境
    python -m venv fargan-rvq-env
    source fargan-rvq-env/bin/activate
    
    # 升级 pip
    pip install --upgrade pip
    
    # 安装依赖
    pip install -r requirements.txt
fi

echo "✅ Environment setup complete!"
echo "🔍 Verifying installation..."

python -c "
import torch
import torchaudio
import numpy as np
import librosa
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ TorchAudio: {torchaudio.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU count: {torch.cuda.device_count()}')
print('✅ All dependencies installed successfully!')
"

echo ""
echo "🎉 Setup complete! To activate the environment:"
if command -v conda &> /dev/null; then
    echo "   conda activate fargan-rvq"
else
    echo "   source fargan-rvq-env/bin/activate"
fi
```

```bash
# 运行安装脚本
chmod +x setup.sh
./setup.sh
```

## 🐋 Docker 环境（可选）

### Dockerfile
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/workspace

# 默认命令
CMD ["/bin/bash"]
```

```bash
# 构建镜像
docker build -t fargan-rvq .

# 运行容器
docker run --gpus all -it --rm -v $(pwd):/workspace fargan-rvq

# 或使用 docker-compose
# docker-compose.yml
version: '3.8'
services:
  fargan-rvq:
    build: .
    volumes:
      - .:/workspace
    working_dir: /workspace
    environment:
      - CUDA_VISIBLE_DEVICES=0
    runtime: nvidia
```

## ⚠️ 常见问题

### conda 与 pip 冲突（重要！）

**问题**：在 conda 环境中使用 pip 后，torch 包消失或版本冲突。

**原因**：pip 可能会覆盖 conda 安装的包，导致依赖链断裂。

#### 解决方案一：纯 conda 安装（推荐）

```bash
# 删除有问题的环境
conda env remove -n fargan-rvq

# 重新创建，尽量使用 conda 安装所有包
conda create -n fargan-rvq python=3.9 -y
conda activate fargan-rvq

# 安装 PyTorch
conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 优先使用 conda 安装的包
conda install numpy scipy matplotlib pandas pyyaml tqdm -c conda-forge -y

# 只对 conda 没有的包使用 pip
pip install soundfile librosa scikit-learn tensorboard wandb pytest black isort
```

#### 解决方案二：使用 environment.yml

创建 `environment_fixed.yml`：
```yaml
name: fargan-rvq
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch>=2.0.0
  - torchaudio>=2.0.0
  - pytorch-cuda=11.8
  - numpy>=1.21.0
  - scipy>=1.7.0
  - matplotlib>=3.5.0
  - pandas>=1.5.0
  - pyyaml>=6.0
  - tqdm>=4.64.0
  - pip>=22.0
  - pip:
    - soundfile>=0.12.1
    - librosa>=0.10.0
    - scikit-learn>=1.1.0
    - tensorboard>=2.10.0
    - wandb>=0.13.0
    - pytest>=7.0.0
    - black>=22.0.0
    - isort>=5.10.0
```

```bash
# 删除旧环境
conda env remove -n fargan-rvq

# 使用 yaml 文件创建
conda env create -f environment_fixed.yml
conda activate fargan-rvq
```

#### 解决方案三：修复现有环境

```bash
# 激活环境
conda activate fargan-rvq

# 检查当前状态
conda list torch
pip list | grep torch

# 如果 torch 缺失，重新安装
conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y --force-reinstall

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

#### 解决方案四：conda-pip 最佳实践

```bash
# 1. 创建环境
conda create -n fargan-rvq python=3.9 -y
conda activate fargan-rvq

# 2. 先安装所有 conda 包
conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install numpy scipy matplotlib pandas pyyaml tqdm -c conda-forge -y

# 3. 锁定 conda 包，防止 pip 覆盖
conda env export --no-builds | grep -E "^(pytorch|torchaudio|numpy|scipy)" > conda_lock.txt

# 4. 最后安装 pip 包
pip install soundfile librosa scikit-learn tensorboard

# 5. 验证关键包没有被覆盖
python -c "
import torch
import numpy as np
print(f'PyTorch: {torch.__version__} (should be conda version)')
print(f'NumPy: {np.__version__} (should be conda version)')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

### 快速修复脚本

创建 `fix_environment.sh`：
```bash
#!/bin/bash
set -e

echo "🔧 Fixing conda-pip conflicts..."

# 检查当前环境
if [[ "$CONDA_DEFAULT_ENV" != "fargan-rvq" ]]; then
    echo "❌ Please activate fargan-rvq environment first:"
    echo "   conda activate fargan-rvq"
    exit 1
fi

# 检查 torch 是否存在
if python -c "import torch" 2>/dev/null; then
    echo "✅ PyTorch is available"
    python -c "import torch; print(f'Version: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
else
    echo "❌ PyTorch missing, reinstalling..."
    
    # 重新安装 PyTorch
    conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y --force-reinstall
    
    echo "✅ PyTorch reinstalled"
fi

# 验证其他关键包
echo "🔍 Checking other packages..."
python -c "
try:
    import numpy, scipy, matplotlib, pandas, yaml, tqdm
    print('✅ Core packages OK')
except ImportError as e:
    print(f'❌ Missing package: {e}')
    
try:
    import soundfile, librosa, sklearn
    print('✅ Audio packages OK')
except ImportError as e:
    print(f'⚠️ Audio package issue: {e}')
    print('Run: pip install soundfile librosa scikit-learn')
"

echo "✅ Environment check complete!"
```

```bash
# 运行修复
chmod +x fix_environment.sh
./fix_environment.sh
```

### 预防措施

1. **使用 conda-forge 优先**：
```bash
# 设置 channel 优先级
conda config --add channels conda-forge
conda config --set channel_priority strict
```

2. **创建 .condarc 配置**：
```yaml
# ~/.condarc
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
channel_priority: strict
pip_interop_enabled: true
```

3. **使用 mamba（更快的 conda）**：
```bash
# 安装 mamba
conda install mamba -c conda-forge

# 使用 mamba 替代 conda
mamba create -n fargan-rvq python=3.9
mamba install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### CUDA 版本不匹配
```bash
# 检查系统 CUDA 版本
nvidia-smi

# 检查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 重新安装匹配版本
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 依赖冲突
```bash
# 清理缓存
pip cache purge
conda clean --all

# 重新创建环境
conda env remove -n fargan-rvq
# 然后重新创建
```

### 权限问题
```bash
# 使用用户安装
pip install --user -r requirements.txt

# 或修改权限
sudo chown -R $USER:$USER /path/to/conda
```

## 🔍 验证安装

### test_installation.py
```python
#!/usr/bin/env python3
"""验证 FarGanRVQ 环境安装"""

def test_imports():
    """测试核心依赖导入"""
    try:
        import torch
        import torchaudio
        import numpy as np
        import scipy
        import soundfile as sf
        import librosa
        import matplotlib.pyplot as plt
        import sklearn
        import pandas as pd
        import yaml
        import tqdm
        
        print("✅ All core dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_cuda():
    """测试 CUDA 可用性"""
    import torch
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.version.cuda}")
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        print(f"✅ Current device: {torch.cuda.current_device()}")
        
        # 简单计算测试
        x = torch.randn(1000, 1000).cuda()
        y = torch.mm(x, x.t())
        print("✅ CUDA computation test passed")
        return True
    else:
        print("⚠️  CUDA not available, using CPU")
        return False

def test_audio():
    """测试音频处理"""
    import numpy as np
    import soundfile as sf
    import librosa
    
    # 生成测试音频
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
    
    # 测试 soundfile
    sf.write('test_audio.wav', audio, sr)
    audio_loaded, sr_loaded = sf.read('test_audio.wav')
    assert sr == sr_loaded
    
    # 测试 librosa
    audio_librosa, sr_librosa = librosa.load('test_audio.wav', sr=sr)
    assert abs(sr - sr_librosa) < 1e-6
    
    print("✅ Audio processing test passed")
    
    # 清理
    import os
    os.remove('test_audio.wav')
    return True

if __name__ == "__main__":
    print("🔍 Testing FarGanRVQ installation...\n")
    
    success = True
    success &= test_imports()
    success &= test_cuda()
    success &= test_audio()
    
    if success:
        print("\n🎉 All tests passed! Environment is ready.")
    else:
        print("\n❌ Some tests failed. Please check the installation.")
        exit(1)
```

```bash
# 运行验证
python test_installation.py
```

## 📝 开发环境配置

### .gitignore
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
fargan-rvq-env/

# PyTorch
*.pth
*.pt

# Data
*.wav
*.pcm
*.f32
data/
checkpoints/
outputs/
experiments/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

### .pre-commit-config.yaml
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203"]
```

记住每次使用前激活环境：
```bash
# conda 用户
conda activate fargan-rvq

# venv 用户  
source fargan-rvq-env/bin/activate

# Poetry 用户
poetry shell
``` 
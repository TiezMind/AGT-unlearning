# $AGT^{AO}$: Robust and Stabilized LLM Unlearning via Adversarial Gating Training with Adaptive Orthogonality

[](https://opensource.org/licenses/MIT)
[](https://www.google.com/search?q=https://arxiv.org/abs/placeholder)
[](https://aclweb.org)

This repository contains the official implementation for the paper: **"$AGT^{AO}$: Robust and Stabilized LLM Unlearning via Adversarial Gating Training with Adaptive Orthogonality"**

$AGT^{AO}$ is a novel unlearning framework that simultaneously addresses **catastrophic forgetting** and **superficial forgetting** (hidden knowledge recovery). It achieves State-of-the-Art (SOTA) performance on TOFU, MUSE, and WMDP benchmarks.

<figure style="text-align: center;">
  <img src="arc.png" alt="arc" width="100%">
  <figcaption>Figure 1: Overview of the $AGT^{AO}$ framework.</figcaption>
  </figure>

## 🚀 Key Features

  * **🛡️ Adversarial Gating Training (AGT):** A min-max game in the latent space that generates worst-case perturbations to ensure deep erasure and prevent adversarial recovery.
  * **⚖️ Adaptive Orthogonality (AO):** A dynamic regularization term that projects forget gradients orthogonally to retain gradients only when conflicts occur, preserving model utility.
  * **📈 Gradient-Norm-Based Gating (GBG):** A curriculum-learning inspired mechanism that stabilizes the adversarial training process
  <figure style="text-align: center;">
  <img src="GBG.png" alt="GBG" width="60%">
  <figcaption>Figure 2: Overview of the Gradient-Norm-Based Gating.</figcaption>
  </figure>

## 🛠️ Installation

We recommend using `Conda` to manage the environment.

```bash
conda create -n agt python=3.10
conda activate agt

# Install PyTorch (Adjust cuda version according to your driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

**Requirements (`requirements.txt`):**

```text
transformers>=4.30.0
peft>=0.4.0
datasets
accelerate
scipy
hydra-core  # For configuration management
wandb       # For logging
```

## 📂 Data Preparation

The code supports three major unlearning benchmarks:

1.  **TOFU (Task of Fictitious Unlearning):** Fictional author biographies.
2.  **MUSE:** News and Books for copyright unlearning.
3.  **WMDP:** Cybersecurity and hazardous knowledge.

Please download the datasets and place them in the `./data` folder.

```bash
# Example structure
data/
  ├── tofu/
  ├── muse/
  └── wmdp/
```

## 🏃 Quick Start

We use **LoRA (Low-Rank Adaptation)** for efficient unlearning

### 1\. Model Configuration

Supported models include `Llama-2-7b-chat`, `Gemma-2b-it`, `Zephyr-7b-beta`, and `ICLM-7b`.

### 2\. Running $AGT^{AO}$

To replicate the main results (e.g., on TOFU with Llama-2-7b), run the following command. The hyperparameters are set according to Appendix A.3.

```bash
python main.py \
    --method agt \
    --model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \
    --dataset tofu \
    --lr 1e-4 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_epochs 5 \
    --ao_gamma 1.0 \
    --tau_grad 2.0 \
    --warmup_steps 50 \
    --perturb_layer 10 \
    --inner_loop_steps 4
```

### 3\. Hyperparameters

Key hyperparameters for $AGT^{AO}$ reproduction:

  * **Learning Rate:** `1e-4`
  * **Batch Size:** `1`
  * **AO Parameter ($\gamma$):** `1`
  * **Gradient Threshold** : $\tau_{grad} = \rho \cdot \left\| \nabla \mathcal{L}_{N_{\text{warmup}}} \right\|_2$
  * **Warm-up Steps ($N_{warm\_up}$):** `1 epoch`
  * **Perturbation Layer:** `10` (The "Semantic Entry" layer)
  * **Inner Loop Steps:** `4`(derive from our ablation study)

## 📊 Method Details

### Adaptive Orthogonality (AO)

AO mitigates collateral damage by penalizing the cosine similarity between forget gradients ($g_f$) and retain gradients ($g_r$) only when they conflict ($g_f \cdot g_r < 0$).

$$\mathcal{R}_{AO} = \mathbb{I}(g_f \cdot g_r < 0) \left(\frac{1 - \cos(g_f, g_r)}{2}\right)^\gamma$$

### Adversarial Gating Training (AGT)

AGT formulates unlearning as a bi-level optimization problem to defend against latent perturbations $\delta$

$$\min_{\theta} \max_{||\delta||_p \le \epsilon} (\mathcal{L}_{unlearn}(h_f^{(l)} + \delta,h_{r};\theta))$$

## 🧪 Experimental Results

### TOFU Benchmark (Llama-2-7b-chat)

Comparison of $AGT^{AO}$ against baselines (GA, NPO, SimNPO, LAT, PGU).

# TOFU 基准测试主要结果（Llama-2-7B-chat）
注：结果为三次评估的平均值，↑表示数值越高越好，↓表示数值越低越好，→0.5表示越接近0.5越理想；加粗为最优结果，下划线为次优结果

| method       | 遗忘质量（Forget quality）↑ | KUR↓       | 模型效用（Model utility）↑ | 流畅度（fluency）↑ | 隐私泄露率（PLR）→0.5 |
|--------------|-----------------------------|------------|----------------------------|-------------------|-----------------------|
| target       | -46.91                      | 0.91       | 0.59                       | 0.87              | 0.98                  |
| retrain      | 0.00                        | 0.29       | 0.58                       | 0.91              | 0.47                  |
| GA           | -50.29                      | 0.48       | 0.00                       | 0.00              | 0.45                  |
| GA_GDR       | -51.16                      | 1.44       | 0.51                       | 0.27              | 0.08                  |
| GA_KLR       | -31.85                      | 0.69       | 0.00                       | 0.29              | 0.59                  |
| NPO          | -19.78                      | 0.30       | 0.00                       | 0.02              | 0.40                  |
| NPO_GDR      | -13.80                      | 0.20       | 0.53                       | 0.16              | 0.19                  |
| NPO_KLR      | -30.51                      | 0.38       | 0.45                       | 0.89              | 0.81                  |
| SimNPO_GDR   | -13.96                      | 0.20       | 0.52                       | 0.21              | 0.18                  |
| PGU          | -15.39                      | 0.23       | 0.47                       | 0.83              | 0.55                  |
| RMU          | -14.20                      | 0.14       | 0.45                       | 0.76              | 0.59                  |
| LAT          | -12.50                      | 0.05       | 0.41                       | 0.70              | 0.55                  |
| **$AGT^{AO}$**      | **-9.43**                   | **0.01**   | **0.59**                   | **0.90**          | **0.53**              |

### 补充说明
- 评估维度包括三类：遗忘效果（遗忘质量、知识遗忘比率KUR）、效用与质量（模型效用、流畅度）、隐私（隐私泄露率PLR）。
- KUR和PLR的子指标组成及详情见论文附录
- Note: $AGT^{AO}$ achieves the best balance between unlearning efficacy and utility, with a Privacy Leakage Ratio (PLR) close to the ideal 0.5.

## 📝 Citation

If you find this code useful, please cite our paper:

<!-- ```bibtex
@inproceedings{anonymous2025agt,
  title={$AGT^{AO}$: Robust and Stabilized LLM Unlearning via Adversarial Gating Training with Adaptive Orthogonality},
  author={Pengyu Li, Lingling Zhang, Zhitao Gao, Bifan Wei,Jun Liu, Yaqiang Wu},
  booktitle={ACL Submission},
  year={2026}
}
``` -->

## 🙏 Acknowledgements

Our code builds upon the following repositories:

  * [TOFU: A Task of Fictitious Unlearning](https://github.com/locuslab/tofu) 
  * [MUSE](https://github.com/locuslab/tofu) 
  * [WMDP Benchmark](https://github.com/centerforaisafety/wmdp) 

-----

*Disclaimer: This repository is for research purposes only.*

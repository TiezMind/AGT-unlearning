# AGO: Robust and Stabilized LLM Unlearning

[](https://opensource.org/licenses/MIT)
[](https://www.google.com/search?q=https://arxiv.org/abs/placeholder)
[](https://aclweb.org)

This repository contains the official implementation for the paper: **"AGO: Robust and Stabilized LLM Unlearning via Latent Adversarial Gating and Adaptive Orthogonality"**

AGO is a novel unlearning framework that simultaneously addresses **catastrophic forgetting** and **superficial forgetting** (hidden knowledge recovery). It achieves State-of-the-Art (SOTA) performance on TOFU, MUSE, and WMDP benchmarks.

![Introduction](arc.png)

 Figure 1: Overview of the AGO framework combining Adaptive Soft Orthogonality (ASO) and Latent Adversarial Gating (LAG). *

## 🚀 Key Features

  * **🛡️ Latent Adversarial Gating (LAG):** A min-max game in the latent space that generates worst-case perturbations to ensure deep erasure and prevent adversarial recovery.
  * **⚖️ Adaptive Soft Orthogonality (ASO):** A dynamic regularization term that projects forget gradients orthogonally to retain gradients only when conflicts occur, preserving model utility.
  * **📈 Gradient-Norm-Based Gating (GBG):** A curriculum-learning inspired mechanism that stabilizes the adversarial training process

## 🛠️ Installation

We recommend using `Conda` to manage the environment.

```bash
conda create -n ago python=3.10
conda activate ago

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
2.  **MUSE:** News and Books for copyright unlearning[cite: 624].
3.  **WMDP (Weapons of Mass Destruction Proxy):** Cybersecurity and hazardous knowledge.

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

### 2\. Running AGO

To replicate the main results (e.g., on TOFU with Llama-2-7b), run the following command. The hyperparameters are set according to Appendix A.3.

```bash
python main.py \
    --method ago \
    --model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \
    --dataset tofu \
    --lr 1e-4 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_epochs 5 \
    --aso_gamma 1.0 \
    --tau_grad 2.0 \
    --warmup_steps 50 \
    --perturb_layer 10 \
    --inner_loop_steps 4
```

### 3\. Hyperparameters

Key hyperparameters for AGO reproduction:

  * **Learning Rate:** `1e-4`
  * **Batch Size:** `1`
  * **ASO Parameter ($\gamma$):** `1`
  * **Gradient Threshold ($\tau_{grad}$):** `2`
  * **Warm-up Steps ($N_{warm\_up}$):** `50`
  * **Perturbation Layer:** `10` (The "Semantic Entry" layer)
  * **Inner Loop Steps:** `4`

## 📊 Method Details

### Adaptive Soft Orthogonality (ASO)

ASO mitigates collateral damage by penalizing the cosine similarity between forget gradients ($g_f$) and retain gradients ($g_r$) only when they conflict ($g_f \cdot g_r < 0$).

$$\mathcal{R}_{ASO} = \mathbb{I}(g_f \cdot g_r < 0) \left(\frac{1 - \cos(g_f, g_r)}{2}\right)^\gamma$$

### Latent Adversarial Gating (LAG)

LAG formulates unlearning as a bi-level optimization problem to defend against latent perturbations $\delta$

$$\min_{\theta} \max_{||\delta||_p \le \epsilon} (\mathcal{L}_{unlearn}(\theta, h_f^{(l)} + \delta))$$

## 🧪 Experimental Results

### TOFU Benchmark (Llama-2-7b-chat)

Comparison of AGO against baselines (GA, NPO, SimNPO).

| Method | Forget Quality (↓) | Model Utility (↑) | Fluency (↑) | PLR (→ 0.5) |
| :--- | :---: | :---: | :---: | :---: |
| Target | -46.91 | 0.59 | 0.98 | - |
| Retrain | 0.00 | 0.91 | 0.47 | - |
| GA | 0.48 | 0.00 | 0.45 | - |
| NPO | -19.78 | 0.02 | 0.40 | - |
| **AGO (Ours)** | **-9.43** | **0.59** | **0.90** | **0.53** |

*\> Note: AGO achieves the best balance between unlearning efficacy and utility, with a Privacy Leakage Ratio (PLR) close to the ideal 0.5.*

## 📝 Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{anonymous2025ago,
  title={AGO: Robust and Stabilized LLM Unlearning via Latent Adversarial Gating and Adaptive Orthogonality},
  author={Anonymous},
  booktitle={ACL Submission},
  year={2025}
}
```

## 🙏 Acknowledgements

Our code builds upon the following repositories:

  * [TOFU: A Task of Fictitious Unlearning](https://github.com/locuslab/tofu) 
  * [WMDP Benchmark](https://github.com/centerforaisafety/wmdp) 

-----

*Disclaimer: This repository is for research purposes only.*

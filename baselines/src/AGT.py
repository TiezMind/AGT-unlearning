import torch
from torch import nn
import torch.nn.functional as F
from transformers import Trainer
import copy
import deepspeed
from typing import Any, Dict, Union
from transformers.training_args import OptimizerNames
from transformers.utils import is_sagemaker_mp_enabled

class AdversarialHook:
    """
    一个上下文管理器，用于在指定的 Transformer 层注册一个前向 hook 来添加对抗扰动。
    """
    def __init__(self, model, perturb_layer_idx, delta_tensor):
        self.model = model
        self.perturb_layer_idx = perturb_layer_idx
        self.delta = delta_tensor
        self.hook_handle = None
        self.target_module = None

    def _get_target_module(self):
        """
        根据常见的模型结构 (如 Llama, GPT-2) 找到目标 Transformer 模块。
        """
        # 解包模型 (例如，处理 DDP 或 FSDP 包装)
        unwrapped_model = self.model
        if hasattr(unwrapped_model, "module"): unwrapped_model = unwrapped_model.module

        blocks = None
        if hasattr(unwrapped_model, 'base_model') and hasattr(unwrapped_model.base_model, 'model') and hasattr(unwrapped_model.base_model.model, 'model') and hasattr(unwrapped_model.base_model.model.model, 'layers'):
            blocks = unwrapped_model.base_model.model.model.layers
        else:
            blocks = unwrapped_model.model.layers

        if blocks is None or not (0 < self.perturb_layer_idx <= len(blocks)):
            raise ValueError(f"无法在模型中找到指定的层 {self.perturb_layer_idx}。请检查模型结构和 `perturb_layer` 参数。")
            
        return blocks[self.perturb_layer_idx - 1]

    def _hook_fn(self, module, inputs, outputs):
        """
        这个 hook 函数会在目标模块前向传播后被调用。
        它将 delta 添加到模块的输出上。
        """
        # Transformer 层的输出通常是一个元组 (hidden_state, ...)，我们只修改第一个元素
        hidden_state = outputs[0]
        
        # 确保 delta 和 hidden_state 在同一设备上
        assert hidden_state.device == self.delta.device, \
            f"Device mismatch! hidden_state is on {hidden_state.device} but delta is on {self.delta.device}"
        
        # 添加扰动
        perturbed_hidden_state = hidden_state + self.delta
        perturbed_hidden_state.requires_grad_(True)
        # 返回修改后的输出元组
        if isinstance(outputs, tuple):
            return (perturbed_hidden_state,) + outputs[1:]
        else:
            return perturbed_hidden_state

    def __enter__(self):
        """
        进入上下文时，找到目标模块并注册 hook。
        """
        self.target_module = self._get_target_module()
        self.hook_handle = self.target_module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出上下文时，移除 hook，避免影响后续操作。
        """
        if self.hook_handle:
            self.hook_handle.remove()

def get_batch_loss(logits, labels):
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss

def get_loss(logits, labels):
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    # get the sum loss for each sequence in a batch
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss

def calculate_conflict_penalty(
    gu: torch.Tensor,
    gr: torch.Tensor,
    lambda_0: float,
    gamma: float,
    w_t: float
) -> torch.Tensor:
    """
    根据提供的公式计算自适应冲突惩罚损失。

    Args:
        gu (torch.Tensor): 遗忘梯度 (unlearning gradient)，一个扁平化的张量。
        gr (torch.Tensor): 保留梯度 (retaining gradient)，一个扁平化的张量。
        lambda_0 (float): 基础惩罚超参数 λ₀。
        gamma (float): 聚焦参数 γ (通常 > 1)。
        w_t (float): 当前训练步骤 t 的退火调度值 w(t)。

    Returns:
        torch.Tensor: 计算出的惩罚损失（一个标量）。
    """
    # 确保梯度是一维向量
    if gu.dim() > 1 or gr.dim() > 1:
        raise ValueError("输入梯度 'gu' 和 'gr' 必须是扁平化的1D张量。")

    # 1. 计算点积 <gu, gr>
    # <gu, gr>
    dot_product = torch.dot(gu, gr)

    # 2. 计算铰链损失 (Hinge Loss) 部分
    # max(0, -<gu, gr>)
    # 只有当点积为负（梯度冲突）时，此项才大于0
    # hinge_loss = torch.relu(-dot_product)

    # 3. 计算自适应惩罚系数 λ(t, gu, gr)
    # 3.1 计算余弦相似度 cos(gu, gr)
    # F.cosine_similarity 需要 (1, D) 形状的输入，并添加一个小的 epsilon 防止除以零
    cosine_sim = F.cosine_similarity(gu.unsqueeze(0), gr.unsqueeze(0), eps=1e-8)
    
    # 3.2 计算核心的自适应项
    # ((1 - cos(gu, gr)) / 2) ^ γ
    adaptive_term = ((1 - cosine_sim) / 2).pow(gamma)

    # 3.3 计算完整的 λ
    # λ₀ * adaptive_term * w(t)
    lambda_val = lambda_0 * adaptive_term * w_t

    # 4. 计算最终的惩罚损失
    # λ * max(0, -<gu, gr>)
    # penalty_loss = lambda_val * hinge_loss
    penalty_loss = lambda_val
    
    if cosine_sim.item() < 0:
        penalty_loss = penalty_loss
    else:
        penalty_loss = torch.tensor(0.0, device=gu.device)
    return penalty_loss,cosine_sim

class AGTUnlearner(Trainer):
    """
    一个实现了“潜在对抗遗忘学习”算法的Hugging Face训练器。
    该算法源于您提供的研究文档，其核心思想是在模型的隐空间（Latent Space）
    引入一个min-max对抗博弈，以实现更鲁棒的模型遗忘。

    该训练器新增了一种损失类型: 'latent_adv'。
    """

    def __init__(self, *args, **kwargs):
        # --- 对抗性训练相关的超参数 ---
        # 从kwargs中弹出自定义参数，避免传入Trainer基类时出错
        self.loss_type = kwargs.pop("loss_type", "ga")
        self.ref_model = kwargs.pop("ref_model", None)
        self.beta = kwargs.pop("beta", 0.1)  # NPO超参数
        self.delta = kwargs.pop("delta", 0.0)  # sim_npo
        self.perturb_layer = kwargs.pop("perturb_layer", 4)
        
        # 潜在对抗遗忘 ('latent_adv') 专用参数
        self.adv_epsilon = kwargs.pop("adv_epsilon", 1e-2)  # 对抗扰动 δ 的最大范数 (ε)1e-2
        self.adv_steps = kwargs.pop("adv_steps", 4)         # 内部攻击循环（最大化步骤）的步数
        self.adv_alpha = kwargs.pop("adv_alpha", 5e-3)       # 内部攻击循环的学习率 (α) 5e-3

        # 梯度范数的阈值
        self.max_grad_norm = kwargs.pop("max_grad_norm", float('inf'))
        self.adv_update_threshold = kwargs.pop("adv_update_threshold", 2) 
        # 预热的步数
        self.warmup_steps = kwargs.pop("warmup_steps", 50)   

        # 扰动向量
        self.adversarial_delta = None

        # AO专用参数
        self.gradient_accumulation_steps = kwargs.pop("gradient_accumulation_steps", 8)
        self.gradient_edit = kwargs.pop("gradient_edit", True)
        self.flattened_gradient = 0
        self.flattened_memory_accumulation = 0.0
        self.steps = 0
        # AO 梯度累积器
        self.gradient_accum = {}
        self.memory_grad = {}

        super().__init__(*args, **kwargs)

        # 如果使用KL散度作为保留损失，需要一个参考模型
        if self.ref_model is not None:
            assert 'klr' in self.loss_type or 'AGT' in self.loss_type, \
                "Reference model is only needed for loss types with KL regularization."
            # 使用DeepSpeed准备参考模型，并设置为评估模式
            self.ref_model = self.e_prepare_deepspeed(self.ref_model)


    def e_prepare_deepspeed(self, model):
        """
        使用与主模型相同的DeepSpeed配置来准备参考模型。
        这确保了在分布式训练（尤其是ZeRO Stage 3）中两个模型的分片方式一致。
        参考模型的所有参数都将被冻结（不计算梯度）。
        """
        # 深度拷贝DeepSpeed配置以避免修改原始配置
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        # 针对ZeRO-3的一些优化配置
        if model is not None and hasattr(model, "config"):
            hidden_size = getattr(model.config, "hidden_size", None)
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                config_kwargs.update({
                    "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                    "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                    "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                })

        # 如果不是ZeRO-3，则将参考模型视为普通模型（stage 0）
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        
        # 参考模型不需要优化器
        config_kwargs["optimizer"] = {"type": None}
        
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        
        # 冻结参考模型的所有参数
        for param in model.parameters():
            param.requires_grad = False
        
        return model

    def _compute_kl_loss(self, model_logits, ref_model_logits):
        """
        计算模型输出与参考模型输出之间的KL散度损失（保留损失）。
        """
        # 使用log_softmax将logits转换为对数概率分布
        model_log_probs = F.log_softmax(model_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_model_logits, dim=-1)
        
        # 计算KL散度，'batchmean'表示对batch中的所有样本取平均
        # log_target=True 表示输入的是对数概率
        kl_div = F.kl_div(
            model_log_probs,
            ref_log_probs,
            reduction='batchmean',
            log_target=True
        )
        return kl_div

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        核心函数：根据指定的loss_type计算总损失。
        这里我们重点实现 'latent_adv' 逻辑。
        """
        # =================================================================
        # == 实现：潜在对抗遗忘学习 (Latent Adversarial Unlearning) ==
        # =================================================================
        self.steps += 1
        if self.loss_type in ['AGT']:
            x_forget, x_preserve = inputs

            if self.adversarial_delta is None:
                if self.is_world_process_zero():
                    print(f"\n首次运行，正在为第 {self.perturb_layer} 层确定扰动形状...")
                with torch.no_grad():
                    outputs = model(**x_forget, output_hidden_states=True)
                    # hidden_states 元组的第 N 个元素是第 N 层 transformer 的输出
                    target_hidden_state = outputs.hidden_states[self.perturb_layer]
                    self.adversarial_delta = torch.zeros_like(target_hidden_state, requires_grad=False)
                    if self.is_world_process_zero():
                        print(f"成功初始化第 {self.perturb_layer} 层的扰动，形状为: {self.adversarial_delta.shape}")
            
            # 使用 hook 执行主要的`forget`数据前向传播
            with AdversarialHook(model, self.perturb_layer, self.adversarial_delta):
                outputs_f = model(**x_forget)
            
            with torch.no_grad():
                with AdversarialHook(self.ref_model, self.perturb_layer, self.adversarial_delta):
                    outputs_f_ref = self.ref_model(**x_forget)

            outputs_r = model(**x_preserve)
            loss_r = outputs_r.loss
            if self.gradient_edit:
                loss_r_scaled = loss_r / self.gradient_accumulation_steps
                self.store_grads(model, loss=loss_r_scaled, typ="retain")


            outputs_f_loss = get_batch_loss(outputs_f.logits, x_forget['labels'])
            outputs_f_ref_loss = get_batch_loss(outputs_f_ref.logits, x_forget['labels'])
            neg_log_ratio = outputs_f_loss - outputs_f_ref_loss
            # neg_log_ratio = outputs_f_loss - self.delta
            loss_npo = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta 
            current_unlearn_loss = loss_npo + loss_r
            if self.gradient_edit:
                loss_u_scaled = current_unlearn_loss / self.gradient_accumulation_steps
                self.store_grads(model, loss=loss_u_scaled, typ="objective")

            # if self.state.global_step > 0:
            if self.state.global_step > self.warmup_steps:
                current_unlearn_loss.backward(retain_graph=True)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.max_grad_norm
                )
                log_dict = {f"adv_grad_norm": grad_norm.item()}
                self.log(log_dict)

                model.zero_grad() 
                
                if grad_norm < self.adv_update_threshold:
                    if self.is_world_process_zero():
                        print(f"\nUpdating adversarial delta (grad_norm: {grad_norm:.4f} < threshold: {self.adv_update_threshold})")

                    delta = self.adversarial_delta.clone().detach().requires_grad_(True)
                    for _ in range(self.adv_steps):

                        # 在内部循环中，同样使用 hook 来注入可学习的 delta
                        with AdversarialHook(model, self.perturb_layer, delta):
                            inner_outputs_forget_adv = model(**x_forget)
                        
                        # with torch.no_grad():
                        with AdversarialHook(self.ref_model, self.perturb_layer, delta):
                            inner_outputs_f_ref = self.ref_model(**x_forget)

                        inner_outputs_preserve = model(**x_preserve)
                        inner_loss = inner_outputs_preserve.loss

                        outputs_f_loss = get_batch_loss(inner_outputs_forget_adv.logits, x_forget['labels'])
                        outputs_f_ref_loss = get_batch_loss(inner_outputs_f_ref.logits, x_forget['labels'])
                        neg_log_ratio = outputs_f_loss - outputs_f_ref_loss
                        # neg_log_ratio = outputs_f_loss - self.delta
                        loss_npo = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta
                        loss_for_attack = loss_npo + inner_loss

                        loss_for_attack.backward(retain_graph=True)
                        # self.accelerator.backward(loss_for_attack)
                        
                        delta_grad = delta.grad.detach()

                        delta.data = delta.data + self.adv_alpha * delta_grad.sign()


                        delta.data = torch.clamp(delta.data, -self.adv_epsilon, self.adv_epsilon)
                        delta.grad.zero_()
                    

                    # ================= wandb 日志记录开始 =================
                    delta_norm = torch.norm(delta.data).item()
                    log_dict = {f"delta_norm": delta_norm}
                    self.log(log_dict)
                    # ================= wandb 日志记录结束 =================
                    self.adversarial_delta = delta.detach()
                    model.zero_grad()
                else:
                    model.zero_grad()
            return (current_unlearn_loss, outputs_r) if return_outputs else current_unlearn_loss
        else:
             raise NotImplementedError(f"Loss type '{self.loss_type}' is not implemented in this custom trainer.")

    # ========================================================================
    # == 以下是 GRU 源代码中的必要函数，被 `training_step` 所调用 ==
    # ========================================================================

    def store_grads(self, model, loss=None, typ=None):
        """
        累积指定层的梯度，保留其原始形状。
        (来自 GRU 源码)
        """
        if loss:
            loss.backward(retain_graph=True)

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                
                if typ == "objective":
                    target_dict = self.gradient_accum
                elif typ == "retain":
                    target_dict = self.memory_grad
                else:
                    raise ValueError("Invalid type specified for gradient storage")

                if name not in target_dict:
                    target_dict[name] = torch.zeros_like(param.grad)
                
                target_dict[name] += param.grad.detach()

        if loss:
            model.zero_grad()

            
    def flatten_and_store_grads(self):
        """
        将累积的梯度展平，移动到 CPU，并存储结构图。
        (来自 GRU 源码)
        """
        def flatten_to_cpu_and_record_structure(gradient_dict):
            flattened_grads = []
            structure_map = []
            for name, grad in gradient_dict.items():
                if grad is not None:
                    grad_flat = grad.view(-1)
                    flattened_grads.append(grad_flat)
                    structure_map.append((name, grad.shape))
            
            if flattened_grads:
                return torch.cat(flattened_grads).to('cpu'), structure_map
            else:
                return torch.tensor([], dtype=torch.float32).to('cpu'), []

        self.flattened_gradient, self.structure_map = flatten_to_cpu_and_record_structure(self.gradient_accum)
        self.flattened_memory_accumulation, _ = flatten_to_cpu_and_record_structure(self.memory_grad)

               
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], extra_arg=None) -> torch.Tensor:
        """
        重载的 training_step，以实现 GRU 逻辑。
        (来自 GRU 源码)
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            # ================================================================
            # 1. COMPUTE LOSS & STORE GRADS
            #    我们修改后的 compute_loss 将在这里被调用。
            #    如果 self.gradient_edit=True，它将：
            #    a) 调用 store_grads(..., typ="retain") 来填充 self.memory_grad
            #    b) 调用 store_grads(..., typ="objective") 来填充 self.gradient_accum
            # ================================================================
            loss = self.compute_loss(model, inputs)

        del inputs
        
        kwargs = {}
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()

        # ====================================================================
        # 2. GRU GRADIENT RECTIFICATION
        # ====================================================================
        if self.gradient_edit:
            # 仅在累积步骤的最后一步执行 GRU 逻辑
            if self.steps % self.gradient_accumulation_steps == 0:
                
                # a. 将所有累积的梯度展平并移至 CPU
                self.flatten_and_store_grads()
                
                # b. 清空累积器以便下一轮
                self.gradient_accum = {}
                self.memory_grad = {}

                penalty,cosine_sim = calculate_conflict_penalty(
                    gu=self.flattened_gradient,
                    gr=self.flattened_memory_accumulation,
                    lambda_0=1.0,
                    gamma=2.0,
                    w_t=1.0
                )
                # ================= wandb 日志记录开始 =================
                if self.is_world_process_zero():
                    log_dict = {
                        f"penalty": penalty.item(),
                        f"cosine_sim": cosine_sim.item(),
                    }
                    self.log(log_dict)
                # ================= wandb 日志记录结束 =================
                loss = loss + penalty.to(loss.device)
                self.accelerator.backward(loss, **kwargs)
            else:
                # 对于中间的累积步骤，我们只对原始 loss 进行反向传播
                # Accelerator 会自动处理梯度的累积
                self.accelerator.backward(loss, **kwargs)
        else:            
            self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from molfeat.trans.pretrained import PretrainedDGLTransformer
import torch.nn.functional as F

class PretrainedTransformerWrapper(nn.Module):
    def __init__(self, kind, dtype=float):
        super(PretrainedTransformerWrapper, self).__init__()
        self.transformer = PretrainedDGLTransformer(kind=kind, dtype=dtype)

    def forward(self, smiles_list):
        return self.transformer(smiles_list)


class SparseDispatcher:
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    ## 모든 expert를 combine 해서 output  
    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)

        expected_batch_size = self._gates.size(0)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined[:expected_batch_size]

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
    
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)  
        self.dropout1 = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)  
        out = self.relu(out)
        out = self.dropout1(out)  
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class MoE(nn.Module):
    def __init__(self, transformer_kinds ,  output_size = 1, num_experts = 4, hidden_size = 256, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = k
        
        # Pretrained GIN Feature Extractors
        self.transformers = nn.ModuleList([
            PretrainedTransformerWrapper(kind=k, dtype=float) for k in transformer_kinds
        ])
        # Dynamically determine input size after feature extraction
        self.input_size = len(transformer_kinds) * 300  # Each transformer outputs 300 features

        # MLP Experts
        self.experts = nn.ModuleList([
            MLP(self.input_size, output_size, hidden_size) for _ in range(num_experts)
        ])
        
        # Gating Mechanism
        self.w_gate = nn.Parameter(torch.randn(self.input_size, num_experts) * 0.05, requires_grad=True)
        self.w_noise = nn.Parameter(torch.randn(self.input_size, num_experts) * 0.05, requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def _extract_gin_features(self, smiles_list, device):
        """Batch-wise GIN feature extraction"""
        with torch.no_grad():
            features = [torch.tensor(transformer(smiles_list), dtype=torch.float32).to(device) 
                        for transformer in self.transformers]
        return torch.cat(features, dim=1)  


    ##  Coefficient of Variation Squared (CV², 변동 계수의 제곱)
    """
    x는 각 Expert에 할당된 데이터의 중요도 (게이트 확률 합 등)를 의미.
    cv_squared(x)가 크면 일부 Expert에 많은 샘플이 몰리고, 작은 값이면 모든 Expert가 균등하게 선택
    """
    def cv_squared(self, x):
        eps = 1e-10

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    ## 각 전문가가 선택된 횟수를 계산.
    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    ## 특정 전문가가 Top-K 내에 포함될 확률을 계산.
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    ##각 전문가에 대한 게이트 확률을 계산하고, k개의 전문가를 선택.

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        # 각 샘플 x에 대해 Expert의 선택 확률(logits)을 계산
        clean_logits = x @ self.w_gate

        # Noisy Gating을 추가하여 확률값을 조정
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # Softmax를 적용하여 각 Expert에 대한 확률값을 생성
        logits = self.softmax(logits)
        # Top-K 개수의 Expert를 선택
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        # 선택된 k개의 Expert에 대한 확률 정규화
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)

        # 전체 Expert 게이트 행렬 생성 (할당된 곳만 확률값, 나머지는 0)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)

        return gates, load

    ## 전체 MoE 파이프라인을 실행하여 최종 예측을 생성., Gating -> Expert 선택 -> Expert 실행 -> 결과 결합
    def forward(self, smiles_list, loss_coef=0.5):
        device = next(self.parameters()).device
        
        # Extract pretrained GIN features
        x = self._extract_gin_features(smiles_list, device)
        
        # MoE gating
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        # Dispatching to experts
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        
        # Expert Forward Pass
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        
        return y, loss



class MoE_regression(nn.Module):
    def __init__(self, transformer_kinds, output_size=1, num_experts=4, hidden_size=256, noisy_gating=True, k=4):
        super(MoE_regression, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = k
        
        # Pretrained GIN Feature Extractors
        self.transformers = nn.ModuleList([
            PretrainedTransformerWrapper(kind=k, dtype=float) for k in transformer_kinds
        ])
        self.input_size = len(transformer_kinds) * 300  # Each transformer outputs 300 features

        # MLP Experts
        self.experts = nn.ModuleList([
            MLP(self.input_size, output_size, hidden_size) for _ in range(num_experts)
        ])
        
        # Gating Mechanism
        self.w_gate = nn.Parameter(torch.randn(self.input_size, num_experts) * 0.05, requires_grad=True)
        self.w_noise = nn.Parameter(torch.randn(self.input_size, num_experts) * 0.05, requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def _extract_gin_features(self, smiles_list, device):
        with torch.no_grad():
            features = [torch.tensor(transformer(smiles_list), dtype=torch.float32).to(device) 
                        for transformer in self.transformers]
        return torch.cat(features, dim=1)  

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k, self.num_experts), dim=1)
        top_k_gates = top_logits / (top_logits.sum(1, keepdim=True) + 1e-6)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_indices, top_k_gates)
        return gates

    def forward(self, smiles_list, target=None, loss_coef=0.5):
        device = next(self.parameters()).device
        x = self._extract_gin_features(smiles_list, device)
        gates = self.noisy_top_k_gating(x, self.training)

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)

        # Calculate loss if target is provided
        loss = None
        if target is not None:
            loss_mse = F.mse_loss(y.squeeze(), target, reduction='mean')
            importance = gates.sum(0)
            load = (gates > 0).sum(0)
            loss_load = self.cv_squared(importance) + self.cv_squared(load)
            loss = loss_mse + loss_coef * loss_load
        
        return y, loss



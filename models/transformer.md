# Transformer Small
- num_hidden_layers: 2
- attention dropout: 0.1
- batch size: 4096 -> 512
- dropout: 0.2
- filter size: 2048 -> 512 (?)
- hidden size: 512 -> 256 (이건 어쩔 수 없음)

## Learning rate
- learning_rate: 0.2
- constant: 2.0
- cosine_steps: 250000
- decay_rate: 1.0
- minimum: None
- schedule: constant-linear_warmup-rsqrt_decay-rsqrt_hidden_size
- decay_scheme: noam
- stairsize: False
- decay_steps: 5000
- warmup_steps: 16000
- max_length: 256 -> 20
- moe_hidden_size: 2048
- moe_loss_coef: 0.001
- moe_num_experts: 16
- optimizer_adam_beta1: 0.9
- optimizer_adam_beta2: 0.997
- optimizer_adam_epsilon: 1e-09
- relu_dropout: 0.1

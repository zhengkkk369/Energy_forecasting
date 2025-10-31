
# 电力/新能源预测中的概念漂移与持续学习项目规范（Codex-Friendly）

> 一句话目标：**在单模型框架下**，同时处理 _周期性漂移_ 与 _突变性/渐进性漂移_，并**可量化**灾难性遗忘与适应速度。

---

## 0. 快速导航（TL;DR）
- **核心问题**：
  1) 如何度量周期性漂移下的灾难性遗忘？  
  2) 在**不使用模型集成**的前提下，单模型如何同时适应周期与突变？  
  3) 渐进性漂移是否需要专门设计/修改模型？  
- **解决思路**：**Backbone + 逐层适配器（Fast） + 上下文条件化（FiLM） + 关联记忆检索 + 稳定-可塑正则（EWC） + 自适应窗口/软更新（渐进漂移）**。  
- **评估指标**：在线误差曲线、误差跳升幅度 Δ_spike、恢复时间 T_rec、遗忘率 FR、后向迁移 BWT、适应速度 AS。
- **产出**：完整实验框架（数据流模拟、在线训练与评估脚本）、可复现实验与图表。

---

## 1. 背景与目标
电力负荷与风光发电预测具有强周期性（昼夜/周末/季节）与显著非平稳性（极端天气、节假日、设备异常）。概念漂移导致离线模型退化；目标是在**在线/增量学习**框架内：
- 快速响应**突变性漂移**（误差突增后迅速回落）；
- 长期稳态下**保持旧模式**（循环出现时尽量不遗忘）；
- 对**渐进性漂移**实现平滑过渡（不过拟合噪声、不滞后）。

---

## 2. 数据与场景设定

### 2.1 数据源（示例）
- 负荷/发电功率：小时级时间序列。
- 气象：温度、风速、风向、辐照（或云量）、湿度、降水。
- 日历与社会特征：小时、星期、月份、节假日、工作日/周末等。

### 2.2 缺失与标签延迟
- 缺失值：采用插补（前向填充、样条、KNN）+ 缺失掩码（boolean mask）共同输入模型。
- 标签延迟：预测先行，真实值延后到达；采用**无监督/弱监督漂移检测**与**伪标签**/自监督辅助更新。

### 2.3 漂移类型（在实验中需显式标注/注入）
- 周期性漂移：季节幅度/相位变化、周末模式等。
- 突变性漂移：极端天气、突发停电、政策/事件冲击。
- 渐进性漂移：基础负荷缓慢抬升/下降、设备老化效率变化等。

---

## 3. 模型设计（单模型，无集成）

### 3.1 总体结构
```
Input (x_t, context_t) 
  -> Embedding(time, holiday, weather, missing_mask) 
  -> Backbone (TCN/Transformer/LSTM)
  -> [Per-Layer] Adapter (Fast) + FiLM (context-conditioned)
  -> Memory Retrieval (top-k adapter params by context key)
  -> Gated Fusion (current adapter ⊕ retrieved adapter)
  -> Output Head -> y_hat_t
```

- **Backbone（慢权重，Slow）**：稳定捕捉长期/跨变量依赖，学习率小，带正则。
- **Adapter（快权重，Fast）**：逐层小瓶颈（bottleneck）或 FiLM；快速更新、参数量小。
- **FiLM/条件归一化**：让上下文（时间/天气/节假日等）直接调制层输出。
- **关联记忆**：存储 {context_key → adapter_params}；相似上下文检索并与当前适配融合。
- **稳定-可塑正则（EWC/L2）**：对主干重要参数施加更强保护，防止灾难性遗忘。

### 3.2 Adapter 参考实现
```python
class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck=0.25):
        super().__init__()
        d_hidden = int(d_model * bottleneck)
        self.down = nn.Linear(d_model, d_hidden)
        self.act  = nn.GELU()
        self.up   = nn.Linear(d_hidden, d_model)
    def forward(self, x):
        return self.up(self.act(self.down(x)))
```

### 3.3 FiLM（特征级调制）
给定上下文嵌入 `c`，计算 \(\gamma,\beta\) 并调制：  
\(\mathrm{FiLM}(h) = \gamma(c) \odot h + \beta(c)\)

### 3.4 关联记忆检索（示意）
- Key：上下文指纹（时间/天气 embedding 的拼接，或 EMA 统计）。
- 检索：cosine 相似度 top-k；融合：\( \hat{a} = \alpha a_{\text{current}} + (1-\alpha)\sum w_i a_i^{\text{mem}} \)。  
- 触发：当无监督漂移分数 \(D_t\) 超阈值或情景标签变化时增强记忆权重。

---

## 4. 在线学习与更新策略

### 4.1 双速学习率
- 主干：小 lr（如 1e-5～3e-5）+ EWC 正则。
- 适配器/FiLM/门控/记忆读写：大 lr（如 5e-4～1e-3）。

### 4.2 漂移检测（无监督/弱监督）
- 分布距离（KS、Wasserstein）比较滑窗内输入/隐表征分布。
- 自监督重构/预测损失的异常上升作为漂移信号。
- 有延迟标签时，到达后进行“纠偏重放”。

### 4.3 回放与伪回放
- 小容量代表性缓冲区（reservoir/coreset）；或用生成模型近似历史样本。

### 4.4 渐进性漂移的自适应（必做）
- **权重样本衰减**：\( w_i=\exp(-\lambda \Delta t) \)，\(\lambda\) 随漂移速率自适应。
- **软参数更新**：\( \theta_{t+1} = (1-\alpha_t)\theta_t + \alpha_t \theta_{\text{new}} \)；\(\alpha_t\) 与漂移强度正相关。
- **卡尔曼/贝叶斯更新**：将某些层参数视作时变状态。

---

## 5. 评估指标（可直接编码）

设在情景切换点 \(t_0\) 处：

- **在线误差曲线**：记录 MAE/MSE@time，绘制 L(t)。  
- **误差跳升幅度**：\(\Delta_{\text{spike}} = L(t_0^+) - L(t_0^-)\)。  
- **恢复时间**：\(T_{\text{rec}}\)：从 \(t_0^+\) 到 \(L(t)\le L(t_0^-)+\epsilon\) 所需步数。  
- **遗忘率（FR）**：  
  \[ FR = \frac{P_{\text{before}} - P_{\text{after}}}{P_{\text{before}}} \]  
  （P 可取 RMSE 的负号或 R^2 等性能指标；用误差时改成相对增幅。）
- **后向迁移（BWT）**：新学习对旧情景性能影响的平均值。  
- **适应速度（AS）**：切换后达到给定误差阈值的步数（可与 \(T_{\text{rec}}\) 合并）。

> 建议提供统一 `metrics.py`，给定切换时间戳与误差序列，输出上述指标。

---

## 6. 实验设计

### 6.1 数据与扰动
- 真实数据 + 合成扰动：注入（a）幅度/相位周期变化；（b）突发极端天气片段；（c）缓慢趋势漂移。

### 6.2 方案对比
- Baseline-Static：离线训练，固定参数。  
- Baseline-Incremental：所有参数同速率微调。  
- **Proposed-Single**：Backbone + Adapter + FiLM + 记忆 + EWC（推荐）。  
- 消融：去掉记忆/FiLM/适配器/回放/正则分别对比。

### 6.3 评价
- 逐点在线评估 + 阶段性汇总；绘制误差曲线与关键指标表；统计显著性检验。

---

## 7. 代码结构（建议）
```
project/
  configs/
    default.yaml
    dataset.yaml
    model.yaml
    train_online.yaml
  data/
    raw/  processed/
  src/
    dataio/        # 数据加载、缺失处理、扰动注入
    features/      # 上下文构造、embedding
    models/
      backbone.py
      adapter.py
      film.py
      memory.py
      gating.py
      ewc.py
    drift/
      detectors.py  # KS/Wasserstein、自监督异常分数
      schedulers.py # 触发策略/阈值自适应
    train/
      online_loop.py
      replay_buffer.py
    eval/
      metrics.py
      plots.py
    utils/
      seed.py log.py config.py
  notebooks/       # 速验与可视化
  scripts/
    run_online.sh
    grid_search.sh
  README.md
```

---

## 8. 关键伪代码

### 8.1 在线训练主循环（含漂移触发与记忆）
```python
for t in range(T):
    x_t, ctx_t = get_input(t)
    y_hat_t = model.predict(x_t, ctx_t)     # 前向
    log_error(y_hat_t)                      # 暂存在线误差

    if label_available(t):                  # 标签延迟时可能为 False
        y_t = get_label(t)
        loss = crit(y_hat_t, y_t)

        # 漂移检测（无/弱监督）：输入分布 & 自监督辅助
        D_t = drift_score(x_t, ctx_t, model.hidden_state)
        trigger = (D_t > tau) or periodic_hook(t)

        # 回放批（少量历史样本）
        batch = make_batch(current=(x_t, y_t, ctx_t),
                           replay=replay_buffer.sample(k))

        # 先更新适配器/FiLM/门控（快路）
        loss_fast = loss_on_fast_modules(batch) + reg_fast(batch)
        optim_fast.zero_grad(); loss_fast.backward(); optim_fast.step()

        # 主干慢更新（含 EWC）
        loss_slow = loss_on_backbone(batch) + ewc_penalty(model)
        optim_slow.zero_grad(); loss_slow.backward(); optim_slow.step()

        # 记忆读写（在触发时更积极）
        if trigger:
            memory.write(ctx_t, model.adapter_params_snapshot())
        retrieved = memory.retrieve(ctx_t, topk=3)
        model.fuse_adapters(retrieved)

        replay_buffer.push(x_t, y_t, ctx_t)
```

### 8.2 渐进漂移下的软更新与样本加权
```python
lambda_t = adapt_decay_rate(drift_velocity)
weights  = exp_decay_weights(batch.times, now=t, lam=lambda_t)
loss     = (weights * loss_vector).sum()

alpha_t  = adapt_soft_ratio(drift_strength)
theta    = (1 - alpha_t) * theta + alpha_t * theta_new
```

---

## 9. 配置与超参（示例）
- `adapter.bottleneck`: 0.25（相对 d_model）  
- `optim_fast.lr`: 1e-3；`optim_slow.lr`: 3e-5  
- `drift.ks_window`: 168 小时；`tau`: 根据验证集自动选取（P95/P99）  
- `memory.topk`: 3；`replay.size`: 2k～10k 样本  
- `ewc.lambda`: 10～200（网格搜索）

---

## 10. 结果展示（建议图表）
- 在线 MAE 曲线（含切换点标注）。  
- \(\Delta_{\text{spike}}\)、\(T_{\text{rec}}\) 条形图（各情景）。  
- FR、BWT 雷达图或表格。  
- 消融实验对比表（±记忆/适配器/FiLM/回放/EWC）。

---

## 11. 里程碑
1. **Week 1-2**：数据管道 + 基线模型；在线评估框架搭好。  
2. **Week 3-4**：Adapter/FiLM + 漂移检测/回放；初步结果。  
3. **Week 5-6**：记忆模块 + 渐进漂移自适应；消融与稳健性测试。  
4. **Week 7**：整理指标、绘图与论文撰写材料。

---

## 12. 参考实现要点（Checklist）
- [ ] 缺失值：显式 mask 输入；插补仅作辅助。  
- [ ] 标签延迟：在无标签阶段启用自监督辅助损失；到达后纠偏更新。  
- [ ] 双速优化器：分参数组设置 lr/weight_decay。  
- [ ] EWC/正则只施加于主干；适配器放宽约束。  
- [ ] 漂移触发与阈值自适应：基于验证集或分位数。  
- [ ] 指标统一 API，便于复现实验与画图。

---

## 13. 许可与复现
- 代码与配置尽量保持确定性（固定随机种子、版本锁定）。  
- 建议提供 `environment.yml` 或 `requirements.txt`。

> 科研界的“黑天鹅”很多，但我们至少可以让模型学会记住黑天鹅长什么样：**记忆**与**快速适应**并重，就是本项目的灵魂。

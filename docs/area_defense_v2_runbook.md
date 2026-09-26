# 区域防守 v2：服务器运行说明

v2 是独立的 VMAS 任务 `task=vmas/area_defense_v2`，保留旧任务和旧检查点。三名蓝方从中部附近出发，每人每局最多拦截一名红方；蓝方观察中包含队友是否还有拦截资格。

`none` 与 `oracle` 使用相同的环境规则、奖励、网络和 MAPPO 参数，但分别从头训练。`none` 的战术提示槽恒为零；`oracle` 每一步都知道红方真实的最终主攻线路或分散意图。这是理想信息价值测试，不是真人介入实验。

## 准备：同步代码

先在本地 PowerShell 推送开发分支：

```powershell
cd "D:\codex\marl应用领域\BenchMARL"
git push origin feat/intent-feasibility-pilot
```

再到服务器 Bash 中执行：

```bash
conda activate pytorch-2.1.1
cd ~/BenchMARL
git status --short
```

如果 `git status --short` 有输出，先确认这些改动的来源，不要继续合并。若工作区干净，执行：

```bash
git fetch https://github.com/nan538238/BenchMARL.git feat/intent-feasibility-pilot
git merge --ff-only FETCH_HEAD
git rev-parse --short HEAD
```

v2 代码提交号应为 `590926c` 或其后续提交。下面的命令都在这个服务器环境和目录中执行。

## 第一步：环境接口和零动作检查

```bash
python -m unittest discover -s test -p test_area_defense_vmas.py -v
python examples/check_area_defense_v2.py --episodes 50 --seed 10000
```

第一个测试验证旧版接口，并检查 v2 在三类训练场景中不能靠零动作获胜。第二个脚本在每类场景上测试 50 个种子，报告零动作成功局数和平均回报。`concentrated`、`spread`、`feint` 属于当前训练分布，`late_switch` 单独作为压力测试。

如果前三类又接近 `50/50`，说明任务仍存在捷径，应停止后续训练。零动作失败只说明任务不再能靠原地站位取胜，尚不能证明任务可学。

## 第二步：各跑 2000 帧冒烟训练

下面每条训练命令各占一整行；末尾不要加反斜杠 `\`。先跑 `none`，结束后再跑 `oracle`。

```bash
python benchmarl/run.py algorithm=mappo task=vmas/area_defense_v2 seed=0 task.guidance_mode=none experiment.max_n_frames=2000 experiment.on_policy_collected_frames_per_batch=1000 experiment.on_policy_n_envs_per_worker=2 experiment.on_policy_n_minibatch_iters=1 experiment.on_policy_minibatch_size=250 experiment.evaluation=true experiment.evaluation_interval=1000 experiment.evaluation_episodes=2 experiment.render=false 'experiment.loggers=[csv]' experiment.create_json=false experiment.checkpoint_at_end=true hydra.run.dir="$PWD/outputs/v2_smoke_none_seed0"
python benchmarl/run.py algorithm=mappo task=vmas/area_defense_v2 seed=0 task.guidance_mode=oracle experiment.max_n_frames=2000 experiment.on_policy_collected_frames_per_batch=1000 experiment.on_policy_n_envs_per_worker=2 experiment.on_policy_n_minibatch_iters=1 experiment.on_policy_minibatch_size=250 experiment.evaluation=true experiment.evaluation_interval=1000 experiment.evaluation_episodes=2 experiment.render=false 'experiment.loggers=[csv]' experiment.create_json=false experiment.checkpoint_at_end=true hydra.run.dir="$PWD/outputs/v2_smoke_oracle_seed0"
find outputs/v2_smoke_none_seed0 outputs/v2_smoke_oracle_seed0 -name 'checkpoint_2000.pt' -print
```

`find` 应打印两个 `checkpoint_2000.pt` 路径。任何一组报错都先停止，把完整报错贴出。2000 帧只检查训练、评估和保存流程，不能据此比较效果。

## 第三步：相同预算的初步训练

两组冒烟训练都通过后，再分别从头训练 20 万帧。旧版检查点的环境规则和观察维度不同，不能用于续训 v2。

```bash
python benchmarl/run.py algorithm=mappo task=vmas/area_defense_v2 seed=0 task.guidance_mode=none experiment.max_n_frames=200000 experiment.on_policy_collected_frames_per_batch=2000 experiment.on_policy_n_envs_per_worker=10 experiment.on_policy_n_minibatch_iters=4 experiment.on_policy_minibatch_size=500 experiment.evaluation=true experiment.evaluation_interval=20000 experiment.evaluation_episodes=20 experiment.render=false 'experiment.loggers=[csv]' experiment.create_json=false experiment.checkpoint_at_end=true hydra.run.dir="$PWD/outputs/v2_train_none_seed0"
python benchmarl/run.py algorithm=mappo task=vmas/area_defense_v2 seed=0 task.guidance_mode=oracle experiment.max_n_frames=200000 experiment.on_policy_collected_frames_per_batch=2000 experiment.on_policy_n_envs_per_worker=10 experiment.on_policy_n_minibatch_iters=4 experiment.on_policy_minibatch_size=500 experiment.evaluation=true experiment.evaluation_interval=20000 experiment.evaluation_episodes=20 experiment.render=false 'experiment.loggers=[csv]' experiment.create_json=false experiment.checkpoint_at_end=true hydra.run.dir="$PWD/outputs/v2_train_oracle_seed0"
find outputs/v2_train_none_seed0 outputs/v2_train_oracle_seed0 -name 'checkpoint_200000.pt' -print
```

## 第四步：同场景配对评估

把上一步打印的两个检查点路径分别填入下面的命令。尖括号内是占位符，执行前必须替换成真实路径：

```bash
python examples/evaluate_area_defense.py --baseline "<none检查点路径>" --oracle "<oracle检查点路径>" --episodes 200 --seed 10000 --output outputs/v2_eval_seed0_200
```

脚本会检查两组都来自 v2，并确认除 `guidance_mode` 外的任务参数一致。两套策略会在相同场景种子下评估。重点看 `summary.json` 中的两组成功率、配对成功率差及其区间，同时检查逐局结果。单个训练种子只用于初筛；若两组都可学且 oracle 有稳定增益，再扩展到多个独立训练种子。若两组成功率长期都接近 0% 或 100%，先检查任务难度，不要据此判断真人介入价值。

## 两组均未获胜时：检查任务可完成性与部分进展

下面的规则控制器读取与 `oracle` 策略相同的四分类战术提示：集中或佯攻时派三名蓝方守最终线路，分散时每条线路派一人。它不训练参数，只用于检查当前动作接口与时限下能否完成拦截。运行后分别查看成功局数和平均拦截人数；规则控制器失败并不能单独证明任务物理上不可能，也可能是这个控制器不够好。

```bash
python examples/check_area_defense_v2_rule.py --episodes 50 --seed 10000
```

评估脚本现已额外输出每局 `baseline_captures`、`oracle_captures`，汇总中包含两组平均拦截人数及 0、1、2、3 人的局数分布。旧评估输出不会自动更新；对已有检查点重新评估时必须选择新的输出目录。例如：

```bash
python examples/evaluate_area_defense.py --baseline "<none检查点路径>" --oracle "<oracle检查点路径>" --episodes 200 --seed 10000 --output outputs/v2_eval_seed0_captures
```

如果规则控制器能赢、MAPPO 两组仍是 0 胜率，应先检查奖励塑形和训练难度，不增加真人介入模块或直接扩大训练种子。

## v3 诊断：先验证修正后的奖励能否学会固定 spread

已有 v2 固定 `spread` 诊断在 20 万帧时只拦截 1/3，评估回报从约 -2.43 降到 -2.71。v2 的距离奖励会把红方自行移动造成的距离缩短记为蓝方进步，也会计入已失去拦截资格的蓝方。v3 只修正这项塑形：在同一时刻的红方位置上比较蓝方移动前后，且只计入仍有拦截资格的蓝方。v2 环境及检查点保持不变；v2 和 v3 的回报数值不可直接比较，应比较成功率和拦截人数。

先在本地推送开发分支，再按“准备：同步代码”中的步骤在服务器拉取。确认服务器已更新至含 v3 的提交后，在 `pytorch-2.1.1` 环境运行：

```bash
python -m unittest discover -s test -p test_area_defense_v3.py -v
```

两个测试都通过后，跑 v3 固定 `spread`、`none` 提示的 2000 帧冒烟训练；命令占一整行：

```bash
python benchmarl/run.py algorithm=mappo task=vmas/area_defense_v3 seed=0 task.guidance_mode=none task.opponent_style=spread experiment.max_n_frames=2000 experiment.on_policy_collected_frames_per_batch=1000 experiment.on_policy_n_envs_per_worker=2 experiment.on_policy_n_minibatch_iters=1 experiment.on_policy_minibatch_size=250 experiment.evaluation=true experiment.evaluation_interval=1000 experiment.evaluation_episodes=2 experiment.render=false 'experiment.loggers=[csv]' experiment.create_json=false experiment.checkpoint_at_end=true hydra.run.dir="$PWD/outputs/v3_smoke_spread_none_seed0"
```

冒烟训练和测试都正常，再以与 v2 固定 `spread` 相同的预算从头训练；不能用 v2 检查点续训 v3：

```bash
python benchmarl/run.py algorithm=mappo task=vmas/area_defense_v3 seed=0 task.guidance_mode=none task.opponent_style=spread experiment.max_n_frames=200000 experiment.on_policy_collected_frames_per_batch=2000 experiment.on_policy_n_envs_per_worker=10 experiment.on_policy_n_minibatch_iters=4 experiment.on_policy_minibatch_size=500 experiment.evaluation=true experiment.evaluation_interval=20000 experiment.evaluation_episodes=20 experiment.render=false 'experiment.loggers=[csv]' experiment.create_json=false experiment.checkpoint_at_end=true hydra.run.dir="$PWD/outputs/v3_diag_spread_none_seed0"
```

查看评估曲线，并对最终检查点做一个固定场景能力检查：

```bash
find outputs/v3_diag_spread_none_seed0 -name 'eval_reward_episode_reward_mean.csv' -print -exec cat {} \;
python -c 'from pathlib import Path; from examples.evaluate_area_defense import _load,_one; p=next(Path("outputs/v3_diag_spread_none_seed0").glob("*/checkpoints/checkpoint_200000.pt")); e=_load(p); print(_one(e,10000)); e.close()'
```

判定重点是 `captures` 和 `success`，不要用 v2/v3 原始 return 比高低。固定 `spread` 的 `scenario_id` 已是同一任务，单局结果是能力诊断；若 v3 仍只拦截 0–1 人，就停止 `oracle` 组和多随机种子训练，进一步检查策略动作、时间窗口和奖励权重。若 v3 能稳定拦截 3 人，再测试多个场景种子，随后回到 `mixed` 下做同预算 `none`/`oracle` 对比。

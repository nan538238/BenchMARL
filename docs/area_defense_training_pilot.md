# 可训练的区域防守信息上限试验

这一步使用同一 VMAS 区域防守任务和同样的 MAPPO 配置，分别训练两个策略：`guidance_mode=none` 时每个蓝方的战术提示槽恒为零；`guidance_mode=oracle` 时提示槽为红方真实计划进攻方向的四分类 one-hot。两组观察维度、动作、奖励和网络结构相同。oracle 提示是环境隐藏信息，真人在当前界面下不可能直接获得；它只测试完美战术信息的乐观上限。

## 服务器冒烟测试

先在 `~/BenchMARL`、`pytorch-2.1.1` 环境下运行 VMAS 接口测试，再确认新任务可训练 2000 帧：

```bash
python -m unittest discover -s test -p test_area_defense_vmas.py -v
```

```bash
python benchmarl/run.py algorithm=mappo task=vmas/area_defense seed=0 task.guidance_mode=none experiment.max_n_frames=2000 experiment.on_policy_collected_frames_per_batch=1000 experiment.on_policy_n_envs_per_worker=2 experiment.on_policy_n_minibatch_iters=1 experiment.on_policy_minibatch_size=250 experiment.evaluation=true experiment.evaluation_interval=1000 experiment.evaluation_episodes=2 experiment.render=false 'experiment.loggers=[csv]' experiment.create_json=false experiment.checkpoint_at_end=true hydra.run.dir="$PWD/outputs/smoke_area_defense_none"
```

这条冒烟命令成功并生成 `checkpoint_2000.pt` 后，再把 `task.guidance_mode=none` 改为 `task.guidance_mode=oracle`，输出目录改为 `outputs/smoke_area_defense_oracle`，确认两种模式都能跑。若有异常，停止长训并贴完整 traceback。

## 正式的小规模训练对照

通过冒烟后，采用相同种子、训练量和硬件训练两组。第一次先跑种子 0，检查学习曲线，然后扩展到 5 个独立训练种子。

```bash
python benchmarl/run.py algorithm=mappo task=vmas/area_defense seed=0 task.guidance_mode=none experiment.max_n_frames=200000 experiment.on_policy_collected_frames_per_batch=2000 experiment.on_policy_n_envs_per_worker=10 experiment.on_policy_n_minibatch_iters=4 experiment.on_policy_minibatch_size=500 experiment.evaluation=true experiment.evaluation_interval=20000 experiment.evaluation_episodes=20 experiment.render=false 'experiment.loggers=[csv]' experiment.create_json=false experiment.checkpoint_at_end=true hydra.run.dir="$PWD/outputs/train_area_defense_none_seed0"
python benchmarl/run.py algorithm=mappo task=vmas/area_defense seed=0 task.guidance_mode=oracle experiment.max_n_frames=200000 experiment.on_policy_collected_frames_per_batch=2000 experiment.on_policy_n_envs_per_worker=10 experiment.on_policy_n_minibatch_iters=4 experiment.on_policy_minibatch_size=500 experiment.evaluation=true experiment.evaluation_interval=20000 experiment.evaluation_episodes=20 experiment.render=false 'experiment.loggers=[csv]' experiment.create_json=false experiment.checkpoint_at_end=true hydra.run.dir="$PWD/outputs/train_area_defense_oracle_seed0"
```

训练完成后，找到每组 `checkpoint_200000.pt` 的绝对路径，运行配对评估。评估脚本每局使用相同随机种子分别运行两种策略，并输出逐局胜负、回报、步数和配对成功率差。

```bash
find outputs/train_area_defense_none_seed0 outputs/train_area_defense_oracle_seed0 -name 'checkpoint_200000.pt' -print
python examples/evaluate_area_defense.py --baseline /绝对路径/none/checkpoint_200000.pt --oracle /绝对路径/oracle/checkpoint_200000.pt --episodes 200 --seed 10000 --output outputs/eval_area_defense_seed0
```

不要根据第一个训练种子的评估结果反复改测试对手。若两组学习曲线都平、成功率极端接近 0 或 1，应先诊断动作幅度、奖励和难度；若两组都可学而 oracle 有稳定增益，才值得继续开发“同一冻结策略上的有限预算请求”和部署可得信息的模拟指挥员。两套分别训练的策略之间的差距**不是**单次人类介入的因果效应，也不是论文主实验。

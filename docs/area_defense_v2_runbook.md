# Area-defense v2: server runbook

Version 2 is a separate VMAS task (`task=vmas/area_defense_v2`). It leaves the
original task and checkpoints unchanged. Three blue defenders start close to
the centre line. A blue defender can intercept only one red attacker per
episode, and all defenders observe which teammates still have capacity.

The `none` and `oracle` runs use the same dynamics and MAPPO settings. The
oracle sees the scripted red team's true final lane or spread intention in
every observation. It is an optimistic information test, not a human study.

## Gate 1: environment and zero-action diagnosis

Run from `~/BenchMARL` in `pytorch-2.1.1` after fetching the new commit:

```bash
python -m unittest discover -s test -p test_area_defense_vmas.py -v
python examples/check_area_defense_v2.py --episodes 50 --seed 10000
```

The first test verifies the v1 interface and confirms that zero action fails
against each v2 training style. The second reports zero-action success over
50 seeds for all four styles. If a training style is again near 50/50, do not
train; the task still has a shortcut. A zero-action failure only establishes
nontriviality, not learnability.

## Gate 2: short MAPPO runs

Use single-line commands in Bash; do not append a trailing backslash.

```bash
python benchmarl/run.py algorithm=mappo task=vmas/area_defense_v2 seed=0 task.guidance_mode=none experiment.max_n_frames=2000 experiment.on_policy_collected_frames_per_batch=1000 experiment.on_policy_n_envs_per_worker=2 experiment.on_policy_n_minibatch_iters=1 experiment.on_policy_minibatch_size=250 experiment.evaluation=true experiment.evaluation_interval=1000 experiment.evaluation_episodes=2 experiment.render=false 'experiment.loggers=[csv]' experiment.create_json=false experiment.checkpoint_at_end=true hydra.run.dir="$PWD/outputs/v2_smoke_none_seed0"
python benchmarl/run.py algorithm=mappo task=vmas/area_defense_v2 seed=0 task.guidance_mode=oracle experiment.max_n_frames=2000 experiment.on_policy_collected_frames_per_batch=1000 experiment.on_policy_n_envs_per_worker=2 experiment.on_policy_n_minibatch_iters=1 experiment.on_policy_minibatch_size=250 experiment.evaluation=true experiment.evaluation_interval=1000 experiment.evaluation_episodes=2 experiment.render=false 'experiment.loggers=[csv]' experiment.create_json=false experiment.checkpoint_at_end=true hydra.run.dir="$PWD/outputs/v2_smoke_oracle_seed0"
find outputs/v2_smoke_none_seed0 outputs/v2_smoke_oracle_seed0 -name 'checkpoint_2000.pt' -print
```

Stop if either run errors. The short runs check engineering only; do not
compare their rewards as a result.

## Gate 3: paired training pilot

Only after both smoke runs pass, train both policies from scratch. The v1
checkpoints cannot be resumed into v2 because the rules and observations
changed.

```bash
python benchmarl/run.py algorithm=mappo task=vmas/area_defense_v2 seed=0 task.guidance_mode=none experiment.max_n_frames=200000 experiment.on_policy_collected_frames_per_batch=2000 experiment.on_policy_n_envs_per_worker=10 experiment.on_policy_n_minibatch_iters=4 experiment.on_policy_minibatch_size=500 experiment.evaluation=true experiment.evaluation_interval=20000 experiment.evaluation_episodes=20 experiment.render=false 'experiment.loggers=[csv]' experiment.create_json=false experiment.checkpoint_at_end=true hydra.run.dir="$PWD/outputs/v2_train_none_seed0"
python benchmarl/run.py algorithm=mappo task=vmas/area_defense_v2 seed=0 task.guidance_mode=oracle experiment.max_n_frames=200000 experiment.on_policy_collected_frames_per_batch=2000 experiment.on_policy_n_envs_per_worker=10 experiment.on_policy_n_minibatch_iters=4 experiment.on_policy_minibatch_size=500 experiment.evaluation=true experiment.evaluation_interval=20000 experiment.evaluation_episodes=20 experiment.render=false 'experiment.loggers=[csv]' experiment.create_json=false experiment.checkpoint_at_end=true hydra.run.dir="$PWD/outputs/v2_train_oracle_seed0"
find outputs/v2_train_none_seed0 outputs/v2_train_oracle_seed0 -name 'checkpoint_200000.pt' -print
```

Use the two printed checkpoint paths with `examples/evaluate_area_defense.py`
for 200 paired episodes. The script verifies that both checkpoints are v2 and
all task settings except `guidance_mode` match. Report success rates and the
paired gain, not only return. A single training seed is diagnostic; repeat
with independent training seeds before making a publication claim.

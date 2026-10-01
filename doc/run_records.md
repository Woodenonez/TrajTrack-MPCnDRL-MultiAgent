# Run records and project boundaries

Pass `--output-dir runs/NAME` to the experiment launcher to reserve a new directory. Existing directories and files are not overwritten. `metadata.json` stores workflow selections, source revision, dependencies, MPC YAML, and checkpoint/native-solver paths and hashes. Each selected trial/method writes its own numeric NPZ, readable with `numpy.load(path, allow_pickle=False)`; fleet runs write `trial_001_mpc_fleet.npz` with robot-prefixed keys.

Single-robot records sample at the end of completed outer control iterations:

| Field | Meaning |
| --- | --- |
| `environment_states` | Environment robot state; it may still reflect synchronization before the latest MPC solve. |
| `mpc_states` | State after MPC stepping, on the controller's own clock. |
| `rl_actions` | Applied pure-RL policy output: DDPG acceleration or DQN discrete index. |
| `mpc_actions` | Applied MPC velocity control, absent in pure-RL runs. |
| `chosen_references` | References selected by the existing control loop. Hybrid policy proposals are not applied RL actions. |
| `timing_ms` | Existing runner timing values and labels; recording does not add inference calls. |
| `success`, `collided`, `budget_exhausted`, `termination` | Episode outcome and reason for stopping. A bounded `step_limit` is not successful task completion. |

Do not pad or resample independent state/action streams to make their shapes match. Fleet `done` records whether a robot has completed during the run; completion is sticky in the original simulator, which can continue stepping that robot while others finish. Fleet termination records use that same meaning. A controller reporting completion without a new action ends the episode; it does not restart it. Solver errors retain available evidence and make the launcher fail. Smoke checks additionally require executed samples.

Environment validation runs on an independent scene copy. Each validation reset receives a fresh map copy; the actual experiment then initializes its intended map. This repairs the original checker-induced mutation/reset hang without changing the general training map generator or the numerical control loop.

## Relationship to the EBM warehouse project

The command layout and optional run records follow the related `DyObAv-MPCnEBM-Warehouse` project's conventions. These shared conventions will help later integration. The numerical models, checkpoints, and generated solvers remain project-specific.

| Boundary | EBM warehouse | TrajTrack |
| --- | --- | --- |
| Launcher | `scripts/run_simulation.py` | `scripts/run_experiment.py` |
| Selection | Scenario and predictor/tracker | Workflow, scene triple or fleet case |
| Budget | Simulation duration | Outer steps; fleet advances by configured `action_steps` |
| Repetition | Repeats of one configuration | Trials; quantitative comparisons run trial first, then each method |
| Neural output | Motion prediction | DQN/DDPG action or hybrid reference proposal |
| Native optimizer | `mpc_fast` | `navi_default` |
| Stored histories | Combined trajectories NPZ with object/repeat keys | Per-trial/method NPZ or robot-prefixed fleet NPZ |

A future integration should adapt launch selections and read existing records with their units and sampling points preserved. Matching sampling intervals or horizon lengths do not make the solvers interchangeable. Shipped `.pt` policy state dictionaries also differ from training-produced Stable-Baselines3 archives and from EBM prediction checkpoints. No cross-project numerical conversion is implied by the shared directory structure.

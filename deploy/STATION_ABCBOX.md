# abcbox station — setup and inference runbook

Bring-up notes for the `abcbox_config` YAM dual-arm station: what its hardware
is, what had to change in the shared deploy code to support it, and the exact
commands that were verified against the real arms.

Everything below was measured on this station on 2026-09-19 with
`cache/vla_abc130k_200000_v2.pt` (ABC-VLA, abc130k step 200000).

## Hardware

| Part | Value |
| --- | --- |
| GPU | RTX 5090 Laptop, 23.4 GiB usable, driver 595.84 |
| Cameras | 3x RealSense D405, FW 5.15.1.55, USB 3.2 |
| `top` | `260322279434` (overhead, both arms in frame) |
| `left` | `260322279288` (left wrist) |
| `right` | `260522273363` (right wrist) |
| Followers | 2x Geschwister Schneider (gs_usb) CAN, 1 Mbit/s |
| left CAN | serial `20523381594E5018`, interface `can_follower_l` |
| right CAN | serial `207A37B045465006`, interface `can_follower_r` |
| Grippers | Flex Point adaptive gripper (`flexible_4310`) on both arms, replacing the `linear_4310` used for the 2026-09-19 measurements |
| Leaders | none attached — teleop, data collection and DAgger are unavailable here |

Python 3.11 in `.venv`; `torch 2.11.0+cu128` runs on this GPU (sm_120).

## What changed in the shared code

`deploy/robot/config.py`:

1. `_profile()` gained a `follower_channels` keyword, defaulting to
   `("can_l_foll", "can_r_foll")`. The follower CAN interface name was
   previously hardcoded as `f"can_{side[0]}_foll"`.
2. `abcbox_config` passes `("can_follower_l", "can_follower_r")`.
3. `abcbox_config` sets `gripper_type="flexible_4310"`.

`deploy/robot/followers/yam_follower.py`:

The pinned i2rt fork (`arthurallshire/i2rt-drivers@852f6ff`) has no
`FLEXIBLE_4310` gripper type, so `flexible_4310` used to raise `KeyError` at
follower boot. Upstream i2rt's `flexible_4310` config is the `linear_4310`
config with the gripper motor direction set to `-1` and nothing else changed
(same DM4310, kp 20 / kd 0.5, calibrated limits, same force limiter). The
follower therefore maps `flexible_4310` to the fork's `LINEAR_4310` and flips
the gripper motor direction to `-1` while `get_yam_robot` builds the motor
chain. The flip has to happen before construction because
`detect_gripper_limits` uses the direction to decide which detected limit is
open and which is closed.

The `-1` was verified on both arms on 2026-10-05: after calibration the
gripper reads `0.0` closed and `1.0` open, which is i2rt's convention
(`gripper_limits = [closed, open]`, mapped by `JointMapper` to 0..1). With
direction `+1` the detected limits come out in the opposite order and the
gripper observation is inverted. Detected raw limits were `[-0.25, 5.12]` rad
(left) and `[-0.33, 5.14]` rad (right), with commanded 0 / 1 reading
`0.007 / 0.999` and `0.009 / 0.997`.

This station already had `/etc/udev/rules.d/90-can.rules` naming its adapters
`can_follower_l` / `can_follower_r`, and that rules file sorts before the
`99-can-names.rules` that `deploy/scripts/setup_can_names.sh` writes — udev
honours only the first `NAME=`, so the repo script could not have renamed them
without removing the existing file first. Making the profile name the
interfaces it finds avoids touching system configuration at all.

`_ABCBOX_CAN` is therefore all empty strings: `setup_can_names.sh` has nothing
to rename for this profile and is a no-op, which is the intended behaviour. The
adapter serials live in a comment next to it.

All six pre-existing profiles keep `can_l_foll` / `can_r_foll` via the default
argument and are unaffected.

## Setup

```bash
uv sync --extra deploy
uv run prepare.py --vla-pretrained      # ~8.8 GB + sim assets for its five tasks
export ROBOT_PROFILE=abcbox_config
```

The Gemma tokenizer ships in the repo (`abc_minimal/gemma/tokenizer/`), so no
separate download is needed for ABC-VLA. The v2 checkpoint embeds its
`norm_stats`; `--norm-stats-path` is not required.

Check the CAN interfaces are up before a run:

```bash
ip -br link show type can     # expect can_follower_l / can_follower_r, both UP
```

`flow_base.rules` auto-configures anything matching `can*` at 1 Mbit/s, so both
come up on plug-in.

## Running inference

Verified working, with grasping:

```bash
export ROBOT_PROFILE=abcbox_config
CUDA_VISIBLE_DEVICES=0 uv run deploy/deploy_policy.py \
    --checkpoint-path=cache/vla_abc130k_200000_v2.pt \
    --prompt='fold and stack the t shirts' \
    --diffusion-steps=10 --fast-inference \
    --rtc --rtc-prefix-length=4 --rtc-inference-lead-steps=7 \
    --execute-chunk-dim=16
```

The rollout prints a per-joint first-action table and waits on ENTER before
anything moves. It needs a real terminal: launched without a TTY it reports
`No terminal available for policy_rollout` and then dies on the confirmation.

`--debug` runs cameras and inference without commanding the followers, but note
it also skips the follower nodes entirely, so it does not exercise the arms or
read their real joint angles.

## Prompt strings matter more than anything else here

Prompts must match the ABC-130k training task name with underscores as spaces.
The checkpoint metadata is explicit:

> Prompt a task any other way and the policy is off-distribution with no error
> to tell you so.

The task vocabulary is the directory listing of the dataset:
`https://huggingface.co/api/datasets/XDOF/ABC-130k/tree/main/data/train?limit=1000`
(~200 real teleop tasks). Folding is the largest category — 36 tasks, 883 h of
the 3553 h of real teleop — so cloth tasks are the best supported on this
hardware: `fold_and_stack_the_t_shirts`, `fold_and_stack_the_towels`,
`fold_and_stack_the_mixed_laundry_pile`, `roll_the_towels`,
`place_the_t_shirt_on_the_hanger`, and so on.

Invented phrasings such as `pick up bottle` are not task names and behaved
markedly worse (see below).

## Measured

| Metric | Value |
| --- | --- |
| VLA load | 5.5 s, 8.24 GiB allocated |
| Peak VRAM | 8.29 GiB of 23.4 GiB |
| Inference, non-RTC | 51 ms median at `--diffusion-steps=10` |
| Inference, first chunk with `--fast-inference` | 50 ms |
| RTC collect window | median 221 ms, p95 251 ms, max 274 ms |

The `deploy/README.md` figure of "~24 GB" for VLA inference is conservative:
the Gemma/SigLIP backbone loads in bf16 (`abc_minimal/policy.py`), and actual
peak here was 8.3 GiB.

The RTC number spans the asynchronous overlap window, not pure compute — it
tracks the `--rtc-inference-lead-steps=7` budget of 233 ms rather than measuring
the model. The 50 ms first chunk is the real compute cost.

`--fast-inference` compiles with `max-autotune` and takes several minutes on
first launch. It logs many
`OutOfMemoryError: out of resource: triton_mm Required: N Hardware limit: 101376`
lines; these are Triton shared-memory limits per SM, not GPU VRAM exhaustion.
The affected candidates are skipped and compilation proceeds.

### Two 2-minute rollouts, same station and scene

| | `pick up bottle`, no RTC | `fold and stack the t shirts`, RTC + fast |
| --- | --- | --- |
| chunks | 200 | 242 |
| L gripper range | 0.014 rad | 0.982 rad (0.018 → 0.999) |
| R gripper range | 0.026 rad | 0.985 rad (0.014 → 1.000) |
| largest joint | L_J3 0.765 rad | L_J1 1.828 rad (104.7 deg) |
| behaviour | arms reach, grippers never actuate | grasps and manipulates the shirt |

Prompt and RTC changed together between these runs, so the contribution of each
was not isolated. Both are plausible: an in-distribution task name, and RTC's
prefix conditioning giving the chunk sequence enough temporal consistency to
commit to a grasp rather than re-planning the approach every 16 steps.

## Measurement traps

Two mistakes cost real time during bring-up:

- **A single action chunk is 0.53 s.** Per-chunk deltas of 0.03-0.09 rad look
  like a static hold but integrate to ~100 deg of joint travel over a 2-minute
  run. Judge motion from the executed state over the whole run, not one chunk.
- **Raw i2rt motor reads give uncalibrated gripper values** (around -5.0 rad),
  while the rollout feeds the policy calibrated ones (0..1) after
  `detect_gripper_limits`. Offline tests built from raw reads are invalid in
  the gripper dimension. Use `obs_state` from the rollout's `inference_events`
  ZMQ topic instead.

## Known issues

- A rollout ended with
  `SubscribeTimeout: no message on camera:left within 500ms`. Whether this was
  a genuine dropout of the left D405 or just the shutdown race after Ctrl-C
  (the camera node exits first, then the rollout times out on it) has not been
  established. Three D405s at 640x480@30 share one USB 3 controller here, so a
  real bandwidth problem is plausible and would matter for long runs.
- No GELLO leaders: `run_teleop.py`, `run_data_record.py` and `dagger.py`
  cannot run. Collecting finetuning data on this station is blocked until
  leaders are attached and their devices added to the profile.
- `init_q` is the generic `_DEFAULT_INIT_Q`, not tuned for this workspace.
  `DEPLOY_INIT_Q` overrides it with 14 values without editing the profile.
- The measurements above were taken with `linear_4310` grippers. The Flex
  Point gripper has the same 0.096 m stroke, so the 0..1 gripper command is
  nominally compatible, but ABC-130k was collected with rigid fingers and the
  soft tips change grasp behaviour. No rollout has been measured with the
  flexible gripper yet.

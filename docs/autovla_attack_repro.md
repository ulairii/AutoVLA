# AutoVLA nuScenes Attack Reproduction Notes

## Literature Snapshot

VLA-specific attack work is the best fit for AutoVLA because the output is an action-token trajectory, not just text. I did not find a public repository that already plugs an attack into AutoVLA on nuScenes end to end, so this reproduction uses a VLA-style visual patch attack adapter against AutoVLA image inputs and action outputs.

| Priority | Work | Code status | Relation to this reproduction |
| --- | --- | --- |
| VLA | BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models via Objective-Decoupled Optimization | Paper/project/code: arXiv `2505.16640`, NeurIPS/OpenReview, `Zxy-MLlab/BadVLA`, `eggplantsalt/hijack` | Direct VLA attack; most relevant family conceptually, but robotics VLA rather than autonomous-driving AutoVLA. |
| VLA | Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics | Paper/project/code: arXiv `2411.13587`, ICCV 2025, `William-wAng618/roboticAttack`, `vlaattacker.github.io` | Direct visual adversarial robustness study for VLA action outputs; closest to this image-patch reproduction. |
| VLA | Model-agnostic Adversarial Attack and Defense for Vision-Language-Action Models | Paper/project trail found; proposes Disruption Patch Attack / EDPA | VLA patch-style threat model, relevant to image-input perturbations. |
| VLA | On Robustness of Vision-Language-Action Model against Multi-modal Perturbations | Code: `gakakulicc/RobustVLA`; arXiv `2510.00037` | Robustness benchmark across action, instruction, environment, and observation perturbations. |
| VLA / robot-control | RoboPAIR / Jailbreaking LLM-Controlled Robots | Paper/project: arXiv `2410.13691`, `robopair.org` | Safety/jailbreak attack on robot-control agents; less direct for trajectory perturbation but useful VLA-adjacent reference. |
| VLA index | Awesome Attacks on VLA Models | Code/list: `jonastbrg/awesome-vla-attacks` | Curated index of VLA attack/backdoor/jailbreak/safety papers and code. |
| VLM fallback | BadVLMDriver: Physical Backdoor Attack can Jeopardize Driving with Vision-Large-Language Models | Public paper/code trail found | Autonomous-driving security setting, but VLM not VLA; useful only as fallback evidence. |
| VLM fallback | ADvLM / AD-VLM-style visual attack work on autonomous-driving VLMs | Public code found in the attack family | Driving image attack setting, but not action-token VLA. |
| Baseline | Adversarial Patch, Brown et al., 2017 | Foundational public implementation family | Threat model used for this first AutoVLA bridge. |

## Local Paths

- AutoVLA repo: `/home/runw/Project/AutoVLA`
- Raw nuScenes root found locally: `/mnt/indigo/tigersec/runw/datasets/nuscenes/data_full/nuscenes`
- Work directory: `/mnt/indigo/tigersec/runw/workdirs/autovla_attack`
- Default preprocessed scene directory expected by the attack script: `/mnt/indigo/tigersec/runw/workdirs/autovla_attack/nuscenes_val`

## Prepare nuScenes JSON Samples

AutoVLA needs preprocessed JSON scene files before evaluation or attack generation.

```bash
cd /home/runw/Project/AutoVLA
conda activate nusc_preprocess
python tools/preprocessing/nusc_sample_generation.py \
  --nuscenes_path /mnt/indigo/tigersec/runw/datasets/nuscenes/data_full/nuscenes \
  --output_dir /mnt/indigo/tigersec/runw/workdirs/autovla_attack/nuscenes_val \
  --split val \
  --version v1.0-trainval
```

If DriveLM annotations are available, add `--drivelm_path /path/to/v1_1_train_nus.json`.

## Generate Attack Samples and Visualizations

This does not require a model checkpoint. It copies the referenced camera frames into the workdir, applies a visible adversarial patch to the copies, writes adversarial scene JSON files, and saves PNG visualizations.

```bash
cd /home/runw/Project/AutoVLA
NUM_SAMPLES=8 bash scripts/run_nusc_patch_attack.sh
```

Outputs:

- `/mnt/indigo/tigersec/runw/workdirs/autovla_attack/nusc_patch_attack/adv_scenes`
- `/mnt/indigo/tigersec/runw/workdirs/autovla_attack/nusc_patch_attack/adv_images`
- `/mnt/indigo/tigersec/runw/workdirs/autovla_attack/nusc_patch_attack/visualizations`
- `/mnt/indigo/tigersec/runw/workdirs/autovla_attack/nusc_patch_attack/attack_results.jsonl`

## Run with AutoVLA Prediction

The compatible `autovla_codeclean` environment, Qwen2.5-VL-3B base model, and official AutoVLA checkpoint were configured locally:

- Qwen: `/mnt/indigo/tigersec/runw/workdirs/autovla_attack/models/qwen/Qwen2.5-VL-3B-Instruct`
- AutoVLA checkpoint: `/mnt/indigo/tigersec/runw/workdirs/autovla_attack/models/autovla_ckpt/AutoVLA_PDMS_89.ckpt`
- Attack config: `/home/runw/Project/AutoVLA/config/eval/qwen2.5-vl-3B-nusc-sft-autovla-attack.yaml`

```bash
cd /home/runw/Project/AutoVLA
export CONFIG=/home/runw/Project/AutoVLA/config/eval/qwen2.5-vl-3B-nusc-sft-autovla-attack.yaml
export CHECKPOINT=/mnt/indigo/tigersec/runw/workdirs/autovla_attack/models/autovla_ckpt/AutoVLA_PDMS_89.ckpt
export NUM_SAMPLES=8
conda run -n autovla_codeclean bash scripts/run_nusc_patch_attack.sh
```

The JSONL includes clean and adversarial action-token outputs plus decoded trajectories, and the PNGs overlay GT, clean prediction, and adversarial prediction.

## Verified Results

Output files:

- Main JSONL: `/mnt/indigo/tigersec/runw/workdirs/autovla_attack/nusc_patch_attack/attack_results.jsonl`
- Shift summary: `/mnt/indigo/tigersec/runw/workdirs/autovla_attack/nusc_patch_attack/trajectory_shift_summary.csv`
- Visualizations: `/mnt/indigo/tigersec/runw/workdirs/autovla_attack/nusc_patch_attack/visualizations`

All 8 evaluated samples produced 10 clean action tokens and 10 adversarial action tokens. Final clean-vs-attack XY trajectory shifts ranged from `0.575 m` to `10.898 m`.

## RTPT-like Defense Check

The local `/home/runw/Project/R-TPT` codebase is not directly compatible with AutoVLA because it is a CLIP classification-time prompt-tuning method, not an autoregressive action-token trajectory model. Its assumptions include:

- trainable prompt tokens inside a CLIP classifier
- entropy-based confident-sample selection over class logits
- image-classification-style TTA batches

To still test the core idea, I built an `rtpt_like` AutoVLA defense that keeps the same spirit:

- attacked image input only
- multiple test-time image variants (`identity`, `median3`, `blur`, `brightness_down`, `contrast_up`)
- AutoVLA trajectory prediction on each variant
- consistency-based trajectory selection as the defended output

Outputs:

- `/mnt/indigo/tigersec/runw/workdirs/autovla_attack/rtpt_like_defense/defense_results.jsonl`
- `/mnt/indigo/tigersec/runw/workdirs/autovla_attack/rtpt_like_defense/defense_shift_summary.csv`
- `/mnt/indigo/tigersec/runw/workdirs/autovla_attack/rtpt_like_defense/visualizations`

I evaluated the top 5 hardest attack samples by final clean-vs-attack trajectory shift. Result: the RTPT-like defense improved only `1/5` samples and worsened `3/5`, with `1/5` unchanged. So it is a runnable baseline, but not a good defense for AutoVLA in its current form.

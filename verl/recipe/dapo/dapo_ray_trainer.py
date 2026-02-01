# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # reload for mix
        def select_batch_by_index(batch:DataProto, index2idx:dict, indexes:list, n=16) -> DataProto:
            """Select a subset of DataProto by given data indexes of DAPO."""
            selected_batch_idxs = []
            for sample_index in indexes:
                data_idxs = index2idx[sample_index]
                if len(data_idxs) < n:
                    raise ValueError(f"Not enough data for sample index {sample_index}: {len(data_idxs)} < {n}")
                else:
                    selected_batch_idxs.extend(data_idxs[:n])
            return batch.select_idxs(selected_batch_idxs)

        reload_for_mix = self.config.algorithm.mix_rollout.get("reload_for_mix", False)
        if reload_for_mix:
            reload_path = self.config.algorithm.mix_rollout.reload_for_mix_path
            print(f"Reloading mix batch (gen by ref) from {reload_path} for mix rollout.")
            reload_mix_batch = DataProto.load_from_disk(reload_path)
            sample_indexes2batch_idxs = defaultdict(list)
            for batch_idx, sample_idx in enumerate(reload_mix_batch.non_tensor_batch["index"]):
                sample_indexes2batch_idxs[sample_idx].append(batch_idx)
            reload_mix_batch = reload_mix_batch.select(batch_keys=['prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids'],
                                                       non_tensor_batch_keys=[])

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.trainer.profile_steps
            if self.config.trainer.profile_steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    orig_gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    orig_gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch = orig_gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    if self.config.algorithm.mix_rollout.enable:
                        # NOTE: kx mixed rollout
                        def merge_data(data_1, data_2, rpt_1, rpt_2, add_source=False):
                            from tensordict import TensorDict
                            """
                            Merge two DataProto, which are repeated for rpt_1 and rpt_2 times from the same original data.
                            """
                            assert data_1.batch.keys() == data_2.batch.keys()
                            assert data_1.non_tensor_batch.keys() == data_2.non_tensor_batch.keys()
                            # tensor batch
                            merged_batch = {}
                            for k in data_1.batch.keys():
                                tensor_1, tensor_2 = data_1.batch[k], data_2.batch[k]
                                assert tensor_1.shape[0] % rpt_1 == 0 and tensor_2.shape[0] % rpt_2 == 0, \
                                    f"{tensor_1.shape[0]=}, {rpt_1=}, {tensor_2.shape[0]=}, {rpt_2=}"
                                orig_dim_0 = tensor_1.shape[0] // rpt_1
                                merged_tensor = torch.concat([tensor_1.view(orig_dim_0, rpt_1, -1), tensor_2.view(orig_dim_0, rpt_2, -1)], dim=1)
                                merged_tensor = merged_tensor.view(orig_dim_0 * (rpt_1 + rpt_2), -1)
                                merged_batch[k] = merged_tensor
                                print(f'Merged key {k}: {tensor_1.shape=} and {tensor_2.shape=}, result {merged_tensor.shape=}')
                            orig_bsz = data_1.batch.batch_size[0] // rpt_1
                            merged_bsz = orig_bsz * (rpt_1 + rpt_2)
                            print(f'Merged batch size from {data_1.batch.batch_size} and {data_2.batch.batch_size} to {merged_bsz}')
                            merged_batch = TensorDict(source=merged_batch, batch_size=(merged_bsz,))
                            # non-tensor batch
                            merged_non_tensor_batch = {}
                            for k in data_1.non_tensor_batch.keys():
                                arr_1, arr_2 = data_1.non_tensor_batch[k], data_2.non_tensor_batch[k]
                                assert len(arr_1.shape) == len(arr_2.shape) == 1, f"{arr_1.shape=}, {arr_2.shape=}"
                                assert len(arr_1) % rpt_1 == 0 and len(arr_2) % rpt_2 == 0, \
                                    f"{len(arr_1)=}, {rpt_1=}, {len(arr_2)=}, {rpt_2=}"
                                orig_len = len(arr_1) // rpt_1
                                merged_arr = np.concatenate([arr_1.reshape(orig_len, rpt_1), arr_2.reshape(orig_len, rpt_2)], axis=1)
                                merged_arr = merged_arr.reshape(-1)
                                merged_non_tensor_batch[k] = merged_arr
                                print(f'Merged non-tensor key {k}: {arr_1.shape=} and {arr_2.shape=}, result {merged_arr.shape=}')
                            # add another non-tensor key to indicate the source
                            if add_source:
                                source_1, source_2 = np.zeros(orig_dim_0 * rpt_1, dtype=int), np.ones(orig_dim_0 * rpt_2, dtype=int)
                                merged_source = np.concatenate([source_1.reshape(orig_dim_0, rpt_1), source_2.reshape(orig_dim_0, rpt_2)], axis=1)
                                merged_source = merged_source.reshape(-1)
                                merged_non_tensor_batch['source'] = merged_source
                            return DataProto(
                                batch=merged_batch,
                                non_tensor_batch=merged_non_tensor_batch,
                            )
                            
                        # 0. apply mix-ratio decay if needed
                        decay_steps = self.config.algorithm.mix_rollout.get("decay_steps", 0)
                        if decay_steps > 0:
                            # decay the main_ratio to 1 linearly within decay_steps
                            default_ratio = self.config.algorithm.mix_rollout.main_ratio
                            main_ratio = default_ratio + (1.0 - default_ratio) * (self.global_steps / decay_steps)
                            main_ratio = min(main_ratio, 1.0)  # cap to 1.0
                            print(f"Decay mix_rollout main_ratio from {default_ratio} to {main_ratio} at step {self.global_steps}/{decay_steps}")
                        else:
                            main_ratio = self.config.algorithm.mix_rollout.main_ratio
                        actor_n = int(main_ratio * self.config.actor_rollout_ref.rollout.n)
                        ref_n = self.config.algorithm.mix_rollout.get("ref_n", self.config.actor_rollout_ref.rollout.n - actor_n)
                        # ref_n = self.config.actor_rollout_ref.rollout.n - actor_n
                        # set the actual ratio for mix probs
                        actual_main_ratio = actor_n / self.config.actor_rollout_ref.rollout.n
                        if self.config.algorithm.mix_rollout.get("auto_mix_ratio", True):
                            self.actor_rollout_wg._update_mix_ratio(actual_main_ratio)
                        print(f"{self.global_steps=}, {main_ratio=}, {actor_n=}, {ref_n=}, {actual_main_ratio=}")
                        if ref_n > 0 and actor_n > 0:
                            ## use both actor and reference to generate
                            # 1. re-construct the gen_batch with 2 parts: for actor and ref
                            gen_batch_actor = orig_gen_batch.repeat(repeat_times=actor_n, interleave=True)
                            gen_batch_ref = orig_gen_batch.repeat(repeat_times=ref_n, interleave=True)
                            # 2. generate sequences separately
                            with marked_timer("gen_actor", timing_raw, "red"):
                                gen_batch_output_actor = self.actor_rollout_wg.generate_sequences(gen_batch_actor)
                                timing_raw.update(gen_batch_output_actor.meta_info["timing"])
                                gen_batch_output_actor.meta_info.pop("timing", None)
                            if reload_for_mix:
                                print(f"Using reloaded mix batch for ref generation.")
                                gen_batch_output_ref = select_batch_by_index(reload_mix_batch, sample_indexes2batch_idxs,
                                                                             new_batch.non_tensor_batch["index"], n=ref_n)
                                print(f"Selected reloaded mix batch with {len(gen_batch_output_ref.batch)} samples.")
                            else:
                                with marked_timer("gen_ref", timing_raw, "red"):
                                    gen_batch_output_ref = self.actor_rollout_wg.generate_sequences_ref(gen_batch_ref)
                                    timing_raw.update(gen_batch_output_ref.meta_info["timing"])
                                    gen_batch_output_ref.meta_info.pop("timing", None)
                            # 3. combine the results
                            gen_batch_output = merge_data(gen_batch_output_actor, gen_batch_output_ref, actor_n, ref_n, add_source=True)
                        elif ref_n > 0:
                            ## only use reference to generate
                            print(f"Only use reference to generate.")
                            if reload_for_mix:
                                print(f"Using reloaded mix batch for ref generation.")
                                gen_batch_output = select_batch_by_index(reload_mix_batch, sample_indexes2batch_idxs,
                                                                            new_batch.non_tensor_batch["index"])
                                print(f"Selected reloaded mix batch with {len(gen_batch_output.batch)} samples.")
                                # and add source non-tensor batch
                                source_arr = np.ones(len(gen_batch_output.batch), dtype=int)
                                gen_batch_output.non_tensor_batch['source'] = source_arr
                            else:
                                with marked_timer("gen_ref", timing_raw, "red"):
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences_ref(gen_batch)
                                    # and add source non-tensor batch
                                    source_arr = np.ones(len(gen_batch_output.batch), dtype=int)
                                    gen_batch_output.non_tensor_batch['source'] = source_arr
                                    timing_raw.update(gen_batch_output.meta_info["timing"])
                                    gen_batch_output.meta_info.pop("timing", None)
                        else:
                            ## only use actor to generate
                            print(f"Only use actor to generate.")
                            with marked_timer("gen", timing_raw, "red"):
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                timing_raw.update(gen_batch_output.meta_info["timing"])
                                gen_batch_output.meta_info.pop("timing", None)
                    else:
                        with marked_timer("gen", timing_raw, "red"):
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    if self.config.algorithm.mix_rollout.enable:
                        repeat_times = actor_n + ref_n
                    else:
                        repeat_times = self.config.actor_rollout_ref.rollout.n
                    new_batch = new_batch.repeat(repeat_times=repeat_times, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    # NOTE: kx add saving raw generations before filtering
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        import os, json
                        with marked_timer("dump_rollout_generations_raw", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(new_batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(new_batch.batch["responses"], skip_special_tokens=True)
                            scores = new_batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in new_batch
                            ]
                            sources = new_batch.non_tensor_batch.get("source", None)
                            if sources is not None:
                                reward_extra_infos_dict["rollout_source"] = sources.tolist()

                            if "request_id" in new_batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    new_batch.non_tensor_batch["request_id"].tolist(),
                                )
                            
                            os.makedirs(rollout_data_dir, exist_ok=True)
                            filename = os.path.join(rollout_data_dir, f"raw_{self.gen_steps}.jsonl")
                            n = len(inputs)
                            base_data = dict(input=inputs, output=outputs, gts=sample_gts, score=scores, gen_step=[self.gen_steps]*n)
                            for k, v in reward_extra_infos_dict.items():
                                if len(v) == n: base_data[k] = v
                            lines = []
                            for i in range(n):
                                entry = {k: v[i] for k, v in base_data.items()}
                                lines.append(json.dumps(entry, ensure_ascii=False))
                            
                            with open(filename, "w") as f:
                                f.write("\n".join(lines) + "\n")
                            print(f"Dumped raw generations to {filename}")

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        sources = new_batch.non_tensor_batch.get("source", None)
                        if self.config.algorithm.mix_rollout.enable and self.config.algorithm.mix_rollout.get("filter_by_main", False) and sources is not None:
                            # Compared to filtering by all trajectories, we additionally remove those prompts
                            # whose main model generations are all solved.
                            print("Dynamic sampling by main model's generation only.")
                            assert metric_name == "acc", "Only acc metric is supported for mix rollout filtering now."
                            prompt_uid2accs = defaultdict(list)
                            prompt_uid2solved, prompt_uid2ref_solved = defaultdict(int), defaultdict(int)  # default to 0
                            for uid, acc, source in zip(
                                new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], sources, strict=True
                            ):
                                if source == 0: 
                                    prompt_uid2accs[uid].append(acc)  # we only consider main model's generation for filtering
                                    prompt_uid2solved[uid] += acc
                                else:
                                    prompt_uid2ref_solved[uid] += acc
                            # we choose prompts with both solved and unsolved results in main model's generations,
                            # or those completely unsolved in main model but at least one solved in ref model.
                            kept_prompt_uids = [
                                uid
                                for uid, accs in prompt_uid2accs.items()
                                if min(accs) != max(accs) or (prompt_uid2solved[uid] <= 0 and prompt_uid2ref_solved[uid] >= 1)
                            ]

                            num_prompt_in_batch += len(kept_prompt_uids)

                            kept_traj_idxs = []
                            for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                                if traj_from_prompt_uid in kept_prompt_uids:
                                    kept_traj_idxs.append(idx)
                        elif self.config.algorithm.mix_rollout.enable and ref_n + actor_n > self.config.actor_rollout_ref.rollout.n:
                            # In this actor_n > ref_n case, we delibrately choose correct response from ref model's generation
                            print("Dynamic sampling by selecting only one correct response from ref model's generation.")
                            final_n = self.config.actor_rollout_ref.rollout.n
                            assert actual_main_ratio == (final_n - 1)/final_n, f"{actual_main_ratio=}, {final_n=}, {actor_n=}, {ref_n=}"
                            assert metric_name == "acc", "Only acc metric is supported for mix rollout filtering now."
                            assert sources is not None, "Source info must be available for mix rollout filtering."
                            prompt_uid2actor_solved, prompt_uid2ref_solved = defaultdict(int), defaultdict(int)
                            for uid, acc, source in zip(
                                new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], sources, strict=True
                            ):
                                if source == 0:
                                    prompt_uid2actor_solved[uid] += acc
                                else:
                                    prompt_uid2ref_solved[uid] += acc
                            if self.config.algorithm.mix_rollout.get("weak_filter", False):
                                # we choose: actor_solved <= actor_n-1 and ref_solved >=1
                                print("Using weaker filtering strategy for mix rollout: actor_solved <= actor_n - 1 and ref_solved >=1")
                                kept_prompt_uids = [
                                    uid
                                    for uid in prompt_uid2actor_solved
                                    if prompt_uid2actor_solved[uid] <= actor_n - 1 and prompt_uid2ref_solved[uid] >= 1
                                ]
                            else:
                                # we choose: actor_solved in [1, actor_n-1] and ref_solved >=1
                                print("Using original filtering strategy for mix rollout: 1 <= actor_solved <= actor_n - 1 and ref_solved >=1")
                                kept_prompt_uids = [
                                    uid
                                    for uid in prompt_uid2actor_solved
                                    if 1 <= prompt_uid2actor_solved[uid] <= actor_n - 1 and prompt_uid2ref_solved[uid] >= 1
                                ]
                            num_prompt_in_batch += len(kept_prompt_uids)

                            kept_traj_idxs = []
                            # at the same time, we reduce ref's generation to 1 solved case
                            ref_used_flag = defaultdict(bool)
                            for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                                if traj_from_prompt_uid in kept_prompt_uids:
                                    if sources[idx] == 0:
                                        # always keep actor's generation
                                        kept_traj_idxs.append(idx)
                                    else:
                                        cur_acc = new_batch.non_tensor_batch[metric_name][idx]
                                        if cur_acc >= 1 and not ref_used_flag[traj_from_prompt_uid]:
                                            kept_traj_idxs.append(idx)
                                            ref_used_flag[traj_from_prompt_uid] = True
                        else:
                            # Collect the sequence reward for each trajectory
                            prompt_uid2metric_vals = defaultdict(list)
                            for uid, metric_val in zip(
                                new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                            ):
                                prompt_uid2metric_vals[uid].append(metric_val)

                            prompt_uid2metric_std = {}
                            for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                                prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                            kept_prompt_uids = [
                                uid
                                for uid, std in prompt_uid2metric_std.items()
                                if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                            ]

                            num_prompt_in_batch += len(kept_prompt_uids)

                            kept_traj_idxs = []
                            for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                                if traj_from_prompt_uid in kept_prompt_uids:
                                    kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch_order_by = self.config.algorithm.filter_groups.get("order_by", "none")
                            print(f"Filtered batch size: {len(batch)}, selecting {traj_bsz} by {batch_order_by}")
                            if batch_order_by in ["hard_first", "easy_first"]:
                                # select the bottom ones
                                prompt_uid2accs = defaultdict(list)
                                for uid, acc in zip(batch.non_tensor_batch["uid"], batch.non_tensor_batch['acc'], strict=True):
                                    prompt_uid2accs[uid].append(acc)
                                prompt_uid2mean_acc = {uid: np.mean(accs) for uid, accs in prompt_uid2accs.items()}
                                if batch_order_by == "hard_first":
                                    sorted_prompts = sorted(prompt_uid2mean_acc.items(), key=lambda x: x[1])
                                else:  # easy_first
                                    sorted_prompts = sorted(prompt_uid2mean_acc.items(), key=lambda x: x[1], reverse=True)
                                selected_prompts = [uid for uid, _ in sorted_prompts[:prompt_bsz]]
                                selected_idxs = []
                                for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
                                    if uid in selected_prompts:
                                        selected_idxs.append(idx)
                                batch = batch[selected_idxs]
                                assert len(batch) == traj_bsz, f"{len(batch)=}, {traj_bsz=}"
                            else:
                                batch = batch[:traj_bsz]

                    # NOTE: We copy the rollout saving from ray_ppo_trainer.py#L1282.
                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]
                            sources = batch.non_tensor_batch.get("source", None)
                            if sources is not None:
                                reward_extra_infos_dict["rollout_source"] = sources.tolist()

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                            if self.config.algorithm.get("save_rollout_batch", False):  # save the whole batch, default False
                                filename = os.path.join(rollout_data_dir, f"batch_{self.global_steps}.pkl")
                                batch.save_to_disk(filename)
                                print(f"Dumped rollout batch data to {filename}")
                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            if self.config.algorithm.mix_rollout.enable:
                                sync_steps = self.config.algorithm.get("ref_policy_sync_steps", 0)
                                if sync_steps > 0:
                                    print("Mix rollout + sync ref enabled, we compute ref log prob from the colocated ref model in actor_rollout_wg.")
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                        # if mix rollout is enabled, track the kl term
                        sources = batch.non_tensor_batch.get("source", None)
                        response_mask = batch.batch["response_mask"]
                        if self.config.algorithm.mix_rollout.enable or self.config.algorithm.get("use_ref_model", False):
                            kl_old_ref = batch.batch["old_log_probs"] -  batch.batch["ref_log_prob"]
                            if sources is not None:
                                kl_on_old = kl_old_ref[sources == 0]
                                kl_on_ref = -kl_old_ref[sources == 1]
                                kl_on_old = torch.sum(kl_on_old * response_mask[sources == 0]) / torch.sum(response_mask[sources == 0])
                                kl_on_ref = torch.sum(kl_on_ref * response_mask[sources == 1]) / torch.sum(response_mask[sources == 1])
                                metrics['mix_kl/kl_on_old'] = kl_on_old.item()
                                metrics['mix_kl/kl_on_ref'] = kl_on_ref.item()
                                metrics['mix_kl/main_ratio'] = (sources == 0).sum().item() / sources.shape[0]
                            else: # fallback to only kl_on_old
                                kl_on_old = torch.sum(kl_old_ref * response_mask) / torch.sum(response_mask)
                                metrics['mix_kl/kl_on_old'] = kl_on_old.item()
                                metrics['mix_kl/main_ratio'] = 1.0
                                

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm
                        )

                    if self.config.algorithm.mix_rollout.enable and self.config.algorithm.mix_rollout.get("mask_out_neg_off", False):
                        # mask out negative off-policy samples (by setting adv to 0)
                        sources = batch.non_tensor_batch.get("source", None)  # bsz, (numpy array)
                        if sources is not None:
                            print("Masking out off-policy samples with negative advantages.")
                            advs = batch.batch["advantages"]  # bsz, seq_len
                            sources_tensor = torch.from_numpy(sources).unsqueeze(-1)  # bsz, 1
                            mask_out = (sources_tensor == 1) & (advs < 0)  # only mask out ref model's negative advantages
                            masked_advs = advs.masked_fill(mask_out, 0.0)
                            batch.batch["advantages"] = masked_advs
                            # record the ratio of masked samples in off-policy ones
                            seq_masked = torch.any(mask_out, dim=-1)  # bsz,
                            masked_ratio = seq_masked.sum().item() / max(1, (sources_tensor == 1).sum().item())
                            metrics['mix_kl/neg_off_masked_ratio'] = masked_ratio

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if self.use_reference_policy and self.config.algorithm.mix_rollout.enable:
                        sync_steps = self.config.algorithm.get("ref_policy_sync_steps", 0)
                        sync_strategy = self.config.algorithm.get("ref_policy_sync_strategy", None)
                        if sync_strategy is not None and sync_steps > 0:
                            print(f"Sync strategy: {sync_strategy}\nSync steps: {sync_steps}")
                            if sync_strategy == "log2":
                                # sync at log2 steps
                                if np.log2(self.global_steps).is_integer():
                                    print(f"Syncing reference policy to actor at step {self.global_steps} (log2 strategy).")
                                    self.actor_rollout_wg._sync_actor_to_ref()
                        else:
                            # no strategy, just sync every sync_steps
                            if sync_steps > 0 and self.global_steps % sync_steps == 0:
                                self.actor_rollout_wg._sync_actor_to_ref()

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.trainer.profile_steps
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1

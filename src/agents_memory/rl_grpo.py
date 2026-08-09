"""Typed 4x2 sampling + two-level advantage GRPO trainer (plan M1).

Subclasses trl's GRPOTrainer (pinned 1.9.1) and overrides
`_generate_and_score_completions` at exactly two sites; everything else is a
line-by-line copy of the trl original:

1. **Prompt expansion** (site 1, trl L2133): each 8-run block of identical
   inputs (RepeatSampler guarantees local 8-runs) is expanded into
   `[add, add, update, update, delete, delete, none, none]` typed prompts;
   a per-completion `op_type` column is written back into `inputs` so reward
   functions receive it via the standard kwargs passthrough. Generation runs
   with n=1 (temporarily overriding `num_generations` / `num_generations_eval`),
   which is correct for both vLLM backends:
   - colocate: every local prompt is generated directly with n=1;
   - server:   `all_prompts[::1]` keeps all 8 typed prompts, each with n=1.
   After generation the attributes are restored, so the group-size arithmetic
   (rewards.view(-1, 8), KL, loss) is untouched.

2. **Advantage block** (site 2, trl L2540-2563): when `advantage_mode ==
   "twolayer"` compute
       A_global(i) = (r_i - mu8) / (sigma8 + eps)     # stock group term
       A_local(i)  = (r_i - mu_pair) / (sigma_pair)   # population std, ddof=0
       A_i = w_global * A_global(i) + w_local * A_local(i)
   with pairs (2k, 2k+1) inside each group (same typed prompt). The local term
   uses population std so |A_local| == 1 for any pair with two distinct
   rewards; pairs with std == 0 (identical rewards) or NaN members collapse to
   A_local == 0 (the +eps division yields 0 when the numerator is 0, NaN pairs
   are zeroed by the final nan_to_num, matching trl's unscorable handling).
   `advantage_mode == "group"` (the default) keeps the exact trl path, so the
   override is a no-op for existing experiments.

Everything new is opt-in and off by default; `standard + group` is bit-identical
to the stock trainer.
"""

from __future__ import annotations

import inspect
import math

import torch
from accelerate.utils import gather_object, is_peft_model
from trl import GRPOTrainer
from trl.data_utils import is_conversational, prepare_multimodal_messages
from trl.models.utils import disable_gradient_checkpointing
from trl.trainer.utils import nanmax, nanmin, nanstd, pad

# Group-internal layout produced by the typed expansion (4 types x 2 samples).
# Position k of the 8-run maps to TYPED_OPS[k]; pairs are (2k, 2k+1).
TYPED_OPS = ("ADD", "ADD", "UPDATE", "UPDATE", "DELETE", "DELETE", "NONE", "NONE")

EPS = 1e-4


def compute_group_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    scale_rewards: str = "group",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stock trl `sum_then_normalize` advantage (copy of grpo_trainer L2540-2563).

    Args:
        rewards: (B,) aggregated rewards; NaN marks unscorable rows (they were
            excluded from the per-group baseline via nanmean/nanstd).
        num_generations: group size (8).
        scale_rewards: "group", "batch" or "none" — same semantics as trl.

    Returns:
        (advantages, is_std_zero) both (B,).
    """
    mean_grouped_rewards = torch.nanmean(rewards.view(-1, num_generations), dim=1)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
    if scale_rewards in ["group", "none"]:
        # If scale_rewards = "none", we'll only use std_rewards to check for zero std for logging
        if num_generations > 1:
            std_rewards = nanstd(rewards.view(-1, num_generations), dim=1)
            std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
        else:  # doesn't occur during training, but could occur in eval when num_generations_eval=1
            std_rewards = torch.zeros_like(rewards)
    elif scale_rewards == "batch":
        # Compute global std
        if rewards.numel() > 1:
            std_rewards = nanstd(rewards).expand_as(rewards)
        else:  # doesn't occur during training, but could occur in eval when num_generations_eval=batch_size=1
            std_rewards = torch.zeros_like(rewards)
    else:
        raise ValueError(
            f"Invalid value for scale_rewards: {scale_rewards}. Must be one of 'batch', 'group', or 'none'."
        )

    advantages = rewards - mean_grouped_rewards
    if scale_rewards != "none":
        advantages = advantages / (std_rewards + EPS)
    is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # for logging
    return advantages, is_std_zero


def compute_two_level_advantages(
    rewards: torch.Tensor,
    num_generations: int = 8,
    w_global: float = 0.5,
    w_local: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Two-level mixed advantage (plan section 1.1).

    A_global is bit-identical to `compute_group_advantages(..., "group")`.
    A_local normalizes each (2k, 2k+1) pair with population std (ddof=0), so
    |A_local| == 1 whenever the pair has two distinct rewards; pairs with
    identical rewards or NaN members yield A_local == 0.

    Args:
        rewards: (B,) aggregated rewards; NaN marks unscorable rows. B must be
            a multiple of num_generations and num_generations a multiple of 2.
        num_generations: group size (8).
        w_global / w_local: mixture weights (default 0.5 / 0.5).

    Returns:
        (advantages, is_std_zero, pair_nonzero_std_frac):
        advantages (B,); is_std_zero (B,) on the *global* std (logging parity
        with the stock path); pair_nonzero_std_frac in [0, 1] = fraction of
        pairs whose local std is > 0 (design monitor mm_pair_nonzero_std_frac).
    """
    if num_generations % 2 != 0 or rewards.numel() % num_generations != 0:
        raise ValueError(
            f"num_generations must be even and divide len(rewards); got "
            f"num_generations={num_generations}, len(rewards)={rewards.numel()}"
        )

    # Global term: identical to compute_group_advantages(scale_rewards="group").
    mean_grouped = torch.nanmean(rewards.view(-1, num_generations), dim=1)
    mean_grouped = mean_grouped.repeat_interleave(num_generations, dim=0)
    if num_generations > 1:
        std_grouped = nanstd(rewards.view(-1, num_generations), dim=1)
        std_grouped = std_grouped.repeat_interleave(num_generations, dim=0)
    else:
        std_grouped = torch.zeros_like(rewards)
    a_global = (rewards - mean_grouped) / (std_grouped + EPS)

    # Local term: pairs (2k, 2k+1) inside each group, population std (ddof=0).
    # A pair with std == 0 has a zero numerator, so (+EPS) yields exactly 0;
    # a pair containing NaN produces NaN here and is zeroed below.
    grouped = rewards.view(-1, num_generations)
    pairs = grouped.reshape(-1, 2)  # (G * num_generations/2, 2)
    pair_mean = torch.nanmean(pairs, dim=1, keepdim=True)
    pair_std = torch.sqrt(torch.nanmean((pairs - pair_mean) ** 2, dim=1, keepdim=True))
    a_local = (pairs - pair_mean) / (pair_std + EPS)
    a_local = torch.nan_to_num(a_local, nan=0.0)
    a_local = a_local.reshape(rewards.shape)  # (B,) — flatten to match a_global

    advantages = w_global * a_global + w_local * a_local
    is_std_zero = torch.isclose(std_grouped, torch.zeros_like(std_grouped))
    valid_pair = (pair_std > 0) & ~torch.isnan(pair_std)
    pair_nonzero_std_frac = valid_pair.float().mean().item()
    return advantages, is_std_zero, pair_nonzero_std_frac


class TypedTwoLevelGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with typed 4x2 sampling and two-level advantage (opt-in).

    Extra kwargs (all default to the stock behavior):
        mm_gen_mode: "standard" (default) | "typed" — typed prompt expansion.
        advantage_mode: "group" (default) | "twolayer" — advantage scheme.
        adv_global_weight / adv_local_weight: float, default 0.5 / 0.5.
    """

    def __init__(
        self,
        *args,
        mm_gen_mode: str = "standard",
        advantage_mode: str = "group",
        adv_global_weight: float = 0.5,
        adv_local_weight: float = 0.5,
        **kwargs,
    ):
        if mm_gen_mode not in ("standard", "typed"):
            raise ValueError(f"mm_gen_mode must be 'standard' or 'typed', got {mm_gen_mode!r}")
        if advantage_mode not in ("group", "twolayer"):
            raise ValueError(f"advantage_mode must be 'group' or 'twolayer', got {advantage_mode!r}")
        if mm_gen_mode == "standard" and advantage_mode == "twolayer":
            raise ValueError("advantage_mode='twolayer' requires mm_gen_mode='typed' (pairs need typed prompts)")
        super().__init__(*args, **kwargs)
        self.mm_gen_mode = mm_gen_mode
        self.advantage_mode = advantage_mode
        self.adv_global_weight = adv_global_weight
        self.adv_local_weight = adv_local_weight

    # ------------------------------------------------------------------
    # Site 1 helpers: typed prompt expansion and n=1 generation
    # ------------------------------------------------------------------

    def _expand_typed_prompts(self, inputs: list[dict]) -> list[object]:
        """Expand each local 8-run block of identical inputs into typed prompts.

        RepeatSampler yields every decision point 8 times consecutively
        (`mini_repeat_count=num_generations`), so row i belongs to decision
        point i // 8 with in-group position i % 8. Position k maps to
        TYPED_OPS[k]; the dataset's typed branch stores the four rendered
        prompts under `prompt_add` / `prompt_update` / `prompt_delete` /
        `prompt_none`. The per-completion `op_type` is written back into each
        input row so `_calculate_rewards` forwards it to the reward function.
        """
        expanded = []
        for i, x in enumerate(inputs):
            op = TYPED_OPS[i % len(TYPED_OPS)]
            expanded.append(x[f"prompt_{op.lower()}"])
            x["op_type"] = op
        return expanded

    def _generate_with_n1(self, prompts):
        """Call self._generate with n=1 while the typed expansion is active.

        Both vLLM backends then produce exactly one completion per (distinct)
        typed prompt. num_generations is restored before the advantage block
        reads it, so the group-size arithmetic stays at 8.
        """
        saved_ng = self.num_generations
        saved_nge = self.num_generations_eval
        self.num_generations = 1
        self.num_generations_eval = 1
        try:
            return self._generate(prompts)
        finally:
            self.num_generations = saved_ng
            self.num_generations_eval = saved_nge

    # ------------------------------------------------------------------
    # Overridden: copy of trl 1.9.1 `_generate_and_score_completions`
    # with exactly two modified sites (marked SITE 1 / SITE 2).
    # ------------------------------------------------------------------

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | object]]
    ) -> dict[str, torch.Tensor | object]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        # SITE 1 (typed expansion) ------------------------------------
        if self.mm_gen_mode == "typed":
            prompts = self._expand_typed_prompts(inputs)
        # -------------------------------------------------------------

        # Resolve each example's environment and draw one reusable instance per rollout from the pool, creating more
        # only when this batch needs more concurrent instances of an environment than exist. `_batch_environments`
        # records each example's environment so `_tokenize_prompts` can render the matching tool schema.
        if self.environment_factories is not None:
            self._batch_environments = [x.get("environment") if self._multi_environment else None for x in inputs]
            if self._multi_environment:
                for name in set(self._batch_environments):
                    if name not in self.environment_factories:
                        raise ValueError(
                            f"Example has `environment={name!r}`, which is not among the environments passed to "
                            f"`environment_factory`. Expected one of: {list(self.environment_factories)}."
                        )
            self.environments = []
            pool_cursor = {}  # how many instances of each environment have been handed out so far this batch
            for name in self._batch_environments:
                pool = self._environment_pool[name]
                index = pool_cursor.get(name, 0)
                if index == len(pool):
                    pool.append(self.environment_factories[name]())
                pool_cursor[name] = index + 1
                self.environments.append(pool[index])

        # Build the per-rollout tool dicts for this batch: the standalone tools plus, for each rollout, the methods of
        # its environment. Done here (not at init) because each example's environment, hence its tools, is data-dependent.
        if self.tools:
            self._sync_tool_dicts = []
            self._async_tool_dicts = []
            for i in range(len(inputs)):
                methods = []
                if self.environments:
                    methods = [
                        member
                        for member_name, member in inspect.getmembers(self.environments[i], predicate=inspect.ismethod)
                        if member_name not in ("reset", "get_reward") and not member_name.startswith("_")
                    ]
                sync_tool_dict, async_tool_dict = {}, {}
                for tool in self._standalone_tools + methods:
                    if inspect.iscoroutinefunction(tool):
                        async_tool_dict[tool.__name__] = tool
                    else:
                        sync_tool_dict[tool.__name__] = tool
                self._sync_tool_dicts.append(sync_tool_dict)
                self._async_tool_dicts.append(async_tool_dict)

        if self.environments:
            for prompt, environment, x in zip(prompts, self.environments, inputs, strict=True):
                # `environment` is a control field in multi-environment mode, so it is not forwarded to `reset`.
                reset_kwargs = {k: v for k, v in x.items() if k != "environment"} if self._multi_environment else x
                observation = environment.reset(**reset_kwargs)
                if observation is None:
                    continue
                if isinstance(observation, list) and isinstance(prompt[-1]["content"], str):
                    prompt[-1]["content"] = [{"type": "text", "text": prompt[-1]["content"]}]
                if isinstance(observation, str) and isinstance(prompt[-1]["content"], list):
                    observation = [{"type": "text", "text": observation}]
                prompt[-1]["content"] += observation

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        if images is not None:
            if not is_conversational(inputs[0]):
                raise ValueError(
                    "Multimodal training requires conversational prompts. It looks like the dataset contains "
                    "non-conversational inputs, likely because a chat template was applied before passing the dataset "
                    "to the trainer. Please provide the raw conversational prompts and let the trainer apply the chat "
                    "template internally."
                )
            prompts = [
                prepare_multimodal_messages(prompt, images=image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        dataset_images = images  # preserve dataset images before _generate may overwrite

        # SITE 1 (n=1 generation) -------------------------------------
        if self.mm_gen_mode == "typed":
            (
                prompt_ids_list,
                completion_ids_list,
                tool_mask_list,
                completions,
                sampling_per_token_logps_list,
                extra_fields,
                images,
                tool_images,
            ) = self._generate_with_n1(prompts)
        else:
            (
                prompt_ids_list,
                completion_ids_list,
                tool_mask_list,
                completions,
                sampling_per_token_logps_list,
                extra_fields,
                images,
                tool_images,
            ) = self._generate(prompts)
        # -------------------------------------------------------------

        if images is None:
            images = dataset_images  # restore dataset images (rollout_func path returns None)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(
            prompt_ids,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        prompt_mask = pad(
            prompt_mask, padding_value=0, padding_side="left", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device=device)
        completion_ids = [torch.tensor(ids) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(
            completion_ids,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        completion_mask = pad(
            completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device=device)
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(
                sampling_per_token_logps,
                padding_value=0.0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            ).to(device=device)
        else:
            sampling_per_token_logps = None
        if tool_mask_list is not None:
            tool_mask = [torch.tensor(mask) for mask in tool_mask_list]
            tool_mask = pad(
                tool_mask, padding_value=1, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            ).to(device=device)
        else:
            tool_mask = None

        # If mask_truncated_completions is enabled, zero out truncated completions for attention and loss masking
        if self.mask_truncated_completions:
            eos_and_pad = [self._tokenizer.eos_token_id, self._tokenizer.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            # Mask completion_mask for attention masking
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
            # Also mask tool_mask for consistency in multi-turn training
            if tool_mask is not None:
                tool_mask = tool_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        loss_mask = completion_mask if tool_mask is None else completion_mask * tool_mask
        num_items_in_batch = self.accelerator.gather(loss_mask.sum()).sum()

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) if img_list else 0 for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs.
        # When tool images are present (from _tool_call_loop), use image_processor directly and build
        # mm_token_type_ids from prompt_completion_ids. Otherwise, use the full processor pipeline
        # which returns model-specific keys (image_sizes, pixel_attention_mask, etc.).
        if self.tools and any(imgs for imgs in tool_images) and self._is_vlm:
            flat_images = [img for img_list in images if img_list for img in img_list]
            image_inputs = self.processing_class.image_processor(images=flat_images, return_tensors="pt")
            image_inputs = super()._prepare_inputs(image_inputs)
            forward_kwargs = dict(image_inputs)
        elif images is not None:
            if self.environment_factories is not None:
                per_prompt_tools = [self._env_tools[name] for name in self._batch_environments]
            else:
                per_prompt_tools = [self.tools] * len(prompts)
            prompts_text = [
                apply_chat_template(
                    {"prompt": prompt}, self.processing_class, tools=tools, **self.chat_template_kwargs
                )["prompt"]
                for prompt, tools in zip(prompts, per_prompt_tools, strict=True)
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # Recover LFM2-VL tile counts; the full processor drops row/column metadata.
        num_tiles = None
        if images is not None and "spatial_shapes" in forward_kwargs:
            image_info = self.processing_class.image_processor(
                images=images, return_tensors="pt", return_row_col_info=True
            )
            tiles_per_image = image_info["image_rows"] * image_info["image_cols"]
            if self.processing_class.image_processor.use_thumbnail:
                tiles_per_image = tiles_per_image + (tiles_per_image > 1).to(tiles_per_image.dtype)
            num_tiles = [group.sum().item() for group in torch.split(tiles_per_image, num_images)]

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            if self.pad_to_multiple_of is not None:
                # Needed only with pad_to_multiple_of: otherwise prompt_ids and token_type_ids must have equal len
                padding_size = prompt_ids.size(1) - token_type_ids.size(1)
                if padding_size > 0:
                    token_type_ids = torch.cat(
                        [token_type_ids.new_zeros((token_type_ids.size(0), padding_size)), token_type_ids], dim=1
                    )
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )
        # If mm_token_type_ids are used, extend them with zeros for the completion part
        if "mm_token_type_ids" in forward_kwargs:
            mm_token_type_ids = forward_kwargs["mm_token_type_ids"]
            if self.pad_to_multiple_of is not None:
                # Needed only with pad_to_multiple_of: otherwise prompt_ids and mm_token_type_ids must have equal len
                padding_size = prompt_ids.size(1) - mm_token_type_ids.size(1)
                if padding_size > 0:
                    mm_token_type_ids = torch.cat(
                        [mm_token_type_ids.new_zeros((mm_token_type_ids.size(0), padding_size)), mm_token_type_ids],
                        dim=1,
                    )
            forward_kwargs["mm_token_type_ids"] = torch.cat(
                [mm_token_type_ids, mm_token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        # For VLM tool images: build token type IDs from the full prompt_completion_ids.
        # This must happen AFTER the token_type_ids/mm_token_type_ids extension blocks above,
        # because our version already covers the full sequence (images are in the completion,
        # not just the prompt).
        if self.tools and any(imgs for imgs in tool_images) and self._is_vlm:
            mm_ids = torch.zeros_like(prompt_completion_ids)
            if self._image_pad_token_id is not None:
                mm_ids[prompt_completion_ids == self._image_pad_token_id] = 1
            if self._video_pad_token_id is not None:
                mm_ids[prompt_completion_ids == self._video_pad_token_id] = 2

            # Use the same key the model expects: token_type_ids for models like Gemma,
            # mm_token_type_ids for models like Qwen.
            image_grid_thw = forward_kwargs.get("image_grid_thw")
            if image_grid_thw is not None:
                forward_kwargs["mm_token_type_ids"] = mm_ids
            else:
                forward_kwargs["token_type_ids"] = mm_ids

            # Truncation safety (Qwen-style models with image_grid_thw only): if
            # max_completion_length truncated some image tokens, the number of image pad tokens
            # in input_ids won't match pixel_values features. Check per-sample and drop ALL
            # images for any sample with a mismatch (safe fallback).
            if image_grid_thw is not None and num_images is not None:
                merge_length = getattr(self.processing_class.image_processor, "merge_size", 2) ** 2
                img_offset = 0
                has_mismatch = False
                for b in range(mm_ids.shape[0]):
                    sample_tokens = (mm_ids[b] == 1).sum().item()
                    sample_features = 0
                    for i in range(num_images[b]):
                        grid_idx = img_offset + i
                        if grid_idx < image_grid_thw.shape[0]:
                            sample_features += image_grid_thw[grid_idx].prod().item() // merge_length
                    if sample_tokens != sample_features:
                        has_mismatch = True
                        break
                    img_offset += num_images[b]

                if has_mismatch:
                    # Drop all images: safer than partial trim which is error-prone
                    forward_kwargs.pop("pixel_values", None)
                    forward_kwargs.pop("image_grid_thw", None)
                    mm_ids.zero_()
                    forward_kwargs["mm_token_type_ids"] = mm_ids
                    num_images = None

        # When gradient checkpointing is enabled with use_reentrant=True (non default), calling the model inside a
        # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
        # Temporarily disable checkpointing to avoid this warning during inference.
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    num_tiles=num_tiles,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask, spatial_shapes, image_sizes, image_position_ids
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                mask = completion_mask if tool_mask is None else completion_mask * tool_mask
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

                sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                if sequence_level_is:
                    per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                    logps_diff = per_sequence_logps_diff
                else:
                    logps_diff = per_token_logps_diff

                vllm_importance_sampling_ratio = torch.exp(logps_diff)

                # vllm_importance_sampling_ratio.shape:
                #   token_* modes:     (B, T)  (per-token ratio)
                #   sequence_* modes:  (B, 1)  (per-sequence ratio)

                if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                    vllm_importance_sampling_ratio = torch.clamp(
                        vllm_importance_sampling_ratio,
                        min=self.vllm_importance_sampling_clip_min,
                        max=self.vllm_importance_sampling_clip_max,
                    )
                elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                    min_val = (
                        self.vllm_importance_sampling_clip_min
                        if self.vllm_importance_sampling_clip_min is not None
                        else -math.inf
                    )
                    max_val = (
                        self.vllm_importance_sampling_clip_max
                        if self.vllm_importance_sampling_clip_max is not None
                        else math.inf
                    )

                    invalid_mis_mask = (vllm_importance_sampling_ratio < min_val) | (
                        vllm_importance_sampling_ratio > max_val
                    )
                    vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                        invalid_mis_mask, value=0.0
                    )
                else:
                    raise ValueError(
                        f"Unknown vLLM importance sampling level: {self.vllm_importance_sampling_mode}. Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', and 'sequence_mask'."
                    )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        num_tiles=num_tiles,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask, spatial_shapes, image_sizes, image_position_ids
                    )
                else:
                    # When training a PEFT adapter, how we obtain the reference depends on the setup:
                    # - New adapter: disabling adapters yields the base model.
                    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                    model = self.accelerator.unwrap_model(self.model)
                    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                        ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            num_tiles=num_tiles,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask, spatial_shapes, image_sizes, image_position_ids
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        # A completion for which every reward function returned None is unscorable. nansum would collapse it to 0,
        # which both biases the per-group baseline and hands the completion a spurious advantage. Mark these rows NaN
        # so they're excluded from the (nan-aware) baseline below; their advantage is forced to 0 afterwards.
        unscorable_mask = torch.isnan(rewards_per_func).all(dim=1)

        if self.multi_objective_aggregation == "sum_then_normalize":
            # Apply weights to each reward function's output and sum
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            rewards[unscorable_mask] = torch.nan

            # SITE 2 (two-level advantage) -----------------------------
            if self.advantage_mode == "twolayer":
                advantages, is_std_zero, pair_nonzero_std_frac = compute_two_level_advantages(
                    rewards,
                    num_generations=num_generations,
                    w_global=self.adv_global_weight,
                    w_local=self.adv_local_weight,
                )
                self._metrics[mode]["mm_pair_nonzero_std_frac"].append(pair_nonzero_std_frac)
            else:
                # ---- stock trl path (L2540-2563), bit-identical ----
                advantages, is_std_zero = compute_group_advantages(
                    rewards, num_generations=num_generations, scale_rewards=self.scale_rewards
                )
            # ----------------------------------------------------------

        elif self.multi_objective_aggregation == "normalize_then_sum":
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
            std_k = nanstd(grouped, dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(mean_k)
            reward_k = (grouped - mean_k) / (std_k + EPS)
            reward_k = reward_k.view(-1, len(self.reward_funcs))
            rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            rewards[unscorable_mask] = torch.nan
            std_rewards = nanstd(rewards).expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            advantages = (rewards - torch.nanmean(rewards)) / (std_rewards + EPS)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # for logging

        else:
            raise ValueError(
                f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}. Must be "
                "'sum_then_normalize' or 'normalize_then_sum'."
            )

        # Unscorable completions (every reward func returned None) carry no learning signal: their reward is NaN here,
        # so zero their advantage to keep them from moving the policy.
        advantages = torch.nan_to_num(advantages, nan=0.0)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        rewards = (rewards_per_func * self.reward_weights.to(rewards_per_func.device).unsqueeze(0)).nansum(dim=1)
        rewards[unscorable_mask] = torch.nan  # exclude unscorable rows from the logged reward stats
        self._metrics[mode]["reward"].append(torch.nanmean(rewards).item())
        self._metrics[mode]["reward_std"].append(nanstd(rewards).item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        # Flush user-logged extra columns (from log_extra), gathering across processes.
        # Keys must be sorted so that all ranks call gather_object in the same order, otherwise values
        # get mis-attributed across columns (dict insertion order may differ between processes).
        for column in sorted(self._pending_extra_logs):
            self._logs["extra"][column].extend(gather_object(self._pending_extra_logs[column]))
        self._pending_extra_logs.clear()

        # Flush user-logged metrics (from log_metric), averaging across processes.
        # Keys must be sorted so that all ranks call accelerator.gather in the same order, otherwise values
        # get mis-attributed across metrics (dict insertion order may differ between processes).
        for name in sorted(self._pending_metrics):
            values = self._pending_metrics[name]
            local_mean = sum(values) / len(values)
            global_mean = self.accelerator.gather(torch.tensor(local_mean, device=device)).mean().item()
            self._metrics[mode][name].append(global_mean)
        self._pending_metrics.clear()

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            mask = completion_mask.bool() if tool_mask is None else (completion_mask * tool_mask).bool()
            delta = delta[mask]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )
            if sequence_level_is:
                flat_is_ratio = vllm_importance_sampling_ratio.flatten()
            else:
                flat_is_ratio = vllm_importance_sampling_ratio[mask]

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if sampling_per_token_logps is not None:
            output["sampling_per_token_logps"] = sampling_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "spatial_shapes" in forward_kwargs:
            output["spatial_shapes"] = forward_kwargs["spatial_shapes"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if "mm_token_type_ids" in forward_kwargs:
            output["mm_token_type_ids"] = forward_kwargs["mm_token_type_ids"]
        if "image_position_ids" in forward_kwargs:
            output["image_position_ids"] = forward_kwargs["image_position_ids"]
        if images is not None:
            output["num_images"] = num_images
            if num_tiles is not None:
                output["num_tiles"] = num_tiles
        if tool_mask is not None:
            output["tool_mask"] = tool_mask
        return output

"""Regression tests for MOSS-Image (OpenMOSS) LoRA training support in the GUI.

MOSS-Image is trained via `sd-scripts/moss_image_train_network.py` (sd-scripts
>= 0.12.1) with the generic `networks.lora_generic` module, which auto-detects
the MOSS-Image DiT architecture. Follows the Lumina/Anima/HunyuanImage model
family pattern: source-model checkbox + class_moss_image widget + command/TOML
wiring.
"""

import unittest
from unittest.mock import patch

from kohya_gui import lora_gui
from conftest import (
    build_train_model_kwargs,
    run_train_model_and_load_toml,
    mock_executor,
)

FIXTURE = "test/config/locon-AdamW8bit-toml.json"

NUMERIC_FIXUPS = (
    "max_train_steps",
    "max_train_epochs",
    "num_processes",
    "num_machines",
    "gradient_accumulation_steps",
    "seed",
    "vae_batch_size",
    "save_every_n_steps",
    "save_every_n_epochs",
    "save_last_n_steps",
    "save_last_n_steps_state",
    "save_last_n_epochs",
    "save_last_n_epochs_state",
    "lr_scheduler_num_cycles",
    "min_bucket_reso",
    "max_bucket_reso",
    "bucket_reso_steps",
    "max_data_loader_n_workers",
    "keep_tokens",
    "clip_skip",
    "max_token_length",
    "caption_dropout_every_n_epochs",
    "min_snr_gamma",
    "min_timestep",
    "max_timestep",
    "sample_every_n_steps",
    "sample_every_n_epochs",
    "gpu_ids",
    "main_process_port",
    "num_cpu_threads_per_process",
    "t5xxl_max_token_length",
    "block_lr_zero_threshold",
    "moss_image_max_token_length",
)
STRING_OVERRIDES = (
    "ae",
    "clip_l",
    "t5xxl",
    "sd3_clip_l",
    "sd3_t5xxl",
    "t5xxl_device",
    "t5xxl_dtype",
    "huggingface_repo_id",
    "huggingface_token",
    "huggingface_repo_type",
    "huggingface_repo_visibility",
    "huggingface_path_in_repo",
    "metadata_author",
    "metadata_description",
    "metadata_license",
    "metadata_tags",
    "metadata_title",
    "log_with",
    "log_config",
    "loss_type",
    "huber_schedule",
    "lr_scheduler_type",
    "dynamo_backend",
    "dynamo_mode",
    "extra_accelerate_launch_args",
    "network_weights",
    "model_prediction_type",
    "timestep_sampling",
    "train_blocks",
    "weighting_scheme",
    "in_dims",
    "moss_image_text_encoder",
    "moss_image_vae",
    "moss_image_tokenizer_cache_dir",
    "moss_image_timestep_sampling",
    "moss_image_attn_mode",
    "moss_image_convrot_int8_bwd",
    "moss_image_sample_uncond_skip_layers",
)

MOSS_OVERRIDES = {
    "LoRA_type": "MOSS-Image",
    "moss_image_checkbox": True,
    "moss_image_text_encoder": "/models/qwen3.5_9b.safetensors",
    "moss_image_vae": "/models/flux2-vae.safetensors",
    "moss_image_max_token_length": 4096,
    "moss_image_discrete_flow_shift": 1.0,
    "moss_image_timestep_sampling": "sigmoid",
    "moss_image_sigmoid_scale": 1.0,
    "moss_image_attn_mode": "torch",
    "moss_image_split_attn": False,
    "moss_image_convrot_int8": False,
    "moss_image_convrot_int8_text_encoder": False,
    "moss_image_convrot_int8_bwd": "bf16",
    "moss_image_cache_text_encoder_outputs": True,
    "moss_image_cache_text_encoder_outputs_to_disk": False,
}


class TestMossImageLoraConfigOutput(unittest.TestCase):
    """train_model(print_only=True) writes a real TOML we inspect for the
    MOSS-Image specific keys."""

    def _run_and_load_toml(self, overrides):
        kwargs = build_train_model_kwargs(
            lora_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={**MOSS_OVERRIDES, **overrides},
        )
        return run_train_model_and_load_toml(lora_gui, kwargs)

    def test_network_module_is_lora_generic(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("network_module"), "networks.lora_generic")

    def test_model_paths_are_forwarded(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("text_encoder"), "/models/qwen3.5_9b.safetensors")
        self.assertEqual(config.get("vae"), "/models/flux2-vae.safetensors")

    def test_max_token_length_uses_moss_key(self):
        config = self._run_and_load_toml({})
        self.assertEqual(config.get("moss_max_token_length"), 4096)
        # the SD-style max_token_length must not leak through
        self.assertNotIn("max_token_length", config)

    def test_flow_matching_params_are_forwarded(self):
        config = self._run_and_load_toml(
            {
                "moss_image_discrete_flow_shift": 3.0,
                "moss_image_timestep_sampling": "truncated_logit_normal",
                "moss_image_sigmoid_scale": 2.0,
            }
        )
        self.assertEqual(config.get("discrete_flow_shift"), 3.0)
        self.assertEqual(config.get("timestep_sampling"), "truncated_logit_normal")
        self.assertEqual(config.get("sigmoid_scale"), 2.0)

    def test_text_encoder_cache_flags_are_forwarded(self):
        config = self._run_and_load_toml({})
        self.assertTrue(config.get("cache_text_encoder_outputs"))
        self.assertNotIn("cache_text_encoder_outputs_to_disk", config)

    def test_convrot_int8_flags_default_off(self):
        config = self._run_and_load_toml({})
        self.assertNotIn("convrot_int8", config)
        self.assertNotIn("convrot_int8_text_encoder", config)
        self.assertNotIn("convrot_int8_bwd", config)

    def test_convrot_int8_flags_forwarded_when_set(self):
        config = self._run_and_load_toml(
            {
                "moss_image_convrot_int8": True,
                "moss_image_convrot_int8_text_encoder": True,
                "moss_image_convrot_int8_bwd": "int8",
            }
        )
        self.assertTrue(config.get("convrot_int8"))
        self.assertTrue(config.get("convrot_int8_text_encoder"))
        self.assertEqual(config.get("convrot_int8_bwd"), "int8")

    def test_convrot_int8_bwd_omitted_without_convrot_base(self):
        # bwd=int8 only matters with a ConvRot base; must not be emitted alone
        config = self._run_and_load_toml({"moss_image_convrot_int8_bwd": "int8"})
        self.assertNotIn("convrot_int8_bwd", config)

    def test_attn_mode_torch_is_omitted(self):
        config = self._run_and_load_toml({})
        self.assertNotIn("attn_mode", config)

    def test_attn_mode_forwarded_when_not_torch(self):
        config = self._run_and_load_toml(
            {"moss_image_attn_mode": "flash", "moss_image_split_attn": True}
        )
        self.assertEqual(config.get("attn_mode"), "flash")
        self.assertTrue(config.get("split_attn"))

    def test_tokenizer_cache_dir_forwarded_when_set(self):
        config = self._run_and_load_toml(
            {"moss_image_tokenizer_cache_dir": "/models/moss_tokenizer"}
        )
        self.assertEqual(config.get("tokenizer_cache_dir"), "/models/moss_tokenizer")

    def test_tokenizer_cache_dir_omitted_when_empty(self):
        config = self._run_and_load_toml({})
        self.assertNotIn("tokenizer_cache_dir", config)

    def test_incompatible_sd_args_are_excluded(self):
        config = self._run_and_load_toml({})
        self.assertNotIn("max_token_length", config)
        self.assertNotIn("clip_skip", config)

    def test_fp8_base_excluded_for_moss(self):
        # MOSS-Image does not support fp8; ConvRot INT8 is the alternative.
        config = self._run_and_load_toml({"fp8_base": True})
        self.assertNotIn("fp8_base", config)

    def test_ddpm_only_loss_args_suppressed_even_with_stale_values(self):
        # A config migrated from SD/SDXL may carry values for args the MOSS
        # trainer ignores (post_process_loss is a no-op; the flow noise path
        # does not apply noise_offset/multires/adaptive). The emitted TOML
        # must not contain them.
        config = self._run_and_load_toml(
            {
                "debiased_estimation_loss": True,
                "min_snr_gamma": 5,
                "v_pred_like_loss": 0.5,
                "scale_v_pred_loss_like_noise_pred": True,
                "noise_offset": 0.1,
                "noise_offset_random_strength": True,
                "adaptive_noise_scale": 0.1,
                "multires_noise_iterations": 6,
                "multires_noise_discount": 0.3,
            }
        )
        for key in (
            "debiased_estimation_loss",
            "min_snr_gamma",
            "v_pred_like_loss",
            "scale_v_pred_loss_like_noise_pred",
            "noise_offset",
            "noise_offset_random_strength",
            "adaptive_noise_scale",
            "multires_noise_iterations",
            "multires_noise_discount",
        ):
            self.assertNotIn(key, config)

    def test_ip_noise_gamma_still_forwarded_for_moss(self):
        # ip_noise_gamma IS applied by the flow-matching noise path
        # (flux_train_utils.get_noisy_model_input_and_timesteps), so unlike
        # the other noise args it must survive.
        config = self._run_and_load_toml({"ip_noise_gamma": 0.1})
        self.assertEqual(config.get("ip_noise_gamma"), 0.1)

    def test_sample_uncond_skip_layers_default_omitted(self):
        config = self._run_and_load_toml({})
        self.assertNotIn("sample_uncond_skip_layers", config)

    def test_sample_uncond_skip_layers_forwarded_when_changed(self):
        config = self._run_and_load_toml({"moss_image_sample_uncond_skip_layers": "10"})
        self.assertEqual(config.get("sample_uncond_skip_layers"), "10")

    def test_script_selection_invokes_moss_image_train_network(self):
        with patch.object(lora_gui, "print_command_and_toml") as mocked:
            kwargs = build_train_model_kwargs(
                lora_gui.train_model,
                FIXTURE,
                numeric_fixups=NUMERIC_FIXUPS,
                string_overrides=STRING_OVERRIDES,
                overrides=MOSS_OVERRIDES,
            )
            mock_executor(lora_gui)
            with patch.object(
                lora_gui, "get_executable_path", return_value="accelerate"
            ):
                lora_gui.train_model(**kwargs)
            self.assertTrue(mocked.called)
            run_cmd = mocked.call_args[0][0]
            self.assertTrue(
                any("moss_image_train_network.py" in part for part in run_cmd)
            )


class TestMossImageValidation(unittest.TestCase):
    """train_model must reject a MOSS-Image LoRA type without the checkbox and
    vice-versa, mirroring the Lumina/Anima guards."""

    def _run(self, overrides):
        kwargs = build_train_model_kwargs(
            lora_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={**MOSS_OVERRIDES, **overrides},
        )
        mock_executor(lora_gui)
        with patch.object(lora_gui, "get_executable_path", return_value="accelerate"):
            with patch.object(lora_gui, "print_command_and_toml") as mocked:
                lora_gui.train_model(**kwargs)
        return mocked.called

    def test_lora_type_without_checkbox_is_rejected(self):
        self.assertFalse(self._run({"moss_image_checkbox": False}))

    def test_checkbox_without_lora_type_is_rejected(self):
        self.assertFalse(self._run({"LoRA_type": "Standard"}))

    def test_matching_type_and_checkbox_is_accepted(self):
        self.assertTrue(self._run({}))


class TestMossImageNativeLohaLokr(unittest.TestCase):
    """Kohya LoHa/LoKr auto-detect MOSS-Image; the checkbox pair must be
    accepted and route to moss_image_train_network.py."""

    def _run(self, overrides):
        kwargs = build_train_model_kwargs(
            lora_gui.train_model,
            FIXTURE,
            numeric_fixups=NUMERIC_FIXUPS,
            string_overrides=STRING_OVERRIDES,
            overrides={**MOSS_OVERRIDES, **overrides},
        )
        mock_executor(lora_gui)
        with patch.object(lora_gui, "get_executable_path", return_value="accelerate"):
            with patch.object(lora_gui, "print_command_and_toml") as mocked:
                lora_gui.train_model(**kwargs)
        return mocked

    def test_kohya_lokr_with_moss_checkbox_routes_to_moss_script(self):
        mocked = self._run({"LoRA_type": "Kohya LoKr"})
        self.assertTrue(mocked.called)
        run_cmd = mocked.call_args[0][0]
        self.assertTrue(any("moss_image_train_network.py" in p for p in run_cmd))


class TestMossImageAdvancedFieldVisibility(unittest.TestCase):
    """The LoRA-type change handler must hide the Advanced controls that the
    MOSS-Image trainer ignores, and show them again for every other type.

    Hidden for MOSS-Image: the whole noise-offset row (noise_offset_type,
    noise_offset, multires, adaptive_noise_scale, ip_noise_gamma), clip_skip,
    max_token_length, min_snr_gamma, debiased_estimation_loss,
    v_pred_like_loss, scale_v_pred_loss_like_noise_pred.
    """

    @classmethod
    def setUpClass(cls):
        import gradio as gr

        from kohya_gui.class_gui_config import KohyaSSGUIConfig

        with gr.Blocks() as demo:
            lora_gui.lora_tab(headless=True, config=KohyaSSGUIConfig())
        cls.dep_fn = None
        for fn in demo.fns.values():
            if getattr(fn.fn, "__name__", "") == "update_LoRA_settings":
                cls.dep_fn = fn
                break
        assert cls.dep_fn is not None, "update_LoRA_settings not wired"

    def _visibility(self, lora_type):
        """Return {label-or-component: visible} for handler outputs."""
        results = self.dep_fn.fn(lora_type)
        assert len(results) == len(self.dep_fn.outputs)
        vis = {}
        for update, comp in zip(results, self.dep_fn.outputs):
            label = getattr(comp, "label", None) or type(comp).__name__
            vis[label] = update.constructor_args.get("visible")
        return vis

    HIDDEN_FOR_MOSS = (
        "Clip skip",
        "Max Token Length",
        "Min SNR gamma",
        "Debiased Estimation loss",
        "V Pred like loss",
        "Scale v prediction loss",
    )

    def test_moss_hides_ddpm_only_advanced_controls(self):
        vis = self._visibility("MOSS-Image")
        for label in self.HIDDEN_FOR_MOSS:
            self.assertIs(
                vis.get(label), False, f"{label} must be hidden for MOSS-Image"
            )

    def test_moss_hides_noise_offset_row(self):
        # The wiring ends with [..., loraplus, noise_offset_row, clip_skip,
        # max_token_length, min_snr_gamma, debiased_estimation_loss,
        # v_pred_like_loss, scale_v_pred_loss_like_noise_pred] — so the
        # noise-offset Row is the 7th output from the end (a bare gr.Row
        # with no label to key on).
        results = self.dep_fn.fn("MOSS-Image")
        noise_row_update = results[-7]
        self.assertEqual(type(self.dep_fn.outputs[-7]).__name__, "Row")
        self.assertIs(
            noise_row_update.constructor_args.get("visible"),
            False,
            "noise_offset_row must be hidden for MOSS-Image",
        )
        results_std = self.dep_fn.fn("Standard")
        self.assertIs(
            results_std[-7].constructor_args.get("visible"),
            True,
            "noise_offset_row must be visible again for Standard",
        )

    def test_other_types_keep_advanced_controls_visible(self):
        for lora_type in ("Standard", "Kohya LoCon", "LyCORIS/LoCon", "Flux1"):
            vis = self._visibility(lora_type)
            for label in self.HIDDEN_FOR_MOSS:
                self.assertIs(
                    vis.get(label),
                    True,
                    f"{label} must stay visible for {lora_type}",
                )

    def test_handler_output_arity_matches_wiring(self):
        results = self.dep_fn.fn("MOSS-Image")
        self.assertEqual(len(results), len(self.dep_fn.outputs))


if __name__ == "__main__":
    unittest.main()

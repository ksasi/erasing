import inspect
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput


@torch.no_grad()
def esd_sd35_call(
    self,
    num_inference_steps: int = 28,
    guidance_scale: float = 4.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,          # Passed from context["sample_prompt_embeds"]
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,   # Passed from context["sample_pooled_prompt_embeds"]
    negative_prompt_embeds: Optional[torch.FloatTensor] = None, # Passed from context["null_prompt_embeds"]
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None, # Passed from context["null_pooled_prompt_embeds"]
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 256,
    run_from_timestep: int = 0,
    run_till_timestep: Optional[int] = None,
    start_latents: Optional[torch.FloatTensor] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
):
    """Fixed generation loop for ESD targeting SD3.5 that leverages pre-computed context embeddings."""
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # Extract dynamic shapes directly from pre-computed embeddings
    batch_size = prompt_embeds.shape[0]
    device = self.transformer.device

    # Handle Classifier-Free Guidance (CFG) configuration
    do_classifier_free_guidance = guidance_scale > 1.0

    # Ensure empty/negative projections are available if CFG is flagged active
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        raise ValueError("CFG scale is > 1.0, but pre-computed negative/null embeddings were not provided.")

    # Prepare Latents
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    if start_latents is not None:
        latents = start_latents

    # Prepare Timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    
    # Slice the inference timeline safely
    run_till_timestep = run_till_timestep if run_till_timestep is not None else len(timesteps)
    timesteps = timesteps[run_from_timestep:run_till_timestep]

    # Denoising Loop
    with self.progress_bar(total=len(timesteps)) as progress_bar:
        for i, t in enumerate(timesteps):
            if self._interrupt:
                continue

            # Expand latents if doing CFG
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            timestep = t.expand(latent_model_input.shape[0])

            # Concatenate conditions for parallel conditional/unconditional passes
            if do_classifier_free_guidance:
                current_encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                current_pooled_projections = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            else:
                current_encoder_hidden_states = prompt_embeds
                current_pooled_projections = pooled_prompt_embeds

            # Forward pass through MMDiT
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=current_encoder_hidden_states,
                pooled_projections=current_pooled_projections,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0] # Extract raw output tensor from tuple structure

            # Perform CFG blending
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute backstep
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            progress_bar.update()

    if output_type == "latent":
        image = latents
    else:
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return StableDiffusion3PipelineOutput(images=image)
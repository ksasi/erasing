import inspect
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput


@torch.no_grad()
def esd_sd35_call(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 4.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 256,
    run_from_timestep: int = 0,
    run_till_timestep: Optional[int] = None,
    start_latents: Optional[torch.FloatTensor] = None,
):
    """Partial generation loop for Erasing Stable Diffusion (ESD) targeting SD3.5."""
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs
    self.check_inputs(
        prompt, prompt_2, prompt_3, height, width,
        negative_prompt=negative_prompt, negative_prompt_2=negative_prompt_2, negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self.transformer.device

    # 2. Encode Prompts
    (
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt, prompt_2=prompt_2, prompt_3=prompt_3,
        prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt=negative_prompt, negative_prompt_2=negative_prompt_2, negative_prompt_3=negative_prompt_3,
        negative_prompt_embeds=negative_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device, num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )

    # 3. Handle Classifier-Free Guidance (CFG)
    do_classifier_free_guidance = guidance_scale > 1.0

    # 4. Prepare Latents
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

    # 5. Prepare Timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    
    # Slice the inference timeline for backward distillation steps
    run_till_timestep = run_till_timestep if run_till_timestep is not None else len(timesteps)
    timesteps = timesteps[run_from_timestep:run_till_timestep]

    # 6. Denoising Loop
    with self.progress_bar(total=len(timesteps)) as progress_bar:
        for i, t in enumerate(timesteps):
            if self._interrupt:
                continue

            # Expand latents if doing CFG
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            timestep = t.expand(latent_model_input.shape[0])

            # Broadcast conditions
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
            )[0]

            # Perform CFG combining
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute step backwards
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            progress_bar.update()

    if output_type == "latent":
        image = latents
    else:
        # Standard VAE Decode (SD3.5 scales latents but does not require unpacking shifts like Flux)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return StableDiffusion3PipelineOutput(images=image)
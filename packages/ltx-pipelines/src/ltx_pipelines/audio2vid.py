# packages/ltx-pipelines/src/ltx_pipelines/audio2vid.py

"""
Audio-to-Video Pipeline for LTX-2

Generates lip-synced video from:
- A single reference image (identity / face lock)
- An external audio file (speech)
- A text prompt (scene context)

Core idea:
- Encode speech into audio latents using LTX-2 audio VAE
- Freeze audio latents during diffusion
- Denoise ONLY video latents
- Let audio↔video cross-attention force lip sync
"""

import torch
import torch.nn.functional as F
import torchaudio
from typing import Optional

from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline


class Audio2VidPipeline(TI2VidTwoStagesPipeline):
    """
    Audio-driven video generation pipeline with frozen audio latents.

    This pipeline:
    1. Encodes external speech audio using the built-in LTX-2 audio VAE
    2. Uses a single reference image for identity preservation
    3. Freezes audio latents during denoising
    4. Lets cross-attention drive lip motion
    """

    # ------------------------------------------------------------
    # Audio encoding
    # ------------------------------------------------------------

    def encode_audio(
        self,
        audio_path: str,
        num_frames: int,
        frame_rate: float,
    ) -> torch.Tensor:
        """
        Encode an audio file into LTX-2 audio latents.

        Output shape:
            [B, C, T] where T == num_frames (after alignment)
        """

        # --------------------------------------------------------
        # Load waveform
        # --------------------------------------------------------
        audio, sr = torchaudio.load(audio_path)

        # Resample to 16 kHz (LTX-2 requirement)
        if sr != 16000:
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)

        # Ensure stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)

        # --------------------------------------------------------
        # Duration alignment
        # --------------------------------------------------------
        video_duration = num_frames / frame_rate
        expected_samples = int(video_duration * 16000)

        if audio.shape[1] < expected_samples:
            audio = F.pad(audio, (0, expected_samples - audio.shape[1]))
        else:
            audio = audio[:, :expected_samples]

        # --------------------------------------------------------
        # Move to model device
        # --------------------------------------------------------
        audio = audio.unsqueeze(0).to(
            device=self.device,
            dtype=self.dtype,
        )

        # --------------------------------------------------------
        # Encode using built-in LTX-2 Audio VAE
        # --------------------------------------------------------
        # NOTE:
        # - self.audio_vae is already loaded by the parent pipeline
        # - Do NOT invent a new encoder
        # - Do NOT pull from trainer code
        with torch.no_grad():
            audio_latents = self.audio_vae.encode(audio).latent_dist.sample()

        # --------------------------------------------------------
        # Temporal sanity check
        # --------------------------------------------------------
        if audio_latents.shape[-1] != num_frames:
            raise ValueError(
                f"Audio latent length ({audio_latents.shape[-1]}) "
                f"must match num_frames ({num_frames}). "
                f"Chunk or interpolate audio if needed."
            )

        return audio_latents

    # ------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        audio_path: str,
        reference_image: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        cfg_guidance_scale: float,
        tiling_config=None,
    ):
        """
        Generate a lip-synced video.

        Args:
            prompt: Scene / context prompt
            negative_prompt: Optional negative prompt
            audio_path: Path to speech audio (.wav)
            reference_image: Path to identity image
            seed: Random seed
            height, width: Output resolution
            num_frames: Number of frames to generate
            frame_rate: Frames per second
            num_inference_steps: Diffusion steps
            cfg_guidance_scale: CFG scale
            tiling_config: Optional tiling config
        """

        # --------------------------------------------------------
        # Encode audio → latents
        # --------------------------------------------------------
        audio_latents = self.encode_audio(
            audio_path=audio_path,
            num_frames=num_frames,
            frame_rate=frame_rate,
        )

        # --------------------------------------------------------
        # Identity conditioning (first-frame lock)
        # --------------------------------------------------------
        # Format: (image_path, frame_index, strength)
        images = [(reference_image, 0, 1.0)]

        # --------------------------------------------------------
        # Call parent pipeline with frozen audio
        # --------------------------------------------------------
        # IMPORTANT:
        # - Parent pipeline MUST accept:
        #     external_audio_latents
        #     freeze_audio
        # - Audio scheduler stepping must be disabled when freeze_audio=True
        return super().__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            cfg_guidance_scale=cfg_guidance_scale,
            images=images,
            tiling_config=tiling_config,
            # NEW PARAMETERS (must be wired in parent)
            external_audio_latents=audio_latents,
            freeze_audio=True,
        )

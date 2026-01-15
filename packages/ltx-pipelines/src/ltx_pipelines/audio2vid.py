"""
Audio-to-Video Pipeline for LTX-2

Generates lip-synced video from:
- A single reference image (identity)
- An audio file (speech to sync)
- A text prompt (scene description)
"""

import torch
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.helpers import (
    frozen_audio_euler_denoising_loop,
    audio_conditioned_denoising_func,
)

class Audio2VidPipeline(TI2VidTwoStagesPipeline):
    """
    Audio-driven video generation with frozen audio latents.
    
    Inherits from TI2VidTwoStagesPipeline but:
    1. Accepts external audio file
    2. Encodes audio to latents
    3. Freezes audio during video denoising
    4. Cross-attention forces lip sync
    """
    
    def encode_audio(
        self,
        audio_path: str,
        num_frames: int,
        frame_rate: float,
    ) -> torch.Tensor:
        """
        Encode audio file to LTX-2 audio latents.
        
        Uses the audio VAE encoder from the model.
        """
        import torchaudio
        
        # Load and preprocess audio
        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        
        # Stereo conversion
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        
        # Calculate expected duration
        video_duration = num_frames / frame_rate
        expected_samples = int(video_duration * 16000)
        
        # Pad or truncate
        if audio.shape[1] < expected_samples:
            audio = torch.nn.functional.pad(audio, (0, expected_samples - audio.shape[1]))
        else:
            audio = audio[:, :expected_samples]
        
        # Move to device
        audio = audio.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        
        # Encode through audio VAE
        # NOTE: You need to expose audio_encoder in ModelLedger
        # OR use the existing trainer preprocessing code
        audio_encoder = self._get_audio_encoder()
        with torch.no_grad():
            audio_latents = audio_encoder(audio)
        
        return audio_latents
    
    def _get_audio_encoder(self):
        """
        Get the audio VAE encoder.
        
        This may require looking at:
        - ltx_trainer/scripts/process_dataset.py (uses audio encoding)
        - ComfyUI-LTXVideo nodes (LTXVAudioVAEEncode)
        """
        # Check if trainer code has audio encoding
        # The trainer precomputes audio_latents, so the encoder exists
        
        # Option 1: Extract from trainer
        # from ltx_trainer.preprocessing import encode_audio
        
        # Option 2: Build from checkpoint (like video VAE)
        # The audio VAE weights are in the same checkpoint
        
        raise NotImplementedError(
            "Extract audio encoder from trainer or ComfyUI code. "
            "The encoder exists - see ltx_trainer/scripts/process_dataset.py"
        )
    
    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str,
        audio_path: str,           # NEW: Path to speech audio
        reference_image: str,      # Path to identity image
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        cfg_guidance_scale: float,
        tiling_config=None,
    ):
        # 1. Encode audio
        audio_latents = self.encode_audio(audio_path, num_frames, frame_rate)
        
        # 2. Set up image conditioning
        images = [(reference_image, 0, 1.0)]  # Frame 0, full strength
        
        # 3. Call parent pipeline with frozen audio
        # This requires modifying the parent class to accept
        # external_audio_latents and freeze_audio parameters
        
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
            # NEW PARAMS
            external_audio_latents=audio_latents,
            freeze_audio=True,
        )

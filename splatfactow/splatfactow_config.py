"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from functools import partial
from dn_splatter.dn_model import DNSplatterModelConfig
from splatfactow.splatfactow_datamanager import (
    SplatfactoWDatamanagerConfig,
)
from splatfactow.splatfactow_model import SplatfactoWModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from splatfactow.nerfw_dataparser import NerfWDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

# Import DN-Splatter components for hybrid configurations  
from dn_splatter.dn_datamanager import DNSplatterManagerConfig
from dn_splatter.dn_pipeline import DNSplatterPipelineConfig
from dn_splatter.data.normal_nerfstudio import NormalNerfstudioConfig

_dn_config = partial(DNSplatterModelConfig, regularization_strategy="dn-splatter")
_splatfactow_model = partial(SplatfactoWModelConfig, eval_right_half=True)
_splatfactow_light_model = partial(
    SplatfactoWModelConfig, 
    appearance_embed_dim=24,
    appearance_features_dim=32,
    app_layer_width=128,
    app_num_layers=2,
    bg_layer_width=128,
    bg_num_layers=2,
    sh_degree_interval=1000,
    bg_sh_degree=4,
    enable_bg_model=False,
    enable_alpha_loss=False,
    enable_robust_mask=False,
    never_mask_upper=0.0,
    use_avg_appearance=True,
    reset_alpha_every=30,
    stop_screen_size_at=4000,
    stop_split_at=15000,
)
_splatfactow_big_model = partial(
    SplatfactoWModelConfig,
    cull_alpha_thresh=0.005,  # Lower threshold for more Gaussians
    continue_cull_post_densification=False,  # Keep more Gaussians
)
_dn_config=partial(
    DNSplatterModelConfig,
    regularization_strategy="dn-splatter",
    use_normal_loss=True,
    normal_lambda=0.1,
    use_normal_tv_loss=True,
    use_depth_loss=True,
    use_normal_cosine_loss=True,
    depth_lambda=1.0,
    normal_supervision="depth",
)

splatfactow_config = MethodSpecification(
    description="Splatfacto in the wild",
    config=TrainerConfig(
        method_name="splatfacto-w",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=55000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=SplatfactoWDatamanagerConfig(
                dataparser=NerfWDataParserConfig(),
                cache_images_type="uint8",
            ),
            model=_splatfactow_model(),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-5, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-7,
                    max_steps=30000,
                ),
            },
            "appearance_features": {
                "optimizer": AdamOptimizerConfig(lr=0.02, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=40000,
                ),
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.03, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=30000
                ),
            },
            "field_background_encoder": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000
                ),
            },
            "field_background_base": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=2e-4, max_steps=30000
                ),
            },
            "field_background_rest": {
                "optimizer": AdamOptimizerConfig(lr=2e-3 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=2e-4 / 20, max_steps=30000
                ),
            },
            "appearance_model_encoder": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000
                ),
            },
            "appearance_model_base": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000
                ),
            },
            "appearance_model_rest": {
                "optimizer": AdamOptimizerConfig(lr=2e-3 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4 / 20, max_steps=30000
                ),
            },
            "appearance_embed": {
                "optimizer": AdamOptimizerConfig(lr=0.02, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=3e-4, max_steps=40000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
)

splatfactow_light_config = MethodSpecification(
    description="Splatfacto in the wild (light)",
    config=TrainerConfig(
        method_name="splatfacto-w-light",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(
                    load_3D_points=True,
                ),
                cache_images_type="uint8",
            ),
            model=_splatfactow_light_model(),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-5, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-7,
                    max_steps=30000,
                ),
            },
            "appearance_features": {
                "optimizer": AdamOptimizerConfig(lr=0.02, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=40000,
                ),
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.03, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=15000
                ),
            },
            "field_background_encoder": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=15000
                ),
            },
            "field_background_base": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=2e-4, max_steps=15000
                ),
            },
            "field_background_rest": {
                "optimizer": AdamOptimizerConfig(lr=2e-3 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=2e-4 / 20, max_steps=15000
                ),
            },
            "appearance_model_encoder": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=15000
                ),
            },
            "appearance_model_base": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=15000
                ),
            },
            "appearance_model_rest": {
                "optimizer": AdamOptimizerConfig(lr=2e-3 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4 / 20, max_steps=15000
                ),
            },
            "appearance_embed": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=3e-4, max_steps=15000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
)

splatfactow_dn_light_config = MethodSpecification(
    description="Splatfacto in the wild (light) with DN-Splatter depth and normal regularization",
    config=TrainerConfig(
        method_name="splatfacto-w-dn-light",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,  # Light version - shorter training
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100, "color": 10, "shs": 10},
        pipeline=DNSplatterPipelineConfig(
            datamanager=DNSplatterManagerConfig(
                dataparser=NormalNerfstudioConfig(load_3D_points=True)
            ),
            model=_splatfactow_light_model(
                dn_config=_dn_config(),
                num_downscales=0,
            )
        ),
        optimizers={
            # Reuse SplatfactoW light optimizers + add normals optimizer for DN-Splatter
            **splatfactow_light_config.config.optimizers,
            "normals": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
)


# Hybrid configurations combining SplatfactoW with DN-Splatter
# These use proper inheritance for type safety and linting compatibility

splatfactow_dn_config = MethodSpecification(
    description="Splatfacto in the wild with DN-Splatter depth and normal regularization",
    config=TrainerConfig(
        method_name="splatfacto-w-dn",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=55000,  # SplatfactoW's longer training
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100, "color": 10, "shs": 10},
        pipeline=DNSplatterPipelineConfig(
            datamanager=DNSplatterManagerConfig(
                dataparser=NormalNerfstudioConfig(load_3D_points=True)
            ),
            model=_splatfactow_model(
                dn_config=_dn_config(),
                num_downscales=0,
            ),
        ),
        optimizers={
            # Reuse SplatfactoW optimizers + add normals optimizer for DN-Splatter
            **splatfactow_config.config.optimizers,
            "normals": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
)

splatfactow_dn_big_config = MethodSpecification(
    description="Splatfacto in the wild with DN-Splatter depth and normal regularization (Big variant)",
    config=TrainerConfig(
        method_name="splatfacto-w-dn-big",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=55000,
        mixed_precision=False,
        pipeline=DNSplatterPipelineConfig(
            datamanager=DNSplatterManagerConfig(
                dataparser=NormalNerfstudioConfig(load_3D_points=True)
            ),
            model=_splatfactow_big_model(
                dn_config=_dn_config(),
                num_downscales=0,
            ),
        ),
        optimizers={
            # Reuse SplatfactoW optimizers + add normals optimizer for DN-Splatter
            **splatfactow_config.config.optimizers,
            "normals": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
)

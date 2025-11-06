import argparse

import numpy as np
import torch
import trimesh
from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, UNet2DConditionModel

from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
from mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils import get_orthogonal_camera, make_image_grid, tensor_to_image
from mvadapter.utils.mesh_utils import NVDiffRastContextWrapper, load_mesh, render
from mvadapter.utils.mesh_utils.render import get_clip_space_position


def extract_individual_meshes(glb_path):
    """Extract individual mesh objects from GLB file."""
    scene = trimesh.load(glb_path, force="scene", process=False)
    
    individual_meshes = []
    if isinstance(scene, trimesh.Scene):
        # Scene contains multiple geometry objects
        for name, geometry in scene.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                individual_meshes.append({
                    'name': name,
                    'mesh': geometry
                })
    elif isinstance(scene, trimesh.Trimesh):
        # Single mesh
        individual_meshes.append({
            'name': 'Mesh_0',
            'mesh': scene
        })
    
    return individual_meshes


def prepare_pipeline(
    base_model,
    vae_model,
    unet_model,
    lora_model,
    adapter_path,
    scheduler,
    num_views,
    device,
    dtype,
):
    # Load vae and unet if provided
    pipe_kwargs = {}
    if vae_model is not None:
        pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(vae_model)
    if unet_model is not None:
        pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(unet_model)

    # Prepare pipeline
    pipe: MVAdapterT2MVSDXLPipeline
    pipe = MVAdapterT2MVSDXLPipeline.from_pretrained(base_model, **pipe_kwargs)

    # Load scheduler if provided
    scheduler_class = None
    if scheduler == "ddpm":
        scheduler_class = DDPMScheduler
    elif scheduler == "lcm":
        scheduler_class = LCMScheduler

    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=scheduler_class,
    )
    pipe.init_custom_adapter(
        num_views=num_views, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0
    )
    pipe.load_custom_adapter(
        adapter_path, weight_name="mvadapter_tg2mv_sdxl.safetensors"
    )

    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)

    # load lora if provided
    if lora_model is not None:
        model_, name_ = lora_model.rsplit("/", 1)
        pipe.load_lora_weights(model_, weight_name=name_)

    # vae slicing for lower memory usage
    pipe.enable_vae_slicing()

    return pipe


def run_pipeline(
    pipe,
    mesh_path,
    num_views,
    text,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    seed,
    negative_prompt,
    lora_scale=1.0,
    device="cuda",
):
    # Prepare cameras
    cameras = get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
        distance=[1.8] * num_views,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
        device=device,
    )
    ctx = NVDiffRastContextWrapper(device=device, context_type="cuda")

    # Load full mesh for main rendering and get transformation parameters
    mesh, transform_offset, transform_scale = load_mesh(
        mesh_path, rescale=True, device=device, return_transform=True
    )
    render_out = render(
        ctx,
        mesh,
        cameras,
        height=height,
        width=width,
        render_attr=False,
        normal_background=0.0,
    )
    pos_images = tensor_to_image((render_out.pos + 0.5).clamp(0, 1), batched=True)
    normal_images = tensor_to_image(
        (render_out.normal / 2 + 0.5).clamp(0, 1), batched=True
    )
    
    # Extract combined mask
    combined_mask_images = tensor_to_image(render_out.mask.float(), batched=True)
    
    control_images = (
        torch.cat(
            [
                (render_out.pos + 0.5).clamp(0, 1),
                (render_out.normal / 2 + 0.5).clamp(0, 1),
            ],
            dim=-1,
        )
        .permute(0, 3, 1, 2)
        .to(device)
    )

    pipe_kwargs = {}
    if seed != -1:
        pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

    images = pipe(
        text,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_views,
        control_image=control_images,
        control_conditioning_scale=1.0,
        negative_prompt=negative_prompt,
        cross_attention_kwargs={"scale": lora_scale},
        **pipe_kwargs,
    ).images

    # Extract individual meshes and use MATERIAL ID approach
    individual_mesh_objects = extract_individual_meshes(mesh_path)
    individual_masks = {}
    
    print(f"\nRendering individual masks for {len(individual_mesh_objects)} meshes...")
    
    # Replicate load_mesh concatenation logic: concatenate first, then transform
    combined_trimesh = trimesh.Trimesh()
    vertex_to_mesh_id = []
    
    mesh_id_to_name = {}
    num_meshes = len(individual_mesh_objects)
    
    for mesh_idx, mesh_data in enumerate(individual_mesh_objects):
        mesh_name = mesh_data['name']
        trimesh_obj = mesh_data['mesh']
        
        # Assign unique ID (0-1 normalized)
        mesh_id = (mesh_idx + 1) / (num_meshes + 1)  # e.g., 0.33, 0.67 for 2 meshes
        mesh_id_to_name[mesh_id] = mesh_name
        
        # Track which mesh each vertex belongs to
        num_vertices = len(trimesh_obj.vertices)
        vertex_to_mesh_id.extend([mesh_id] * num_vertices)
        
        # Concatenate (same as load_mesh does)
        combined_trimesh = trimesh.util.concatenate([combined_trimesh, trimesh_obj])
    
    # Apply same transformations as load_mesh
    vertices = combined_trimesh.vertices.copy()
    if transform_offset is not None:
        offset = transform_offset.cpu().numpy() if hasattr(transform_offset, 'cpu') else transform_offset
        vertices = vertices - offset
    if transform_scale is not None:
        scale_val = transform_scale.item() if hasattr(transform_scale, 'item') else transform_scale
        max_scale = scale_val * 0.5
        vertices = vertices / max_scale * 0.5
    
    # Create vertex colors for material IDs
    vertex_to_mesh_id = np.array(vertex_to_mesh_id, dtype=np.float32)
    vertex_colors = np.column_stack([vertex_to_mesh_id, vertex_to_mesh_id, vertex_to_mesh_id])
    
    try:
        # Convert to torch tensors
        v_pos_torch = torch.from_numpy(vertices).float().to(device)
        t_pos_idx_torch = torch.from_numpy(combined_trimesh.faces).long().to(device)
        vertex_colors_torch = torch.from_numpy(vertex_colors).float().to(device)
        
        # Rasterize and interpolate vertex attributes (material IDs)
        v_pos_clip = get_clip_space_position(v_pos_torch, cameras.mvp_mtx)
        rast, _ = ctx.rasterize(v_pos_clip, t_pos_idx_torch, (height, width), grad_db=True)
        
        # Expand vertex colors to match batch dimension
        vertex_colors_batch = vertex_colors_torch[None].expand(rast.shape[0], -1, -1)
        id_attr_interp, _ = ctx.interpolate(vertex_colors_batch, rast, t_pos_idx_torch)
        
        # Extract material ID map
        material_id_map = id_attr_interp  # Shape: [num_views, height, width, 3]
        
        if material_id_map is not None and material_id_map.numel() > 0:
            # Use first channel for material IDs
            material_ids = material_id_map[..., 0]
            
            # Extract mask for each mesh by comparing IDs
            tolerance = 0.15  # Account for interpolation precision
            for mesh_id, mesh_name in mesh_id_to_name.items():
                mesh_mask = torch.abs(material_ids - mesh_id) < tolerance
                mask_images = tensor_to_image(mesh_mask.float(), batched=True)
                individual_masks[mesh_name] = mask_images
                print(f"  ✓ {mesh_name}: {mesh_mask.sum().item():,} pixels")
        else:
            raise ValueError("Could not render material IDs")
                
    except Exception as e:
        print(f"  ✗ Material ID approach failed: {e}")
        # Fallback: use combined mask for all meshes
        for mesh_data in individual_mesh_objects:
            individual_masks[mesh_data['name']] = combined_mask_images

    return images, pos_images, normal_images, combined_mask_images, individual_masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Models
    parser.add_argument(
        "--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
    )
    parser.add_argument(
        "--vae_model", type=str, default="madebyollin/sdxl-vae-fp16-fix"
    )
    parser.add_argument("--unet_model", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default="huanngzh/mv-adapter")
    parser.add_argument("--num_views", type=int, default=6)
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    # Inference
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="watermark, ugly, deformed, noisy, blurry, low contrast",
    )
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="output.png")
    args = parser.parse_args()

    pipe = prepare_pipeline(
        base_model=args.base_model,
        vae_model=args.vae_model,
        unet_model=args.unet_model,
        lora_model=args.lora_model,
        adapter_path=args.adapter_path,
        scheduler=args.scheduler,
        num_views=args.num_views,
        device=args.device,
        dtype=torch.float16,
    )
    images, pos_images, normal_images, combined_mask, individual_masks = run_pipeline(
        pipe,
        mesh_path=args.mesh,
        num_views=args.num_views,
        text=args.text,
        height=768,
        width=768,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        lora_scale=args.lora_scale,
        device=args.device,
    )
    
    # Save main outputs
    make_image_grid(images, rows=1).save(args.output)
    make_image_grid(pos_images, rows=1).save(args.output.rsplit(".", 1)[0] + "_pos.png")
    make_image_grid(normal_images, rows=1).save(
        args.output.rsplit(".", 1)[0] + "_nor.png"
    )
    
    # Save combined mask
    make_image_grid(combined_mask, rows=1).save(
        args.output.rsplit(".", 1)[0] + "_mask_combined.png"
    )
    
    # Save individual mesh masks
    print(f"\nSaving {len(individual_masks)} individual mesh masks...")
    for mesh_name, mask_images in individual_masks.items():
        # Sanitize mesh name for filename
        safe_name = mesh_name.replace(" ", "_").replace("/", "_")
        mask_filename = args.output.rsplit(".", 1)[0] + f"_mask_{safe_name}.png"
        make_image_grid(mask_images, rows=1).save(mask_filename)
        print(f"  ✓ Saved: {mask_filename}")

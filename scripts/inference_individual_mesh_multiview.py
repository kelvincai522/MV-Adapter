"""
Individual Mesh Multiview Generation Pipeline

Combines text-guided geometry multiview with partial view refinement to generate
separate, clean multiview images for each mesh in a GLB file.

Pipeline:
1. Generate full scene multiview with per-mesh masks (tg2mv_sdxl)
2. Create partial views by masking the full scene for each mesh
3. Refine each mesh individually using partial view generation (ig2mv_partial)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

# Import existing pipeline functions
from mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from mvadapter.utils import make_image_grid


def extract_individual_meshes(glb_path):
    """
    Extract individual mesh objects from GLB file.
    Returns both base geometry (for material ID) and transformed geometry (for export).
    """
    scene = trimesh.load(glb_path, force="scene", process=False)
    
    individual_meshes = []
    if isinstance(scene, trimesh.Scene):
        # Scene contains multiple geometry objects
        for name, geometry in scene.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                # Get node transformation if it exists
                nodes_with_geom = []
                for node_name in scene.graph.nodes:
                    node_data = scene.graph.transforms.node_data.get(node_name, {})
                    if 'geometry' in node_data and name in node_data['geometry']:
                        nodes_with_geom.append(node_name)
                
                # Create transformed version for export
                transform = np.eye(4)
                mesh_transformed = geometry.copy()
                
                if nodes_with_geom:
                    node_name = nodes_with_geom[0]
                    transform, _ = scene.graph[node_name]
                    mesh_transformed.apply_transform(transform)
                
                individual_meshes.append({
                    'name': name,
                    'mesh': geometry,  # Base geometry for material ID
                    'mesh_transformed': mesh_transformed,  # Transformed for export
                    'transform': transform
                })
    elif isinstance(scene, trimesh.Trimesh):
        # Single mesh
        individual_meshes.append({
            'name': 'Mesh_0',
            'mesh': scene,
            'mesh_transformed': scene.copy(),
            'transform': np.eye(4)
        })
    
    return individual_meshes


def generate_full_scene_multiview(mesh_path, text, output_dir, args):
    """
    Step 1: Generate full scene multiview with per-mesh masks.
    Uses the enhanced tg2mv_sdxl pipeline.
    """
    print("\n" + "=" * 80)
    print("STEP 1: Generating Full Scene Multiview")
    print("=" * 80)
    
    # Import the run_pipeline function from tg2mv_sdxl
    from scripts.inference_tg2mv_sdxl import prepare_pipeline, run_pipeline
    
    # Prepare pipeline
    pipe = prepare_pipeline(
        base_model=args.base_model,
        vae_model=args.vae_model,
        unet_model=None,
        lora_model=None,
        adapter_path=args.adapter_path,
        scheduler=None,
        num_views=args.num_views,
        device=args.device,
        dtype=torch.float16,
    )
    
    # Run pipeline
    images, pos_images, normal_images, combined_mask, individual_masks = run_pipeline(
        pipe,
        mesh_path=mesh_path,
        num_views=args.num_views,
        text=text,
        height=768,
        width=768,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        device=args.device,
    )
    
    # Save full scene outputs
    scene_dir = os.path.join(output_dir, "full_scene")
    os.makedirs(scene_dir, exist_ok=True)
    
    make_image_grid(images, rows=1).save(os.path.join(scene_dir, "multiview.png"))
    make_image_grid(pos_images, rows=1).save(os.path.join(scene_dir, "pos.png"))
    make_image_grid(normal_images, rows=1).save(os.path.join(scene_dir, "nor.png"))
    make_image_grid(combined_mask, rows=1).save(os.path.join(scene_dir, "mask_combined.png"))
    
    # Save individual masks
    masks_dir = os.path.join(output_dir, "full_scene_masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    for mesh_name, mask_images in individual_masks.items():
        safe_name = mesh_name.replace(" ", "_").replace("/", "_")
        mask_path = os.path.join(masks_dir, f"{safe_name}_mask.png")
        make_image_grid(mask_images, rows=1).save(mask_path)
    
    print(f"✓ Full scene saved to: {scene_dir}")
    print(f"✓ Individual masks saved to: {masks_dir}")
    
    return images, individual_masks


def create_partial_views(full_scene_images, individual_masks, output_dir):
    """
    Step 2: Create partial views by applying masks to the full scene.
    Each partial view shows only one mesh (as seen in the full scene).
    Extracts front view (first view) for refinement.
    """
    print("\n" + "=" * 80)
    print("STEP 2: Creating Partial Views")
    print("=" * 80)
    
    partial_dir = os.path.join(output_dir, "partial_views")
    os.makedirs(partial_dir, exist_ok=True)
    
    partial_views = {}
    
    # Convert full scene PIL images to single grid image
    full_scene_grid = make_image_grid(full_scene_images, rows=1)
    full_scene_array = np.array(full_scene_grid)
    
    # Extract front view only (first 768x768 from the 4608x768 grid)
    view_width = 768
    front_view_scene = full_scene_array[:, :view_width, :]
    
    for mesh_name, mask_images in individual_masks.items():
        # Convert mask to grid
        mask_grid = make_image_grid(mask_images, rows=1)
        mask_array = np.array(mask_grid)
        
        # Extract front view mask only
        front_view_mask = mask_array[:, :view_width]
        
        # Convert to grayscale if needed
        if len(front_view_mask.shape) == 3:
            front_view_mask = front_view_mask[:, :, 0]
        
        # Apply mask to front view only
        front_view_partial = front_view_scene.copy()
        
        # Create alpha channel from mask
        alpha = front_view_mask.astype(np.uint8)
        
        # Zero out RGB channels where alpha=0 (only keep masked mesh pixels)
        alpha_mask_3d = np.stack([alpha > 0] * 3, axis=-1)
        front_view_partial = front_view_partial * alpha_mask_3d
        
        # Combine RGB with alpha
        if len(front_view_partial.shape) == 3 and front_view_partial.shape[2] == 3:
            partial_rgba = np.dstack([front_view_partial, alpha])
        else:
            partial_rgba = front_view_partial
        
        # Convert to PIL Image
        partial_image = Image.fromarray(partial_rgba)
        
        # Save front view partial (768x768)
        safe_name = mesh_name.replace(" ", "_").replace("/", "_")
        partial_path = os.path.join(partial_dir, f"{safe_name}_partial_front.png")
        partial_image.save(partial_path)
        
        partial_views[mesh_name] = partial_path
        print(f"✓ Created front view partial for: {mesh_name} ({partial_image.size})")
    
    print(f"✓ Partial views saved to: {partial_dir}")
    
    return partial_views


def refine_individual_mesh(mesh_name, mesh_data, partial_view_path, text, output_dir, args):
    """
    Step 3: Refine individual mesh using partial view generation.
    Uses ig2mv_partial_sdxl to generate clean, complete multiview.
    Exports mesh with scene hierarchy transformations applied.
    """
    print(f"\n→ Refining mesh: {mesh_name}")
    
    # Load partial view image (front view only)
    # Don't convert to RGB - preserve alpha channel for preprocessing
    partial_image = Image.open(partial_view_path)
    
    # Save individual mesh as GLB in output directory
    # Use transformed mesh to preserve scene hierarchy transformations
    safe_name = mesh_name.replace(" ", "_").replace("/", "_")
    mesh_output_dir = os.path.join(output_dir, "individual_multiviews", safe_name)
    os.makedirs(mesh_output_dir, exist_ok=True)
    
    mesh_glb_path = os.path.join(mesh_output_dir, f"{safe_name}.glb")
    mesh_data['mesh_transformed'].export(mesh_glb_path)
    
    # Check if transform was applied
    is_identity = np.allclose(mesh_data['transform'], np.eye(4))
    if not is_identity:
        print(f"  ✓ Exported mesh GLB: {safe_name}.glb (with node transform)")
    else:
        print(f"  ✓ Exported mesh GLB: {safe_name}.glb")
    
    # Import the refinement pipeline
    from scripts.inference_ig2mv_partial_sdxl import prepare_pipeline, run_pipeline
    
    # Prepare pipeline
    pipe = prepare_pipeline(
        base_model=args.base_model,
        vae_model=args.vae_model,
        unet_model=None,
        lora_model=None,
        adapter_path=args.adapter_path,
        scheduler=None,
        num_views=args.num_views,
        device=args.device,
        dtype=torch.float16,
    )
    
    # Run refinement with the saved mesh GLB
    # Use lower guidance scale for image-guided generation (default 3.0 vs 7.0 for text)
    images, pos_images, normal_images, reference_image, transform_info = run_pipeline(
        pipe,
        mesh_path=mesh_glb_path,
        num_views=args.num_views,
        text=text,
        image=partial_image,
        height=768,
        width=768,
        num_inference_steps=args.refine_steps,
        guidance_scale=args.refine_guidance_scale,
        seed=args.seed,
        remove_bg_fn=None,  # Already masked
        negative_prompt=args.negative_prompt,
        device=args.device,
    )
    
    # Save outputs
    make_image_grid(images, rows=1).save(os.path.join(mesh_output_dir, "multiview.png"))
    make_image_grid(pos_images, rows=1).save(os.path.join(mesh_output_dir, "pos.png"))
    make_image_grid(normal_images, rows=1).save(os.path.join(mesh_output_dir, "nor.png"))
    reference_image.save(os.path.join(mesh_output_dir, "reference.png"))
    
    # Save transform info
    with open(os.path.join(mesh_output_dir, "transform.json"), 'w') as f:
        json.dump(transform_info, f, indent=4)
    
    print(f"  ✓ Saved to: {mesh_output_dir}")


def main_pipeline(args):
    """
    Main pipeline orchestrating the three steps.
    """
    print("\n" + "=" * 80)
    print("🎯 INDIVIDUAL MESH MULTIVIEW GENERATION PIPELINE")
    print("=" * 80)
    print(f"Mesh: {args.mesh}")
    print(f"Text: {args.text}")
    print(f"Output: {args.output_dir}")
    print(f"Refine: {args.refine}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract individual meshes
    individual_meshes = extract_individual_meshes(args.mesh)
    print(f"\nFound {len(individual_meshes)} meshes:")
    for mesh_data in individual_meshes:
        print(f"  • {mesh_data['name']}")
    
    # Step 1: Generate full scene with masks
    full_scene_images, individual_masks = generate_full_scene_multiview(
        args.mesh, args.text, args.output_dir, args
    )
    
    # Step 2: Create partial views
    partial_views = create_partial_views(
        full_scene_images, individual_masks, args.output_dir
    )
    
    # Step 3: Refine individual meshes (if enabled)
    if args.refine:
        print("\n" + "=" * 80)
        print("STEP 3: Refining Individual Meshes")
        print("=" * 80)
        
        # Load per-mesh prompts if provided
        mesh_prompts = {}
        if args.mesh_prompts and os.path.exists(args.mesh_prompts):
            with open(args.mesh_prompts, 'r') as f:
                mesh_prompts = json.load(f)
        
        for mesh_data in individual_meshes:
            mesh_name = mesh_data['name']
            
            # Skip if selective refinement and not in list
            if args.refine_only:
                refine_list = [m.strip() for m in args.refine_only.split(',')]
                if mesh_name not in refine_list:
                    print(f"\n→ Skipping {mesh_name} (not in refine_only list)")
                    continue
            
            # Get mesh-specific prompt or use default
            mesh_text = mesh_prompts.get(mesh_name, args.text)
            
            refine_individual_mesh(
                mesh_name,
                mesh_data,  # Pass full mesh_data dict (includes mesh_transformed)
                partial_views[mesh_name],
                mesh_text,
                args.output_dir,
                args
            )
    else:
        print("\n→ Skipping refinement (--refine=False)")
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {args.output_dir}")
    print("\nDirectory structure:")
    print(f"  {args.output_dir}/")
    print(f"    ├── full_scene/")
    print(f"    ├── full_scene_masks/")
    print(f"    ├── partial_views/")
    if args.refine:
        print(f"    └── individual_multiviews/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate individual mesh multiviews using sequential pipeline"
    )
    
    # Required arguments
    parser.add_argument("--mesh", type=str, required=True,
                        help="Path to multi-mesh GLB file")
    parser.add_argument("--text", type=str, required=True,
                        help="Base text prompt for full scene")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for outputs")
    
    # Optional arguments
    parser.add_argument("--mesh_prompts", type=str, default=None,
                        help="JSON file with per-mesh prompts")
    parser.add_argument("--refine", type=lambda x: x.lower() == 'true', default=True,
                        help="Whether to run refinement step")
    parser.add_argument("--refine_only", type=str, default=None,
                        help="Comma-separated list of mesh names to refine")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, 
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--vae_model", type=str, 
                        default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--adapter_path", type=str, 
                        default="huanngzh/mv-adapter")
    
    # Generation arguments
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Steps for full scene generation")
    parser.add_argument("--refine_steps", type=int, default=30,
                        help="Steps for individual mesh refinement")
    parser.add_argument("--guidance_scale", type=float, default=7.0,
                        help="Guidance scale for full scene generation")
    parser.add_argument("--refine_guidance_scale", type=float, default=3.0,
                        help="Guidance scale for individual mesh refinement (lower is better for image-guided)")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--negative_prompt", type=str,
                        default="watermark, ugly, deformed, noisy, blurry, low contrast")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    main_pipeline(args)


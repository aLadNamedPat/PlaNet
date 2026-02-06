#!/usr/bin/env python3
"""
Standalone script to generate RSSM dream videos.

This script creates "dream" videos where the RSSM model imagines/hallucinates
a trajectory purely from its learned world model without any real environment interaction.

Usage:
    python generate_dream.py --checkpoint checkpoints/best_model.pth --action-strategy random
    python generate_dream.py --checkpoint checkpoints/best_model.pth --action-strategy cem --dream-length 300
    python generate_dream.py --checkpoint checkpoints/best_model.pth --action-strategy sine --fps 60
"""

import torch
import numpy as np
import os
import argparse
import sys

# Import the dream generation function from visualize_actions
from visualize_actions import generate_dream_video, load_rssm


def main():
    parser = argparse.ArgumentParser(description='Generate RSSM dream videos')

    # Model and device settings
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to RSSM checkpoint file')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')

    # Dream generation settings
    parser.add_argument('--dream-length', type=int, default=200,
                        help='Number of timesteps for dream sequence')
    parser.add_argument('--action-strategy', type=str, default='random',
                        choices=['random', 'cem', 'zero', 'sine'],
                        help='Action strategy: random, cem (planned), zero (no actions), sine (oscillating)')
    parser.add_argument('--cem-horizon', type=int, default=12,
                        help='Planning horizon when using CEM action strategy')

    # Output settings
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (auto-generated if not specified)')
    parser.add_argument('--output-dir', type=str, default='dream_videos',
                        help='Output directory for dream videos')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video frames per second')

    # Advanced options
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible dreams')
    parser.add_argument('--deterministic-prior', action='store_true',
                        help='Use deterministic sampling from prior (less stochastic)')

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Set random seed to {args.seed}")

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        print("Make sure you have a trained RSSM model checkpoint.")
        print("\nTo train a model first, run:")
        print("  python training.py")
        print("\nOr download a pre-trained checkpoint if available.")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output path if not provided
    if args.output is None:
        filename = f"dream_{args.action_strategy}_{args.dream_length}steps"
        if args.seed is not None:
            filename += f"_seed{args.seed}"
        filename += ".mp4"
        args.output = os.path.join(args.output_dir, filename)

    print(f"\n{'='*60}")
    print("RSSM DREAM VIDEO GENERATOR")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Dream length: {args.dream_length} steps")
    print(f"Action strategy: {args.action_strategy}")
    print(f"Output: {args.output}")
    print(f"FPS: {args.fps}")

    try:
        # Load the RSSM model
        print("\nLoading RSSM model...")
        rssm = load_rssm(args.checkpoint, args.device)
        print("Model loaded successfully!")

        # Generate the dream video
        print(f"\nGenerating dream video...")
        imagined_frames, imagined_rewards, actions_taken = generate_dream_video(
            rssm=rssm,
            device=args.device,
            dream_length=args.dream_length,
            action_strategy=args.action_strategy,
            output_path=args.output,
            fps=args.fps,
            use_cem=(args.action_strategy == 'cem'),
            cem_horizon=args.cem_horizon
        )

        # Print final statistics
        print(f"\n{'='*60}")
        print("DREAM GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Video saved to: {args.output}")
        print(f"Dream duration: {len(imagined_frames)/args.fps:.1f} seconds")
        print(f"Total frames: {len(imagined_frames)}")
        print(f"Total imagined reward: {imagined_rewards.sum():.2f}")
        print(f"Mean reward per step: {imagined_rewards.mean():.4f}")

        # Action statistics
        print(f"\nAction Statistics:")
        action_names = ['R.Hip', 'R.Knee', 'R.Ankle', 'L.Hip', 'L.Knee', 'L.Ankle']
        for i, name in enumerate(action_names):
            if i < actions_taken.shape[1]:
                vals = actions_taken[:, i]
                print(f"  {name:<8}: mean={vals.mean():6.3f}, std={vals.std():6.3f}, "
                      f"min={vals.min():6.3f}, max={vals.max():6.3f}")

        print(f"\nTo view the dream video, open: {args.output}")

    except Exception as e:
        print(f"\nError generating dream video: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_usage_examples():
    """Print some usage examples"""
    print("\nUsage Examples:")
    print("================")
    print("# Basic random dream (most common)")
    print("python generate_dream.py --checkpoint checkpoints/best_model.pth")
    print()
    print("# CEM-planned dream (model tries to maximize reward)")
    print("python generate_dream.py --checkpoint checkpoints/best_model.pth --action-strategy cem")
    print()
    print("# Long sine wave dream")
    print("python generate_dream.py --checkpoint checkpoints/best_model.pth --action-strategy sine --dream-length 500")
    print()
    print("# Reproducible dream with seed")
    print("python generate_dream.py --checkpoint checkpoints/best_model.pth --seed 42")
    print()
    print("# High-FPS dream")
    print("python generate_dream.py --checkpoint checkpoints/best_model.pth --fps 60")


if __name__ == "__main__":
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        print_usage_examples()
        if len(sys.argv) == 1:
            sys.exit(0)

    main()
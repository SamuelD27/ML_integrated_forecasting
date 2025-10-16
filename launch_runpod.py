#!/usr/bin/env python3
"""
Simple RunPod Training Launcher
================================
Launches training on RunPod cloud GPUs using their API.

Usage:
    python launch_runpod.py --gpu "RTX A6000"
"""

import os
import sys
import argparse
import requests
import time
import json
from pathlib import Path
from typing import Dict, Optional

# RunPod API Configuration
RUNPOD_API_URL = "https://api.runpod.io/graphql"
DEFAULT_GPU = "NVIDIA RTX A6000"
DEFAULT_IMAGE = "runpod/pytorch:2.1.0-py3.11-cuda11.8.0-devel"

# GPU Type mapping
GPU_TYPES = {
    "RTX 3090": "NVIDIA GeForce RTX 3090",
    "RTX A6000": "NVIDIA RTX A6000",
    "A100": "NVIDIA A100-PCIE-40GB",
    "A100-80GB": "NVIDIA A100-SXM4-80GB",
}


class RunPodLauncher:
    """Simple RunPod API wrapper for launching training jobs"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "RunPod API key not found. Set RUNPOD_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _graphql_query(self, query: str, variables: Dict = None) -> Dict:
        """Execute GraphQL query against RunPod API"""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        response = requests.post(
            RUNPOD_API_URL,
            json=payload,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def create_pod(self,
                   name: str = "stock-analysis-training",
                   gpu_type: str = "RTX A6000",
                   image: str = DEFAULT_IMAGE,
                   volume_size: int = 50,
                   use_spot: bool = True) -> Dict:
        """
        Create a new RunPod instance.

        Args:
            name: Pod name
            gpu_type: GPU type (RTX 3090, RTX A6000, A100, A100-80GB)
            image: Docker image to use
            volume_size: Persistent volume size in GB
            use_spot: Use spot instances (cheaper but can be interrupted)

        Returns:
            Pod information dictionary
        """
        gpu_name = GPU_TYPES.get(gpu_type, DEFAULT_GPU)

        print(f"🚀 Creating RunPod instance...")
        print(f"   GPU: {gpu_type} ({gpu_name})")
        print(f"   Image: {image}")
        print(f"   Volume: {volume_size}GB")
        print(f"   Type: {'Spot' if use_spot else 'On-Demand'}")

        # Note: This is a simplified example. Real RunPod API integration
        # requires more complex setup with pod templates, etc.
        # For now, we'll provide instructions for manual setup.

        print("\n⚠️  Automated RunPod pod creation via API is not yet implemented.")
        print("    Please use the RunPod web interface or CLI for now.")
        print()
        return self._provide_manual_instructions(gpu_type, image, volume_size)

    def _provide_manual_instructions(self, gpu_type: str, image: str, volume_size: int) -> Dict:
        """Provide manual setup instructions"""

        instructions = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                   MANUAL RUNPOD SETUP INSTRUCTIONS                       ║
╚══════════════════════════════════════════════════════════════════════════╝

Since the RunPod GraphQL API requires complex pod templates, here's how to
set up training manually:

OPTION 1: RunPod Web Interface (Easiest)
──────────────────────────────────────────

1. Go to: https://www.runpod.io/console/pods

2. Click "+ Deploy" or "New Pod"

3. Select GPU:
   - Choose: {gpu_type}
   - Enable: Spot Instance (for cost savings)

4. Select Template:
   - Search for: "PyTorch"
   - Or use image: {image}

5. Configure Storage:
   - Volume Size: {volume_size}GB
   - Container Disk: 20GB

6. Environment Variables:
   Add these in the "Environment Variables" section:

   WANDB_API_KEY={os.getenv('WANDB_API_KEY', 'your_wandb_key')}
   PROJECT_NAME=stock_analysis
   WANDB_PROJECT=hybrid-trading-model

7. Expose Ports:
   - 8888 (Jupyter)
   - 6006 (TensorBoard)

8. Click "Deploy"


OPTION 2: RunPod CLI (Advanced)
────────────────────────────────

Install RunPod CLI:
  pip install runpodctl

Configure:
  runpodctl config --apiKey {self.api_key[:8]}...

Create pod:
  runpodctl create pod \\
    --name stock-analysis \\
    --gpuType "{gpu_type}" \\
    --imageName {image} \\
    --volumeSize {volume_size} \\
    --env WANDB_API_KEY={os.getenv('WANDB_API_KEY', 'your_key')[:8]}...


ONCE POD IS RUNNING:
────────────────────────────────

1. Connect to the pod (click "Connect" in web UI)

2. Open terminal and run:

   # Clone your repo
   git clone https://github.com/SamuelD27/ML_integrated_forecasting.git
   cd ML_integrated_forecasting

   # Install dependencies
   pip install -r requirements_training.txt

   # Set environment variables (if not set via UI)
   export WANDB_API_KEY=your_key

   # Run training
   python training/train_hybrid.py --config training/config.yaml

3. Monitor training:
   - TensorBoard: http://[pod-ip]:6006
   - Weights & Biases: https://wandb.ai


DOWNLOADING TRAINED MODELS:
────────────────────────────────

After training completes:

1. In RunPod terminal:
   tar -czf checkpoints.tar.gz checkpoints/

2. Download via RunPod web interface or use scp/rsync


ESTIMATED COSTS:
────────────────────────────────

{gpu_type}:
  - Spot: ~$0.50/hour
  - On-Demand: ~$0.80/hour

For 10 hour training: $5-8


Need help? Check: https://docs.runpod.io/


Press Ctrl+C to exit or Enter to continue...
        """

        print(instructions)

        try:
            input()
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)

        return {
            "status": "manual_setup_required",
            "gpu_type": gpu_type,
            "image": image,
            "volume_size": volume_size
        }


def main():
    parser = argparse.ArgumentParser(
        description="Launch training on RunPod cloud GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_runpod.py
  python launch_runpod.py --gpu "RTX A6000" --spot
  python launch_runpod.py --gpu "A100" --volume 100

This script provides instructions for setting up RunPod training.
For automated deployment, consider using RunPod's CLI or web interface.
        """
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="RTX A6000",
        choices=list(GPU_TYPES.keys()),
        help="GPU type to use"
    )

    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE,
        help="Docker image to use"
    )

    parser.add_argument(
        "--volume",
        type=int,
        default=50,
        help="Volume size in GB"
    )

    parser.add_argument(
        "--spot",
        action="store_true",
        help="Use spot instances (cheaper)"
    )

    args = parser.parse_args()

    # Check for API key
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("❌ Error: RUNPOD_API_KEY not found in environment")
        print("   Make sure you ran: source activate_env.sh")
        sys.exit(1)

    print(f"✓ RunPod API Key found: {api_key[:8]}...")
    print()

    # Create launcher
    launcher = RunPodLauncher(api_key)

    # Launch (or provide instructions)
    result = launcher.create_pod(
        gpu_type=args.gpu,
        image=args.image,
        volume_size=args.volume,
        use_spot=args.spot
    )

    print("\n✅ Setup information provided!")
    print("   Follow the instructions above to start training on RunPod.")


if __name__ == "__main__":
    main()

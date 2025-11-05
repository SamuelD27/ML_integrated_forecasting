---
name: runpod-quick-connect
description: Instantly connect VS Code Remote SSH to any RunPod pod with SSH configured
version: 1.0.0
dependencies: curl, jq
---

# RunPod Quick Connect

Automatically fetch SSH connection details and update VS Code Remote SSH config for any RunPod pod.

## Overview

Works with any pod template that has SSH enabled (TCP port 22 or any SSH-enabled template).
- Queries RunPod API for current connection details
- Updates SSH config automatically
- Supports frequent pod switching
- API key pre-configured for instant use

## Prerequisites

- SSH key registered in RunPod account
- Pod with SSH enabled (custom templates with TCP Port 22 work perfectly)
- VS Code with Remote-SSH extension
- `jq` installed (`brew install jq` on macOS, `apt install jq` on Linux)

## Installation

1. Extract the skill files to your preferred location:
   ```bash
   unzip runpod-quick-connect.zip -d ~/runpod-quick-connect
   cd ~/runpod-quick-connect
   chmod +x connect.sh
   ```

2. Verify jq is installed:
   ```bash
   jq --version
   ```

3. Test the connection:
   ```bash
   ./connect.sh list
   ```

## Usage

### List Available Pods
```bash
./connect.sh list
```
Shows all pods with their status (RUNNING/STOPPED).

### Connect to a Pod
```bash
# By pod name
./connect.sh "your-pod-name"

# By pod ID
./connect.sh "abc123xyz"

# With custom SSH alias
./connect.sh "your-pod-name" my-custom-alias
```

### Using with Claude Code

Simply ask Claude:
- "Connect to my RunPod pod training-server"
- "Update SSH for pod ml-training"
- "Switch to RunPod pod dev-env"

Claude will automatically execute the script and update your SSH config.

### Connecting in VS Code

After running the script:
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
2. Type "Remote-SSH: Connect to Host"
3. Select `runpod` (or your custom alias)
4. VS Code will connect via Remote-SSH

### Terminal Connection
```bash
ssh runpod
```

## Multiple Pod Aliases

If you frequently switch between multiple pods:
```bash
./connect.sh "training-pod" runpod-train
./connect.sh "test-pod" runpod-test
./connect.sh "dev-pod" runpod-dev
```

Then you'll have multiple aliases available:
- `ssh runpod-train`
- `ssh runpod-test`
- `ssh runpod-dev`

## How It Works

1. **API Query**: Fetches all pod details from RunPod GraphQL API
2. **Pod Matching**: Finds your pod by name or ID
3. **Validation**: Checks if pod is running and has SSH port 22 exposed
4. **Config Update**: Backs up and updates `~/.ssh/config` with connection details
5. **Ready**: You can now connect via `ssh runpod` or VS Code Remote-SSH

## Troubleshooting

### "Pod not found"
- Verify pod name/ID with `./connect.sh list`
- Check that pod exists in your RunPod account

### "Pod is not running"
- Start the pod in RunPod web interface
- Wait for initialization to complete

### "SSH port (22) not found"
- Ensure your template has TCP Port 22 exposed
- Check pod network configuration in RunPod

### "jq required"
- Install jq: `brew install jq` (macOS) or `apt install jq` (Linux)

### Connection refused
- Pod may still be initializing (wait 30-60 seconds)
- Check SSH service is running in pod
- Verify firewall settings

## Security Notes

- API key is embedded in the script for convenience
- SSH config uses `StrictHostKeyChecking no` for RunPod's dynamic IPs
- Keep your API key secure and don't commit to public repositories
- Backs up SSH config before modifications (`~/.ssh/config.bak`)

## Integration with VS Code Tasks (Optional)

Add to `.vscode/tasks.json`:
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Connect to RunPod",
      "type": "shell",
      "command": "${userHome}/runpod-quick-connect/connect.sh",
      "args": ["${input:podName}", "runpod"],
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    },
    {
      "label": "List RunPod Pods",
      "type": "shell",
      "command": "${userHome}/runpod-quick-connect/connect.sh",
      "args": ["list"]
    }
  ],
  "inputs": [
    {
      "id": "podName",
      "type": "promptString",
      "description": "Enter pod name or ID:"
    }
  ]
}
```

Then use: Terminal → Run Task → Connect to RunPod

## Compatibility

- ✅ Custom templates with TCP Port 22 exposed
- ✅ Official RunPod templates with SSH
- ✅ Any template with SSH over TCP
- ✅ macOS, Linux, WSL2

---

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Author**: MASUKA V2 Development

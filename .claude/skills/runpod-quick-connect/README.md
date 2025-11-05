# RunPod Quick Connect - Quick Start

Your RunPod API key is pre-configured and ready to use!

## Installation (30 seconds)

```bash
# 1. Extract the zip file
unzip runpod-quick-connect.zip -d ~/runpod-quick-connect
cd ~/runpod-quick-connect

# 2. Make script executable (if not already)
chmod +x connect.sh

# 3. Install jq (if not installed)
# macOS:
brew install jq

# Linux:
sudo apt install jq

# 4. Test it
./connect.sh list
```

## Usage

### List your pods
```bash
./connect.sh list
```

### Connect to a pod
```bash
# By name
./connect.sh "my-pod-name"

# By ID
./connect.sh "abc123xyz"

# With custom alias
./connect.sh "my-pod-name" my-alias
```

### Connect in VS Code
After running the script:
1. Press `Ctrl+Shift+P` (Cmd+Shift+P on macOS)
2. Select "Remote-SSH: Connect to Host"
3. Choose `runpod` (or your custom alias)

### Connect in Terminal
```bash
ssh runpod
```

## File Structure
```
runpod-quick-connect/
├── README.md         # This file (quick start)
├── SKILL.md          # Full documentation
└── connect.sh        # Connection script (API key embedded)
```

## Troubleshooting

**"jq not found"**
→ Install jq: `brew install jq` (macOS) or `sudo apt install jq` (Linux)

**"Pod not found"**
→ Run `./connect.sh list` to see available pods

**"Pod is not running"**
→ Start the pod in RunPod web interface

**"SSH port not found"**
→ Ensure your template exposes TCP Port 22

## Security Note

Your API key is embedded in `connect.sh` for convenience. Keep this file secure and don't commit it to public repositories.

## Full Documentation

See `SKILL.md` for complete documentation, advanced usage, and VS Code integration.

---

**Need help?** Check `SKILL.md` or ask Claude Code!

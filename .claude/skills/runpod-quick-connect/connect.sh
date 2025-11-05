#!/bin/bash
# RunPod Quick Connect - Works with any SSH-enabled template
# Usage: ./connect.sh <pod_name_or_id> [ssh_alias]
#        ./connect.sh list

set -e

# API key from environment variable or .env file
# Set RUNPOD_API_KEY in your environment or load from .env
API_KEY="${RUNPOD_API_KEY:-}"

# Check for list command
if [[ "$1" == "list" ]]; then
    echo "📋 Fetching pod list..."
    RESPONSE=$(curl -s --request POST \
        --header 'content-type: application/json' \
        --url "https://api.runpod.io/graphql?api_key=$API_KEY" \
        --data '{"query": "query { myself { pods { id name runtime { uptimeInSeconds } } } }"}')
    
    echo ""
    echo "Available pods:"
    echo "$RESPONSE" | jq -r '.data.myself.pods[] | "  • \(.name) (\(.id)) - \(if .runtime then "RUNNING (\(.runtime.uptimeInSeconds)s uptime)" else "STOPPED" end)"'
    echo ""
    exit 0
fi

# Connection mode
POD="${1:?Usage: ./connect.sh <pod_name_or_id> [ssh_alias] OR ./connect.sh list}"
ALIAS="${2:-runpod}"
SSH_CONFIG="$HOME/.ssh/config"
SSH_KEY="$HOME/.ssh/id_ed25519"

# Check jq is installed
if ! command -v jq >/dev/null 2>&1; then
    echo "❌ Error: jq is required but not installed."
    echo ""
    echo "Install jq:"
    echo "  macOS:  brew install jq"
    echo "  Linux:  sudo apt install jq"
    echo ""
    exit 1
fi

echo "🔍 Fetching pod details..."

# Query all pods (handles both ID and name)
RESPONSE=$(curl -s --request POST \
    --header 'content-type: application/json' \
    --url "https://api.runpod.io/graphql?api_key=$API_KEY" \
    --data '{"query": "query Pods { myself { pods { id name runtime { ports { ip privatePort publicPort type } } } } }"}')

# Check for API errors
if echo "$RESPONSE" | jq -e '.errors' >/dev/null 2>&1; then
    echo "❌ API Error:"
    echo "$RESPONSE" | jq -r '.errors[].message'
    exit 1
fi

# Find matching pod by ID or name
POD_DATA=$(echo "$RESPONSE" | jq -r ".data.myself.pods[] | select(.id == \"$POD\" or .name == \"$POD\")")

if [[ -z "$POD_DATA" ]] || [[ "$POD_DATA" == "null" ]]; then
    echo "❌ Pod not found: $POD"
    echo ""
    echo "Available pods:"
    echo "$RESPONSE" | jq -r '.data.myself.pods[] | "  • \(.name) (\(.id)) - \(if .runtime then "RUNNING" else "STOPPED" end)"'
    echo ""
    echo "Usage: ./connect.sh <pod_name_or_id> [ssh_alias]"
    exit 1
fi

# Check if running
if [[ "$(echo "$POD_DATA" | jq -r '.runtime')" == "null" ]]; then
    POD_NAME=$(echo "$POD_DATA" | jq -r '.name')
    echo "❌ Pod '$POD_NAME' is not running."
    echo ""
    echo "Please start the pod in RunPod web interface and try again."
    exit 1
fi

# Extract SSH connection info (port 22)
SSH_INFO=$(echo "$POD_DATA" | jq -r '.runtime.ports[] | select(.privatePort == 22 and .type == "tcp")')

if [[ -z "$SSH_INFO" ]] || [[ "$SSH_INFO" == "null" ]]; then
    POD_NAME=$(echo "$POD_DATA" | jq -r '.name')
    echo "❌ SSH port (22) not found on pod '$POD_NAME'."
    echo ""
    echo "Ensure your template has TCP Port 22 exposed for SSH."
    exit 1
fi

IP=$(echo "$SSH_INFO" | jq -r '.ip')
PORT=$(echo "$SSH_INFO" | jq -r '.publicPort')
NAME=$(echo "$POD_DATA" | jq -r '.name')

echo "✅ Found: $NAME"
echo "✅ SSH: root@$IP:$PORT"

# Update SSH config
echo "📝 Updating SSH config..."

# Create SSH directory if it doesn't exist
mkdir -p "$(dirname "$SSH_CONFIG")"

# Backup existing config
if [[ -f "$SSH_CONFIG" ]]; then
    cp "$SSH_CONFIG" "$SSH_CONFIG.bak"
    echo "   Backup created: $SSH_CONFIG.bak"
fi

# Create config if it doesn't exist
touch "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"

# Remove old entry for this alias
if grep -q "^Host $ALIAS$" "$SSH_CONFIG" 2>/dev/null; then
    # Create temp file without the old entry
    awk -v alias="$ALIAS" '
        BEGIN { skip = 0 }
        /^Host / { 
            if ($2 == alias) { 
                skip = 1 
            } else { 
                skip = 0 
            }
        }
        !skip { print }
    ' "$SSH_CONFIG" > "$SSH_CONFIG.tmp"
    mv "$SSH_CONFIG.tmp" "$SSH_CONFIG"
    echo "   Removed old entry for '$ALIAS'"
fi

# Add new entry
cat >> "$SSH_CONFIG" << EOF

# RunPod: $NAME (auto-generated $(date +%Y-%m-%d))
Host $ALIAS
    HostName $IP
    User root
    Port $PORT
    IdentityFile $SSH_KEY
    IdentitiesOnly yes
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 5
EOF

echo "   Added new entry for '$ALIAS'"
echo ""
echo "🎉 SSH config updated successfully!"
echo ""
echo "Connect now:"
echo "  Terminal:  ssh $ALIAS"
echo "  VS Code:   Ctrl+Shift+P → Remote-SSH: Connect to Host → $ALIAS"
echo ""

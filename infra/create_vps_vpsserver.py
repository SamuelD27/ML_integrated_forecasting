#!/usr/bin/env python3
"""
VPSServer.com VPS Provisioning Script
======================================

Creates and manages VPS instances on VPSServer.com for the Alpaca trading bot.

Usage:
    # Create a new VPS
    python infra/create_vps_vpsserver.py --create

    # List existing servers
    python infra/create_vps_vpsserver.py --list

    # Destroy a server by ID
    python infra/create_vps_vpsserver.py --destroy <server_id>

    # Show available options (datacenters, images, etc.)
    python infra/create_vps_vpsserver.py --options

Environment Variables Required:
    VPSSERVER_CLIENT_ID    - Your VPSServer API client ID
    VPSSERVER_SECRET       - Your VPSServer API secret
    VPS_ROOT_PASSWORD      - Root password for the new VPS (min 8 chars, mixed case + numbers)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

try:
    import requests
except ImportError:
    print("Error: 'requests' package not installed. Run: pip install requests")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VPSConfig:
    """VPS configuration for trading bot."""
    # Datacenter: US-NY2 = New York (closest to Alpaca for low latency)
    datacenter: str = "US-NY2"

    # OS Image: Ubuntu Server 24.04 64-bit
    image_id: str = "US-NY2:6000C29549da189eaef6ea8a31001a34"
    image_description: str = "ubuntu_server_24.04_64-bit"

    # Resources: 2 vCPU (Type A), 4GB RAM, 20GB SSD
    cpu: str = "2A"      # 2 cores, Type A
    ram: int = 4096      # 4GB in MB
    disk: int = 20       # 20GB SSD

    # Billing
    billing: str = "hourly"  # or "monthly"

    # Network
    traffic: str = "t5000"  # 5TB traffic
    network: str = "wan"    # Public network

    # Hostname
    hostname: str = "alpaca-bot-1"


# API Configuration
API_BASE_URL = "https://console.vpsserver.com/service"


# =============================================================================
# API Client
# =============================================================================

class VPSServerClient:
    """Client for VPSServer.com API."""

    def __init__(self, client_id: str, secret: str):
        """
        Initialize the client.

        Args:
            client_id: VPSServer API client ID
            secret: VPSServer API secret
        """
        self.client_id = client_id
        self.secret = secret
        self._token: Optional[str] = None
        self._token_expires: int = 0

    def _authenticate(self) -> str:
        """Authenticate and get a token."""
        response = requests.post(
            f"{API_BASE_URL}/authenticate",
            headers={"Content-Type": "application/json"},
            json={"clientId": self.client_id, "secret": self.secret},
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"Authentication failed: {response.status_code} - {response.text}")

        data = response.json()
        if "errors" in data:
            raise Exception(f"Authentication error: {data['errors']}")

        self._token = data["authentication"]
        self._token_expires = data.get("expires", 0)
        return self._token

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with valid auth token."""
        # Re-authenticate if token is missing or expired
        if not self._token or time.time() > self._token_expires - 60:
            self._authenticate()

        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json"
        }

    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make an API request."""
        url = f"{API_BASE_URL}/{endpoint}"
        headers = self._get_headers()

        if method == "GET":
            response = requests.get(url, headers=headers, timeout=60)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=120)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=60)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if response.status_code not in [200, 201, 202, 204]:
            raise Exception(f"API error: {response.status_code} - {response.text}")

        if response.text:
            result = response.json()
            if "errors" in result:
                raise Exception(f"API error: {result['errors']}")
            return result
        return {}

    def get_server_options(self) -> Dict[str, Any]:
        """Get available server configuration options."""
        return self._request("GET", "server")

    def list_servers(self) -> List[Dict]:
        """List all servers."""
        return self._request("GET", "servers") or []

    def get_server(self, server_id: str) -> Dict:
        """Get server details."""
        return self._request("GET", f"server/{server_id}")

    def create_server(
        self,
        name: str,
        datacenter: str,
        image: str,
        cpu: str,
        ram: int,
        disk: int,
        password: str,
        billing: str = "hourly",
        traffic: str = "t5000",
        network: str = "wan",
    ) -> Dict:
        """
        Create a new server.

        Args:
            name: Server hostname
            datacenter: Datacenter ID (e.g., "US-NY2")
            image: Disk image ID
            cpu: CPU type (e.g., "2A")
            ram: RAM in MB
            disk: Disk size in GB
            password: Root password
            billing: "hourly" or "monthly"
            traffic: Traffic quota
            network: Network type

        Returns:
            Server creation response with ID and details
        """
        payload = {
            "name": name,
            "datacenter": datacenter,
            "image": image,
            "cpu": cpu,
            "ram": ram,
            "disk": disk,
            "password": password,
            "billing": billing,
            "traffic": traffic,
            "network": network,
        }

        return self._request("POST", "server", payload)

    def destroy_server(self, server_id: str) -> Dict:
        """Destroy a server."""
        return self._request("DELETE", f"server/{server_id}")

    def wait_for_server(self, server_id: str, timeout: int = 300) -> Dict:
        """Wait for server to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            server = self.get_server(server_id)
            status = server.get("status", "").lower()

            if status == "on" or status == "running":
                return server
            elif status in ["error", "failed"]:
                raise Exception(f"Server creation failed: {server}")

            print(f"  Waiting for server... (status: {status})")
            time.sleep(10)

        raise Exception(f"Timeout waiting for server {server_id}")


# =============================================================================
# Main Functions
# =============================================================================

def create_vps(client: VPSServerClient, config: VPSConfig, password: str) -> Dict:
    """Create a new VPS with the specified configuration."""
    print(f"Creating VPS '{config.hostname}'...")
    print(f"  Datacenter: {config.datacenter}")
    print(f"  Image: {config.image_description}")
    print(f"  CPU: {config.cpu}")
    print(f"  RAM: {config.ram}MB")
    print(f"  Disk: {config.disk}GB")
    print(f"  Billing: {config.billing}")
    print()

    result = client.create_server(
        name=config.hostname,
        datacenter=config.datacenter,
        image=config.image_id,
        cpu=config.cpu,
        ram=config.ram,
        disk=config.disk,
        password=password,
        billing=config.billing,
        traffic=config.traffic,
        network=config.network,
    )

    server_id = result.get("id") or result.get("serverId")
    if not server_id:
        # Sometimes the response structure varies
        print(f"Server creation initiated: {result}")
        return result

    print(f"Server creation initiated with ID: {server_id}")
    print("Waiting for server to be ready...")

    try:
        server = client.wait_for_server(server_id)
        ip_address = server.get("ip") or server.get("networks", [{}])[0].get("ips", ["N/A"])[0]

        print()
        print("=" * 60)
        print("VPS CREATED SUCCESSFULLY!")
        print("=" * 60)
        print(f"  Server ID:    {server_id}")
        print(f"  Hostname:     {config.hostname}")
        print(f"  IP Address:   {ip_address}")
        print(f"  Datacenter:   {config.datacenter}")
        print(f"  OS:           {config.image_description}")
        print()
        print("SSH Access:")
        print(f"  ssh root@{ip_address}")
        print()
        print("=" * 60)

        return server
    except Exception as e:
        print(f"Warning: Server created but status check failed: {e}")
        print(f"Check the VPSServer console for server ID: {server_id}")
        return result


def list_servers(client: VPSServerClient) -> None:
    """List all servers."""
    servers = client.list_servers()

    if not servers:
        print("No servers found.")
        return

    print("=" * 80)
    print("EXISTING SERVERS")
    print("=" * 80)

    for server in servers:
        server_id = server.get("id", "N/A")
        name = server.get("name", "N/A")
        status = server.get("status", "N/A")
        ip = server.get("ip", "N/A")
        datacenter = server.get("datacenter", "N/A")

        print(f"\nServer: {name}")
        print(f"  ID:         {server_id}")
        print(f"  Status:     {status}")
        print(f"  IP:         {ip}")
        print(f"  Datacenter: {datacenter}")

    print()


def show_options(client: VPSServerClient) -> None:
    """Show available configuration options."""
    options = client.get_server_options()

    print("=" * 60)
    print("AVAILABLE DATACENTERS")
    print("=" * 60)
    for dc_id, dc_name in sorted(options.get("datacenters", {}).items()):
        print(f"  {dc_id}: {dc_name}")

    print()
    print("=" * 60)
    print("CPU OPTIONS (Type A = Standard, B = High-Memory, T = High-CPU)")
    print("=" * 60)
    cpu_opts = options.get("cpu", [])[:15]  # First 15
    print(f"  {', '.join(cpu_opts)}...")

    print()
    print("=" * 60)
    print("RAM OPTIONS (in MB)")
    print("=" * 60)
    ram_opts = options.get("ram", {})
    if isinstance(ram_opts, dict):
        for cpu_type, values in ram_opts.items():
            print(f"  Type {cpu_type}: {values[:8]}...")

    print()
    print("=" * 60)
    print("DISK OPTIONS (in GB)")
    print("=" * 60)
    disk_opts = options.get("disk", [])[:15]
    print(f"  {disk_opts}")

    print()
    print("=" * 60)
    print("UBUNTU IMAGES FOR US-NY2")
    print("=" * 60)
    images = options.get("diskImages", {}).get("US-NY2", [])
    for img in images:
        desc = img.get("description", "")
        if "ubuntu" in desc.lower() and "server" in desc.lower():
            if not desc.startswith("apps_") and not desc.startswith("service_") and "desktop" not in desc.lower():
                print(f"  {img['id']}: {desc}")


def destroy_server(client: VPSServerClient, server_id: str) -> None:
    """Destroy a server by ID."""
    print(f"Destroying server {server_id}...")

    # Confirm
    confirm = input(f"Are you sure you want to destroy server {server_id}? (yes/no): ")
    if confirm.lower() != "yes":
        print("Cancelled.")
        return

    client.destroy_server(server_id)
    print(f"Server {server_id} destroyed.")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VPSServer.com VPS Provisioning for Alpaca Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--create", action="store_true", help="Create a new VPS")
    parser.add_argument("--list", action="store_true", help="List existing servers")
    parser.add_argument("--destroy", type=str, help="Destroy a server by ID")
    parser.add_argument("--options", action="store_true", help="Show available options")

    # Override defaults
    parser.add_argument("--datacenter", type=str, help="Datacenter ID (default: US-NY2)")
    parser.add_argument("--cpu", type=str, help="CPU type (default: 2A)")
    parser.add_argument("--ram", type=int, help="RAM in MB (default: 4096)")
    parser.add_argument("--disk", type=int, help="Disk in GB (default: 20)")
    parser.add_argument("--hostname", type=str, help="Server hostname (default: alpaca-bot-1)")
    parser.add_argument("--billing", choices=["hourly", "monthly"], help="Billing type")

    args = parser.parse_args()

    # Get credentials from environment
    client_id = os.environ.get("VPSSERVER_CLIENT_ID")
    secret = os.environ.get("VPSSERVER_SECRET")

    if not client_id or not secret:
        print("Error: Missing environment variables.")
        print("  export VPSSERVER_CLIENT_ID='your_client_id'")
        print("  export VPSSERVER_SECRET='your_secret'")
        sys.exit(1)

    # Create client
    client = VPSServerClient(client_id, secret)

    # Execute command
    if args.list:
        list_servers(client)

    elif args.options:
        show_options(client)

    elif args.destroy:
        destroy_server(client, args.destroy)

    elif args.create:
        # Get password
        password = os.environ.get("VPS_ROOT_PASSWORD")
        if not password:
            import getpass
            password = getpass.getpass("Enter root password for new VPS: ")

        if len(password) < 8:
            print("Error: Password must be at least 8 characters")
            sys.exit(1)

        # Build config
        config = VPSConfig()
        if args.datacenter:
            config.datacenter = args.datacenter
        if args.cpu:
            config.cpu = args.cpu
        if args.ram:
            config.ram = args.ram
        if args.disk:
            config.disk = args.disk
        if args.hostname:
            config.hostname = args.hostname
        if args.billing:
            config.billing = args.billing

        create_vps(client, config, password)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

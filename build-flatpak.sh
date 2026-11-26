#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/bus"

echo "[1/3] Vendoring Cargo crates inside my-ubuntu..."
podman exec my-ubuntu bash -lc 'export PATH="$HOME/.cargo/bin:$PATH"; cd /workspace/vulkanscope-pro && cargo vendor flatpak/vendor > /dev/null'

echo "[2/3] Building Flatpak inside Freedesktop 25.08 SDK..."
flatpak run org.flatpak.Builder \
  --force-clean \
  --repo=flatpak/repo \
  flatpak/build-dir \
  flatpak/io.github.vulkanscope.VulkanScopePro.yml

echo "[3/3] Exporting single-file bundle vulkanscope-pro.flatpak..."
flatpak build-bundle \
  flatpak/repo \
  vulkanscope-pro.flatpak \
  io.github.vulkanscope.VulkanScopePro \
  --runtime-repo=https://flathub.org/repo/flathub.flatpakrepo

ls -lh vulkanscope-pro.flatpak
echo "Done! Install locally with: flatpak install --user --reinstall -y vulkanscope-pro.flatpak"

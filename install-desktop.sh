#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT/target/release/vulkanscope-pro"

mkdir -p "$HOME/.local/bin" \
         "$HOME/.local/share/applications" \
         "$HOME/.local/share/icons/hicolor/scalable/apps"

ln -sf "$BIN" "$HOME/.local/bin/vulkanscope-pro"
cp "$ROOT/assets/io.github.vulkanscope.VulkanScopePro.svg" \
   "$HOME/.local/share/icons/hicolor/scalable/apps/io.github.vulkanscope.VulkanScopePro.svg"

sed "s|^Exec=.*|Exec=$BIN|" \
   "$ROOT/assets/io.github.vulkanscope.VulkanScopePro.desktop" \
   > "$HOME/.local/share/applications/io.github.vulkanscope.VulkanScopePro.desktop"

command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$HOME/.local/share/applications" || true
echo "Installed VulkanScope Pro desktop entry & icon!"

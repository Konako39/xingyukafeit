#!/bin/zsh
# 编译 EventKit 日历助手，并打包成 .app。
# 打包成 .app 是必须的：裸命令行程序在后台被调用时 macOS 不会弹授权窗，
# 只有正经的 .app 身份双击运行才能拿到日历与提醒事项权限。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_DIR="$PROJECT_ROOT/应用/后端"
OUTPUT_DIR="$PROJECT_ROOT/数据/客户端"
APP_DIR="$OUTPUT_DIR/日历助手.app"
mkdir -p "$APP_DIR/Contents/MacOS"

swiftc -O -swift-version 5 \
  "$SOURCE_DIR/日历助手.swift" \
  -o "$APP_DIR/Contents/MacOS/日历助手" \
  -framework EventKit -framework AppKit -framework Foundation

cp "$SOURCE_DIR/日历助手-Info.plist" "$APP_DIR/Contents/Info.plist"
codesign --force --deep --sign - "$APP_DIR" 2>/dev/null || true

# 兼容旧路径：保留一个指向包内可执行文件的软链接。
ln -sf "日历助手.app/Contents/MacOS/日历助手" "$OUTPUT_DIR/日历助手"

echo "已生成：$APP_DIR"
echo "首次使用请双击它一次，按提示允许日历与提醒事项。"

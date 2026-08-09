#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
PROJECT_ROOT=${SCRIPT_DIR:h}
APP_NAME="星语茶话屋.app"
APP_TARGET="$PROJECT_ROOT/$APP_NAME"
NATIVE_STATE_DIR="$PROJECT_ROOT/数据/客户端"
NATIVE_HASH_FILE="$NATIVE_STATE_DIR/原生构建.sha256"
NATIVE_HASH=$(
  shasum -a 256 \
    "$PROJECT_ROOT/应用/原生/LocalAIStudio.swift" \
    "$PROJECT_ROOT/应用/原生/Info.plist" \
    "$PROJECT_ROOT/图片资源/icon.png" \
  | shasum -a 256 | awk '{print $1}'
)

# 后端和 Web 界面由应用在项目目录中直接加载，不需要重签原生 app。
# 原生源码未变化时保留同一个可执行文件/CDHash，避免 macOS 把已经授予的
# 屏幕录制权限当成一个全新的程序。
if [[ -x "$APP_TARGET/Contents/MacOS/LocalAIStudio" \
      && -f "$NATIVE_HASH_FILE" \
      && "$(<"$NATIVE_HASH_FILE")" == "$NATIVE_HASH" ]]; then
  echo "$APP_TARGET（原生代码未变化，保留签名与屏幕权限）"
  exit 0
fi

# 当前是 ad-hoc 签名：重编译会改变 CDHash，macOS 会把它当成
# 新应用并丢失已有的屏幕录制授权。已存在客户端时默认拒绝
# 这种破坏性重建；只有主人明确决定迁移签名时才能手动放行。
if [[ -x "$APP_TARGET/Contents/MacOS/LocalAIStudio" \
      && -f "$NATIVE_HASH_FILE" \
      && "$(<"$NATIVE_HASH_FILE")" != "$NATIVE_HASH" \
      && "${STAR_TEAHOUSE_ALLOW_NATIVE_REBUILD:-0}" != "1" ]]; then
  echo "已检测到原生源码变化，但未重建当前 App：这会改变 CDHash 并让 macOS 重新要求屏幕权限。" >&2
  echo "当前客户端与已有权限已保留。若确定迁移签名，再显式设置 STAR_TEAHOUSE_ALLOW_NATIVE_REBUILD=1。" >&2
  exit 3
fi

BUILD_DIR=$(mktemp -d /tmp/starry-teahouse-build.XXXXXX)
APP_DIR="$BUILD_DIR/$APP_NAME"
CONTENTS="$APP_DIR/Contents"
RESOURCES="$CONTENTS/Resources"

mkdir -p "$CONTENTS/MacOS" "$RESOURCES/runtime/后端" "$RESOURCES/runtime/界面"

xcrun swiftc \
  -O \
  -framework AppKit \
  -framework CoreGraphics \
  -framework ScreenCaptureKit \
  -framework ServiceManagement \
  -framework WebKit \
  "$PROJECT_ROOT/应用/原生/LocalAIStudio.swift" \
  -o "$CONTENTS/MacOS/LocalAIStudio"

cp "$PROJECT_ROOT/应用/原生/Info.plist" "$CONTENTS/Info.plist"
ditto "$PROJECT_ROOT/应用/后端" "$RESOURCES/runtime/后端"
ditto "$PROJECT_ROOT/应用/界面" "$RESOURCES/runtime/界面"

cp "$PROJECT_ROOT/图片资源/icon.png" "$RESOURCES/MenuBarIcon.png"

ICON_WORK="$BUILD_DIR/icon-work"
ICONSET="$ICON_WORK/AppIcon.iconset"
mkdir -p "$ICONSET"
ICON_SOURCE="$PROJECT_ROOT/图片资源/icon.png"
sips -z 16 16 "$ICON_SOURCE" --out "$ICONSET/icon_16x16.png" >/dev/null
sips -z 32 32 "$ICON_SOURCE" --out "$ICONSET/icon_16x16@2x.png" >/dev/null
sips -z 32 32 "$ICON_SOURCE" --out "$ICONSET/icon_32x32.png" >/dev/null
sips -z 64 64 "$ICON_SOURCE" --out "$ICONSET/icon_32x32@2x.png" >/dev/null
sips -z 128 128 "$ICON_SOURCE" --out "$ICONSET/icon_128x128.png" >/dev/null
sips -z 256 256 "$ICON_SOURCE" --out "$ICONSET/icon_128x128@2x.png" >/dev/null
sips -z 256 256 "$ICON_SOURCE" --out "$ICONSET/icon_256x256.png" >/dev/null
sips -z 512 512 "$ICON_SOURCE" --out "$ICONSET/icon_256x256@2x.png" >/dev/null
sips -z 512 512 "$ICON_SOURCE" --out "$ICONSET/icon_512x512.png" >/dev/null
sips -z 1024 1024 "$ICON_SOURCE" --out "$ICONSET/icon_512x512@2x.png" >/dev/null
iconutil -c icns "$ICONSET" -o "$RESOURCES/AppIcon.icns"

codesign --force --deep --sign - "$APP_DIR" >/dev/null
ditto "$APP_DIR" "$APP_TARGET"
mkdir -p "$NATIVE_STATE_DIR"
print -r -- "$NATIVE_HASH" > "$NATIVE_HASH_FILE"

echo "$APP_TARGET"

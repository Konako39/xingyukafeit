ObjC.import("AppKit");
ObjC.import("Foundation");
ObjC.import("CoreGraphics");

function value(dictionary, key, fallback) {
  const item = dictionary.objectForKey(key);
  if (!item) return fallback;
  try { return ObjC.unwrap(item); } catch (_) { return fallback; }
}

const workspace = $.NSWorkspace.sharedWorkspace;
const application = workspace.frontmostApplication;
const pid = application ? Number(application.processIdentifier) : 0;
const bundle = application && application.bundleIdentifier
  ? ObjC.unwrap(application.bundleIdentifier)
  : "";
const name = application && application.localizedName
  ? ObjC.unwrap(application.localizedName)
  : "";

const screens = [];
const screenList = $.NSScreen.screens;
for (let index = 0; index < Number(screenList.count); index += 1) {
  const frame = screenList.objectAtIndex(index).frame;
  screens.push({
    width: Number(frame.size.width),
    height: Number(frame.size.height),
  });
}

let fullscreen = false;
let largestRatio = 0;
let largestLayer = 0;
if (pid > 0) {
  const options = 1 | 16; // onScreenOnly | excludeDesktopElements
  const windowList = ObjC.castRefToObject(
    $.CGWindowListCopyWindowInfo(options, 0)
  );
  for (let index = 0; index < Number(windowList.count); index += 1) {
    const windowInfo = windowList.objectAtIndex(index);
    if (Number(value(windowInfo, "kCGWindowOwnerPID", 0)) !== pid) continue;
    const alpha = Number(value(windowInfo, "kCGWindowAlpha", 1));
    if (alpha <= 0.01) continue;
    const boundsObject = windowInfo.objectForKey("kCGWindowBounds");
    if (!boundsObject) continue;
    const bounds = ObjC.deepUnwrap(boundsObject);
    const width = Number(bounds.Width || 0);
    const height = Number(bounds.Height || 0);
    if (width < 240 || height < 180) continue;
    const layer = Number(value(windowInfo, "kCGWindowLayer", 0));
    for (const screen of screens) {
      const widthRatio = width / Math.max(1, screen.width);
      const heightRatio = height / Math.max(1, screen.height);
      const ratio = Math.min(widthRatio, heightRatio);
      if (ratio > largestRatio) {
        largestRatio = ratio;
        largestLayer = layer;
      }
      // 标准全屏、无边框全屏和游戏独占窗口都会覆盖至少
      // 97% 的某个显示器。普通“最大化”会留出菜单栏或 Dock。
      if (widthRatio >= 0.97 && heightRatio >= 0.97) fullscreen = true;
    }
  }
}

JSON.stringify({
  state_known: true,
  fullscreen_active: fullscreen,
  frontmost_pid: pid,
  frontmost_bundle: bundle,
  frontmost_app: name,
  frontmost_window_ratio: Math.round(largestRatio * 1000) / 1000,
  frontmost_window_layer: largestLayer,
});

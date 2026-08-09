import AppKit
import CoreGraphics
import ScreenCaptureKit
import ServiceManagement
import WebKit

final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, NSPopoverDelegate, WKNavigationDelegate, WKDownloadDelegate, WKScriptMessageHandler {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var quickWebView: WKWebView!
    private var statusItem: NSStatusItem!
    private let quickPopover = NSPopover()
    private var singleClickWorkItem: DispatchWorkItem?
    private var loadingView: NSVisualEffectView!
    private var statusLabel: NSTextField!
    private var spinner: NSProgressIndicator!
    private var healthAttempts = 0
    private var pageRecoveryAttempts = 0
    private var screenWatchTimer: Timer?
    private var screenWatchBusy = false
    private let serviceURL = URL(string: "http://127.0.0.1:11435/app/")!
    private let quickURL = URL(string: "http://127.0.0.1:11435/app/quick.html")!
    private let healthURL = URL(string: "http://127.0.0.1:11435/health")!
    private let openMainAtLaunchKey = "openMainAtLaunch"
    private var serviceReady = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        configureMenu()
        buildWindow()
        buildQuickPopover()
        buildStatusItem()
        startLocalServices()
        if UserDefaults.standard.bool(forKey: openMainAtLaunchKey) {
            openMainWindow(nil)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        sender.orderOut(nil)
        return false
    }

    private func webConfiguration() -> WKWebViewConfiguration {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        configuration.preferences.setValue(true, forKey: "developerExtrasEnabled")
        configuration.userContentController.add(self, name: "nativeBridge")
        return configuration
    }

    private func buildWindow() {
        webView = WKWebView(frame: .zero, configuration: webConfiguration())
        webView.navigationDelegate = self
        webView.setValue(false, forKey: "drawsBackground")
        webView.autoresizingMask = [.width, .height]

        let style: NSWindow.StyleMask = [
            .titled, .closable, .miniaturizable, .resizable
        ]
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 820),
            styleMask: style,
            backing: .buffered,
            defer: false
        )
        window.title = "星语茶话屋"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        // Keep a real AppKit title bar above the WKWebView.  Extending the web
        // content into the title bar makes WebKit compete with AppKit for mouse
        // events, which can break both window dragging and the traffic lights.
        window.isMovableByWindowBackground = false
        window.backgroundColor = NSColor(calibratedRed: 0.031, green: 0.039, blue: 0.059, alpha: 1)
        window.minSize = NSSize(width: 980, height: 640)
        window.center()
        window.contentView = webView
        window.isReleasedWhenClosed = false
        window.delegate = self

        loadingView = NSVisualEffectView(frame: webView.bounds)
        loadingView.autoresizingMask = [.width, .height]
        loadingView.material = .underWindowBackground
        loadingView.blendingMode = .behindWindow
        loadingView.state = .active

        spinner = NSProgressIndicator()
        spinner.style = .spinning
        spinner.controlSize = .regular
        spinner.translatesAutoresizingMaskIntoConstraints = false
        spinner.startAnimation(nil)

        let title = NSTextField(labelWithString: "✦ 星语茶话屋 ✦")
        title.font = NSFont.systemFont(ofSize: 13, weight: .bold)
        title.textColor = NSColor(calibratedRed: 0.67, green: 0.61, blue: 1, alpha: 1)
        title.alignment = .center

        statusLabel = NSTextField(labelWithString: "正在启动本地模型服务…")
        statusLabel.font = NSFont.systemFont(ofSize: 12, weight: .regular)
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.alignment = .center

        let stack = NSStackView(views: [spinner, title, statusLabel])
        stack.orientation = .vertical
        stack.alignment = .centerX
        stack.spacing = 12
        stack.translatesAutoresizingMaskIntoConstraints = false
        loadingView.addSubview(stack)
        webView.addSubview(loadingView)
        NSLayoutConstraint.activate([
            stack.centerXAnchor.constraint(equalTo: loadingView.centerXAnchor),
            stack.centerYAnchor.constraint(equalTo: loadingView.centerYAnchor),
            statusLabel.widthAnchor.constraint(lessThanOrEqualToConstant: 430)
        ])
    }

    private func buildQuickPopover() {
        quickWebView = WKWebView(frame: NSRect(x: 0, y: 0, width: 430, height: 610), configuration: webConfiguration())
        quickWebView.navigationDelegate = self
        quickWebView.setValue(false, forKey: "drawsBackground")
        let controller = NSViewController()
        controller.view = quickWebView
        controller.preferredContentSize = NSSize(width: 430, height: 610)
        quickPopover.contentViewController = controller
        quickPopover.contentSize = NSSize(width: 430, height: 610)
        quickPopover.behavior = .transient
        quickPopover.animates = true
        quickPopover.delegate = self
    }

    private func buildStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        guard let button = statusItem.button else { return }
        if let iconURL = Bundle.main.url(forResource: "MenuBarIcon", withExtension: "png"),
           let image = NSImage(contentsOf: iconURL) {
            image.size = NSSize(width: 22, height: 22)
            image.isTemplate = false
            button.image = image
            button.imageScaling = .scaleProportionallyDown
        } else {
            button.image = NSImage(systemSymbolName: "sparkles", accessibilityDescription: "星语茶话屋")
        }
        button.toolTip = "星语茶话屋 · 单击快速对话，双击进入小屋"
        button.setAccessibilityLabel("星语茶话屋快速对话")
        button.setAccessibilityHelp("单击打开快速对话，双击打开主面板，右键显示菜单")
        button.target = self
        button.action = #selector(statusItemClicked(_:))
        button.sendAction(on: [.leftMouseUp, .rightMouseUp])
    }

    @objc private func statusItemClicked(_ sender: Any?) {
        guard let event = NSApp.currentEvent else { return }
        if event.type == .rightMouseUp {
            singleClickWorkItem?.cancel()
            showStatusMenu()
            return
        }
        if event.clickCount >= 2 {
            singleClickWorkItem?.cancel()
            quickPopover.performClose(nil)
            openMainWindow(nil)
            return
        }
        singleClickWorkItem?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.toggleQuickPopover() }
        singleClickWorkItem = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.19, execute: work)
    }

    private func toggleQuickPopover() {
        guard let button = statusItem.button else { return }
        if quickPopover.isShown {
            quickPopover.performClose(nil)
        } else {
            if serviceReady && quickWebView.url == nil {
                quickWebView.load(URLRequest(url: quickURL))
            }
            quickPopover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
            NSApp.activate(ignoringOtherApps: true)
            quickWebView.evaluateJavaScript("window.dispatchEvent(new Event('focus'))")
        }
    }

    func popoverDidShow(_ notification: Notification) {
        // .floating 是给浮动调色板用的层级，会盖住部分中文输入法
        // 的浮动组字/候选窗口。Popover 本身已由 AppKit 管理显隐，
        // 这里必须使用普通层级，再让 Web 输入框成为真正焦点。
        DispatchQueue.main.async { [weak self] in
            guard let self,
                  let popoverWindow = self.quickPopover.contentViewController?.view.window
            else { return }
            popoverWindow.level = .normal
            popoverWindow.collectionBehavior.insert(.fullScreenAuxiliary)
            popoverWindow.makeKey()
            popoverWindow.makeFirstResponder(self.quickWebView)
            self.quickWebView.evaluateJavaScript(
                "document.getElementById('messageInput')?.focus({preventScroll:true})"
            )
        }
    }

    private func showStatusMenu() {
        let menu = NSMenu()
        let open = NSMenuItem(title: "打开主面板", action: #selector(openMainWindow(_:)), keyEquivalent: "")
        open.target = self
        menu.addItem(open)
        let quick = NSMenuItem(title: "快速对话", action: #selector(openQuickFromMenu(_:)), keyEquivalent: "")
        quick.target = self
        menu.addItem(quick)
        menu.addItem(.separator())
        let login = NSMenuItem(title: "登录时自动启动", action: #selector(toggleLaunchAtLoginFromMenu(_:)), keyEquivalent: "")
        login.target = self
        login.state = launchAtLoginEnabled ? .on : .off
        menu.addItem(login)
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "退出星语茶话屋", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        menu.addItem(quit)
        statusItem.menu = menu
        statusItem.button?.performClick(nil)
        statusItem.menu = nil
    }

    @objc private func openQuickFromMenu(_ sender: Any?) {
        if !quickPopover.isShown { toggleQuickPopover() }
    }

    @objc func openMainWindow(_ sender: Any?) {
        quickPopover.performClose(nil)
        if serviceReady && webView.url == nil {
            webView.load(URLRequest(url: serviceURL))
        }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func startLocalServices() {
        guard let resources = Bundle.main.resourceURL else {
            showStartupError("找不到应用资源。")
            return
        }
        let root = Bundle.main.bundleURL.deletingLastPathComponent()
        let bundledBackend = resources
            .appendingPathComponent("runtime")
            .appendingPathComponent("后端")
            .appendingPathComponent("memory_api_server.py")
        let bundledStaticDirectory = resources
            .appendingPathComponent("runtime")
            .appendingPathComponent("界面")
        // 放在项目根目录运行时优先使用外部源码。这样后端和界面升级不再
        // 改动应用签名，macOS 的录屏授权也不会因普通更新反复失效；
        // 把 app 单独搬走时仍可使用包内 runtime 独立运行。
        let externalApplicationDirectory = root.appendingPathComponent("应用")
        let externalBackend = externalApplicationDirectory
            .appendingPathComponent("后端")
            .appendingPathComponent("memory_api_server.py")
        let externalStaticDirectory = externalApplicationDirectory
            .appendingPathComponent("界面")
        let hasExternalRuntime = FileManager.default.fileExists(atPath: externalBackend.path)
            && FileManager.default.fileExists(
                atPath: externalStaticDirectory.appendingPathComponent("index.html").path
            )
        let backend = hasExternalRuntime ? externalBackend : bundledBackend
        let staticDirectory = hasExternalRuntime
            ? externalStaticDirectory
            : bundledStaticDirectory
        let dataDirectory = root.appendingPathComponent("数据", isDirectory: true)
        let uploadDirectory = dataDirectory.appendingPathComponent("上传", isDirectory: true)
        let logDirectory = dataDirectory.appendingPathComponent("日志", isDirectory: true)

        do {
            try FileManager.default.createDirectory(
                at: uploadDirectory,
                withIntermediateDirectories: true
            )
            try FileManager.default.createDirectory(
                at: logDirectory,
                withIntermediateDirectories: true
            )
        } catch {
            showStartupError("无法创建数据目录：\(error.localizedDescription)")
            return
        }

        let pythonCandidates = [
            URL(fileURLWithPath: "/opt/homebrew/bin/python3"),
            URL(fileURLWithPath: "/usr/local/bin/python3")
        ]
        guard let python = pythonCandidates.first(where: {
            FileManager.default.isExecutableFile(atPath: $0.path)
        }) else {
            showStartupError("找不到 Python 运行环境。请确认 /opt/homebrew/bin/python3 存在。")
            return
        }

        let process = Process()
        process.executableURL = python
        process.arguments = [
            backend.path,
            "--daemon",
            "--db", dataDirectory.appendingPathComponent("对话记忆.sqlite3").path,
            "--uploads", uploadDirectory.path,
            "--static", staticDirectory.path,
            "--log", logDirectory.appendingPathComponent("长期记忆API.log").path,
            "--ollama-log", logDirectory.appendingPathComponent("Ollama后台.log").path
        ]
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["LOCAL_AI_DATA_DIR"] = dataDirectory.path
        process.environment = environment
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice

        do {
            try process.run()
            pollHealth()
        } catch {
            showStartupError("服务启动失败：\(error.localizedDescription)")
        }
    }

    private func pollHealth() {
        healthAttempts += 1
        var request = URLRequest(url: healthURL)
        request.timeoutInterval = 1
        URLSession.shared.dataTask(with: request) { [weak self] data, response, _ in
            guard let self else { return }
            let online = (response as? HTTPURLResponse)?.statusCode == 200
                && data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }?["version"] as? Int ?? 0 >= 2
            DispatchQueue.main.async {
                if online {
                    self.serviceReady = true
                    self.statusLabel.stringValue = "服务已就绪"
                    self.webView.load(URLRequest(url: self.serviceURL))
                    self.quickWebView.load(URLRequest(url: self.quickURL))
                    self.statusItem.button?.toolTip = "星语茶话屋 · 艾莉和沙雅已就绪"
                    self.startScreenWatchPolling()
                } else if self.healthAttempts < 60 {
                    self.statusLabel.stringValue = self.healthAttempts < 15
                        ? "正在启动 Ollama 后台服务…"
                        : "正在载入本地 AI 工作台…"
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
                        self.pollHealth()
                    }
                } else {
                    self.showStartupError("本地服务启动超时。请查看 数据/日志。")
                }
            }
        }.resume()
    }

    private func showStartupError(_ message: String) {
        spinner.stopAnimation(nil)
        spinner.isHidden = true
        statusLabel.stringValue = message
        statusLabel.textColor = .systemRed
    }

    func applicationWillTerminate(_ notification: Notification) {
        screenWatchTimer?.invalidate()
        screenWatchTimer = nil
        webView?.configuration.userContentController.removeScriptMessageHandler(forName: "nativeBridge")
        quickWebView?.configuration.userContentController.removeScriptMessageHandler(forName: "nativeBridge")
    }

    private func startScreenWatchPolling() {
        guard screenWatchTimer == nil else { return }
        screenWatchTimer = Timer.scheduledTimer(
            withTimeInterval: 120,
            repeats: true
        ) { [weak self] _ in
            self?.pollScreenWatch()
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 6) { [weak self] in
            self?.pollScreenWatch()
        }
    }

    private func pollScreenWatch() {
        guard !screenWatchBusy,
              let url = URL(string: "http://127.0.0.1:11435/api/gui/screen-watch/claim")
        else { return }
        screenWatchBusy = true
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("{}".utf8)
        request.timeoutInterval = 6
        URLSession.shared.dataTask(with: request) { [weak self] data, response, _ in
            guard let self else { return }
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
            guard statusCode == 200,
                  let data,
                  let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  payload["due"] as? Bool == true,
                  let requestID = payload["request_id"] as? String
            else {
                DispatchQueue.main.async { self.screenWatchBusy = false }
                return
            }
            DispatchQueue.main.async {
                self.captureAndSubmitScreen(requestID: requestID)
            }
        }.resume()
    }

    private func captureAndSubmitScreen(requestID: String) {
        // 后台轮询绝不能反复触发系统授权弹窗。授权只由主人在系统设置中
        // 明确完成；这里仅做无副作用的状态检查。
        let permitted = CGPreflightScreenCaptureAccess()
        guard permitted else {
            let appName = Bundle.main.object(
                forInfoDictionaryKey: "CFBundleDisplayName"
            ) as? String ?? "本地 AI 客户端"
            submitScreenCaptureFailure(
                requestID: requestID,
                message: "尚未获得屏幕录制权限；后台不会重复弹窗。请在系统设置 → 隐私与安全性 → 屏幕与系统音频录制中允许 \(appName)"
            )
            return
        }
        guard #available(macOS 14.0, *) else {
            submitScreenCaptureFailure(
                requestID: requestID,
                message: "屏幕观察需要 macOS 14 或更高版本；瞬时截图未保存"
            )
            return
        }
        Task { [weak self] in
            guard let self else { return }
            do {
                let content = try await SCShareableContent.excludingDesktopWindows(
                    false,
                    onScreenWindowsOnly: true
                )
                guard !content.displays.isEmpty else {
                    throw NSError(
                        domain: "LocalAIStudio.ScreenWatch",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "找不到可观察的显示器"]
                    )
                }
                // 主显示器排在第一位，其余显示器按 displayID 稳定排序。逐屏使用
                // ScreenCaptureKit 截图不会激活窗口、切换 Space 或退出全屏。
                let displays = content.displays.sorted { left, right in
                    if left.displayID == CGMainDisplayID() { return true }
                    if right.displayID == CGMainDisplayID() { return false }
                    return left.displayID < right.displayID
                }
                var screens: [[String: Any]] = []
                for display in displays.prefix(8) {
                    let filter = SCContentFilter(display: display, excludingWindows: [])
                    let configuration = SCStreamConfiguration()
                    let scale = min(
                        1.0,
                        1600.0 / Double(max(display.width, display.height))
                    )
                    configuration.width = max(
                        1,
                        Int((Double(display.width) * scale).rounded())
                    )
                    configuration.height = max(
                        1,
                        Int((Double(display.height) * scale).rounded())
                    )
                    configuration.showsCursor = true
                    configuration.captureResolution = .best
                    let image = try await SCScreenshotManager.captureImage(
                        contentFilter: filter,
                        configuration: configuration
                    )
                    guard let encoded = self.resizedJPEG(
                        from: image,
                        maxDimension: 1600
                    ) else {
                        throw NSError(
                            domain: "LocalAIStudio.ScreenWatch",
                            code: 2,
                            userInfo: [NSLocalizedDescriptionKey: "无法压缩瞬时屏幕画面"]
                        )
                    }
                    screens.append([
                        "display_id": Int(display.displayID),
                        "image_base64": encoded.data.base64EncodedString(),
                        "width": encoded.width,
                        "height": encoded.height
                    ])
                }
                self.submitScreenCapture(
                    requestID: requestID,
                    screens: screens
                )
            } catch {
                self.submitScreenCaptureFailure(
                    requestID: requestID,
                    message: "屏幕读取失败：\(error.localizedDescription)；瞬时截图未保存"
                )
            }
        }
    }

    private func resizedJPEG(
        from image: CGImage,
        maxDimension: CGFloat
    ) -> (data: Data, width: Int, height: Int)? {
        let sourceWidth = CGFloat(image.width)
        let sourceHeight = CGFloat(image.height)
        let scale = min(1, maxDimension / max(sourceWidth, sourceHeight))
        let width = max(1, Int((sourceWidth * scale).rounded()))
        let height = max(1, Int((sourceHeight * scale).rounded()))
        let source = NSImage(
            cgImage: image,
            size: NSSize(width: sourceWidth, height: sourceHeight)
        )
        let target = NSImage(size: NSSize(width: width, height: height))
        target.lockFocus()
        NSGraphicsContext.current?.imageInterpolation = .high
        source.draw(
            in: NSRect(x: 0, y: 0, width: width, height: height),
            from: .zero,
            operation: .copy,
            fraction: 1
        )
        target.unlockFocus()
        guard let tiff = target.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiff),
              let jpeg = bitmap.representation(
                using: .jpeg,
                properties: [.compressionFactor: 0.62]
              )
        else { return nil }
        return (jpeg, width, height)
    }

    private func submitScreenCapture(
        requestID: String,
        screens: [[String: Any]]
    ) {
        let payload: [String: Any] = [
            "request_id": requestID,
            "screens": screens
        ]
        sendScreenPayload(payload)
    }

    private func submitScreenCaptureFailure(requestID: String, message: String) {
        sendScreenPayload(["request_id": requestID, "error": message])
    }

    private func sendScreenPayload(_ payload: [String: Any]) {
        guard let url = URL(string: "http://127.0.0.1:11435/api/gui/screen-watch/submit"),
              let body = try? JSONSerialization.data(withJSONObject: payload)
        else {
            screenWatchBusy = false
            return
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        request.timeoutInterval = 20
        URLSession.shared.dataTask(with: request) { [weak self] _, _, _ in
            DispatchQueue.main.async { self?.screenWatchBusy = false }
        }.resume()
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        guard webView.url?.path.hasPrefix("/app") == true else { return }
        webView.evaluateJavaScript("Boolean(document.querySelector('.app-shell'))") {
            [weak self] result, _ in
            guard let self else { return }
            if (result as? Bool) == true {
                self.pageRecoveryAttempts = 0
                NSAnimationContext.runAnimationGroup { context in
                    context.duration = 0.24
                    self.loadingView.animator().alphaValue = 0
                } completionHandler: {
                    self.loadingView.removeFromSuperview()
                }
            } else if self.pageRecoveryAttempts < 3 {
                self.pageRecoveryAttempts += 1
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.45) {
                    var request = URLRequest(url: self.serviceURL)
                    request.cachePolicy = .reloadIgnoringLocalCacheData
                    self.webView.load(request)
                }
            } else {
                self.showStartupError("界面载入失败，正在等待手动刷新。")
            }
        }
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
        guard let url = navigationAction.request.url else {
            decisionHandler(.cancel)
            return
        }
        if let host = url.host, host != "127.0.0.1" && host != "localhost" {
            NSWorkspace.shared.open(url)
            decisionHandler(.cancel)
            return
        }
        decisionHandler(.allow)
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationResponse: WKNavigationResponse,
        decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void
    ) {
        if let response = navigationResponse.response as? HTTPURLResponse,
           response.value(forHTTPHeaderField: "Content-Disposition")?.contains("attachment") == true {
            decisionHandler(.download)
        } else {
            decisionHandler(.allow)
        }
    }

    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload) {
        download.delegate = self
    }

    func download(
        _ download: WKDownload,
        decideDestinationUsing response: URLResponse,
        suggestedFilename: String,
        completionHandler: @escaping (URL?) -> Void
    ) {
        let downloads = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)[0]
        var destination = downloads.appendingPathComponent(suggestedFilename)
        if FileManager.default.fileExists(atPath: destination.path) {
            let base = destination.deletingPathExtension().lastPathComponent
            let ext = destination.pathExtension
            destination = downloads.appendingPathComponent("\(base)-\(Int(Date().timeIntervalSince1970)).\(ext)")
        }
        completionHandler(destination)
    }

    private var launchAtLoginEnabled: Bool {
        if #available(macOS 13.0, *) {
            return SMAppService.mainApp.status == .enabled
        }
        return false
    }

    private func nativePreferencePayload(message: String? = nil, error: String? = nil) -> [String: Any] {
        var payload: [String: Any] = [
            "launchAtLogin": launchAtLoginEnabled,
            "openMainAtLaunch": UserDefaults.standard.bool(forKey: openMainAtLaunchKey),
            "menuBarMode": true
        ]
        if let message { payload["message"] = message }
        if let error { payload["error"] = error }
        return payload
    }

    private func sendNativeEvent(_ name: String, payload: [String: Any], to target: WKWebView? = nil) {
        guard JSONSerialization.isValidJSONObject(payload),
              let data = try? JSONSerialization.data(withJSONObject: payload),
              let json = String(data: data, encoding: .utf8)
        else { return }
        let script = "window.dispatchEvent(new CustomEvent('\(name)', {detail: \(json)}));"
        if let target {
            target.evaluateJavaScript(script)
        } else {
            webView.evaluateJavaScript(script)
            quickWebView.evaluateJavaScript(script)
        }
    }

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.name == "nativeBridge",
              let body = message.body as? [String: Any],
              let action = body["action"] as? String
        else { return }
        let source = message.webView
        switch action {
        case "getPreferences":
            sendNativeEvent("local-ai-native-preferences", payload: nativePreferencePayload(), to: source)
        case "setOpenMainAtLaunch":
            let enabled = body["enabled"] as? Bool ?? false
            UserDefaults.standard.set(enabled, forKey: openMainAtLaunchKey)
            sendNativeEvent(
                "local-ai-native-result",
                payload: nativePreferencePayload(message: enabled ? "已设置启动时打开主面板" : "启动后将只驻留菜单栏"),
                to: source
            )
        case "setLaunchAtLogin":
            setLaunchAtLogin(body["enabled"] as? Bool ?? false, target: source)
        case "openScreenPrivacy":
            if let url = URL(
                string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
            ) {
                NSWorkspace.shared.open(url)
                sendNativeEvent(
                    "local-ai-native-result",
                    payload: nativePreferencePayload(
                        message: "已打开屏幕录制设置；本应用后台不会主动反复弹窗"
                    ),
                    to: source
                )
            }
        case "openMainWindow":
            openMainWindow(nil)
        case "closeQuickPanel":
            quickPopover.performClose(nil)
        default:
            sendNativeEvent(
                "local-ai-native-result",
                payload: nativePreferencePayload(error: "未知的原生操作"),
                to: source
            )
        }
    }

    private func setLaunchAtLogin(_ enabled: Bool, target: WKWebView? = nil) {
        guard #available(macOS 13.0, *) else {
            sendNativeEvent(
                "local-ai-native-result",
                payload: nativePreferencePayload(error: "登录自启动需要 macOS 13 或更高版本"),
                to: target
            )
            return
        }
        if enabled {
            do {
                try SMAppService.mainApp.register()
                let needsApproval = SMAppService.mainApp.status == .requiresApproval
                sendNativeEvent(
                    "local-ai-native-result",
                    payload: nativePreferencePayload(message: needsApproval ? "请在系统设置的登录项中允许星语茶话屋" : "登录自启动已开启"),
                    to: target
                )
            } catch {
                sendNativeEvent(
                    "local-ai-native-result",
                    payload: nativePreferencePayload(error: "无法开启登录自启动：\(error.localizedDescription)"),
                    to: target
                )
            }
        } else {
            Task { [weak self, weak target] in
                guard let self else { return }
                do {
                    try await SMAppService.mainApp.unregister()
                    await MainActor.run {
                        self.sendNativeEvent(
                            "local-ai-native-result",
                            payload: self.nativePreferencePayload(message: "登录自启动已关闭"),
                            to: target
                        )
                    }
                } catch {
                    await MainActor.run {
                        self.sendNativeEvent(
                            "local-ai-native-result",
                            payload: self.nativePreferencePayload(error: "无法关闭登录自启动：\(error.localizedDescription)"),
                            to: target
                        )
                    }
                }
            }
        }
    }

    @objc private func toggleLaunchAtLoginFromMenu(_ sender: Any?) {
        setLaunchAtLogin(!launchAtLoginEnabled)
    }

    private func configureMenu() {
        let mainMenu = NSMenu()
        let appItem = NSMenuItem()
        mainMenu.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "关于星语茶话屋", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "退出星语茶话屋", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu

        let editItem = NSMenuItem()
        mainMenu.addItem(editItem)
        let editMenu = NSMenu(title: "编辑")
        editMenu.addItem(withTitle: "撤销", action: Selector(("undo:")), keyEquivalent: "z")
        editMenu.addItem(withTitle: "重做", action: Selector(("redo:")), keyEquivalent: "Z")
        editMenu.addItem(.separator())
        editMenu.addItem(withTitle: "剪切", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        editMenu.addItem(withTitle: "复制", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "粘贴", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(withTitle: "全选", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = editMenu

        NSApp.mainMenu = mainMenu
    }
}

let application = NSApplication.shared
let delegate = AppDelegate()
application.delegate = delegate
application.setActivationPolicy(.accessory)
application.run()

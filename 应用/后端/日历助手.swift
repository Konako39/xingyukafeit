// 日历助手：基于 EventKit 的命令行工具，供后端 Agent 以 JSON 方式增删查 Mac 日历。
// AppleScript 控制“日历”应用极易超时，EventKit 直接读写系统日历数据库，稳定得多。
//
// 编译（工具/构建日历助手.sh 会自动执行）：
//   swiftc -O -swift-version 5 日历助手.swift -o 日历助手 \
//     -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker 日历助手-Info.plist
//
// 用法（全部输出 JSON，一行一个结果对象）：
//   日历助手 calendars
//   日历助手 events --from 2026-08-01T00:00:00 --to 2026-08-08T00:00:00 [--calendar 名称] [--query 关键词]
//   日历助手 add --title 标题 --start 2026-08-03T14:00:00 [--end ISO] [--calendar 名称]
//              [--notes 备注] [--location 地点] [--all-day] [--alarm 分钟]
//   日历助手 delete --id 事件标识 [--future]   （--future 表示删除重复事件的这次及以后）
//   日历助手 update --id 事件标识 [--title ...] [--start ...] [--end ...] [--notes ...] [--location ...]

import AppKit
import EventKit
import Foundation

let store = EKEventStore()

// 双击 .app（无参数）时进入授权向导：依次申请日历与提醒事项权限并弹窗告知结果。
// 裸命令行进程在后台被 Python 调用时 macOS 不会弹授权窗，必须由这里一次性完成。
func runAuthorizationWizard() -> Never {
    var lines: [String] = []
    for (entity, label) in [(EKEntityType.event, "日历"), (EKEntityType.reminder, "提醒事项")] {
        let semaphore = DispatchSemaphore(value: 0)
        var granted = false
        var failure = ""
        let handler: (Bool, Error?) -> Void = { ok, error in
            granted = ok
            failure = error?.localizedDescription ?? ""
            semaphore.signal()
        }
        if #available(macOS 14.0, *) {
            if entity == .reminder {
                store.requestFullAccessToReminders(completion: handler)
            } else {
                store.requestFullAccessToEvents(completion: handler)
            }
        } else {
            store.requestAccess(to: entity, completion: handler)
        }
        _ = semaphore.wait(timeout: .now() + 120)
        if granted {
            lines.append("✅ \(label)：已允许")
        } else {
            let suffix = failure.isEmpty ? "" : "（\(failure)）"
            lines.append("❌ \(label)：未允许\(suffix)")
        }
    }
    let allGranted = !lines.contains { $0.hasPrefix("❌") }
    let alert = NSAlert()
    alert.messageText = allGranted ? "星语茶话屋已获得授权" : "还有权限没有打开"
    alert.informativeText = lines.joined(separator: "\n")
        + (allGranted
            ? "\n\n现在回到星语茶话屋，直接说「明天下午三点提醒我开会」就能用了。"
            : "\n\n请到 系统设置 › 隐私与安全性 › 日历 / 提醒事项，"
              + "打开「星语茶话屋日历助手」的开关后再双击一次本程序。")
    alert.alertStyle = allGranted ? .informational : .warning
    alert.addButton(withTitle: "好")
    if !allGranted {
        alert.addButton(withTitle: "打开系统设置")
    }
    NSApplication.shared.setActivationPolicy(.regular)
    NSApplication.shared.activate(ignoringOtherApps: true)
    let response = alert.runModal()
    if !allGranted, response == .alertSecondButtonReturn {
        let target = "x-apple.systempreferences:com.apple.preference.security?Privacy_Calendars"
        if let url = URL(string: target) {
            NSWorkspace.shared.open(url)
        }
    }
    exit(allGranted ? 0 : 1)
}

// 通过 open 以独立 app 身份运行时拿不到 stdout，结果必须写进 --out 指定的文件。
var resultPath = ""

func emitResult(_ text: String) {
    print(text)
    if !resultPath.isEmpty {
        try? text.write(toFile: resultPath, atomically: true, encoding: .utf8)
    }
}

func fail(_ message: String) -> Never {
    let payload: [String: Any] = ["ok": false, "error": message]
    if let data = try? JSONSerialization.data(withJSONObject: payload),
       let text = String(data: data, encoding: .utf8) {
        emitResult(text)
    } else {
        emitResult("{\"ok\":false,\"error\":\"未知错误\"}")
    }
    exit(1)
}

func output(_ payload: [String: Any]) {
    var object = payload
    object["ok"] = true
    guard let data = try? JSONSerialization.data(withJSONObject: object),
          let text = String(data: data, encoding: .utf8) else {
        fail("结果无法序列化为 JSON")
    }
    emitResult(text)
}

func requestAccess(_ entity: EKEntityType) {
    let semaphore = DispatchSemaphore(value: 0)
    var granted = false
    var failure = ""
    let handler: (Bool, Error?) -> Void = { ok, error in
        granted = ok
        failure = error?.localizedDescription ?? ""
        semaphore.signal()
    }
    let label = entity == .reminder ? "提醒事项" : "日历"
    if #available(macOS 14.0, *) {
        if entity == .reminder {
            store.requestFullAccessToReminders(completion: handler)
        } else {
            store.requestFullAccessToEvents(completion: handler)
        }
    } else {
        store.requestAccess(to: entity, completion: handler)
    }
    if semaphore.wait(timeout: .now() + 60) == .timedOut {
        fail("等待\(label)权限超时；请在 系统设置 > 隐私与安全性 > \(label) 中允许访问")
    }
    if !granted {
        let suffix = failure.isEmpty ? "" : "（\(failure)）"
        fail("\(label)访问未授权\(suffix)；请在 系统设置 > 隐私与安全性 > \(label) 中允许")
    }
}

let isoWithZone = ISO8601DateFormatter()
let localFormats = [
    "yyyy-MM-dd'T'HH:mm:ss", "yyyy-MM-dd'T'HH:mm", "yyyy-MM-dd HH:mm:ss",
    "yyyy-MM-dd HH:mm", "yyyy-MM-dd",
]

func parseDate(_ text: String) -> Date? {
    if let date = isoWithZone.date(from: text) { return date }
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.timeZone = TimeZone.current
    for format in localFormats {
        formatter.dateFormat = format
        if let date = formatter.date(from: text) { return date }
    }
    return nil
}

func formatDate(_ date: Date) -> String {
    let formatter = DateFormatter()
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.timeZone = TimeZone.current
    formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ssZZZZZ"
    return formatter.string(from: date)
}

func eventJSON(_ event: EKEvent) -> [String: Any] {
    return [
        "id": event.eventIdentifier ?? "",
        "title": event.title ?? "",
        "start": event.startDate.map(formatDate) ?? "",
        "end": event.endDate.map(formatDate) ?? "",
        "all_day": event.isAllDay,
        "calendar": event.calendar?.title ?? "",
        "location": event.location ?? "",
        "notes": event.notes ?? "",
        "recurring": event.hasRecurrenceRules,
    ]
}

// 简易参数解析：--key value 与 --flag。
var positional: [String] = []
var options: [String: String] = [:]
var flags: Set<String> = []
let knownFlags: Set<String> = ["all-day", "future"]
var index = 1
let arguments = CommandLine.arguments
while index < arguments.count {
    let argument = arguments[index]
    if argument.hasPrefix("--") {
        let key = String(argument.dropFirst(2))
        if knownFlags.contains(key) {
            flags.insert(key)
        } else if index + 1 < arguments.count {
            options[key] = arguments[index + 1]
            index += 1
        } else {
            fail("参数 --\(key) 缺少值")
        }
    } else {
        positional.append(argument)
    }
    index += 1
}

resultPath = options["out"] ?? ""

// 无参数（双击 .app 启动）或显式 authorize：进入授权向导。
if positional.isEmpty || positional.first == "authorize" {
    runAuthorizationWizard()
}

guard let command = positional.first else {
    fail("缺少子命令：calendars / events / add / delete / update / "
        + "reminders / reminder-add / reminder-done / reminder-delete")
}

requestAccess(command.hasPrefix("reminder") ? .reminder : .event)

func writableCalendar(named name: String?) -> EKCalendar {
    let calendars = store.calendars(for: .event)
    if let name = name, !name.isEmpty {
        if let match = calendars.first(where: { $0.title == name && $0.allowsContentModifications }) {
            return match
        }
        fail("找不到可写日历「\(name)」")
    }
    if let preferred = store.defaultCalendarForNewEvents, preferred.allowsContentModifications {
        return preferred
    }
    if let fallback = calendars.first(where: { $0.allowsContentModifications }) {
        return fallback
    }
    fail("没有任何可写日历")
}

switch command {
case "calendars":
    let items = store.calendars(for: .event).map { calendar -> [String: Any] in
        [
            "title": calendar.title,
            "writable": calendar.allowsContentModifications,
            "type": String(describing: calendar.type),
            "default": calendar.calendarIdentifier
                == store.defaultCalendarForNewEvents?.calendarIdentifier,
        ]
    }
    output(["calendars": items])

case "events":
    guard let fromText = options["from"], let from = parseDate(fromText) else {
        fail("events 需要 --from（如 2026-08-01T00:00:00）")
    }
    guard let toText = options["to"], let to = parseDate(toText), to > from else {
        fail("events 需要晚于 --from 的 --to")
    }
    var calendars: [EKCalendar]? = nil
    if let name = options["calendar"], !name.isEmpty {
        let matched = store.calendars(for: .event).filter { $0.title == name }
        if matched.isEmpty { fail("找不到日历「\(name)」") }
        calendars = matched
    }
    let predicate = store.predicateForEvents(withStart: from, end: to, calendars: calendars)
    var events = store.events(matching: predicate)
    if let query = options["query"], !query.isEmpty {
        let needle = query.lowercased()
        events = events.filter { event in
            (event.title ?? "").lowercased().contains(needle)
                || (event.location ?? "").lowercased().contains(needle)
                || (event.notes ?? "").lowercased().contains(needle)
        }
    }
    events.sort { ($0.startDate ?? .distantPast) < ($1.startDate ?? .distantPast) }
    output(["events": events.prefix(200).map(eventJSON), "count": events.count])

case "add":
    guard let title = options["title"], !title.isEmpty else { fail("add 需要 --title") }
    guard let startText = options["start"], let start = parseDate(startText) else {
        fail("add 需要 --start（如 2026-08-03T14:00:00）")
    }
    let event = EKEvent(eventStore: store)
    event.title = title
    event.calendar = writableCalendar(named: options["calendar"])
    event.isAllDay = flags.contains("all-day")
    event.startDate = start
    if let endText = options["end"], let end = parseDate(endText) {
        if end <= start { fail("--end 必须晚于 --start") }
        event.endDate = end
    } else {
        event.endDate = event.isAllDay
            ? Calendar.current.date(byAdding: .day, value: 1, to: start)!
            : start.addingTimeInterval(3600)
    }
    if let notes = options["notes"], !notes.isEmpty { event.notes = notes }
    if let location = options["location"], !location.isEmpty { event.location = location }
    if let alarmText = options["alarm"], let minutes = Double(alarmText), minutes >= 0 {
        event.addAlarm(EKAlarm(relativeOffset: -minutes * 60))
    }
    do {
        try store.save(event, span: .thisEvent, commit: true)
    } catch {
        fail("保存日程失败：\(error.localizedDescription)")
    }
    output(["event": eventJSON(event)])

case "delete":
    guard let identifier = options["id"], !identifier.isEmpty else { fail("delete 需要 --id") }
    guard let event = store.event(withIdentifier: identifier) else {
        fail("找不到该日程（可能已被删除）")
    }
    let removed = eventJSON(event)
    let span: EKSpan = flags.contains("future") ? .futureEvents : .thisEvent
    do {
        try store.remove(event, span: span, commit: true)
    } catch {
        fail("删除日程失败：\(error.localizedDescription)")
    }
    output(["deleted": removed])

case "update":
    guard let identifier = options["id"], !identifier.isEmpty else { fail("update 需要 --id") }
    guard let event = store.event(withIdentifier: identifier) else {
        fail("找不到该日程（可能已被删除）")
    }
    if let title = options["title"], !title.isEmpty { event.title = title }
    if let startText = options["start"] {
        guard let start = parseDate(startText) else { fail("--start 无法解析") }
        event.startDate = start
    }
    if let endText = options["end"] {
        guard let end = parseDate(endText) else { fail("--end 无法解析") }
        event.endDate = end
    }
    if let endDate = event.endDate, let startDate = event.startDate, endDate <= startDate {
        fail("结束时间必须晚于开始时间")
    }
    if let notes = options["notes"] { event.notes = notes }
    if let location = options["location"] { event.location = location }
    do {
        try store.save(event, span: .thisEvent, commit: true)
    } catch {
        fail("修改日程失败：\(error.localizedDescription)")
    }
    output(["event": eventJSON(event)])

case "reminders":
    let predicate: NSPredicate
    if options["from"] != nil || options["to"] != nil {
        guard let fromText = options["from"], let from = parseDate(fromText),
              let toText = options["to"], let to = parseDate(toText), to > from else {
            fail("reminders 的 --from/--to 需要成对且有效")
        }
        predicate = store.predicateForIncompleteReminders(
            withDueDateStarting: from, ending: to,
            calendars: store.calendars(for: .reminder))
    } else {
        predicate = store.predicateForIncompleteReminders(
            withDueDateStarting: nil, ending: nil,
            calendars: store.calendars(for: .reminder))
    }
    let semaphore = DispatchSemaphore(value: 0)
    var fetched: [EKReminder] = []
    store.fetchReminders(matching: predicate) { reminders in
        fetched = reminders ?? []
        semaphore.signal()
    }
    if semaphore.wait(timeout: .now() + 30) == .timedOut {
        fail("读取提醒事项超时")
    }
    if let query = options["query"], !query.isEmpty {
        let needle = query.lowercased()
        fetched = fetched.filter {
            ($0.title ?? "").lowercased().contains(needle)
                || ($0.notes ?? "").lowercased().contains(needle)
        }
    }
    fetched.sort {
        let left = $0.dueDateComponents?.date ?? .distantFuture
        let right = $1.dueDateComponents?.date ?? .distantFuture
        return left < right
    }
    let items = fetched.prefix(60).map { reminder -> [String: Any] in
        [
            "id": reminder.calendarItemIdentifier,
            "title": reminder.title ?? "",
            "due": reminder.dueDateComponents?.date.map(formatDate) ?? "",
            "notes": reminder.notes ?? "",
            "list": reminder.calendar?.title ?? "",
            "priority": reminder.priority,
        ]
    }
    output(["reminders": items, "count": fetched.count])

case "reminder-add":
    guard let title = options["title"], !title.isEmpty else {
        fail("reminder-add 需要 --title")
    }
    let reminder = EKReminder(eventStore: store)
    reminder.title = title
    if let name = options["list"], !name.isEmpty {
        guard let calendar = store.calendars(for: .reminder)
            .first(where: { $0.title == name && $0.allowsContentModifications }) else {
            fail("找不到可写的提醒列表「\(name)」")
        }
        reminder.calendar = calendar
    } else if let preferred = store.defaultCalendarForNewReminders() {
        reminder.calendar = preferred
    } else if let fallback = store.calendars(for: .reminder)
        .first(where: { $0.allowsContentModifications }) {
        reminder.calendar = fallback
    } else {
        fail("没有任何可写的提醒列表")
    }
    if let dueText = options["due"], let due = parseDate(dueText) {
        reminder.dueDateComponents = Calendar.current.dateComponents(
            [.year, .month, .day, .hour, .minute], from: due)
        reminder.addAlarm(EKAlarm(absoluteDate: due))
    }
    if let notes = options["notes"], !notes.isEmpty { reminder.notes = notes }
    do {
        try store.save(reminder, commit: true)
    } catch {
        fail("保存提醒失败：\(error.localizedDescription)")
    }
    output([
        "reminder": [
            "id": reminder.calendarItemIdentifier,
            "title": reminder.title ?? "",
            "due": reminder.dueDateComponents?.date.map(formatDate) ?? "",
            "list": reminder.calendar?.title ?? "",
        ],
    ])

case "reminder-done", "reminder-delete":
    guard let identifier = options["id"], !identifier.isEmpty else {
        fail("\(command) 需要 --id")
    }
    guard let reminder = store.calendarItem(withIdentifier: identifier) as? EKReminder else {
        fail("找不到该提醒（可能已被删除）")
    }
    let snapshot: [String: Any] = [
        "id": reminder.calendarItemIdentifier,
        "title": reminder.title ?? "",
        "due": reminder.dueDateComponents?.date.map(formatDate) ?? "",
        "list": reminder.calendar?.title ?? "",
    ]
    do {
        if command == "reminder-done" {
            reminder.isCompleted = true
            try store.save(reminder, commit: true)
        } else {
            try store.remove(reminder, commit: true)
        }
    } catch {
        fail("操作提醒失败：\(error.localizedDescription)")
    }
    output([command == "reminder-done" ? "completed" : "deleted": snapshot])

default:
    fail("未知子命令：\(command)")
}

// ScarlettMac HUD window: an always-on-top, terminal-style panel that shows
// tools/scarlett-hud.py --stream, a new frame every 0.1 s. Built and opened by
// tools/scarlett-hud.sh. Drag it anywhere; it stays above the game window and
// over full-screen spaces. Quit with Cmd-Q or the close button.
import AppKit

let script = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "tools/scarlett-hud.py"

final class HUD: NSObject, NSApplicationDelegate {
    var panel: NSPanel!
    var text: NSTextView!

    func applicationDidFinishLaunching(_ note: Notification) {
        let frame = NSRect(x: 40, y: 80, width: 560, height: 480)
        panel = NSPanel(contentRect: frame,
                        styleMask: [.titled, .closable, .resizable, .utilityWindow, .nonactivatingPanel],
                        backing: .buffered, defer: false)
        panel.title = "ScarlettMac HUD"
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.hidesOnDeactivate = false
        panel.isMovableByWindowBackground = true
        panel.backgroundColor = NSColor(calibratedWhite: 0.04, alpha: 0.98)
        panel.isOpaque = false

        let scroll = NSScrollView(frame: panel.contentView!.bounds)
        scroll.autoresizingMask = [.width, .height]
        scroll.drawsBackground = false
        text = NSTextView(frame: scroll.bounds)
        text.isEditable = false
        text.drawsBackground = true; text.backgroundColor = NSColor(calibratedWhite: 0.04, alpha: 1)
        text.font = NSFont.monospacedSystemFont(ofSize: 11.5, weight: .regular)
        text.textColor = NSColor(calibratedRed: 0.55, green: 1.0, blue: 0.6, alpha: 1)
        text.textContainerInset = NSSize(width: 6, height: 6)
        scroll.documentView = text
        panel.contentView!.addSubview(scroll)
        panel.orderFrontRegardless()

        start()
    }

    // One long-lived `scarlett-hud.py --stream`: it writes a frame every 0.1 s,
    // each ended by a form feed; the newest complete frame is shown.
    var buffer = Data()
    func start() {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        p.arguments = ["-I", script, "--stream"]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = FileHandle.nullDevice
        pipe.fileHandleForReading.readabilityHandler = { h in
            let chunk = h.availableData
            if chunk.isEmpty { return }
            self.buffer.append(chunk)
            guard let last = self.buffer.lastIndex(of: 0x0c) else { return }
            let head = self.buffer[..<last]
            let prev = head.lastIndex(of: 0x0c).map { head.index(after: $0) } ?? head.startIndex
            let frame = String(decoding: head[prev...], as: UTF8.self)
            self.buffer = Data(self.buffer[self.buffer.index(after: last)...])
            DispatchQueue.main.async { self.text.string = frame }
        }
        p.terminationHandler = { _ in
            // Restart if the script ever dies (e.g. edited while running).
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) { self.start() }
        }
        do { try p.run() } catch { text.string = "could not run \(script): \(error)" }
        child = p
    }
    var child: Process?

    func applicationWillTerminate(_ note: Notification) { child?.terminationHandler = nil; child?.terminate() }

    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let delegate = HUD()
app.delegate = delegate
app.run()

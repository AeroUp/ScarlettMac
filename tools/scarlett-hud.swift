// ScarlettMac HUD window: an always-on-top, terminal-style panel that shows
// tools/scarlett-hud.py --once, refreshed every second. Built and opened by
// tools/scarlett-hud.sh. Drag it anywhere; it stays above the game window and
// over full-screen spaces. Quit with Cmd-Q or the close button.
import AppKit

let script = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "tools/scarlett-hud.py"

final class HUD: NSObject, NSApplicationDelegate {
    var panel: NSPanel!
    var text: NSTextView!
    var busy = false

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

        refresh()
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in self.refresh() }
    }

    func refresh() {
        if busy { return }
        busy = true
        DispatchQueue.global().async {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
            p.arguments = ["-I", script, "--once"]
            let pipe = Pipe()
            p.standardOutput = pipe
            p.standardError = pipe
            var out = ""
            do {
                try p.run()
                out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
                p.waitUntilExit()
            } catch {
                out = "could not run \(script): \(error)"
            }
            DispatchQueue.main.async {
                self.text.string = out
                self.busy = false
            }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let delegate = HUD()
app.delegate = delegate
app.run()

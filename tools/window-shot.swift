// List a window, or save one as a PNG, without bringing it forward.
//
//   window-shot list                 pid, window number, size, title
//   window-shot shot NUMBER PATH     one window, full resolution
//
// Synara's window ids carry the CGWindowID after the last colon
// (cua:PID:NUMBER), and screencapture -l takes that same number, but only a
// CoreGraphics query works from a script while a run is in progress.
import AppKit
import CoreGraphics
import Foundation

func windows() -> [[String: Any]] {
    let options: CGWindowListOption = [.optionAll, .excludeDesktopElements]
    guard let list = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] else {
        return []
    }
    return list
}

func list() {
    for w in windows() {
        let pid = w[kCGWindowOwnerPID as String] as? Int ?? -1
        let num = w[kCGWindowNumber as String] as? Int ?? -1
        let name = w[kCGWindowName as String] as? String ?? ""
        let owner = w[kCGWindowOwnerName as String] as? String ?? ""
        let b = w[kCGWindowBounds as String] as? [String: Any] ?? [:]
        let width = b["Width"] as? Double ?? 0
        let height = b["Height"] as? Double ?? 0
        // A title window is large; skip the 1x1 and 64x64 helpers.
        if width < 200 || height < 200 { continue }
        print("\(pid)\t\(num)\t\(Int(width))x\(Int(height))\t\(owner)\t\(name)")
    }
}

// CGWindowListCreateImage is obsoleted at compile time on macOS 15+ but the
// symbol is still there; looking it up at run time keeps the tool working
// without pulling in ScreenCaptureKit and its own permission flow.
typealias CreateImage = @convention(c) (CGRect, UInt32, UInt32, UInt32) -> Unmanaged<CGImage>?

func capture(_ number: Int) -> CGImage? {
    guard let handle = dlopen(
        "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics", RTLD_LAZY),
        let sym = dlsym(handle, "CGWindowListCreateImage")
    else { return nil }
    let create = unsafeBitCast(sym, to: CreateImage.self)
    let including: UInt32 = 0x8 // kCGWindowListOptionIncludingWindow
    let options: UInt32 = 0x1 | 0x8 // boundsIgnoreFraming | bestResolution
    return create(.null, including, UInt32(number), options)?.takeRetainedValue()
}

func shot(_ number: Int, _ path: String) -> Int32 {
    guard let image = capture(number) else {
        FileHandle.standardError.write("no image for window \(number)\n".data(using: .utf8)!)
        return 1
    }
    let rep = NSBitmapImageRep(cgImage: image)
    guard let data = rep.representation(using: .png, properties: [:]) else {
        FileHandle.standardError.write("no PNG for window \(number)\n".data(using: .utf8)!)
        return 1
    }
    do {
        try data.write(to: URL(fileURLWithPath: path))
    } catch {
        FileHandle.standardError.write("cannot write \(path): \(error)\n".data(using: .utf8)!)
        return 1
    }
    print("\(path)\t\(rep.pixelsWide)x\(rep.pixelsHigh)")
    return 0
}

let args = Array(CommandLine.arguments.dropFirst())
switch args.first {
case "list":
    list()
    exit(0)
case "shot":
    guard args.count == 3, let number = Int(args[1]) else {
        FileHandle.standardError.write("usage: window-shot shot NUMBER PATH\n".data(using: .utf8)!)
        exit(2)
    }
    exit(shot(number, args[2]))
default:
    FileHandle.standardError.write("usage: window-shot list | shot NUMBER PATH\n".data(using: .utf8)!)
    exit(2)
}

import AVFoundation
import SwiftUI

/// The measurement screen: live camera, model running on every frame it can keep up with,
/// and the numbers that matter — latency, FPS, thermal state.
struct RunView: View {
    @State private var model = RunModel()

    var body: some View {
        ZStack(alignment: .top) {
            CameraPreview(session: model.camera.session)
                .ignoresSafeArea()

            stats
                .padding()
                .background(.ultraThinMaterial, in: .rect(cornerRadius: 16))
                .padding()
        }
        .overlay(alignment: .bottom) {
            Button(model.isRunning ? "Stop" : "Start") {
                Task { await model.toggle() }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .padding(.bottom, 40)
        }
        .task { await model.prepare() }
    }

    private var stats: some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
            row("Model", model.modelName)
            row("Input", model.inputDescription)
            row("Latency", model.latencyText)
            row("FPS", model.fpsText)
            row("Dropped", "\(model.droppedFrames)")
            row("Thermal", model.thermalText)
            if let error = model.errorText {
                GridRow {
                    Text("Error").foregroundStyle(.secondary)
                    Text(error).foregroundStyle(.red)
                }
            }
        }
        .font(.system(.footnote, design: .monospaced))
    }

    private func row(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            Text(value)
        }
    }
}

// MARK: - State

@MainActor
@Observable
final class RunModel {
    let camera = CameraSource()

    private(set) var isRunning = false
    private(set) var modelName = "—"
    private(set) var inputDescription = "—"
    private(set) var latencyMS: Double = 0
    private(set) var fps: Double = 0
    private(set) var droppedFrames = 0
    private(set) var errorText: String?

    private let segmenter = Segmenter()
    private let busy = BusyFlag()
    private var lastFrameTime = CFAbsoluteTimeGetCurrent()

    var latencyText: String { latencyMS > 0 ? String(format: "%.1f ms", latencyMS) : "—" }
    var fpsText: String { fps > 0 ? String(format: "%.1f", fps) : "—" }

    var thermalText: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    func prepare() async {
        do {
            try await segmenter.load()
            modelName = segmenter.modelName
            inputDescription = describe(segmenter.inputKind)
        } catch {
            errorText = error.localizedDescription
        }

        camera.onFrame = { [weak self] buffer in
            self?.handle(buffer)
        }
    }

    func toggle() async {
        if isRunning {
            camera.stop()
            isRunning = false
        } else {
            do {
                try await camera.start()
                isRunning = true
            } catch {
                errorText = error.localizedDescription
            }
        }
    }

    /// Called on the camera queue. Skips the frame if a prediction is already in flight —
    /// the drop count is the honest signal that the model can't keep up.
    private nonisolated func handle(_ buffer: CVPixelBuffer) {
        guard busy.claim() else {
            Task { @MainActor in self.droppedFrames += 1 }
            return
        }
        defer { busy.release() }

        do {
            let elapsed = try segmenter.run(on: buffer)
            Task { @MainActor in self.record(latency: elapsed) }
        } catch {
            Task { @MainActor in self.errorText = error.localizedDescription }
        }
    }

    private func record(latency: Double) {
        let now = CFAbsoluteTimeGetCurrent()
        let interval = now - lastFrameTime
        lastFrameTime = now

        // Plain exponential smoothing; good enough to read off the screen.
        latencyMS = latencyMS == 0 ? latency : latencyMS * 0.9 + latency * 0.1
        if interval > 0 {
            let instant = 1 / interval
            fps = fps == 0 ? instant : fps * 0.9 + instant * 0.1
        }
    }

    private func describe(_ kind: Segmenter.InputKind) -> String {
        switch kind {
        case let .image(name): "\(name) (image)"
        case let .multiArray(name, shape): "\(name) \(shape.map(String.init).joined(separator: "×"))"
        case .none: "—"
        }
    }
}

/// Single-slot gate so only one prediction runs at a time.
private final class BusyFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var isBusy = false

    func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if isBusy { return false }
        isBusy = true
        return true
    }

    func release() {
        lock.lock()
        isBusy = false
        lock.unlock()
    }
}

// MARK: - Preview layer

#if canImport(UIKit)
import UIKit

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}
#else
/// The measurement target is iOS; on Mac the numbers still read, there's just no preview.
struct CameraPreview: View {
    let session: AVCaptureSession
    var body: some View { Color.black }
}
#endif

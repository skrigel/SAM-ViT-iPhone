import AVFoundation
import CoreML
import SwiftUI

/// The measurement screen: live camera with the mask overlaid, drag a box (or tap) to prompt,
/// and the numbers that matter — encode/decode latency, FPS, drops, IoU, thermal state.
struct RunView: View {
    @State private var model = RunModel()
    @State private var dragStart: CGPoint?
    @State private var dragNow: CGPoint?

    var body: some View {
        GeometryReader { geo in
            let fit = model.displayedFrameRect(in: geo.size)
            ZStack {
                CameraPreview(session: model.camera.session)

                if let mask = model.mask, let fit {
                    Image(decorative: mask, scale: 1)
                        .resizable()
                        .interpolation(.none)
                        .frame(width: fit.width, height: fit.height)
                        .position(x: fit.midX, y: fit.midY)
                        .allowsHitTesting(false)
                }

                if let a = dragStart, let b = dragNow {
                    Path { $0.addRect(Self.rect(a, b)) }
                        .stroke(.yellow, lineWidth: 2)
                        .allowsHitTesting(false)
                }
            }
            .contentShape(Rectangle())
            .gesture(promptGesture(fit: fit))
        }
        .ignoresSafeArea()
        .overlay(alignment: .top) {
            stats
                .padding()
                .background(.ultraThinMaterial, in: .rect(cornerRadius: 16))
                .padding()
        }
        .overlay(alignment: .bottom) {
            HStack(spacing: 16) {
                Button(model.isRunning ? "Stop" : "Start") {
                    Task { await model.toggle() }
                }
                .buttonStyle(.borderedProminent)

                if model.hasPrompt {
                    Button("Clear") { model.setPrompt(nil) }
                        .buttonStyle(.bordered)
                }
            }
            .controlSize(.large)
            .padding(.bottom, 40)
        }
        .task { await model.prepare() }
    }

    // MARK: Prompting

    /// Drag = box. Tap = default-size box around the tap. View points -> frame pixels.
    private func promptGesture(fit: CGRect?) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                dragStart = value.startLocation
                dragNow = value.location
            }
            .onEnded { value in
                defer { dragStart = nil; dragNow = nil }
                guard model.isRunning, let fit, let frame = model.frameSize else { return }

                var rect = Self.rect(value.startLocation, value.location)
                if rect.width < 12 || rect.height < 12 {
                    rect = CGRect(x: value.location.x - 60, y: value.location.y - 60, width: 120, height: 120)
                }
                let s = frame.width / fit.width
                let inFrame = CGRect(x: (rect.minX - fit.minX) * s, y: (rect.minY - fit.minY) * s,
                                     width: rect.width * s, height: rect.height * s)
                    .intersection(CGRect(origin: .zero, size: frame))
                model.setPrompt(inFrame.isNull || inFrame.isEmpty ? nil : inFrame)
            }
    }

    private static func rect(_ a: CGPoint, _ b: CGPoint) -> CGRect {
        CGRect(x: min(a.x, b.x), y: min(a.y, b.y), width: abs(a.x - b.x), height: abs(a.y - b.y))
    }

    // MARK: Stats

    private var stats: some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
            row("Model", model.modelName)
            row("Encode", model.encodeText)
            row("Decode", model.decodeText)
            row("FPS", model.fpsText)
            row("Dropped", "\(model.droppedFrames)")
            row("IoU (pred)", model.iouText)
            row("Thermal", model.thermalText)
            if model.isRunning && !model.hasPrompt {
                GridRow {
                    Text("Hint").foregroundStyle(.secondary)
                    Text("Drag a box around an object")
                }
            }
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
    private(set) var hasPrompt = false
    private(set) var modelName = "—"
    private(set) var mask: CGImage?
    private(set) var frameSize: CGSize?
    private(set) var encodeMS: Double = 0
    private(set) var decodeMS: Double = 0
    private(set) var iou: Float = 0
    private(set) var fps: Double = 0
    private(set) var droppedFrames = 0
    private(set) var thermal = ProcessInfo.processInfo.thermalState
    private(set) var errorText: String?

    private var worker: FrameWorker?
    private var didPrepare = false
    private var lastFrameTime: CFTimeInterval = 0

    var encodeText: String { encodeMS > 0 ? String(format: "%.1f ms", encodeMS) : "—" }
    var decodeText: String { decodeMS > 0 ? String(format: "%.1f ms", decodeMS) : "—" }
    var fpsText: String { fps > 0 ? String(format: "%.1f", fps) : "—" }
    var iouText: String { hasPrompt && iou > 0 ? String(format: "%.2f", iou) : "—" }

    var thermalText: String {
        switch thermal {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    /// .task re-runs when the tab reappears; only set up once.
    func prepare() async {
        guard !didPrepare else { return }
        didPrepare = true
        observeThermalState()

        do {
            let segmenter = Segmenter()
            try await segmenter.load()   // pass encoderURL: to try a variant from Documents
            // Encoder on the Neural Engine (accurate in fp16 only with the ANE-safe model) and the fixed-shape decoder
            // on the GPU (the bundled decoder is stuck on the CPU). See research/PROGRESS.md:
            // try await segmenter.load(encoderUnits: .cpuAndNeuralEngine, decoderUnits: .cpuAndGPU,
            //                          encoderURL: Segmenter.aneEncoderURL, decoderURL: Segmenter.fixedDecoderURL)
            modelName = segmenter.encoderName

            let worker = FrameWorker(segmenter: segmenter)
            worker.publish = { @Sendable [weak self] result in
                // Bind before the Task: a `weak self` capture is a var, and referencing a
                // captured var from concurrently-executing code is an error in Swift 6.
                guard let self else { return }
                Task { @MainActor in self.apply(result) }
            }
            worker.attach(to: camera)
            self.worker = worker
        } catch {
            errorText = error.localizedDescription
        }
    }

    func toggle() async {
        if isRunning {
            camera.stop()
            isRunning = false
        } else {
            guard worker != nil else { errorText = errorText ?? "Models not loaded."; return }
            do {
                fps = 0
                lastFrameTime = 0
                try await camera.start()
                isRunning = true
            } catch {
                errorText = error.localizedDescription
            }
        }
    }

    func setPrompt(_ rect: CGRect?) {
        worker?.setPrompt(rect)
        hasPrompt = rect != nil
        if rect == nil { mask = nil; iou = 0 }
    }

    /// Where the frame lands on screen under the preview layer's .resizeAspectFill.
    func displayedFrameRect(in container: CGSize) -> CGRect? {
        guard let frameSize, frameSize.width > 0, frameSize.height > 0 else { return nil }
        let scale = max(container.width / frameSize.width, container.height / frameSize.height)
        let width = frameSize.width * scale
        let height = frameSize.height * scale
        return CGRect(x: (container.width - width) / 2, y: (container.height - height) / 2,
                      width: width, height: height)
    }

    private func apply(_ result: FrameResult) {
        if lastFrameTime > 0, result.timestamp > lastFrameTime {
            let instant = 1 / (result.timestamp - lastFrameTime)
            fps = fps == 0 ? instant : fps * 0.9 + instant * 0.1
        }
        lastFrameTime = result.timestamp

        frameSize = result.frameSize
        droppedFrames = result.dropped
        if let error = result.error { errorText = error }

        guard hasPrompt else { return }   // ignore results from before a Clear
        mask = result.mask
        if result.encoded {
            // Plain exponential smoothing; good enough to read off the screen.
            encodeMS = encodeMS == 0 ? result.encodeMS : encodeMS * 0.9 + result.encodeMS * 0.1
            decodeMS = decodeMS == 0 ? result.decodeMS : decodeMS * 0.9 + result.decodeMS * 0.1
            iou = result.iou
        }
    }

    /// @Observable can't see ProcessInfo, so mirror the thermal state into a property.
    private func observeThermalState() {
        Task { [weak self] in
            for await _ in NotificationCenter.default.notifications(named: ProcessInfo.thermalStateDidChangeNotification) {
                self?.thermal = ProcessInfo.processInfo.thermalState
            }
        }
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

import AVFoundation
import CoreVideo

/// Camera feed. Hands each frame to `onFrame` on a background queue. AVFoundation drops
/// frames that arrive while one is still being processed; `onDrop` reports each one.
final class CameraSource: NSObject, @unchecked Sendable {

    let session = AVCaptureSession()

    /// Set both before `start()`. Called on the frame queue, never the main thread.
    var onFrame: ((CVPixelBuffer) -> Void)?
    var onDrop: (() -> Void)?

    private let sessionQueue = DispatchQueue(label: "camera.session")
    private let frameQueue = DispatchQueue(label: "camera.frames", qos: .userInitiated)
    private let output = AVCaptureVideoDataOutput()
    private var device: AVCaptureDevice?

    func start() async throws {
        guard await AVCaptureDevice.requestAccess(for: .video) else {
            throw CameraError.accessDenied
        }
        // Configure + startRunning both block; do them on the session queue.
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            sessionQueue.async {
                do {
                    try self.configureIfNeeded()
                    if !self.session.isRunning { self.session.startRunning() }
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    /// Frame-rate knob for the adaptive policy. Ignored if the active format can't do `fps`.
    func setFrameRate(_ fps: Int) {
        sessionQueue.async {
            guard let device = self.device else { return }
            let supported = device.activeFormat.videoSupportedFrameRateRanges.contains {
                Double(fps) >= $0.minFrameRate && Double(fps) <= $0.maxFrameRate
            }
            guard supported, (try? device.lockForConfiguration()) != nil else { return }
            let duration = CMTime(value: 1, timescale: CMTimeScale(fps))
            device.activeVideoMinFrameDuration = duration
            device.activeVideoMaxFrameDuration = duration
            device.unlockForConfiguration()
        }
    }

    // Runs on sessionQueue.
    private func configureIfNeeded() throws {
        guard session.inputs.isEmpty else { return }

        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .hd1280x720

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            throw CameraError.noCamera
        }
        session.addInput(input)

        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        // System-level backpressure: never queue a backlog.
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: frameQueue)

        guard session.canAddOutput(output) else { throw CameraError.noCamera }
        session.addOutput(output)

        // Deliver portrait buffers (720×1280) so frame coords match the portrait preview.
        if let connection = output.connection(with: .video),
           connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
        self.device = device
    }

    enum CameraError: LocalizedError {
        case accessDenied
        case noCamera

        var errorDescription: String? {
            switch self {
            case .accessDenied: "Camera access denied. Enable it in Settings."
            case .noCamera: "No usable camera on this device."
            }
        }
    }
}

extension CameraSource: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onFrame?(buffer)
    }

    /// The honest drop signal: called for every frame discarded while we were busy.
    func captureOutput(
        _ output: AVCaptureOutput,
        didDrop sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        onDrop?()
    }
}

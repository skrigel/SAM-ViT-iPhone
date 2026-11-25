import CoreGraphics
import CoreVideo
import Foundation
import QuartzCore

/// One processed frame, handed to the UI.
struct FrameResult: @unchecked Sendable {
    let frameSize: CGSize
    let mask: CGImage?
    let encoded: Bool
    let encodeMS: Double
    let decodeMS: Double
    let iou: Float
    let dropped: Int
    let timestamp: CFTimeInterval
    let error: String?
}

/// Runs on the camera's frame queue, one frame at a time.
/// Owns the prompt and does app-level tracking: each frame's mask box prompts the next frame.
/// This is where StridePolicy (encode or reuse the embedding) plugs in.
final class FrameWorker: @unchecked Sendable {

    var publish: (@Sendable (FrameResult) -> Void)?

    /// Below this predicted IoU, keep the old box instead of following the mask (likely lost).
    var minIoUToTrack: Float = 0.5
    /// Grow each tracked box a little so a moving object stays inside the next prompt.
    /// Keep this small: when the mask just fills the prompt box, the margin compounds every frame.
    var boxMargin: CGFloat = 0.02
    /// Reject an update whose area changes by more than this factor in one frame (runaway / lost).
    var maxAreaChange: CGFloat = 1.25
    /// Reject an update that covers more than this fraction of the frame.
    var maxFrameCoverage: CGFloat = 0.6

    private let segmenter: Segmenter
    private let lock = NSLock()
    private var prompt: CGRect?     // frame pixels; guarded by lock
    private var dropped = 0         // guarded by lock

    init(segmenter: Segmenter) {
        self.segmenter = segmenter
    }

    /// Wired here (a nonisolated type) so the closures aren't inferred as main-actor isolated.
    func attach(to camera: CameraSource) {
        camera.onFrame = { [self] buffer in process(buffer) }
        camera.onDrop = { [self] in lock.withLock { dropped += 1 } }
    }

    func setPrompt(_ rect: CGRect?) {
        lock.withLock { prompt = rect }
    }

    private func process(_ buffer: CVPixelBuffer) {
        let size = CGSize(width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer))
        let (current, drops) = lock.withLock { (prompt, dropped) }

        guard let current else {
            publish?(FrameResult(frameSize: size, mask: nil, encoded: false, encodeMS: 0, decodeMS: 0,
                                 iou: 0, dropped: drops, timestamp: CACurrentMediaTime(), error: nil))
            return
        }

        do {
            let encodeMS = try segmenter.encode(buffer)        // TODO(D): gate with StridePolicy.shouldEncode
            let decoded = try segmenter.decode(box: current)

            let frameArea = size.width * size.height
            let oldArea = max(current.width * current.height, 1)
            let newArea = decoded.box.width * decoded.box.height
            let plausible = decoded.iou >= minIoUToTrack
                && !decoded.box.isEmpty
                && newArea <= oldArea * maxAreaChange
                && newArea >= oldArea / maxAreaChange
                && newArea <= frameArea * maxFrameCoverage

            if plausible {
                let next = decoded.box
                    .insetBy(dx: -boxMargin * decoded.box.width, dy: -boxMargin * decoded.box.height)
                    .intersection(CGRect(origin: .zero, size: size))
                // Don't overwrite a prompt the user set while this frame was in flight.
                lock.withLock { if prompt == current { prompt = next } }
            }

            publish?(FrameResult(frameSize: size, mask: decoded.mask, encoded: true, encodeMS: encodeMS,
                                 decodeMS: decoded.decodeMS, iou: decoded.iou, dropped: drops,
                                 timestamp: CACurrentMediaTime(), error: nil))
        } catch {
            publish?(FrameResult(frameSize: size, mask: nil, encoded: false, encodeMS: 0, decodeMS: 0,
                                 iou: 0, dropped: drops, timestamp: CACurrentMediaTime(),
                                 error: error.localizedDescription))
        }
    }
}

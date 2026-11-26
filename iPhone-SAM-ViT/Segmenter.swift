import CoreImage
import CoreML
import CoreVideo
import QuartzCore

/// EfficientViT-SAM: letterbox -> encoder (expensive, cache the embedding) -> prompt decoder (cheap).
///
/// Model contract (from the conversion, see the old SamModels.swift notes):
///   SAMImageEncoder   in:  `image` 512×512, letterboxed: long side 512, top-left, pad RGB (124,116,104).
///                          Normalization is baked into the model.
///                     out: `image_embeddings` 1×256×64×64
///   SAMPromptDecoder  in:  `image_embeddings`, `point_coords` 1×2×2 (in the 1024 prompt frame),
///                          `point_labels` 1×2
///                     out: `masks` 1×4×256×256 logits, `iou_predictions` 1×4
final class Segmenter: @unchecked Sendable {

    struct Decoded {
        /// Translucent mask covering exactly the frame; stretch it over the displayed frame.
        let mask: CGImage?
        /// Bounding box of the mask in frame pixels (`.zero` if the mask is empty).
        let box: CGRect
        let iou: Float
        let decodeMS: Double
    }

    static let inputSide = 512
    static let promptSide: CGFloat = 1024

    private(set) var encoderName = "—"
    private var encoder: SAMImageEncoder?
    private var decoder: SAMPromptDecoder?

    // Color management off: pixel values must reach the model unchanged, like in Python.
    private let ci = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
    private var inputBuffer: CVPixelBuffer?   // reused 512×512 BGRA buffer

    // Per-frame geometry (set by encode, used by decode).
    private var embedding: MLMultiArray?
    private var frameSize: CGSize = .zero
    private var promptScale: CGFloat = 1      // 1024-prompt-frame px per frame px

    // MARK: - Loading

    /// The Neural Engine-safe encoder (Models/SAMImageEncoderANE.mlpackage, built by research/tools/convert.py).
    /// Same interface as SAMImageEncoder. The stock fp16 encoder is wrong on the ANE (50 vs 84 mIoU on COCO): the ANE's
    /// built-in GELU is too coarse for the high-gain layers in stage 3, and the ANE miscompiles the neck's resize.
    /// This one computes GELU from exp/divide (exact identity) in stages 3-4 and the resize as linear ops:
    /// 83.46 mIoU on the ANE vs 83.56 fp32 (stock model on the ANE: 4.5). See research/PROGRESS.md.
    /// Use it with `encoderUnits: .cpuAndNeuralEngine`.
    static var aneEncoderURL: URL? {
        Bundle.main.url(forResource: "SAMImageEncoderANE", withExtension: "mlmodelc")
    }

    /// Fixed-shape decoder (Models/SAMPromptDecoderFixed.mlpackage, research/tools/convert_decoder_app.py), same interface
    /// as SAMPromptDecoder. The bundled decoder's flexible point shapes keep all of its ops on the CPU whatever
    /// `computeUnits` says (~43 ms per decode on an M5 Pro). This one runs on the GPU in ~1.6 ms with the same masks
    /// (COCO box mIoU 85.90 vs 85.92). Use it with `decoderUnits: .cpuAndGPU`.
    static var fixedDecoderURL: URL? {
        Bundle.main.url(forResource: "SAMPromptDecoderFixed", withExtension: "mlmodelc")
    }

    /// Each model gets its own compute units so placement is an explicit experiment knob.
    /// `encoderURL`: optional variant (e.g. ReLU or int8 encoder imported into Documents)
    /// with the same interface; `nil` uses the bundled SAMImageEncoder. `decoderURL` likewise for the decoder.
    func load(
        encoderUnits: MLComputeUnits = .cpuAndGPU,
        decoderUnits: MLComputeUnits = .all,
        encoderURL: URL? = nil,
        decoderURL: URL? = nil
    ) async throws {
        let encoderConfig = MLModelConfiguration()
        encoderConfig.computeUnits = encoderUnits
        let decoderConfig = MLModelConfiguration()
        decoderConfig.computeUnits = decoderUnits

        if let encoderURL {
            let compiled = encoderURL.pathExtension == "mlmodelc"
                ? encoderURL
                : try await MLModel.compileModel(at: encoderURL)
            encoder = try SAMImageEncoder(contentsOf: compiled, configuration: encoderConfig)
            encoderName = encoderURL.deletingPathExtension().lastPathComponent
        } else {
            encoder = try await SAMImageEncoder.load(configuration: encoderConfig)
            encoderName = "SAMImageEncoder"
        }
        if let decoderURL {
            let compiled = decoderURL.pathExtension == "mlmodelc"
                ? decoderURL
                : try await MLModel.compileModel(at: decoderURL)
            decoder = try SAMPromptDecoder(contentsOf: compiled, configuration: decoderConfig)
        } else {
            decoder = try await SAMPromptDecoder.load(configuration: decoderConfig)
        }

        var buffer: CVPixelBuffer?
        let attributes = [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()] as CFDictionary
        CVPixelBufferCreate(nil, Self.inputSide, Self.inputSide, kCVPixelFormatType_32BGRA, attributes, &buffer)
        guard let buffer else { throw SegmenterError.bufferAllocation }
        inputBuffer = buffer
    }

    // MARK: - Encode (expensive; encoder stride skips this)

    /// Letterboxes the frame, runs the encoder, caches the embedding. Returns ms (letterbox included).
    @discardableResult
    func encode(_ frame: CVPixelBuffer) throws -> Double {
        guard let encoder, let inputBuffer else { throw SegmenterError.notLoaded }
        let start = CACurrentMediaTime()

        let width = CGFloat(CVPixelBufferGetWidth(frame))
        let height = CGFloat(CVPixelBufferGetHeight(frame))
        let side = CGFloat(Self.inputSide)
        let scale = side / max(width, height)
        let scaledHeight = (height * scale).rounded()

        frameSize = CGSize(width: width, height: height)
        promptScale = Self.promptSide / max(width, height)

        let canvas = CGRect(x: 0, y: 0, width: side, height: side)
        let pad = CIImage(color: CIColor(red: 124 / 255, green: 116 / 255, blue: 104 / 255)).cropped(to: canvas)
        // Core Image's origin is bottom-left, so shift up to land the image at the top-left.
        let letterboxed = CIImage(cvPixelBuffer: frame)
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            .transformed(by: CGAffineTransform(translationX: 0, y: side - scaledHeight))
            .cropped(to: canvas)
            .composited(over: pad)
        ci.render(letterboxed, to: inputBuffer, bounds: canvas, colorSpace: nil)

        embedding = try encoder.prediction(input: SAMImageEncoderInput(image: inputBuffer)).image_embeddings
        return (CACurrentMediaTime() - start) * 1000
    }

    // MARK: - Decode (cheap; runs on the cached embedding)

    /// `box` in frame pixels. Sent as a SAM box prompt: two corners, labels 2 and 3.
    /// Verify once in Python that the converted decoder handles box labels like stock SAM.
    func decode(box: CGRect) throws -> Decoded {
        guard let decoder, let embedding else { throw SegmenterError.notEncoded }
        let start = CACurrentMediaTime()

        let coords = try MLMultiArray(shape: [1, 2, 2], dataType: .float32)
        let corners = [box.minX, box.minY, box.maxX, box.maxY].map { Float($0 * promptScale) }
        for (i, value) in corners.enumerated() { coords[i] = NSNumber(value: value) }
        let labels = try MLMultiArray(shape: [1, 2], dataType: .float32)
        labels[0] = 2
        labels[1] = 3

        let out = try decoder.prediction(image_embeddings: embedding, point_coords: coords, point_labels: labels)
        let masks = out.masks
        let ious = out.iou_predictions

        // Highest predicted IoU wins.
        var best = 0
        var bestIoU = -Float.infinity
        for k in 0..<masks.shape[1].intValue {
            let value = ious[[0, NSNumber(value: k)]].floatValue
            if value > bestIoU { bestIoU = value; best = k }
        }

        let (image, maskBox) = Self.render(masks: masks, index: best, frameSize: frameSize, promptScale: promptScale)
        return Decoded(mask: image, box: maskBox, iou: bestIoU,
                       decodeMS: (CACurrentMediaTime() - start) * 1000)
    }

    /// Crops the low-res logits to the unpadded region, thresholds at 0, returns a tinted image
    /// plus the mask's bounding box in frame pixels.
    private static func render(
        masks: MLMultiArray, index: Int, frameSize: CGSize, promptScale: CGFloat
    ) -> (CGImage?, CGRect) {
        let maskSide = masks.shape[3].intValue                     // 256, covers the padded 1024 frame
        let maskPerPrompt = CGFloat(maskSide) / promptSide
        let width = max(1, min(maskSide, Int((frameSize.width * promptScale * maskPerPrompt).rounded())))
        let height = max(1, min(maskSide, Int((frameSize.height * promptScale * maskPerPrompt).rounded())))

        // Pull the chosen mask's logits out once, whatever the output dtype.
        let strides = masks.strides.map(\.intValue)
        var logits = [Float](repeating: 0, count: width * height)
        func offset(_ y: Int, _ x: Int) -> Int { index * strides[1] + y * strides[2] + x * strides[3] }

        switch masks.dataType {
        case .float32:
            masks.withUnsafeBufferPointer(ofType: Float.self) { p in
                for y in 0..<height { for x in 0..<width { logits[y * width + x] = p[offset(y, x)] } }
            }
        case .float16:
            masks.withUnsafeBufferPointer(ofType: Float16.self) { p in
                for y in 0..<height { for x in 0..<width { logits[y * width + x] = Float(p[offset(y, x)]) } }
            }
        default:
            for y in 0..<height {
                for x in 0..<width {
                    logits[y * width + x] = masks[[0, NSNumber(value: index), NSNumber(value: y), NSNumber(value: x)]].floatValue
                }
            }
        }

        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        var minX = width, minY = height, maxX = -1, maxY = -1
        for y in 0..<height {
            for x in 0..<width where logits[y * width + x] > 0 {
                let i = (y * width + x) * 4
                rgba[i] = 0; rgba[i + 1] = 60; rgba[i + 2] = 120; rgba[i + 3] = 120   // premultiplied blue
                minX = min(minX, x); maxX = max(maxX, x)
                minY = min(minY, y); maxY = max(maxY, y)
            }
        }

        let framePerMask = 1 / (maskPerPrompt * promptScale)
        let box = maxX < 0 ? CGRect.zero : CGRect(
            x: CGFloat(minX) * framePerMask, y: CGFloat(minY) * framePerMask,
            width: CGFloat(maxX - minX + 1) * framePerMask, height: CGFloat(maxY - minY + 1) * framePerMask
        )

        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return (nil, box) }
        let image = CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )
        return (image, box)
    }
}

enum SegmenterError: LocalizedError {
    case notLoaded
    case notEncoded
    case bufferAllocation

    var errorDescription: String? {
        switch self {
        case .notLoaded: "Models not loaded."
        case .notEncoded: "Decode called before any frame was encoded."
        case .bufferAllocation: "Couldn't allocate the 512×512 input buffer."
        }
    }
}

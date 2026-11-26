// One encoder, three runtimes: Core ML, Core AI and MPSGraph (Metal), timed the same way.
//
// Build:  swiftc -O -parse-as-library bench/RuntimeBench.swift -o work/runtimebench
// Usage:  runtimebench <coreml|coreai|mpsgraph> <cpu|gpu|ane|all> <model path> [iterations=100] [warmup=10]
//   coreml    .mlpackage (tensor or image input); units map to MLComputeUnits
//   coreai    .aimodel; unit is the preferred ComputeUnitKind of the specialization
//   mpsgraph  .mlpackage loaded by MPSGraphExecutable(coreMLPackageAt:) and run on the GPU (unit ignored)
// Prints one JSON object: load time, first (cold) prediction, then p10/p50/p90/mean over `iterations` warm predictions.
// The same calls exist on iOS 27, so the file drops into an app target for iPhone numbers.

import CoreAI
import CoreML
import CoreVideo
import Foundation
import Metal
import MetalPerformanceShadersGraph

func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
func ms(_ ns: UInt64) -> Double { Double(ns) / 1e6 }

func stats(_ xs: [Double]) -> [String: Double] {
    let s = xs.sorted()
    func p(_ q: Double) -> Double { s[min(s.count - 1, Int(Double(s.count) * q))] }
    return ["p10": p(0.1), "p50": p(0.5), "p90": p(0.9), "mean": s.reduce(0, +) / Double(s.count), "min": s[0]]
}

func randomPixels(_ n: Int) -> [Float] {
    var g = SystemRandomNumberGenerator()
    return (0..<n).map { _ in Float(UInt8.random(in: 0...255, using: &g)) }
}

protocol Runner {
    func run() async throws
}

// MARK: Core ML

final class CoreMLRunner: Runner {
    let model: MLModel
    let input: MLFeatureProvider

    init(path: String, unit: String) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = ["cpu": .cpuOnly, "gpu": .cpuAndGPU, "ane": .cpuAndNeuralEngine][unit] ?? .all
        let compiled = try MLModel.compileModel(at: URL(fileURLWithPath: path))
        model = try MLModel(contentsOf: compiled, configuration: cfg)
        let (name, desc) = model.modelDescription.inputDescriptionsByName.first!
        if let c = desc.imageConstraint {
            var pb: CVPixelBuffer?
            CVPixelBufferCreate(nil, c.pixelsWide, c.pixelsHigh, kCVPixelFormatType_32BGRA,
                                [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pb)
            input = try MLDictionaryFeatureProvider(dictionary: [name: MLFeatureValue(pixelBuffer: pb!)])
        } else {
            let shape = desc.multiArrayConstraint!.shape
            let a = try MLMultiArray(shape: shape, dataType: .float32)
            let px = randomPixels(a.count)
            a.withUnsafeMutableBufferPointer(ofType: Float.self) { p, _ in for i in 0..<px.count { p[i] = px[i] } }
            input = try MLDictionaryFeatureProvider(dictionary: [name: MLFeatureValue(multiArray: a)])
        }
    }

    // Synchronous prediction (what the app's frame queue calls); a sync helper keeps Swift from picking the async overload.
    func predict() throws { _ = try model.prediction(from: input, options: MLPredictionOptions()) }
    func run() async throws { try predict() }
}

// MARK: Core AI

final class CoreAIRunner: Runner {
    let function: InferenceFunction
    let inputName: String
    let input: NDArray   // built once, like a reused camera-frame buffer

    init(path: String, unit: String) async throws {
        let kind: ComputeUnitKind = ["cpu": .cpu, "gpu": .gpu, "ane": .neuralEngine][unit] ?? .gpu
        let model = try await AIModel(contentsOf: URL(fileURLWithPath: path),
                                      options: SpecializationOptions(preferredComputeUnitKind: kind))
        function = try model.loadFunction(named: "main")!
        inputName = function.descriptor.inputNames[0]
        guard case let .ndArray(d) = function.descriptor.inputDescriptor(of: inputName)! else {
            fatalError("input is not an NDArray")
        }
        let px = randomPixels(d.shape.reduce(1, *))
        input = d.scalarType == .float16 ? NDArray(scalars: px.map { Float16($0) }, shape: d.shape)
                                         : NDArray(scalars: px, shape: d.shape)
    }

    func run() async throws {
        _ = try await function.run(inputs: [inputName: input])
    }
}

// MARK: MPSGraph (Metal)

final class MPSGraphRunner: Runner {
    let exe: MPSGraphExecutable
    let queue: MTLCommandQueue
    let inputs: [MPSGraphTensorData]

    init(path: String) throws {
        let device = MTLCreateSystemDefaultDevice()!
        queue = device.makeCommandQueue()!
        let desc = MPSGraphCompilationDescriptor()
        desc.optimizationLevel = .level1
        exe = MPSGraphExecutable(coreMLPackageAtURL: URL(fileURLWithPath: path), descriptor: desc)
        // Executables built from a Core ML package expose no feed tensors; the encoder takes one (1, 3, 512, 512) tensor.
        let mpsDevice = MPSGraphDevice(mtlDevice: device)
        let side = Int(ProcessInfo.processInfo.environment["SIDE"] ?? "512")!
        let shape: [NSNumber] = [1, 3, NSNumber(value: side), NSNumber(value: side)]
        // fp16 Core ML programs take an fp16 tensor after Core ML's input cast is stripped; DTYPE=f32 for fp32 models.
        let f32 = ProcessInfo.processInfo.environment["DTYPE"] == "f32"
        let px = randomPixels(3 * side * side)
        let data = f32 ? px.withUnsafeBufferPointer { Data(buffer: $0) }
                       : px.map { Float16($0) }.withUnsafeBufferPointer { Data(buffer: $0) }
        let dtype: MPSDataType = f32 ? .float32 : .float16
        let datas = [MPSGraphTensorData(device: mpsDevice, data: data, shape: shape, dataType: dtype)]
        let types = [MPSGraphShapedType(shape: shape, dataType: dtype)]
        exe.specialize(with: mpsDevice, inputTypes: types, compilationDescriptor: desc)
        inputs = datas
    }

    func run() async throws {
        _ = exe.run(with: queue, inputs: inputs, results: nil, executionDescriptor: nil)
    }
}

@main
struct Main {
    static func main() async throws {
        let a = CommandLine.arguments
        guard a.count >= 4 else {
            FileHandle.standardError.write("usage: runtimebench <coreml|coreai|mpsgraph> <cpu|gpu|ane|all> <model> [iters] [warmup]\n".data(using: .utf8)!)
            exit(2)
        }
        let (runtime, unit, path) = (a[1], a[2], a[3])
        let iters = a.count > 4 ? Int(a[4])! : 100
        let warm = a.count > 5 ? Int(a[5])! : 10

        let t0 = now()
        var loadAttempts = 1
        let r: Runner
        switch runtime {
        case "coreml": r = try CoreMLRunner(path: path, unit: unit)
        case "coreai":
            // Core AI's ANE specialization fails intermittently on macOS 27.2 (Foundation nilError); retry and report it.
            var attempt = 0
            while true {
                attempt += 1
                do { r = try await CoreAIRunner(path: path, unit: unit); break } catch {
                    let ns = error as NSError
                    FileHandle.standardError.write("Core AI load attempt \(attempt) failed: domain=\(ns.domain) code=\(ns.code)\n".data(using: .utf8)!)
                    if attempt >= 5 { throw error }
                    try await Task.sleep(nanoseconds: 1_000_000_000)
                }
            }
            loadAttempts = attempt
        default: r = try MPSGraphRunner(path: path)
        }
        let loadMs = ms(now() - t0)
        let t1 = now()
        try await r.run()
        let firstMs = ms(now() - t1)
        for _ in 0..<warm { try await r.run() }
        // DURATION=<s>: loop for that long instead of `iters` runs (for power sampling over a steady window).
        let duration = Double(ProcessInfo.processInfo.environment["DURATION"] ?? "0") ?? 0
        var xs: [Double] = []
        let start = now()
        while duration > 0 ? ms(now() - start) < duration * 1000 : xs.count < iters {
            let t = now()
            try await r.run()
            xs.append(ms(now() - t))
        }
        let wallS = ms(now() - start) / 1000
        let out: [String: Any] = ["runtime": runtime, "unit": unit, "model": (path as NSString).lastPathComponent,
                                  "load_ms": loadMs, "load_attempts": loadAttempts, "first_ms": firstMs, "iters": xs.count, "wall_s": wallS,
                                  "latency_ms": stats(xs)]
        let json = try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys])
        print(String(data: json, encoding: .utf8)!)
    }
}

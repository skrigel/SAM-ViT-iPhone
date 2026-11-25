import CoreML
import Foundation

/// M0 utility: reports where Core ML *plans* to execute each operation of a model,
/// and what shapes the model's interface accepts.
///
/// Answers the three day-1 gates in DESIGN.md §11:
///   1. whether `computeUnits` is a live knob at all (placement histogram),
///   2. whether the encoder accepts enumerated input shapes (resolution knob),
///   3. whether the decoder takes an externally supplied embedding (encodeStride knob).
///
/// Placement here is Core ML's *static plan*, not observed execution. Treat it as
/// the hypothesis that Instruments later confirms or refutes.
enum ComputePlanDump {

    /// The settings probed for every model. Order matters: it's the report's column order.
    static let probedComputeUnits: [MLComputeUnits] = [
        .all, .cpuAndNeuralEngine, .cpuAndGPU, .cpuOnly
    ]

    // MARK: - Running

    /// Compiles the model once, then builds one plan per compute-unit setting.
    static func inspect(modelAt url: URL) async throws -> ModelInspection {
        let compiledURL = try await compiledURL(for: url)
        let name = url.deletingPathExtension().lastPathComponent

        let interface = try await interface(ofCompiledModelAt: compiledURL)

        var plans: [PlacementReport] = []
        for units in probedComputeUnits {
            plans.append(try await placement(ofCompiledModelAt: compiledURL, computeUnits: units))
        }

        return ModelInspection(modelName: name, interface: interface, placements: plans)
    }

    /// `MLComputePlan` requires a compiled `.mlmodelc`; `.mlpackage` has to be compiled first.
    private static func compiledURL(for url: URL) async throws -> URL {
        if url.pathExtension == "mlmodelc" { return url }
        return try await MLModel.compileModel(at: url)
    }

    // MARK: - Placement

    private static func placement(
        ofCompiledModelAt url: URL,
        computeUnits: MLComputeUnits
    ) async throws -> PlacementReport {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits

        let plan = try await MLComputePlan.load(contentsOf: url, configuration: configuration)

        guard case let .program(program) = plan.modelStructure else {
            throw DumpError.notAnMLProgram
        }
        guard let main = program.functions["main"] else {
            throw DumpError.missingMainFunction
        }

        var operations: [OperationPlacement] = []
        collect(block: main.block, plan: plan, into: &operations)

        return PlacementReport(computeUnits: computeUnits, operations: operations)
    }

    /// Walks nested blocks so operations inside conditionals and loops are counted too.
    private static func collect(
        block: MLModelStructure.Program.Block,
        plan: MLComputePlan,
        into operations: inout [OperationPlacement]
    ) {
        for operation in block.operations {
            let usage = plan.deviceUsage(for: operation)
            let cost = plan.estimatedCost(of: operation)

            operations.append(
                OperationPlacement(
                    index: operations.count,
                    operatorName: operation.operatorName,
                    outputName: operation.outputs.first?.name ?? "",
                    preferred: usage.map { ComputeDeviceKind($0.preferred) } ?? .unknown,
                    supported: usage.map { $0.supported.map(ComputeDeviceKind.init) } ?? [],
                    costWeight: cost?.weight
                )
            )

            for nested in operation.blocks {
                collect(block: nested, plan: plan, into: &operations)
            }
        }
    }

    // MARK: - Interface

    private static func interface(ofCompiledModelAt url: URL) async throws -> ModelInterface {
        // Interface description is placement-independent, so CPU-only keeps the load cheap.
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly

        let model = try await MLModel.load(contentsOf: url, configuration: configuration)
        let description = model.modelDescription

        return ModelInterface(
            inputs: description.inputDescriptionsByName.values
                .map(FeatureSummary.init).sorted { $0.name < $1.name },
            outputs: description.outputDescriptionsByName.values
                .map(FeatureSummary.init).sorted { $0.name < $1.name }
        )
    }

    enum DumpError: LocalizedError {
        case notAnMLProgram
        case missingMainFunction

        var errorDescription: String? {
            switch self {
            case .notAnMLProgram:
                "Model is not an ML Program. Re-export with coremltools using convert_to=\"mlprogram\"."
            case .missingMainFunction:
                "ML Program has no 'main' function."
            }
        }
    }
}

// MARK: - Report types

enum ComputeDeviceKind: String, CaseIterable, Sendable {
    case cpu = "CPU"
    case gpu = "GPU"
    case neuralEngine = "ANE"
    case unknown = "???"

    init(_ device: MLComputeDevice) {
        switch device {
        case .cpu: self = .cpu
        case .gpu: self = .gpu
        case .neuralEngine: self = .neuralEngine
        @unknown default: self = .unknown
        }
    }
}

struct OperationPlacement: Sendable {
    let index: Int
    let operatorName: String
    let outputName: String
    let preferred: ComputeDeviceKind
    let supported: [ComputeDeviceKind]
    /// Core ML's estimated share of total model cost, if available.
    let costWeight: Double?
}

struct PlacementReport: Sendable {
    let computeUnits: MLComputeUnits
    let operations: [OperationPlacement]

    var operationCounts: [ComputeDeviceKind: Int] {
        operations.reduce(into: [:]) { $0[$1.preferred, default: 0] += 1 }
    }

    /// Cost-weighted share per device. More meaningful than op counts, since one
    /// conv can outweigh fifty reshapes.
    var costShare: [ComputeDeviceKind: Double] {
        let total = operations.compactMap(\.costWeight).reduce(0, +)
        guard total > 0 else { return [:] }
        return operations.reduce(into: [:]) { result, operation in
            result[operation.preferred, default: 0] += (operation.costWeight ?? 0) / total
        }
    }

    /// Operators not planned for `device`, by count — i.e. the fallback list.
    func fallbackOperators(awayFrom device: ComputeDeviceKind) -> [(name: String, count: Int)] {
        let counts = operations
            .filter { $0.preferred != device }
            .reduce(into: [String: Int]()) { $0[$1.operatorName, default: 0] += 1 }
        return counts.sorted { ($0.value, $1.key) > ($1.value, $0.key) }
            .map { (name: $0.key, count: $0.value) }
    }
}

struct FeatureSummary: Sendable {
    let name: String
    let type: String
    /// Human-readable shape constraint — this is what answers DESIGN.md §11 Q2.
    let shape: String

    init(_ description: MLFeatureDescription) {
        name = description.name

        switch description.type {
        case .multiArray:
            type = "MultiArray"
            shape = Self.describe(description.multiArrayConstraint)
        case .image:
            type = "Image"
            shape = Self.describe(description.imageConstraint)
        case .double: type = "Double"; shape = "scalar"
        case .int64: type = "Int64"; shape = "scalar"
        case .string: type = "String"; shape = "scalar"
        case .dictionary: type = "Dictionary"; shape = "-"
        case .sequence: type = "Sequence"; shape = "-"
        case .state: type = "State"; shape = "-"
        case .invalid: type = "invalid"; shape = "-"
        @unknown default: type = "unknown"; shape = "-"
        }
    }

    private static func describe(_ constraint: MLMultiArrayConstraint?) -> String {
        guard let constraint else { return "-" }
        let base = "\(constraint.shape.map(\.stringValue).joined(separator: "×")) \(constraint.dataType)"

        switch constraint.shapeConstraint.type {
        case .enumerated:
            let shapes = constraint.shapeConstraint.enumeratedShapes
                .map { $0.map(\.stringValue).joined(separator: "×") }
                .joined(separator: " | ")
            return "\(base) — enumerated: \(shapes)"
        case .range:
            let ranges = constraint.shapeConstraint.sizeRangeForDimension
                .map { "\($0.rangeValue.location)...\($0.rangeValue.location + $0.rangeValue.length)" }
                .joined(separator: ", ")
            return "\(base) — range: \(ranges)"
        case .unspecified:
            return "\(base) — static"
        @unknown default:
            return base
        }
    }

    private static func describe(_ constraint: MLImageConstraint?) -> String {
        guard let constraint else { return "-" }
        let base = "\(constraint.pixelsWide)×\(constraint.pixelsHigh)"

        switch constraint.sizeConstraint.type {
        case .enumerated:
            let sizes = constraint.sizeConstraint.enumeratedImageSizes
                .map { "\(Int($0.pixelsWide))×\(Int($0.pixelsHigh))" }
                .joined(separator: " | ")
            return "\(base) — enumerated: \(sizes)"
        case .range:
            let w = constraint.sizeConstraint.pixelsWideRange
            let h = constraint.sizeConstraint.pixelsHighRange
            return "\(base) — range: w \(w.location)...\(w.location + w.length), h \(h.location)...\(h.location + h.length)"
        case .unspecified:
            return "\(base) — static"
        @unknown default:
            return base
        }
    }
}

struct ModelInterface: Sendable {
    let inputs: [FeatureSummary]
    let outputs: [FeatureSummary]
}

struct ModelInspection: Sendable, Identifiable {
    let modelName: String
    let interface: ModelInterface
    let placements: [PlacementReport]

    var id: String { modelName }
}

// MARK: - Markdown rendering

extension ModelInspection {
    /// Paste-into-the-writeup report.
    var markdown: String {
        var out = "## \(modelName)\n\n### Interface\n\n"
        out += "| Direction | Name | Type | Shape |\n|---|---|---|---|\n"
        for input in interface.inputs {
            out += "| in | `\(input.name)` | \(input.type) | \(input.shape) |\n"
        }
        for output in interface.outputs {
            out += "| out | `\(output.name)` | \(output.type) | \(output.shape) |\n"
        }

        out += "\n### Planned placement\n\n"
        out += "| Compute units | Ops | ANE | GPU | CPU | ANE cost | GPU cost | CPU cost |\n"
        out += "|---|---|---|---|---|---|---|---|\n"
        for report in placements {
            let counts = report.operationCounts
            let share = report.costShare
            let pct = { (device: ComputeDeviceKind) in
                share[device].map { String(format: "%.1f%%", $0 * 100) } ?? "—"
            }
            out += "| \(report.computeUnits.label) | \(report.operations.count)"
            out += " | \(counts[.neuralEngine] ?? 0) | \(counts[.gpu] ?? 0) | \(counts[.cpu] ?? 0)"
            out += " | \(pct(.neuralEngine)) | \(pct(.gpu)) | \(pct(.cpu)) |\n"
        }

        for report in placements where report.computeUnits == .cpuAndNeuralEngine {
            let fallbacks = report.fallbackOperators(awayFrom: .neuralEngine)
            guard !fallbacks.isEmpty else { continue }
            out += "\n### Operators falling back off ANE (`.cpuAndNeuralEngine`)\n\n"
            out += "| Operator | Count |\n|---|---|\n"
            for fallback in fallbacks.prefix(20) {
                out += "| `\(fallback.name)` | \(fallback.count) |\n"
            }
        }

        return out
    }
}

extension MLComputeUnits {
    var label: String {
        switch self {
        case .all: ".all"
        case .cpuOnly: ".cpuOnly"
        case .cpuAndGPU: ".cpuAndGPU"
        case .cpuAndNeuralEngine: ".cpuAndNeuralEngine"
        @unknown default: "unknown"
        }
    }
}

// MARK: - Model discovery

enum ModelDiscovery {
    static let extensions = ["mlpackage", "mlmodelc"]

    /// Models bundled into the app plus anything dropped into Documents.
    static func availableModels() -> [URL] {
        var urls: [URL] = []

        for ext in extensions {
            urls += Bundle.main.urls(forResourcesWithExtension: ext, subdirectory: nil) ?? []
        }

        if let documents = try? FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false
        ) {
            let contents = (try? FileManager.default.contentsOfDirectory(
                at: documents, includingPropertiesForKeys: nil
            )) ?? []
            urls += contents.filter { extensions.contains($0.pathExtension) }
        }

        return urls.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}

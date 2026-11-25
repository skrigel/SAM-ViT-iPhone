import SwiftUI
import UniformTypeIdentifiers

/// M0 harness UI: pick a model, inspect where Core ML plans to run it, export the report.
struct ComputePlanView: View {
    @State private var models: [URL] = ModelDiscovery.availableModels()
    @State private var inspections: [ModelInspection] = []
    @State private var isInspecting = false
    @State private var errorMessage: String?
    @State private var isImporting = false

    private var report: String {
        inspections.map(\.markdown).joined(separator: "\n---\n\n")
    }

    var body: some View {
        NavigationStack {
            Group {
                if models.isEmpty {
                    ContentUnavailableView(
                        "No models found",
                        systemImage: "cube.transparent",
                        description: Text("Add a .mlpackage to the app target, or import one below.")
                    )
                } else {
                    List {
                        Section("Models") {
                            ForEach(models, id: \.self) { url in
                                Text(url.lastPathComponent)
                                    .font(.system(.body, design: .monospaced))
                            }
                        }

                        if !report.isEmpty {
                            Section("Report") {
                                Text(report)
                                    .font(.system(.caption, design: .monospaced))
                                    .textSelection(.enabled)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Compute Plan")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Inspect", systemImage: "play.fill") {
                        Task { await inspectAll() }
                    }
                    .disabled(models.isEmpty || isInspecting)
                }
                ToolbarItem(placement: .secondaryAction) {
                    Button("Import", systemImage: "folder") { isImporting = true }
                }
                if !report.isEmpty {
                    ToolbarItem(placement: .secondaryAction) {
                        ShareLink(item: report) { Label("Export", systemImage: "square.and.arrow.up") }
                    }
                }
            }
            .overlay {
                if isInspecting {
                    ProgressView("Compiling and planning…")
                        .padding()
                        .background(.regularMaterial, in: .rect(cornerRadius: 12))
                }
            }
            .fileImporter(
                isPresented: $isImporting,
                allowedContentTypes: [.package, .folder, .data],
                allowsMultipleSelection: true
            ) { result in
                handleImport(result)
            }
            .alert("Inspection failed", isPresented: .constant(errorMessage != nil)) {
                Button("OK") { errorMessage = nil }
            } message: {
                Text(errorMessage ?? "")
            }
        }
    }

    private func inspectAll() async {
        isInspecting = true
        defer { isInspecting = false }

        var results: [ModelInspection] = []
        for url in models {
            do {
                results.append(try await ComputePlanDump.inspect(modelAt: url))
            } catch {
                errorMessage = "\(url.lastPathComponent): \(error.localizedDescription)"
            }
        }
        inspections = results
    }

    /// Copies picked models into Documents so later launches find them without re-importing.
    private func handleImport(_ result: Result<[URL], any Error>) {
        guard case let .success(urls) = result else { return }

        let documents = URL.documentsDirectory
        for url in urls where ModelDiscovery.extensions.contains(url.pathExtension) {
            guard url.startAccessingSecurityScopedResource() else { continue }
            defer { url.stopAccessingSecurityScopedResource() }

            let destination = documents.appending(path: url.lastPathComponent)
            try? FileManager.default.removeItem(at: destination)
            try? FileManager.default.copyItem(at: url, to: destination)
        }
        models = ModelDiscovery.availableModels()
    }
}

#Preview {
    ComputePlanView()
}

import Foundation
import SherpaOnnx

/// Owns the sherpa-onnx offline recognizer for GigaAM-v3.
///
/// An `actor` rather than a stored property on the `@MainActor` engine for two
/// reasons that both matter. `SherpaOnnxOfflineRecognizer` is a non-`Sendable`
/// class from a Swift-5-mode module, so it has to stay confined to one isolation
/// domain; and building it compiles the ONNX graphs, which takes seconds — work
/// that must not run on the main thread. Isolating it here gives both: the
/// recognizer never crosses an isolation boundary, and every call to it lands on
/// the actor's own executor.
///
/// `load()` is separate from `init` deliberately. An actor's non-async `init`
/// runs on the *caller's* thread, so constructing the recognizer there would put
/// the graph compile back on the main actor; `init` only records the paths.
actor GigaAMRecognizer {
    private let files: GigaAMModelFiles.Paths
    private var recognizer: SherpaOnnxOfflineRecognizer?

    /// Feature dimension of GigaAM's log-mel front end. Not the sherpa default
    /// (80) — a mismatch here does not fail loudly, it produces garbage text.
    private static let featureDim = 64

    /// The graphs are int8 and CPU-only; eight threads is what the model card
    /// recommends and what the Apple-silicon performance cores can carry without
    /// starving the rest of the pipeline.
    private static let threadCount = 8

    init(files: GigaAMModelFiles.Paths) {
        self.files = files
    }

    /// Build the recognizer. Idempotent — a second call is a no-op, so the
    /// engine's single-flight load and a late `ensureModel()` can both call it.
    func load() {
        guard recognizer == nil else { return }
        recognizer = Self.makeRecognizer(files: files)
    }

    var isLoaded: Bool {
        recognizer != nil
    }

    /// Decode one chunk of 16 kHz mono samples in [-1, 1].
    func decode(samples: [Float]) throws -> String {
        guard let recognizer else { throw TranscriptionError.modelNotLoaded }
        return recognizer.decode(
            samples: samples, sampleRate: AudioConstants.targetSampleRate,
        ).text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The config structs hold raw `const char *` borrowed from autoreleased
    /// bridged `NSString`s (`toCPointer`), so they stay valid only until the
    /// enclosing autorelease pool drains. Everything from the first config to
    /// `SherpaOnnxOfflineRecognizer.init` — which copies the strings on the C++
    /// side — therefore happens in one function body, exactly as the upstream
    /// examples do it.
    private static func makeRecognizer(files: GigaAMModelFiles.Paths) -> SherpaOnnxOfflineRecognizer {
        let transducer = sherpaOnnxOfflineTransducerModelConfig(
            encoder: files.encoder.path,
            decoder: files.decoder.path,
            joiner: files.joiner.path,
        )
        let modelConfig = sherpaOnnxOfflineModelConfig(
            tokens: files.tokens.path,
            transducer: transducer,
            numThreads: threadCount,
            provider: "cpu",
            modelType: "nemo_transducer",
        )
        var config = sherpaOnnxOfflineRecognizerConfig(
            featConfig: sherpaOnnxFeatureConfig(
                sampleRate: AudioConstants.targetSampleRate, featureDim: featureDim,
            ),
            modelConfig: modelConfig,
            decodingMethod: "greedy_search",
        )
        return SherpaOnnxOfflineRecognizer(config: &config)
    }
}

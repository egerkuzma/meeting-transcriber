import Foundation
import os.log

private let logger = Logger(subsystem: AppPaths.logSubsystem, category: "WhisperCppEngine")

/// Transcription engine that runs ggml Whisper models through the prebuilt
/// whisper.cpp framework (Metal-accelerated).
///
/// Exists for **fine-tunes**: WhisperKit runs Apple's CoreML conversions of the
/// stock OpenAI checkpoints, and a community fine-tune (the Russian large-v3
/// this was added for) ships as a ggml `.bin` that nothing else here can load.
/// The model path is therefore a setting rather than a constant — any ggml
/// Whisper model can be dropped in — and so is the language, since which model
/// is installed decides what language it should be told.
///
/// Like `GigaAMEngine` the app **downloads nothing**: a missing file resolves to
/// `EngineModelState.failed` naming the path, not to `.unloaded`.
///
/// **No chunking**, unlike GigaAM: whisper.cpp does its own 30 s windowing with
/// context carried across windows, so the whole buffer goes in one call and the
/// segments come back phrase-shaped with real boundaries. That gives diarization
/// precise timestamps to map onto.
///
/// Live transcription is not implemented — `StreamingTranscribingEngine` is not
/// adopted and `TranscriptionEngineSetting.whisperCpp.supportsLiveTranscription`
/// is `false`. Captions still work: the engine reports a set language, which
/// routes them to the engine-independent Nemotron streaming session.
@MainActor
@Observable
final class WhisperCppEngine: TranscribingEngine {
    private(set) var modelState: EngineModelState = .unloaded
    /// Always 0 — the model is installed by hand, so there is nothing to report
    /// progress for. Part of the protocol, not a stub for a future download.
    private(set) var downloadProgress: Double = 0
    private(set) var transcriptionProgress: Double = 0

    /// Path to the ggml `.bin`. Changing it drops the loaded context, so the
    /// next transcription runs the newly chosen model rather than the one
    /// already in memory.
    var modelPath: String {
        didSet {
            guard modelPath != oldValue else { return }
            context = nil
            modelState = .unloaded
        }
    }

    /// ISO 639-1 code passed to the decoder. Empty means auto-detect, which
    /// whisper.cpp accepts as `""`.
    var language: String

    private var context: WhisperCppContext?
    private let modelLoad = SingleFlight()

    init(modelPath: String = WhisperCppModelFile.defaultPath, language: String = "ru") {
        self.modelPath = modelPath
        self.language = language
    }

    func loadModel() async {
        await modelLoad.run { [self] in
            guard context == nil else { return }
            switch WhisperCppModelFile.resolve(path: modelPath) {
            case let .missing(reason):
                logger.error("Whisper.cpp model unavailable: \(reason, privacy: .public)")
                modelState = .failed(reason)

            case let .ready(url):
                modelState = .loading
                let context = WhisperCppContext()
                guard await context.load(modelPath: url.path) else {
                    let reason = "Whisper.cpp could not load the ggml model at \(url.path)"
                    logger.error("\(reason, privacy: .public)")
                    modelState = .failed(reason)
                    return
                }
                self.context = context
                modelState = .loaded
                logger.info("Whisper.cpp: model loaded")
            }
        }
    }

    private func ensureModel() async throws -> WhisperCppContext {
        if let context { return context }
        logger.info("Whisper.cpp: model not loaded, loading…")
        await loadModel()
        guard let context else {
            logger.error("Whisper.cpp: model load FAILED, state=\(String(describing: self.modelState), privacy: .public)")
            throw TranscriptionError.modelNotLoaded
        }
        return context
    }

    func transcribeSegments(audioPath: URL) async throws -> [TimestampedSegment] {
        let context = try await ensureModel()

        // The pipeline resamples to 16 kHz mono before any engine sees the file.
        // The guard is here because whisper.cpp takes a bare sample count with
        // no rate alongside it — WHISPER_SAMPLE_RATE is assumed — so a file at
        // another rate would transcribe as time-stretched audio, not fail.
        let (samples, sampleRate) = try await AudioMixer.loadAudioAsFloat32(url: audioPath)
        let audio = sampleRate == AudioConstants.targetSampleRate
            ? samples
            : AudioMixer.resample(samples, from: sampleRate, to: AudioConstants.targetSampleRate)
        guard !audio.isEmpty else { return [] }

        transcriptionProgress = 0
        defer { transcriptionProgress = 1.0 }

        let segments = try await context.transcribe(samples: audio, language: language) { fraction in
            Task { @MainActor [weak self] in self?.transcriptionProgress = fraction }
        }
        return segments.map { segment in
            TimestampedSegment(start: segment.start, end: segment.end, text: segment.text)
        }
    }
}

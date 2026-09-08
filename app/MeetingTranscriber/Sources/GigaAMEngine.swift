import Foundation
import os.log

private let logger = Logger(subsystem: AppPaths.logSubsystem, category: "GigaAMEngine")

/// Transcription engine backed by Sber GigaAM-v3 (e2e-RNNT) via sherpa-onnx.
///
/// **Russian only.** GigaAM is a single-language model with no language
/// parameter and no detection, so `AppSettings.activeEngineLanguageOrNil`
/// reports `ru` for it unconditionally rather than exposing a picker.
///
/// Unlike the other two engines this one **downloads nothing**: the ONNX
/// artefacts are installed by hand under `AppPaths.gigaamModelDir`, so a missing
/// folder is a normal state that resolves to `EngineModelState.failed` with the
/// path in the message, not to `.unloaded` (which would read as "not tried yet"
/// and send the user looking for a download that never happens).
///
/// Live transcription is not implemented — `StreamingTranscribingEngine` is not
/// adopted, and `TranscriptionEngineSetting.gigaam.supportsLiveTranscription` is
/// `false`. Captions still work through the language-driven Nemotron streaming
/// backend, which drives its own model and never touches this engine.
@MainActor
@Observable
final class GigaAMEngine: TranscribingEngine {
    private(set) var modelState: EngineModelState = .unloaded
    /// Always 0 — the model is installed by hand, so there is nothing to report
    /// progress for. Part of the protocol, not a stub for a future download.
    private(set) var downloadProgress: Double = 0
    private(set) var transcriptionProgress: Double = 0

    private var recognizer: GigaAMRecognizer?
    private let modelLoad = SingleFlight()
    private let modelDirectory: URL

    /// `modelDirectory` is injectable so tests can point the engine at a fixture
    /// (or at a folder known to be absent) instead of the user's real one.
    init(modelDirectory: URL = AppPaths.gigaamModelDir) {
        self.modelDirectory = modelDirectory
    }

    func loadModel() async {
        await modelLoad.run { [self] in
            guard recognizer == nil else { return }
            switch GigaAMModelFiles.resolve(in: modelDirectory) {
            case let .missing(reason):
                logger.error("GigaAM model unavailable: \(reason, privacy: .public)")
                modelState = .failed(reason)

            case let .ready(paths):
                modelState = .loading
                let recognizer = GigaAMRecognizer(files: paths)
                await recognizer.load()
                guard await recognizer.isLoaded else {
                    let reason = "GigaAM recognizer could not be created from \(modelDirectory.path)"
                    logger.error("\(reason, privacy: .public)")
                    modelState = .failed(reason)
                    return
                }
                self.recognizer = recognizer
                modelState = .loaded
                logger.info("GigaAM: model loaded")
            }
        }
    }

    private func ensureModel() async throws -> GigaAMRecognizer {
        if let recognizer { return recognizer }
        logger.info("GigaAM: model not loaded, loading…")
        await loadModel()
        guard let recognizer else {
            logger.error("GigaAM: model load FAILED, state=\(String(describing: self.modelState), privacy: .public)")
            throw TranscriptionError.modelNotLoaded
        }
        return recognizer
    }

    /// Decode the file chunk by chunk, one `TimestampedSegment` per chunk.
    ///
    /// The segment boundaries are the chunk boundaries, so timestamps are as
    /// coarse as `GigaAMChunking.maxChunkSeconds`. Diarization still works —
    /// it reads the audio itself and maps by timestamp — but a chunk spanning
    /// two speakers is attributed to one of them. The transducer result does
    /// carry per-token timestamps, so a finer split is available later without
    /// changing this engine's interface.
    func transcribeSegments(audioPath: URL) async throws -> [TimestampedSegment] {
        let recognizer = try await ensureModel()

        // The pipeline resamples to 16 kHz mono before any engine sees the file;
        // the guard is here because feeding sherpa a rate its features weren't
        // built for produces plausible-looking wrong text rather than an error.
        let (samples, sampleRate) = try await AudioMixer.loadAudioAsFloat32(url: audioPath)
        let audio = sampleRate == AudioConstants.targetSampleRate
            ? samples
            : AudioMixer.resample(samples, from: sampleRate, to: AudioConstants.targetSampleRate)

        let ranges = GigaAMChunking.chunks(samples: audio, sampleRate: AudioConstants.targetSampleRate)
        guard !ranges.isEmpty else { return [] }

        transcriptionProgress = 0
        defer { transcriptionProgress = 1.0 }

        var segments: [TimestampedSegment] = []
        for (index, range) in ranges.enumerated() {
            let text = try await recognizer.decode(samples: Array(audio[range]))
            transcriptionProgress = Double(index + 1) / Double(ranges.count)
            guard !text.isEmpty else { continue }
            segments.append(TimestampedSegment(
                start: Double(range.lowerBound) / Double(AudioConstants.targetSampleRate),
                end: Double(range.upperBound) / Double(AudioConstants.targetSampleRate),
                text: text,
            ))
        }
        return segments
    }
}

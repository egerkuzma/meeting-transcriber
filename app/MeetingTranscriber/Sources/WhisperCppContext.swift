import Foundation
import os.log
import whisper

private let logger = Logger(subsystem: AppPaths.logSubsystem, category: "WhisperCpp")

/// Owns the `whisper_context` and every call into the whisper.cpp C API.
///
/// **Not an actor**, unlike `GigaAMRecognizer`, and the difference is
/// deliberate. `whisper_full` blocks for as long as the transcription takes —
/// minutes on a long meeting — and an actor runs its body on a cooperative
/// thread-pool thread, where a block that long starves the pool. A dedicated
/// serial `DispatchQueue` blocks a thread that exists for exactly this, and it
/// also gives the serialization the C API requires: whisper.cpp documents
/// `whisper_full` as "not thread safe for same context", which actor
/// reentrancy — suspending inside a call and admitting another — would not
/// guarantee on its own.
///
/// The context is loaded once and reused across files (a 3 GB model takes
/// seconds to map) and freed in `deinit`.
///
/// `@unchecked Sendable` is the honest annotation here: `context` is mutable
/// state, and its safety rests on every access happening on `queue`, which the
/// compiler cannot verify. Nothing else in the type is mutable.
final class WhisperCppContext: @unchecked Sendable {
    /// One decoded segment, already converted out of whisper's centisecond
    /// integers so no caller has to know that detail.
    struct Segment {
        let start: TimeInterval
        let end: TimeInterval
        let text: String
    }

    private let queue = DispatchQueue(label: "com.meetingtranscriber.whispercpp")
    private var context: OpaquePointer?

    /// whisper.cpp's own thread count for the decode. Matches the performance
    /// core count on Apple silicon; the heavy lifting is on the GPU via Metal.
    private static let threadCount: Int32 = 8

    /// Beam search over greedy: on a fine-tuned model the quality difference is
    /// what the fine-tune was for, and the cost is bounded because Metal does
    /// the encoder work. `whisper_full_default_params` seeds `beam_size` for
    /// this strategy; 5 is its default and what upstream's CLI uses.
    private static let beamSize: Int32 = 5

    init() {
        // Route the library's own logging into os.log. Without this a model load
        // writes a screenful to stderr on every launch, and — worse — a real
        // load error would land only there, where a shipped app has no reader.
        whisper_log_set({ level, message, _ in
            guard let message else { return }
            let text = String(cString: message).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return }
            if level.rawValue >= GGML_LOG_LEVEL_WARN.rawValue {
                logger.warning("whisper.cpp: \(text, privacy: .public)")
            } else {
                logger.debug("whisper.cpp: \(text, privacy: .public)")
            }
        }, nil)
    }

    deinit {
        // Safe without hopping to `queue`: every enqueued block captures `self`
        // strongly, so `deinit` cannot run while one is pending or in flight.
        if let context { whisper_free(context) }
    }

    /// Load the model. Idempotent — a second call is a no-op — so the engine's
    /// single-flight load and a late `ensureModel()` can both call it.
    /// Returns whether a usable context exists afterwards.
    func load(modelPath: String) async -> Bool {
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                if context != nil { return continuation.resume(returning: true) }
                var params = whisper_context_default_params()
                params.use_gpu = true
                params.flash_attn = true
                context = modelPath.withCString { whisper_init_from_file_with_params($0, params) }
                continuation.resume(returning: context != nil)
            }
        }
    }

    /// Transcribe a whole 16 kHz mono buffer in one call.
    ///
    /// No chunking, on purpose: whisper.cpp does its own 30 s windowing with
    /// context carried between windows, and the segments it returns are
    /// phrase-shaped with real boundaries — which is what makes the timestamps
    /// worth mapping onto diarization.
    ///
    /// - Parameters:
    ///   - samples: 16 kHz mono PCM in [-1, 1].
    ///   - language: ISO 639-1 code, or empty for auto-detect.
    ///   - progress: called with 0…1 as decoding advances, off the main actor.
    func transcribe(
        samples: [Float],
        language: String,
        progress: @escaping @Sendable (Double) -> Void,
    ) async throws -> [Segment] {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { [self] in
                guard let context else {
                    return continuation.resume(throwing: TranscriptionError.modelNotLoaded)
                }
                do {
                    let segments = try Self.run(
                        context: context, samples: samples, language: language, progress: progress,
                    )
                    continuation.resume(returning: segments)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// The C call itself. `static` so it cannot reach any mutable state beyond
    /// the context it is handed, which is what keeps the queue confinement above
    /// reviewable in one place.
    private static func run(
        context: OpaquePointer,
        samples: [Float],
        language: String,
        progress: @escaping @Sendable (Double) -> Void,
    ) throws -> [Segment] {
        var params = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH)
        params.beam_search.beam_size = beamSize
        params.n_threads = threadCount
        params.no_timestamps = false
        params.single_segment = false
        params.translate = false
        params.print_special = false
        params.print_progress = false
        params.print_realtime = false
        params.print_timestamps = false

        // A C function pointer cannot capture, so the Swift closure travels as
        // `user_data`. `box` is kept alive across the call by the
        // `withExtendedLifetime` below — without it the optimiser is free to
        // release it before whisper.cpp ever invokes the callback.
        let box = ProgressBox(report: progress)
        params.progress_callback_user_data = Unmanaged.passUnretained(box).toOpaque()
        params.progress_callback = { _, _, percent, userData in
            guard let userData else { return }
            Unmanaged<ProgressBox>.fromOpaque(userData)
                .takeUnretainedValue()
                .report(max(0, min(1, Double(percent) / 100)))
        }

        // `language` must outlive the call: `whisper_full_params` stores the
        // pointer, it does not copy the string.
        let status = withExtendedLifetime(box) {
            language.withCString { languageCString -> Int32 in
                params.language = languageCString
                return whisper_full(context, params, samples, Int32(samples.count))
            }
        }
        guard status == 0 else { throw WhisperCppError.decodeFailed(status: Int(status)) }

        return (0 ..< whisper_full_n_segments(context)).compactMap { index in
            guard let raw = whisper_full_get_segment_text(context, index) else { return nil }
            let text = String(cString: raw).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return nil }
            // t0/t1 are centiseconds.
            return Segment(
                start: Double(whisper_full_get_segment_t0(context, index)) / 100,
                end: Double(whisper_full_get_segment_t1(context, index)) / 100,
                text: text,
            )
        }
    }

    /// Carries the progress closure through `void *`. A class because that is
    /// what `Unmanaged` needs; `@unchecked Sendable` because whisper.cpp invokes
    /// the callback from its own worker thread and the only stored value is an
    /// immutable `@Sendable` closure.
    private final class ProgressBox: @unchecked Sendable {
        let report: @Sendable (Double) -> Void

        init(report: @escaping @Sendable (Double) -> Void) {
            self.report = report
        }
    }
}

enum WhisperCppError: LocalizedError {
    case decodeFailed(status: Int)

    var errorDescription: String? {
        switch self {
        case let .decodeFailed(status): "whisper.cpp failed to decode the audio (status \(status))"
        }
    }
}

import Foundation

/// Resolves the single ggml model file `WhisperCppEngine` runs, and says plainly
/// what is wrong when it can't.
///
/// Unlike WhisperKit and Parakeet the app downloads nothing here: the user puts
/// a `.bin` where they like and points `AppSettings.whisperCppModelPath` at it,
/// which is the whole reason the engine exists — any ggml Whisper fine-tune can
/// be dropped in. So "the file isn't there" is a normal, user-actionable state
/// carrying the exact path, not a failure to swallow. Same shape as
/// `GigaAMModelFiles`, one file instead of four.
enum WhisperCppModelFile {
    enum Resolution: Equatable {
        case ready(URL)
        /// Human-readable reason, naming the path that was tried.
        case missing(String)
    }

    /// The model shipped-for by default: the Russian large-v3 fine-tune this
    /// engine was added for. Only a default — the setting is what is read.
    static let defaultFileName = "ggml-large-v3-russian.bin"

    /// `~/Library/Application Support/MeetingTranscriber/models/whisper-cpp/`
    static let defaultDirectory = AppPaths.modelsDir.appendingPathComponent("whisper-cpp")

    static var defaultPath: String {
        defaultDirectory.appendingPathComponent(defaultFileName).path
    }

    /// Check the configured path. `fileManager` is a parameter so tests can
    /// resolve against a fixture without touching the user's real model folder.
    static func resolve(path: String, fileManager: FileManager = .default) -> Resolution {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return .missing("No Whisper.cpp model selected. Choose a ggml .bin file in Settings → Transcribe.")
        }

        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: trimmed, isDirectory: &isDirectory) else {
            return .missing("Whisper.cpp model not found at \(trimmed)")
        }
        guard !isDirectory.boolValue else {
            return .missing("Whisper.cpp model path is a folder, not a ggml .bin file: \(trimmed)")
        }

        return .ready(URL(fileURLWithPath: trimmed))
    }
}

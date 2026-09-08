import Foundation

/// Resolves the four ONNX artefacts GigaAM-v3 needs, and says plainly what is
/// missing when it can't.
///
/// Unlike WhisperKit and FluidAudio there is **no download step**: the model is
/// placed under `AppPaths.gigaamModelDir` by hand. That makes "the folder isn't
/// there" a normal, user-actionable state rather than a failure to report as a
/// crash, so the missing case carries a message naming the exact path — the only
/// thing a user can act on — and the engine surfaces it as
/// `EngineModelState.failed`.
enum GigaAMModelFiles {
    /// The four paths the recognizer is configured with.
    struct Paths: Equatable {
        let encoder: URL
        let decoder: URL
        let joiner: URL
        let tokens: URL
    }

    enum Resolution: Equatable {
        case ready(Paths)
        /// Human-readable reason, naming the directory and the missing files.
        case missing(String)
    }

    static let encoderName = "gigaam_v3_e2e_rnnt_encoder_int8.onnx"
    static let decoderName = "gigaam_v3_e2e_rnnt_decoder.onnx"
    static let joinerName = "gigaam_v3_e2e_rnnt_joint.onnx"
    static let tokensName = "gigaam_v3_e2e_rnnt_tokens.txt"

    /// Check `directory` for the four artefacts. `directory` is a parameter (not
    /// read from `AppPaths` inside) so tests can point it at a fixture without
    /// touching the user's real model folder.
    static func resolve(
        in directory: URL = AppPaths.gigaamModelDir,
        fileManager: FileManager = .default,
    ) -> Resolution {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: directory.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            return .missing(
                "GigaAM model folder not found. Place the GigaAM-v3 e2e-RNNT files in \(directory.path)",
            )
        }

        let names = [encoderName, decoderName, joinerName, tokensName]
        let absent = names.filter {
            !fileManager.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
        guard absent.isEmpty else {
            return .missing(
                "GigaAM model is incomplete in \(directory.path) — missing: \(absent.joined(separator: ", "))",
            )
        }

        return .ready(Paths(
            encoder: directory.appendingPathComponent(encoderName),
            decoder: directory.appendingPathComponent(decoderName),
            joiner: directory.appendingPathComponent(joinerName),
            tokens: directory.appendingPathComponent(tokensName),
        ))
    }
}

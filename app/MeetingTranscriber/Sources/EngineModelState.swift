import Foundation

/// App-owned model lifecycle state for a `TranscribingEngine`, decoupled from
/// any ASR vendor's enum. WhisperKit ships its own `ModelState`; mapping to
/// this type at the engine boundary keeps the protocol — and the
/// FluidAudio-backed engines (e.g. Parakeet) — from importing WhisperKit
/// just to report status.
///
/// The case names are the RPC wire contract, projected by `wireName`:
/// "unloaded"/"downloading"/"loading"/"loaded"/"failed".
/// `scripts/e2e-cpu-load.sh` waits for `modelState == "loaded"` to know preload
/// finished, and `RPCEngineStateTests` pins the spelling.
enum EngineModelState: Equatable {
    case unloaded
    case downloading
    case loading
    case loaded
    /// The load was attempted and cannot succeed until the user acts, with the
    /// reason to show them. Distinct from `.unloaded` on purpose: an engine
    /// whose model is installed by hand (GigaAM) has no download to wait for, so
    /// "not loaded yet" would send the user looking for one.
    case failed(String)

    /// The RPC projection. A dedicated property rather than
    /// `String(describing:).lowercased()`, which would lowercase — and leak the
    /// whole of — a failure message that names a filesystem path.
    var wireName: String {
        switch self {
        case .unloaded: "unloaded"
        case .downloading: "downloading"
        case .loading: "loading"
        case .loaded: "loaded"
        case .failed: "failed"
        }
    }

    /// The reason a load cannot succeed, for the Settings status line. `nil` in
    /// every other state.
    var failureMessage: String? {
        guard case let .failed(message) = self else { return nil }
        return message
    }
}

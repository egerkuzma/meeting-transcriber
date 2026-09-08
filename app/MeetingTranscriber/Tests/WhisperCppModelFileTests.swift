@testable import MeetingTranscriber
import XCTest

/// The model-path resolver is the only part of the Whisper.cpp engine that runs
/// without a 3 GB model and a GPU, and it is the part a user meets first — a
/// mistyped path has to come back as a sentence naming what was tried, not as a
/// silent `.unloaded`.
final class WhisperCppModelFileTests: XCTestCase {
    private var directory = URL(fileURLWithPath: NSTemporaryDirectory())

    override func setUpWithError() throws {
        try super.setUpWithError()
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("WhisperCppModelFileTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
        try super.tearDownWithError()
    }

    func testResolvesAnExistingFile() throws {
        let model = directory.appendingPathComponent("ggml-test.bin")
        try Data("not a real model".utf8).write(to: model)

        XCTAssertEqual(WhisperCppModelFile.resolve(path: model.path), .ready(model))
    }

    func testMissingFileReportsTheAttemptedPath() {
        let path = directory.appendingPathComponent("absent.bin").path

        guard case let .missing(message) = WhisperCppModelFile.resolve(path: path) else {
            return XCTFail("expected .missing")
        }
        XCTAssertTrue(
            message.contains(path),
            "The message is the only thing the user can act on, so it must name the path: \(message)",
        )
    }

    /// A folder passes `fileExists` — without the directory check this would
    /// resolve as ready and fail much later, inside whisper.cpp.
    func testDirectoryIsNotAModel() {
        guard case let .missing(message) = WhisperCppModelFile.resolve(path: directory.path) else {
            return XCTFail("expected .missing")
        }
        XCTAssertTrue(message.contains(directory.path))
    }

    func testEmptyPathAsksTheUserToChooseOne() {
        guard case let .missing(message) = WhisperCppModelFile.resolve(path: "   ") else {
            return XCTFail("expected .missing")
        }
        XCTAssertTrue(message.contains("Settings"), "An unset path should point at where to set it: \(message)")
    }

    func testDefaultPathSitsUnderTheAppModelsDirectory() {
        XCTAssertTrue(WhisperCppModelFile.defaultPath.hasPrefix(AppPaths.modelsDir.path))
        XCTAssertTrue(WhisperCppModelFile.defaultPath.hasSuffix(".bin"))
    }
}

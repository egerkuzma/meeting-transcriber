@testable import MeetingTranscriber
import XCTest

/// The chunk boundary is the whole of GigaAM's long-form handling, so it is
/// pinned here rather than through the engine: it needs no model on disk and no
/// ONNX runtime, which is what makes it runnable on every PR.
final class GigaAMChunkingTests: XCTestCase {
    private let rate = AudioConstants.targetSampleRate

    /// Loud noise everywhere except an explicit silent notch at each of
    /// `silentAt` (seconds), 100 ms wide.
    private func audio(seconds: Double, silentAt: [Double] = []) -> [Float] {
        var samples = [Float](repeating: 0, count: Int(seconds * Double(rate)))
        for index in samples.indices {
            samples[index] = index.isMultiple(of: 2) ? 0.5 : -0.5
        }
        let notch = Int(GigaAMChunking.quietWindowSeconds * Double(rate))
        for start in silentAt {
            let from = Int(start * Double(rate))
            for index in from ..< min(from + notch, samples.count) { samples[index] = 0 }
        }
        return samples
    }

    func testShortAudioIsOneChunk() {
        let samples = audio(seconds: 12)
        XCTAssertEqual(
            GigaAMChunking.chunks(samples: samples, sampleRate: rate),
            [0 ..< samples.count],
        )
    }

    func testAudioAtTheCeilingIsStillOneChunk() {
        let samples = audio(seconds: GigaAMChunking.maxChunkSeconds)
        XCTAssertEqual(
            GigaAMChunking.chunks(samples: samples, sampleRate: rate).count, 1,
            "A file that fits in one decoder pass must not be cut",
        )
    }

    func testEmptyAudioProducesNoChunks() {
        XCTAssertTrue(GigaAMChunking.chunks(samples: [], sampleRate: rate).isEmpty)
    }

    func testChunksAreContiguousAndCoverEverything() {
        let samples = audio(seconds: 95, silentAt: [17, 34.5, 52, 70])
        let chunks = GigaAMChunking.chunks(samples: samples, sampleRate: rate)

        XCTAssertEqual(chunks.first?.lowerBound, 0)
        XCTAssertEqual(chunks.last?.upperBound, samples.count)
        for (previous, next) in zip(chunks, chunks.dropFirst()) {
            XCTAssertEqual(previous.upperBound, next.lowerBound, "Chunks must not overlap or leave gaps")
        }
    }

    func testNoChunkExceedsTheDecoderCeiling() {
        let samples = audio(seconds: 95, silentAt: [17, 34.5, 52, 70])
        let ceiling = Int(GigaAMChunking.maxChunkSeconds * Double(rate))
        for chunk in GigaAMChunking.chunks(samples: samples, sampleRate: rate) {
            XCTAssertLessThanOrEqual(chunk.count, ceiling)
        }
    }

    /// The point of searching rather than cutting on a fixed grid: a pause
    /// inside the search window wins over the grid position.
    func testCutLandsOnTheSilentNotch() {
        let samples = audio(seconds: 40, silentAt: [17])
        let chunks = GigaAMChunking.chunks(samples: samples, sampleRate: rate)

        guard let firstCut = chunks.first?.upperBound else { return XCTFail("expected a cut") }
        let notchCentre = (17 + GigaAMChunking.quietWindowSeconds / 2) * Double(rate)
        XCTAssertEqual(
            Double(firstCut), notchCentre,
            accuracy: GigaAMChunking.quietWindowSeconds * Double(rate),
            "The cut should sit in the pause, not at the 20 s ceiling",
        )
    }

    /// With no pause anywhere the search still has to terminate and stay inside
    /// the bounds — the degenerate case that a "find the silence" splitter can
    /// loop on.
    func testUniformlyLoudAudioStillSplitsWithinBounds() {
        let samples = audio(seconds: 50)
        let chunks = GigaAMChunking.chunks(samples: samples, sampleRate: rate)
        let floor = Int(GigaAMChunking.minChunkSeconds * Double(rate))
        let ceiling = Int(GigaAMChunking.maxChunkSeconds * Double(rate))

        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks.dropLast() {
            XCTAssertGreaterThanOrEqual(chunk.count, floor)
            XCTAssertLessThanOrEqual(chunk.count, ceiling)
        }
    }
}

import Foundation

/// Splits a recording into decoder-sized chunks for `GigaAMEngine`.
///
/// GigaAM-v3's e2e-RNNT graph decodes a whole utterance in one pass and degrades
/// badly on long inputs (the encoder is not streaming, and its state is not
/// carried across calls), so the file has to be cut before it reaches the
/// recognizer. Cutting on a fixed grid would slice mid-word; cutting on the
/// quietest moment inside a bounded window keeps the boundary at a pause
/// wherever the audio offers one, and degrades to "the least loud point" when it
/// doesn't — which is the best a boundary can do without a VAD.
///
/// Pure and value-typed on purpose: the boundary policy is the part worth
/// pinning in tests, and it needs no model, no file and no actor to exercise.
enum GigaAMChunking {
    /// Hard ceiling on a chunk. Chosen against GigaAM's own long-form limit, not
    /// against a memory budget.
    static let maxChunkSeconds: Double = 20

    /// No cut is searched for before this point, so a chunk is never shorter
    /// than this except for the final remainder.
    static let minChunkSeconds: Double = 15

    /// Width of the window whose energy decides the cut. Short enough to sit
    /// inside a between-word pause, long enough not to lock onto a single
    /// zero-crossing.
    static let quietWindowSeconds: Double = 0.1

    /// How far the search window advances between energy measurements. Half the
    /// window, so every position is covered by some measurement.
    private static let searchHopSeconds: Double = 0.05

    /// Chunk boundaries as half-open sample ranges, contiguous and covering
    /// `samples` exactly.
    ///
    /// - Parameters:
    ///   - samples: 16 kHz mono PCM in [-1, 1].
    ///   - sampleRate: samples per second; must be positive.
    static func chunks(samples: [Float], sampleRate: Int) -> [Range<Int>] {
        guard sampleRate > 0, !samples.isEmpty else { return [] }

        let maxChunk = Int(maxChunkSeconds * Double(sampleRate))
        let minChunk = Int(minChunkSeconds * Double(sampleRate))
        let window = max(1, Int(quietWindowSeconds * Double(sampleRate)))
        let hop = max(1, Int(searchHopSeconds * Double(sampleRate)))

        var result: [Range<Int>] = []
        var start = 0
        while start < samples.count {
            // The tail fits in one pass — emit it whole rather than forcing a
            // cut that would leave a scrap behind.
            guard samples.count - start > maxChunk else {
                result.append(start ..< samples.count)
                break
            }
            let cut = quietestCut(
                in: samples,
                searchFrom: start + minChunk,
                searchTo: start + maxChunk,
                window: window,
                hop: hop,
            )
            result.append(start ..< cut)
            start = cut
        }
        return result
    }

    /// Midpoint of the lowest-energy `window`-wide slice whose start lies in
    /// `searchFrom ..< searchTo`. Returns a value strictly greater than
    /// `searchFrom - 1`, so the caller always makes progress.
    private static func quietestCut(
        in samples: [Float],
        searchFrom: Int,
        searchTo: Int,
        window: Int,
        hop: Int,
    ) -> Int {
        // Keep the measured window inside the buffer; `searchTo` is a chunk
        // ceiling, not a promise that a full window fits after it.
        let lastStart = min(searchTo, samples.count - window)
        guard lastStart > searchFrom else { return min(searchTo, samples.count) }

        var bestStart = searchFrom
        var bestEnergy = Float.greatestFiniteMagnitude
        var position = searchFrom
        while position <= lastStart {
            var energy: Float = 0
            for index in position ..< (position + window) {
                let value = samples[index]
                energy += value * value
            }
            if energy < bestEnergy {
                bestEnergy = energy
                bestStart = position
            }
            position += hop
        }
        return min(bestStart + window / 2, samples.count)
    }
}

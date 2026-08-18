import CoreMedia
import Foundation
import Testing
@testable import CirceKit

/// End-to-end Core AI runs, over every export bundle present on this machine.
///
/// Gated on the bundles being present: they are multi-gigabyte local directories
/// and cannot be fetched. Point `CIRCEKIT_COREAI_EXPORTS` at them.
///
/// The two architectures behave differently under the hood — Parakeet TDT is a
/// transducer, Whisper is encoder/decoder with a KV cache — so running both
/// through the same assertions is what proves the wrapper is not accidentally
/// specialized to one of them.
@Suite("Core AI backend", .enabled(if: TestEnv.hasCoreAIExports), .serialized)
struct CoreAIBackendTests {
    @Test("Transcribes the JFK clip accurately", arguments: TestEnv.availableCoreAIModels)
    func transcribesAccurately(model: CoreAIModel) async throws {
        let transcriber = CirceTranscriber(
            backend: .coreAI(model),
            locale: Locale(identifier: "en_US"),
            preset: .transcription
        )
        let (text, results) = try await TestEnv.transcribeJFK(transcriber)

        // Batch engine: the whole clip comes back as a single final result.
        #expect(results.count == 1)
        #expect(results.allSatisfy { $0.isFinal })

        let wer = TestEnv.jfkWER(text)
        #expect(wer < 0.2, "\(model) WER \(wer) too high for: \(text)")
    }

    @Test("The result spans the whole clip", arguments: TestEnv.availableCoreAIModels)
    func resultSpansClip(model: CoreAIModel) async throws {
        let transcriber = CirceTranscriber(
            backend: .coreAI(model),
            locale: Locale(identifier: "en_US"),
            preset: .transcription
        )
        let (_, results) = try await TestEnv.transcribeJFK(transcriber)
        let duration = try AudioDecoder.duration(of: TestEnv.jfkURL)

        let result = try #require(results.first)
        #expect(result.range.start == .zero)
        #expect(abs(result.range.end.seconds - duration) < 0.2)
    }

    @Test("Decode statistics are reported", arguments: TestEnv.availableCoreAIModels)
    func reportsDecodeStats(model: CoreAIModel) async throws {
        let transcriber = CirceTranscriber(
            backend: .coreAI(model),
            locale: Locale(identifier: "en_US"),
            preset: .transcription
        )
        _ = try await TestEnv.transcribeJFK(transcriber)

        let stats = try #require(transcriber.coreAIStats)
        #expect(stats.windowCount >= 1)
        #expect(stats.stepCount > 0)
    }

    @Test("Timing options are reported unsupported", arguments: TestEnv.availableCoreAIModels)
    func reportsUnsupportedOptions(model: CoreAIModel) {
        // Neither architecture emits timings, and both must say so rather than
        // returning results that silently lack the requested attributes.
        let transcriber = CirceTranscriber(
            backend: .coreAI(model),
            locale: Locale(identifier: "en_US"),
            preset: .timeIndexedTranscription
        )
        #expect(transcriber.unsupportedOptions == [.audioTimeRange])
    }
}

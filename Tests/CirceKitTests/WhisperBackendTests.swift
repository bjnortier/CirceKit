import AVFoundation
import CoreMedia
import Foundation
import Testing
@testable import CirceKit

/// End-to-end whisper.cpp runs over the bundled JFK clip.
///
/// Gated on the model being available: already cached, or `CIRCEKIT_TEST_NETWORK=1`
/// to permit a download. `CIRCEKIT_MODEL_DIR` points at an existing model set.
@Suite("whisper.cpp backend", .enabled(if: TestEnv.canRunWhisper), .serialized)
struct WhisperBackendTests {
    @Test("Transcribes the JFK clip accurately")
    func transcribesAccurately() async throws {
        let transcriber = CirceTranscriber(
            backend: .whisperCPP(TestEnv.testModel),
            locale: Locale(identifier: "en_US"),
            preset: .transcription
        )
        let (text, results) = try await TestEnv.transcribeJFK(transcriber)

        #expect(!results.isEmpty)
        #expect(!text.isEmpty)

        let wer = TestEnv.jfkWER(text)
        #expect(wer < 0.2, "WER \(wer) too high for: \(text)")
    }

    @Test("Segment ranges are ordered and within the clip")
    func segmentRangesAreSane() async throws {
        let transcriber = CirceTranscriber(
            backend: .whisperCPP(TestEnv.testModel),
            locale: Locale(identifier: "en_US"),
            preset: .transcription
        )
        let (_, results) = try await TestEnv.transcribeJFK(transcriber)
        let duration = try AudioDecoder.duration(of: TestEnv.jfkURL)

        var previousEnd = CMTime.zero
        for result in results {
            #expect(result.range.duration.seconds >= 0)
            // Segments arrive in order and do not overlap.
            #expect(result.range.start.seconds >= previousEnd.seconds - 0.001)
            previousEnd = result.range.end
        }
        // Whisper can round the final segment slightly past the true end.
        #expect(previousEnd.seconds <= duration + 0.5)
    }

    @Test("Batch decoding marks every result final")
    func allResultsAreFinal() async throws {
        let transcriber = CirceTranscriber(
            backend: .whisperCPP(TestEnv.testModel),
            locale: Locale(identifier: "en_US"),
            preset: .transcription
        )
        let (_, results) = try await TestEnv.transcribeJFK(transcriber)
        // whisper.cpp decodes the whole clip in one pass; nothing is volatile.
        #expect(results.allSatisfy { $0.isFinal })
    }

    @Test("Word timings land inside their segment")
    func wordTimingsAreAttached() async throws {
        let transcriber = CirceTranscriber(
            backend: .whisperCPP(TestEnv.testModel),
            locale: Locale(identifier: "en_US"),
            preset: .timeIndexedTranscription
        )
        let (_, results) = try await TestEnv.transcribeJFK(transcriber)

        // Nothing was requested that this backend cannot deliver.
        #expect(transcriber.unsupportedOptions.isEmpty)

        var timedRuns = 0
        for result in results {
            for run in result.text.runs {
                guard let range = run.audioTimeRange else { continue }
                timedRuns += 1
                #expect(range.start.seconds >= result.range.start.seconds - 0.5)
                #expect(range.end.seconds <= result.range.end.seconds + 0.5)
            }
        }
        #expect(timedRuns > 0, "expected per-token time ranges")
    }

    @Test("Confidence is attached and in range")
    func confidenceIsAttached() async throws {
        let transcriber = CirceTranscriber(
            backend: .whisperCPP(TestEnv.testModel),
            locale: Locale(identifier: "en_US"),
            transcriptionOptions: [],
            reportingOptions: [],
            attributeOptions: [.transcriptionConfidence]
        )
        let (_, results) = try await TestEnv.transcribeJFK(transcriber)

        var scored = 0
        for result in results {
            for run in result.text.runs {
                guard let confidence = run.transcriptionConfidence else { continue }
                scored += 1
                #expect(confidence >= 0 && confidence <= 1)
            }
        }
        #expect(scored > 0, "expected per-token confidences")
    }
}

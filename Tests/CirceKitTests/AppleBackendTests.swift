import CoreMedia
import Foundation
import Testing
@testable import CirceKit

/// End-to-end Apple `SpeechAnalyzer` runs.
///
/// Skipped when English assets are not installed — the test never triggers a
/// multi-hundred-megabyte system download on its own.
@Suite("Apple backend", .serialized)
struct AppleBackendTests {
    @Test("Transcribes the JFK clip accurately")
    func transcribesAccurately() async throws {
        guard await TestEnv.appleEnglishInstalled() else { return }

        let transcriber = CirceTranscriber(
            backend: .apple,
            locale: Locale(identifier: "en_US"),
            preset: .transcription
        )
        let (text, results) = try await TestEnv.transcribeJFK(transcriber)

        #expect(!results.isEmpty)
        let wer = TestEnv.jfkWER(text)
        #expect(wer < 0.2, "WER \(wer) too high for: \(text)")
    }

    @Test("Word timings survive the mapping from Apple's results")
    func wordTimingsPassThrough() async throws {
        guard await TestEnv.appleEnglishInstalled() else { return }

        let transcriber = CirceTranscriber(
            backend: .apple,
            locale: Locale(identifier: "en_US"),
            preset: .timeIndexedTranscription
        )
        let (_, results) = try await TestEnv.transcribeJFK(transcriber)

        #expect(transcriber.unsupportedOptions.isEmpty)

        // CirceKit reuses Apple's own attribute scope, so the attributes on the
        // AttributedString arrive unchanged.
        let timedRuns = results.reduce(0) { total, result in
            total + result.text.runs.count { $0.audioTimeRange != nil }
        }
        #expect(timedRuns > 0)
    }

    @Test("Reports supported locales")
    func reportsSupportedLocales() async throws {
        let supported = await CirceTranscriber.supportedLocales(for: .apple)
        guard !supported.isEmpty else { return }
        #expect(supported.contains { $0.language.languageCode?.identifier == "en" })
    }
}

import CoreMedia
import Foundation
import Testing
import os

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

    @Test("Cancelling stops the analyzer instead of waiting out the file")
    func cancelsPromptly() async throws {
        guard await TestEnv.appleEnglishInstalled() else { return }

        let directory = URL.temporaryDirectory.appending(path: "circe-apple-cancel-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appending(path: "long.wav")
        let seconds = 600.0
        try TestEnv.writeClip(seconds: seconds, to: url)

        let transcriber = CirceFileTranscriber(backend: .apple, locale: Locale(identifier: "en_US"))
        try await transcriber.prepare()
        let emitted = OSAllocatedUnfairLock<[CirceTranscriber.Result]>(initialState: [])
        let cancelledAt = OSAllocatedUnfairLock<ContinuousClock.Instant?>(initialState: nil)
        let handle = OSAllocatedUnfairLock<Task<CirceTranscription, Error>?>(initialState: nil)
        let task = Task {
            try await transcriber.transcribe(fileAt: url) { result in
                emitted.withLock { $0.append(result) }
                // Results arrive on the backend's collector, not the calling task,
                // so cancel the caller through its handle on the first one.
                cancelledAt.withLock { $0 = $0 ?? .now }
                handle.withLock { $0 }?.cancel()
            }
        }
        handle.withLock { $0 = task }
        do {
            _ = try await task.value
            Issue.record("Cancelled analysis returned success")
        } catch is CancellationError {
            // Cancellation must not surface as an analyzer failure.
        }

        let elapsed = try #require(cancelledAt.withLock { $0 }).duration(to: .now)
        // The whole clip takes far longer than this; a cancel that waited it out would not.
        #expect(elapsed < .seconds(5), "cancel took \(elapsed)")
        let latest = emitted.withLock { $0 }.map(\.range.end.seconds).max() ?? 0
        #expect(latest < seconds / 2)

        // The next file runs on a fresh analyzer and is unaffected.
        let result = try await transcriber.transcribe(fileAt: TestEnv.jfkURL)
        #expect(TestEnv.jfkWER(result.text) < 0.2)
    }

    @Test("Reports supported locales")
    func reportsSupportedLocales() async throws {
        let supported = await CirceTranscriber.supportedLocales(for: .apple)
        guard !supported.isEmpty else { return }
        #expect(supported.contains { $0.language.languageCode?.identifier == "en" })
    }
}

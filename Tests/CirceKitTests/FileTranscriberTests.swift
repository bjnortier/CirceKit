import CoreMedia
import Foundation
import Testing
@testable import CirceKit

/// `CirceFileTranscriber` is the batch entry point: one loaded model, many files.
///
/// The reuse test is the important one. `CirceTranscriber` is single-use by
/// design, and every backend had to be checked for run-once state before batch
/// work was possible — `AppleBackend` in particular, whose `SpeechTranscriber`
/// results sequence terminates once a run finalizes.
@Suite("File transcriber", .serialized)
struct FileTranscriberTests {
    /// The backends this machine can run right now.
    static func availableBackends() async -> [CirceTranscriber.Backend] {
        var backends: [CirceTranscriber.Backend] = []
        if await TestEnv.appleEnglishInstalled() { backends.append(.apple) }
        if TestEnv.canRunWhisper { backends.append(.whisperCPP(TestEnv.testModel)) }
        backends.append(contentsOf: TestEnv.availableCoreAIModels.map { .coreAI($0) })
        return backends
    }

    @Test("Transcribes the JFK clip")
    func transcribesOnce() async throws {
        for backend in await Self.availableBackends() {
            let transcriber = CirceFileTranscriber(
                backend: backend,
                locale: Locale(identifier: "en_US"),
                preset: .transcription
            )
            let transcription = try await transcriber.transcribe(fileAt: TestEnv.jfkURL)

            #expect(!transcription.text.isEmpty, "\(backend) produced no text")
            #expect(!transcription.results.isEmpty)
            #expect(abs(transcription.audioDuration.seconds - 11.0) < 1.5)

            let wer = TestEnv.jfkWER(transcription.text)
            #expect(wer < 0.2, "\(backend) WER \(wer): \(transcription.text)")
        }
    }

    /// The premise of the whole batch API: a second file must reuse the loaded
    /// model, not silently return nothing (Apple) or reload it (Core AI).
    @Test("Reuses one loaded model across repeated files")
    func reusesLoadedModel() async throws {
        for backend in await Self.availableBackends() {
            let transcriber = CirceFileTranscriber(
                backend: backend,
                locale: Locale(identifier: "en_US"),
                preset: .transcription
            )
            // Pay the model load up front so it is outside both measurements.
            try await transcriber.prepare()

            let clock = ContinuousClock()

            let firstStart = clock.now
            let first = try await transcriber.transcribe(fileAt: TestEnv.jfkURL)
            let firstElapsed = clock.now - firstStart

            let secondStart = clock.now
            let second = try await transcriber.transcribe(fileAt: TestEnv.jfkURL)
            let secondElapsed = clock.now - secondStart

            let thirdStart = clock.now
            let third = try await transcriber.transcribe(fileAt: TestEnv.jfkURL)
            let thirdElapsed = clock.now - thirdStart

            // Same audio, same model — the text must not drift between runs.
            #expect(!second.text.isEmpty, "\(backend) returned nothing on the second run")
            #expect(!third.text.isEmpty, "\(backend) returned nothing on the third run")
            #expect(second.text == first.text, "\(backend) run 2 differs: \(second.text)")
            #expect(third.text == first.text, "\(backend) run 3 differs: \(third.text)")

            // A reload would show up as a large jump, not the small variance of
            // an already-warm model.
            #expect(
                secondElapsed < firstElapsed * 4 + .seconds(1),
                "\(backend) run 2 (\(secondElapsed)) looks like a reload vs run 1 (\(firstElapsed))"
            )
            #expect(
                thirdElapsed < firstElapsed * 4 + .seconds(1),
                "\(backend) run 3 (\(thirdElapsed)) looks like a reload vs run 1 (\(firstElapsed))"
            )
        }
    }

    @Test("Reports Core AI stats and unsupported options")
    func reportsCapabilities() async throws {
        guard let model = TestEnv.availableCoreAIModels.first else { return }
        let transcriber = CirceFileTranscriber(
            backend: .coreAI(model),
            locale: Locale(identifier: "en_US"),
            preset: .timeIndexedTranscription
        )
        // Core AI emits no timings, and must say so rather than returning results
        // that silently lack the requested attributes.
        #expect(await transcriber.unsupportedOptions == [.audioTimeRange])

        _ = try await transcriber.transcribe(fileAt: TestEnv.jfkURL)
        let stats = try #require(await transcriber.coreAIStats)
        #expect(stats.windowCount >= 1)
    }
}

import Foundation
import Testing

@testable import CirceKit

/// Translation is one token of Whisper's decoder prefix, so only Whisper can
/// offer it — Core AI exports, and multilingual non-turbo whisper.cpp models.
/// The failure mode when it is not honoured is the worst kind: a fluent,
/// plausible transcript in the language that was spoken, which nothing in the
/// output marks as untranslated. These tests pin the loud paths.
@Suite("Translation to English")
struct TranslationTests {
    // MARK: Backend capability

    @Test("Only Whisper backends claim translation")
    func onlyWhisperTranslates() {
        #expect(CirceTranscriber.Backend.coreAI(.whisperLargeV3Turbo).canTranslateToEnglish)
        #expect(CirceTranscriber.Backend.whisperCPP(.tiny).canTranslateToEnglish)
        #expect(!CirceTranscriber.Backend.apple.canTranslateToEnglish)
    }

    /// English-only models have no task to switch, and turbo ignores the switch.
    @Test("whisper.cpp translates only with multilingual, non-turbo models")
    func whisperCPPTranslationFollowsTheModel() {
        let translating = WhisperModel.allCases.filter(\.canTranslateToEnglish)
        #expect(Set(translating) == [.tiny, .base, .small, .medium, .largeV3])
    }

    @Test("Transcribing is available on every backend")
    func transcriptionNeedsNoCapability() throws {
        for backend: CirceTranscriber.Backend in [
            .apple, .whisperCPP(.tinyEN), .coreAI(.whisperLargeV3Turbo),
        ] {
            #expect(throws: Never.self) { try backend.validateTranslation(false) }
        }
    }

    @Test("A backend that cannot translate says so rather than transcribing")
    func rejectsUnsupportedBackends() {
        for backend: CirceTranscriber.Backend in [.apple, .whisperCPP(.tinyEN), .whisperCPP(.largeV3Turbo)] {
            #expect(throws: CirceError.self) { try backend.validateTranslation(true) }
        }
        #expect(throws: Never.self) {
            try CirceTranscriber.Backend.coreAI(.whisperLargeV3Turbo).validateTranslation(true)
            try CirceTranscriber.Backend.whisperCPP(.tiny).validateTranslation(true)
        }
    }

    // MARK: Defaults

    @Test("Transcribers do not translate unless asked")
    func defaultsToTranscription() async {
        let transcriber = CirceTranscriber(backend: .apple)
        #expect(!transcriber.translatesToEnglish)

        let fileTranscriber = CirceFileTranscriber(backend: .apple)
        #expect(await !fileTranscriber.translatesToEnglish)
    }

    @Test("The request survives construction")
    func carriesTheRequest() async {
        let transcriber = CirceTranscriber(
            backend: .coreAI(.whisperLargeV3Turbo), translatesToEnglish: true)
        #expect(transcriber.translatesToEnglish)

        // Through the preset initializer as well, which forwards to the designated one.
        let preset = CirceFileTranscriber(
            backend: .coreAI(.whisperLargeV3Turbo), preset: .transcription,
            translatesToEnglish: true)
        #expect(await preset.translatesToEnglish)
    }

    // MARK: Preparation

    @Test("Preparing to translate on an unsupported backend throws")
    func prepareRejectsUnsupportedBackend() async {
        let transcriber = CirceFileTranscriber(backend: .apple, translatesToEnglish: true)
        await #expect(throws: CirceError.self) { try await transcriber.prepare() }
    }

    /// Whisper writes English out of the same decoder pass, so a translated run
    /// costs what a transcription of the same audio costs — and English audio
    /// comes back unchanged, since the task's target *is* English.
    @Test(
        "Translating English audio is still English",
        .enabled(if: CoreAIModel.whisperLargeV3Turbo.resolvedURL != nil))
    func translatesEnglishAudio() async throws {
        let transcriber = CirceFileTranscriber(
            backend: .coreAI(.whisperLargeV3Turbo),
            locale: Locale(identifier: "en-US"),
            translatesToEnglish: true)
        let result = try await transcriber.transcribe(fileAt: TestEnv.jfkURL)
        #expect(TestEnv.jfkWER(result.text) < 0.2)
    }

    /// Parakeet is a transducer with no task slot to put `<|translate|>` in.
    /// Loud, because the alternative is an untranslated transcript that reads as
    /// a successful run.
    @Test(
        "Parakeet refuses to translate rather than transcribing",
        .enabled(if: CoreAIModel.parakeetTDT06BV3.resolvedURL != nil))
    func parakeetRefusesToTranslate() async throws {
        let transcriber = CirceFileTranscriber(
            backend: .coreAI(.parakeetTDT06BV3),
            locale: Locale(identifier: "en-US"),
            translatesToEnglish: true)
        await #expect(throws: (any Error).self) {
            _ = try await transcriber.transcribe(fileAt: TestEnv.jfkURL)
        }
    }
}

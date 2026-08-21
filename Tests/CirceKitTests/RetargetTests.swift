import Foundation
import Testing

@testable import CirceKit

/// Re-targeting is what lets one loaded model transcribe a corpus that spans
/// languages. Getting it wrong is silent — an engine told the wrong language
/// does not fail, it transcribes as if the audio were in that language — so the
/// rules are pinned here rather than left to a benchmark to discover.
@Suite("Language retargeting")
struct RetargetTests {
    private let french = Locale(identifier: "fr-FR")
    private let english = Locale(identifier: "en-US")

    @Test("A Core AI backend takes a new language without reloading")
    func coreAIRetargets() {
        // No bundle needed: the language is a decoder prefix chosen per run, not
        // something the load bakes in, which is the property under test.
        let backend = CoreAIBackend(model: .bundle(URL(filePath: "/nonexistent")), locale: english)
        #expect(backend.retarget(locale: french))
    }

    @Test("A multilingual whisper.cpp model takes a new language")
    func whisperRetargets() {
        let backend = WhisperBackend(
            model: .largeV3Turbo, locale: english, attributeOptions: []
        )
        #expect(backend.retarget(locale: french))
    }

    @Test("An English-only whisper.cpp model refuses another language")
    func englishOnlyWhisperRefuses() {
        let backend = WhisperBackend(model: .baseEN, locale: english, attributeOptions: [])
        #expect(backend.retarget(locale: french) == false)
        #expect(backend.retarget(locale: Locale(identifier: "en-GB")))
    }

    @Test("Apple's transcriber refuses: its locale is fixed at construction")
    func appleRefuses() {
        let backend = AppleBackend(
            locale: english, transcriptionOptions: [], reportingOptions: [], attributeOptions: []
        )
        #expect(backend.retarget(locale: french) == false)
    }
}

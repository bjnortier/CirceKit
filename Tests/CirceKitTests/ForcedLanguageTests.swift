import Foundation
import Testing
@testable import CirceKit

/// A Core AI Whisper export pins its decoder language and cannot be retargeted at
/// run time. Fed another language it produces fluent English rather than failing,
/// so the pin has to be reported or a benchmark will publish it as a model score.
@Suite("Forced language", .enabled(if: TestEnv.hasCoreAIExports))
struct ForcedLanguageTests {
    @Test("Whisper exports report the language they pin")
    func whisperPinsEnglish() throws {
        guard CoreAIModel.whisperLargeV3Turbo.resolvedURL != nil else { return }
        let forced = try #require(CoreAIModel.whisperLargeV3Turbo.forcedLanguage)
        #expect(forced.language.languageCode?.identifier == "en")

        // The pin is the export's default, not a limit: CoreAISpeech retargets the
        // decoder prefix, so any language in the bundle's vocabulary is available.
        for id in ["en-US", "en-GB", "fr-FR", "de-DE", "es-419", "ru-RU"] {
            #expect(
                CoreAIModel.whisperLargeV3Turbo.supports(Locale(identifier: id)),
                "\(id) should be supported via prefix retargeting")
        }
        // A language Whisper genuinely has no token for.
        #expect(!CoreAIModel.whisperLargeV3Turbo.supports(Locale(identifier: "zz")))
    }

    @Test("Parakeet pins nothing and accepts every language")
    func parakeetIsMultilingual() throws {
        guard CoreAIModel.parakeetTDT06BV3.resolvedURL != nil else { return }
        #expect(CoreAIModel.parakeetTDT06BV3.forcedLanguage == nil)
        for id in ["en-US", "fr-FR", "de-DE", "es-419"] {
            #expect(CoreAIModel.parakeetTDT06BV3.supports(Locale(identifier: id)))
        }
    }

    @Test("A missing bundle reports no pin rather than crashing")
    func missingBundle() {
        let missing = CoreAIModel.bundle(URL(filePath: "/nonexistent/circe-bundle"))
        #expect(missing.forcedLanguage == nil)
        #expect(missing.supports(Locale(identifier: "fr-FR")))
    }
}

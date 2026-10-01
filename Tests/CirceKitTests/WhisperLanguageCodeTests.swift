import Foundation
import Testing

@testable import CirceKit

/// The language reaches whisper.cpp as a code, through `Locale`, which renames some.
@Suite("whisper.cpp language codes")
struct WhisperLanguageCodeTests {
    @Test("Every language whisper.cpp lists reaches it as a code it knows")
    func everySupportedLanguageRoundTrips() async {
        let locales = await CirceTranscriber.supportedLocales(for: .whisperCPP(.small))
        #expect(locales.count > 90)
        for locale in locales {
            let code = WhisperBackend.whisperLanguageCode(for: locale, englishOnly: false)
            #expect(code.map(WhisperBackend.knowsLanguage) == true, "\(locale.identifier) → \(code ?? "nil")")
        }
    }

    @Test("Codes Locale renames go back to whisper.cpp's")
    func renamedCodesMapBack() {
        func code(_ identifier: String, englishOnly: Bool = false) -> String? {
            WhisperBackend.whisperLanguageCode(for: Locale(identifier: identifier), englishOnly: englishOnly)
        }
        #expect(code("no") == "no")
        #expect(code("tl") == "tl")
        #expect(code("jw") == "jw")
        #expect(code("de_DE") == "de")
        #expect(code("und") == nil)
        #expect(code("de", englishOnly: true) == "en")
    }
}

import AVFoundation
import Foundation
import Testing
@testable import CirceKit

/// Shared fixtures and environment probes.
///
/// Model-backed suites gate on these rather than failing on a machine that has
/// no models: `swift test` should be fast and green everywhere, with the heavy
/// runs opted into explicitly.
enum TestEnv {
    /// The bundled JFK clip: 11 s of speech, the canonical whisper.cpp sample.
    static var jfkURL: URL {
        Bundle.module.url(forResource: "jfk", withExtension: "wav", subdirectory: "Resources")
            ?? Bundle.module.url(forResource: "jfk", withExtension: "wav")!
    }

    /// Ground truth for `jfk.wav`.
    static let jfkReference = """
        And so my fellow Americans, ask not what your country can do for you, \
        ask what you can do for your country.
        """

    /// Whether downloading models is permitted. Off by default.
    static var allowsNetwork: Bool {
        ProcessInfo.processInfo.environment["CIRCEKIT_TEST_NETWORK"] == "1"
    }

    /// The small model used for backend tests.
    static let testModel: WhisperModel = .tinyEN

    /// Whether a whisper run is possible: either the model is already on disk,
    /// or we are allowed to fetch it.
    static var canRunWhisper: Bool {
        CirceModelStore.shared.isDownloaded(testModel) || allowsNetwork
    }

    /// The Core AI export bundles present on this machine.
    ///
    /// Each bundle is multi-gigabyte and cannot be fetched, so suites run over
    /// whichever ones happen to be there rather than requiring all of them.
    static let availableCoreAIModels: [CoreAIModel] = [
        .parakeetTDT06BV3,
        .whisperLargeV3Turbo,
    ].filter { $0.resolvedURL != nil }

    /// Whether any Core AI export bundle is present.
    static var hasCoreAIExports: Bool { !availableCoreAIModels.isEmpty }

    /// Whether Apple's transcriber has assets installed for English.
    static func appleEnglishInstalled() async -> Bool {
        guard await CirceTranscriber.isAvailable(.apple) else { return false }
        let installed = await CirceTranscriber.installedLocales(for: .apple)
        return installed.contains { $0.language.languageCode?.identifier == "en" }
    }

    /// Runs one transcriber over the JFK clip and returns the plain transcript
    /// plus every result, in emission order.
    static func transcribeJFK(
        _ transcriber: CirceTranscriber
    ) async throws -> (text: String, results: [CirceTranscriber.Result]) {
        let file = try AVAudioFile(forReading: jfkURL)
        let analyzer = CirceAnalyzer(modules: [transcriber])

        // The results sequence ends only when the analyzer finishes, so collect
        // it concurrently with driving the analyzer.
        let collector = Task {
            var collected: [CirceTranscriber.Result] = []
            for try await result in transcriber.results {
                collected.append(result)
            }
            return collected
        }

        try await analyzer.analyzeSequence(from: file)
        try await analyzer.finalizeAndFinishThroughEndOfInput()

        let results = try await collector.value
        // Only finals contribute to the transcript; volatiles are superseded.
        let text = results
            .filter(\.isFinal)
            .map { String($0.text.characters) }
            .joined()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (text, results)
    }

    /// WER against the JFK reference, using the English normalizer so that
    /// formatting differences between engines don't count as errors.
    static func jfkWER(_ hypothesis: String) -> Double {
        let normalizer = EnglishTextNormalizer.shared
        return WordErrorRate.compute(
            reference: jfkReference,
            hypothesis: hypothesis,
            tokenizer: normalizer.tokenize
        )
    }
}

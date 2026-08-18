import Foundation
import Testing
@testable import CirceKit

/// Runs every locally-available backend over the same clip and compares them.
///
/// This is the test that actually justifies the unified API: the same code path
/// drives all three engines, and the transcripts should agree.
@Suite("Cross-backend", .serialized)
struct CrossBackendTests {
    /// The backends this machine can actually run right now.
    static func availableBackends() async -> [CirceTranscriber.Backend] {
        var backends: [CirceTranscriber.Backend] = []
        if await TestEnv.appleEnglishInstalled() { backends.append(.apple) }
        if TestEnv.canRunWhisper { backends.append(.whisperCPP(TestEnv.testModel)) }
        backends.append(contentsOf: TestEnv.availableCoreAIModels.map { .coreAI($0) })
        return backends
    }

    @Test("Every available backend agrees with the reference and with each other")
    func backendsAgree() async throws {
        let backends = await Self.availableBackends()
        // Nothing to compare on a machine with no models at all.
        guard backends.count >= 1 else { return }

        var transcripts: [(backend: CirceTranscriber.Backend, text: String)] = []
        for backend in backends {
            let transcriber = CirceTranscriber(
                backend: backend,
                locale: Locale(identifier: "en_US"),
                preset: .transcription
            )
            let (text, _) = try await TestEnv.transcribeJFK(transcriber)
            transcripts.append((backend, text))

            let wer = TestEnv.jfkWER(text)
            #expect(wer < 0.2, "\(backend) WER \(wer): \(text)")
        }

        // Engines that both got the reference roughly right must also agree with
        // each other; a large pairwise gap means one of the wrappers is wrong.
        let normalizer = EnglishTextNormalizer.shared
        for i in transcripts.indices {
            for j in transcripts.indices where j > i {
                let wer = WordErrorRate.compute(
                    reference: transcripts[i].text,
                    hypothesis: transcripts[j].text,
                    tokenizer: normalizer.tokenize
                )
                #expect(
                    wer < 0.3,
                    "\(transcripts[i].backend) vs \(transcripts[j].backend) disagree (WER \(wer))"
                )
            }
        }
    }
}

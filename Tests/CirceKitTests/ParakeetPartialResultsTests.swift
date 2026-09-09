import AVFoundation
import Foundation
import Testing
import os
@testable import CirceKit

@Suite struct ParakeetPartialResultsTests {
    @Test(.enabled(if: CoreAIModel.parakeetTDT06BV3.resolvedURL != nil), .timeLimit(.minutes(5)))
    func decoderAndWindowResultsPreserveFinalTranscripts() async throws {
        let temporary = FileManager.default.temporaryDirectory.appending(path: "ParakeetPartials-\(UUID()).caf")
        defer { try? FileManager.default.removeItem(at: temporary) }
        do {
            let input = try AVAudioFile(forReading: TestEnv.jfkURL)
            let buffer = try #require(AVAudioPCMBuffer(
                pcmFormat: input.processingFormat, frameCapacity: AVAudioFrameCount(input.length)))
            try input.read(into: buffer)
            let output = try AVAudioFile(forWriting: temporary, settings: input.processingFormat.settings)
            for _ in 0..<6 { try output.write(from: buffer) }
        }
        let units = CoreAIComputeUnits(encoder: .gpu, decoder: .cpu)
        let baseline = CirceFileTranscriber(
            backend: .coreAI(.parakeetTDT06BV3), locale: Locale(identifier: "und"),
            preset: .transcription, coreAIComputeUnits: units)
        let progressive = CirceFileTranscriber(
            backend: .coreAI(.parakeetTDT06BV3), locale: Locale(identifier: "und"),
            preset: .progressiveTranscription, coreAIComputeUnits: units)
        for url in [TestEnv.jfkURL, temporary] {
            let expected = try await baseline.transcribe(fileAt: url)
            let updates = OSAllocatedUnfairLock<[CirceTranscriber.Result]>(initialState: [])
            let actual = try await progressive.transcribe(fileAt: url) { update in
                updates.withLock { $0.append(update) }
            }
            let emitted = updates.withLock { $0 }
            #expect(!expected.text.isEmpty)
            #expect(actual.text == expected.text)
            #expect(actual.results.count == 1)
            #expect(expected.results.count == 1)
            #expect(emitted.last?.isFinal == true)
            let progress = emitted.compactMap(\.progress)
            #expect(progress.count == emitted.count)
            #expect(progress.allSatisfy { (0...1).contains($0) })
            #expect(zip(progress, progress.dropFirst()).allSatisfy { $0 <= $1 })
            #expect(progress.last == 1)
            #expect(emitted.contains { $0.partialSource == .decoder && !$0.isFinal })
            let windows = emitted.filter { $0.partialSource == .window }
            #expect(windows.count == (url == temporary ? 3 : 1))
            #expect(windows.allSatisfy { !$0.isFinal })
            if url == temporary {
                let firstProgress = try #require(windows.first?.progress)
                #expect(abs(firstProgress - 30 / actual.audioDuration.seconds) < 0.001)
            }
            #expect(String(try #require(windows.last).text.characters) == actual.text)
        }
    }
}

import AVFoundation
import CoreMedia
import Foundation
import Testing
@testable import CirceKit

/// Pure-semantics tests: no models, no audio decoding, no network.
@Suite("API parity")
struct APIParityTests {
    // MARK: isFinal

    @Test("A result finalized at or past its end is final")
    func finalWhenFinalizationReachesEnd() {
        let result = CirceTranscriber.Result(
            range: CMTimeRange(start: .zero, end: CMTime(seconds: 2, preferredTimescale: 100)),
            resultsFinalizationTime: CMTime(seconds: 2, preferredTimescale: 100),
            text: AttributedString("hello")
        )
        #expect(result.isFinal)
    }

    @Test("A result finalized before its end is volatile")
    func volatileWhenFinalizationLagsEnd() {
        let result = CirceTranscriber.Result(
            range: CMTimeRange(start: .zero, end: CMTime(seconds: 2, preferredTimescale: 100)),
            resultsFinalizationTime: CMTime(seconds: 1, preferredTimescale: 100),
            text: AttributedString("hel")
        )
        #expect(!result.isFinal)
    }

    // MARK: Presets

    @Test("Presets carry the options their names promise")
    func presetsMatchNames() {
        #expect(CirceTranscriber.Preset.transcription.attributeOptions.isEmpty)
        #expect(CirceTranscriber.Preset.transcription.reportingOptions.isEmpty)

        #expect(CirceTranscriber.Preset.timeIndexedTranscription.attributeOptions == [.audioTimeRange])

        #expect(CirceTranscriber.Preset.progressiveTranscription.reportingOptions == [.volatileResults])

        let timeIndexedProgressive = CirceTranscriber.Preset.timeIndexedProgressiveTranscription
        #expect(timeIndexedProgressive.reportingOptions == [.volatileResults])
        #expect(timeIndexedProgressive.attributeOptions == [.audioTimeRange])
    }

    @Test("A preset initializer and the explicit initializer agree")
    func presetInitializerMatchesExplicit() {
        let fromPreset = CirceTranscriber(
            backend: .whisperCPP(.tinyEN),
            locale: Locale(identifier: "en_US"),
            preset: .timeIndexedTranscription
        )
        #expect(fromPreset.attributeOptions == [.audioTimeRange])
        #expect(fromPreset.reportingOptions.isEmpty)
    }

    // MARK: Capability reporting

    @Test("Core AI reports timing options as unsupported rather than dropping them")
    func coreAIReportsUnsupportedOptions() {
        let transcriber = CirceTranscriber(
            backend: .coreAI(.parakeetTDT06BV3),
            locale: Locale(identifier: "en_US"),
            preset: .timeIndexedTranscription
        )
        // The engine produces no timings at all, and says so.
        #expect(transcriber.unsupportedOptions == [.audioTimeRange])
    }

    @Test("Unrequested options are never reported as unsupported")
    func unrequestedOptionsAreNotReported() {
        let transcriber = CirceTranscriber(
            backend: .coreAI(.parakeetTDT06BV3),
            locale: Locale(identifier: "en_US"),
            preset: .transcription
        )
        #expect(transcriber.unsupportedOptions.isEmpty)
    }

    @Test("whisper.cpp supports every attribute option")
    func whisperSupportsAllAttributes() {
        let transcriber = CirceTranscriber(
            backend: .whisperCPP(.tinyEN),
            locale: Locale(identifier: "en_US"),
            transcriptionOptions: [],
            reportingOptions: [],
            attributeOptions: [.audioTimeRange, .transcriptionConfidence]
        )
        #expect(transcriber.unsupportedOptions.isEmpty)
    }

    // MARK: Analyzer

    @Test("An unknown module is rejected rather than silently ignored")
    func unknownModuleThrows() async throws {
        let analyzer = CirceAnalyzer(modules: [UnknownModule()])
        let stream = AsyncStream<CirceAnalyzerInput> { $0.finish() }

        await #expect(throws: CirceError.self) {
            try await analyzer.start(inputSequence: stream)
        }
    }

    @Test("analyzeSequence reports where the input ended")
    func analyzeSequenceReportsEndTime() async throws {
        let transcriber = CirceTranscriber(backend: .whisperCPP(.tinyEN))
        let analyzer = CirceAnalyzer(modules: [transcriber])

        // Three half-second buffers of silence: the run should report 1.5 s.
        let format = AudioDecoder.canonicalFormat
        let stream = AsyncStream<CirceAnalyzerInput> { continuation in
            for chunk in 0..<3 {
                let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 8_000)!
                buffer.frameLength = 8_000
                let start = CMTime(value: CMTimeValue(chunk * 8_000), timescale: 16_000)
                continuation.yield(CirceAnalyzerInput(buffer: buffer, bufferStartTime: start))
            }
            continuation.finish()
        }

        let end = try await analyzer.analyzeSequence(stream)
        #expect(end?.seconds == 1.5)
        await analyzer.cancelAndFinishNow()
    }

    @Test("Starting twice is rejected")
    func doubleStartThrows() async throws {
        let transcriber = CirceTranscriber(backend: .whisperCPP(.tinyEN))
        let analyzer = CirceAnalyzer(modules: [transcriber])

        try await analyzer.start(inputSequence: AsyncStream<CirceAnalyzerInput> { $0.finish() })
        await #expect(throws: CirceError.self) {
            try await analyzer.start(inputSequence: AsyncStream<CirceAnalyzerInput> { $0.finish() })
        }
        await analyzer.cancelAndFinishNow()
    }

    // MARK: Model resolution

    @Test("Whisper model filenames and URLs follow the ggml convention")
    func whisperModelNaming() {
        #expect(WhisperModel.tinyEN.filename == "ggml-tiny.en.bin")
        #expect(WhisperModel.largeV3Turbo.filename == "ggml-large-v3-turbo.bin")
        #expect(WhisperModel.tinyEN.isEnglishOnly)
        #expect(!WhisperModel.largeV3Turbo.isEnglishOnly)
        #expect(
            WhisperModel.tinyEN.remoteURL.absoluteString
                == "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
        )
    }

    @Test("A missing Core AI bundle throws rather than returning a bad path")
    func missingCoreAIBundleThrows() {
        let missing = CoreAIModel.bundle(URL(filePath: "/nonexistent/circekit-test-bundle"))
        #expect(missing.resolvedURL == nil)
        #expect(throws: CirceError.self) {
            _ = try missing.requireURL()
        }
    }
}

/// A module the analyzer has no way to drive.
private final class UnknownModule: CirceSpeechModule {
    struct Result: CirceSpeechModuleResult, Sendable {
        var range: CMTimeRange = .zero
        var resultsFinalizationTime: CMTime = .zero
    }

    var results: AsyncThrowingStream<Result, any Error> {
        AsyncThrowingStream { $0.finish() }
    }

    var availableCompatibleAudioFormats: [AVAudioFormat] {
        get async { [] }
    }
}

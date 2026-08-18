import AVFoundation
import CoreMedia
import Foundation
import Testing
@testable import CirceKit

/// End-to-end whisper.cpp runs over the bundled JFK clip.
///
/// Gated on the model being available: already cached, or `CIRCEKIT_TEST_NETWORK=1`
/// to permit a download. `CIRCEKIT_MODEL_DIR` points at an existing model set.
@Suite("whisper.cpp backend", .enabled(if: TestEnv.canRunWhisper), .serialized)
struct WhisperBackendTests {
    @Test("Transcribes the JFK clip accurately")
    func transcribesAccurately() async throws {
        let transcriber = CirceTranscriber(
            backend: .whisperCPP(TestEnv.testModel),
            locale: Locale(identifier: "en_US"),
            preset: .transcription
        )
        let (text, results) = try await TestEnv.transcribeJFK(transcriber)

        #expect(!results.isEmpty)
        #expect(!text.isEmpty)

        let wer = TestEnv.jfkWER(text)
        #expect(wer < 0.2, "WER \(wer) too high for: \(text)")
    }

    @Test("Segment ranges are ordered and within the clip")
    func segmentRangesAreSane() async throws {
        let transcriber = CirceTranscriber(
            backend: .whisperCPP(TestEnv.testModel),
            locale: Locale(identifier: "en_US"),
            preset: .transcription
        )
        let (_, results) = try await TestEnv.transcribeJFK(transcriber)
        let duration = try AudioDecoder.duration(of: TestEnv.jfkURL)

        var previousEnd = CMTime.zero
        for result in results {
            #expect(result.range.duration.seconds >= 0)
            // Segments arrive in order and do not overlap.
            #expect(result.range.start.seconds >= previousEnd.seconds - 0.001)
            previousEnd = result.range.end
        }
        // Whisper can round the final segment slightly past the true end.
        #expect(previousEnd.seconds <= duration + 0.5)
    }

    @Test("Batch decoding marks every result final")
    func allResultsAreFinal() async throws {
        let transcriber = CirceTranscriber(
            backend: .whisperCPP(TestEnv.testModel),
            locale: Locale(identifier: "en_US"),
            preset: .transcription
        )
        let (_, results) = try await TestEnv.transcribeJFK(transcriber)
        // whisper.cpp decodes the whole clip in one pass; nothing is volatile.
        #expect(results.allSatisfy { $0.isFinal })
    }

    /// whisper.cpp rounds segment ends to centiseconds and regularly reports a
    /// last segment ending *past* the end of the audio — 10.00s for a 9.90s clip.
    ///
    /// Finalization used to be set to the audio duration, so those results came
    /// back with `isFinal == false` and any caller filtering on it — including
    /// ``CirceFileTranscriber`` — silently returned an empty transcript for a
    /// perfectly good decode. One such sample in ten took a FLEURS English run
    /// from 3.3% to 18.9% WER.
    @Test(
        "Results stay final when a segment overruns the audio",
        arguments: [9.9, 10.4, 4.3, 7.7]
    )
    func finalWhenSegmentOverrunsAudio(seconds: Double) async throws {
        let directory = URL.temporaryDirectory.appending(path: "circe-overrun-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        // Loop the JFK clip to an awkward length, so the last segment's rounded end
        // is likely to sit beyond the true duration.
        let url = directory.appending(path: "clip.wav")
        try Self.writeClip(seconds: seconds, to: url)

        let transcriber = CirceFileTranscriber(
            backend: .whisperCPP(TestEnv.testModel),
            locale: Locale(identifier: "en_US"),
            preset: .transcription
        )
        let transcription = try await transcriber.transcribe(fileAt: url)

        #expect(!transcription.results.isEmpty, "no results for a \(seconds)s clip")
        // The invariant: a batch engine's output is final regardless of how its
        // segment ends round relative to the audio length.
        #expect(
            transcription.results.allSatisfy { $0.isFinal },
            "a result was left volatile for a \(seconds)s clip"
        )
        // And the joined transcript must actually survive the isFinal filter.
        #expect(!transcription.text.isEmpty, "empty transcript for a \(seconds)s clip")
    }

    @Test("Word timings land inside their segment")
    func wordTimingsAreAttached() async throws {
        let transcriber = CirceTranscriber(
            backend: .whisperCPP(TestEnv.testModel),
            locale: Locale(identifier: "en_US"),
            preset: .timeIndexedTranscription
        )
        let (_, results) = try await TestEnv.transcribeJFK(transcriber)

        // Nothing was requested that this backend cannot deliver.
        #expect(transcriber.unsupportedOptions.isEmpty)

        var timedRuns = 0
        for result in results {
            for run in result.text.runs {
                guard let range = run.audioTimeRange else { continue }
                timedRuns += 1
                #expect(range.start.seconds >= result.range.start.seconds - 0.5)
                #expect(range.end.seconds <= result.range.end.seconds + 0.5)
            }
        }
        #expect(timedRuns > 0, "expected per-token time ranges")
    }

    @Test("Confidence is attached and in range")
    func confidenceIsAttached() async throws {
        let transcriber = CirceTranscriber(
            backend: .whisperCPP(TestEnv.testModel),
            locale: Locale(identifier: "en_US"),
            transcriptionOptions: [],
            reportingOptions: [],
            attributeOptions: [.transcriptionConfidence]
        )
        let (_, results) = try await TestEnv.transcribeJFK(transcriber)

        var scored = 0
        for result in results {
            for run in result.text.runs {
                guard let confidence = run.transcriptionConfidence else { continue }
                scored += 1
                #expect(confidence >= 0 && confidence <= 1)
            }
        }
        #expect(scored > 0, "expected per-token confidences")
    }

    /// Writes a clip of `seconds` by looping the JFK sample.
    ///
    /// The `AVAudioFile` must go out of scope before the file is read: it
    /// finalizes its header on deallocation, and a still-open writer leaves a
    /// zero-length file behind.
    private static func writeClip(seconds: Double, to url: URL) throws {
        let base = try AudioDecoder.decodePCM16kMono(url: TestEnv.jfkURL)
        var pcm: [Float] = []
        let wanted = Int(seconds * AudioDecoder.targetSampleRate)
        while pcm.count < wanted { pcm.append(contentsOf: base.prefix(wanted - pcm.count)) }

        let format = AudioDecoder.canonicalFormat
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format, frameCapacity: AVAudioFrameCount(pcm.count)
        ) else {
            throw AudioDecoder.DecodeError.bufferAllocationFailed
        }
        buffer.frameLength = AVAudioFrameCount(pcm.count)
        pcm.withUnsafeBufferPointer {
            buffer.floatChannelData![0].update(from: $0.baseAddress!, count: pcm.count)
        }
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
    }

}

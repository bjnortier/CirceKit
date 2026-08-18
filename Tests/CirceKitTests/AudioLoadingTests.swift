import AVFoundation
import CoreMedia
import Foundation
import Testing
@testable import CirceKit

@Suite("Audio loading")
struct AudioLoadingTests {
    @Test("Decodes the JFK clip to 16 kHz mono float")
    func decodesToCanonicalFormat() throws {
        let samples = try AudioDecoder.decodePCM16kMono(url: TestEnv.jfkURL)

        // The clip is ~11 s; at 16 kHz that is ~176k samples.
        let seconds = Double(samples.count) / AudioDecoder.targetSampleRate
        #expect(seconds > 10.0)
        #expect(seconds < 12.0)

        // Real audio, not silence.
        let rms = (samples.reduce(0) { $0 + Double($1 * $1) } / Double(samples.count)).squareRoot()
        #expect(rms > 0.001)
    }

    @Test("Reported duration matches the decoded sample count")
    func durationMatchesSamples() throws {
        let samples = try AudioDecoder.decodePCM16kMono(url: TestEnv.jfkURL)
        let duration = try AudioDecoder.duration(of: TestEnv.jfkURL)
        let decodedSeconds = Double(samples.count) / AudioDecoder.targetSampleRate
        #expect(abs(duration - decodedSeconds) < 0.05)
    }

    @Test("Chunked inputs are contiguous and cover the whole file")
    func chunksAreContiguous() throws {
        let file = try AVAudioFile(forReading: TestEnv.jfkURL)
        let inputs = try AudioLoader.readChunks(from: file)

        #expect(inputs.count > 1)

        var expectedStart = CMTime.zero
        var totalFrames: AVAudioFrameCount = 0
        for input in inputs {
            let start = try #require(input.bufferStartTime)
            // Each chunk begins exactly where the previous one ended.
            #expect(start == expectedStart)
            expectedStart = start + input.bufferDuration
            totalFrames += input.buffer.frameLength
        }

        #expect(totalFrames == AVAudioFrameCount(file.length))
        #expect(expectedStart == AudioLoader.duration(of: file))
    }

    @Test("Collecting a chunked stream reproduces a direct decode")
    func collectMatchesDirectDecode() async throws {
        let file = try AVAudioFile(forReading: TestEnv.jfkURL)
        let stream = try AudioLoader.inputStream(from: file)
        let (collected, duration) = try await AudioLoader.collectPCM16kMono(from: stream)
        let direct = try AudioDecoder.decodePCM16kMono(url: TestEnv.jfkURL)

        // Chunked conversion can differ by a few frames at boundaries, but not more.
        #expect(abs(collected.count - direct.count) < 2_000)
        #expect(abs(duration.seconds - Double(direct.count) / 16_000) < 0.15)
    }

    @Test("Input metadata is derived from the buffer")
    func inputDerivesMetadata() throws {
        let format = AudioDecoder.canonicalFormat
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 8_000))
        buffer.frameLength = 8_000

        let input = CirceAnalyzerInput(buffer: buffer)
        #expect(input.bufferStartTime == nil)
        #expect(input.bufferFormat.sampleRate == 16_000)
        // 8000 frames at 16 kHz is exactly half a second.
        #expect(input.bufferDuration.seconds == 0.5)
    }
}

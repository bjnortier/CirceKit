import AVFoundation
import CoreMedia
import Foundation
import Testing
@testable import CirceKit

/// Runs the backends over audio that is *not* already 16 kHz mono.
///
/// The bundled JFK clip is 16 kHz mono, so every other suite takes the fast path
/// and never converts. That masked two real defects: a per-chunk `AVAudioConverter`
/// that discarded resampler state at every buffer boundary, and source-timeline
/// timestamps attached to converted buffers, which drifted until Apple's analyzer
/// rejected the stream as disordered audio.
@Suite("Resampled audio", .serialized)
struct ResampledAudioTests {
    /// Writes the JFK clip as 44.1 kHz stereo — a common recording format, and the
    /// one that exposed the drift.
    static func makeStereo44k(at url: URL) throws {
        let mono = try AudioDecoder.decodePCM16kMono(url: TestEnv.jfkURL)
        let source = AudioDecoder.canonicalFormat
        let target = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 44_100, channels: 2, interleaved: false
        )!

        let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: AVAudioFrameCount(mono.count))!
        input.frameLength = AVAudioFrameCount(mono.count)
        mono.withUnsafeBufferPointer {
            input.floatChannelData![0].update(from: $0.baseAddress!, count: mono.count)
        }

        let converter = AVAudioConverter(from: source, to: target)!
        let capacity = AVAudioFrameCount(Double(mono.count) * 44_100 / 16_000) + 1_024
        let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity)!
        var fed = false
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            fed = true
            status.pointee = .haveData
            return input
        }
        if let error { throw error }

        let file = try AVAudioFile(forWriting: url, settings: target.settings)
        try file.write(from: output)
    }

    static func availableBackends() async -> [CirceTranscriber.Backend] {
        var backends: [CirceTranscriber.Backend] = []
        if await TestEnv.appleEnglishInstalled() { backends.append(.apple) }
        if TestEnv.canRunWhisper { backends.append(.whisperCPP(TestEnv.testModel)) }
        backends.append(contentsOf: TestEnv.availableCoreAIModels.map { .coreAI($0) })
        return backends
    }

    @Test("Every backend transcribes 44.1 kHz stereo audio")
    func transcribesResampledAudio() async throws {
        let directory = URL.temporaryDirectory.appending(path: "circe-resample-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appending(path: "jfk-44k-stereo.wav")
        try Self.makeStereo44k(at: url)

        // Sanity-check the fixture is actually in the awkward format.
        let file = try AVAudioFile(forReading: url)
        #expect(file.fileFormat.sampleRate == 44_100)
        #expect(file.fileFormat.channelCount == 2)

        for backend in await Self.availableBackends() {
            let transcriber = CirceFileTranscriber(
                backend: backend, locale: Locale(identifier: "en_US"), preset: .transcription
            )
            let result = try await transcriber.transcribe(fileAt: url)

            // Resampling must not cost accuracy: the same words, from the same
            // speech, merely arriving at a different rate.
            let wer = TestEnv.jfkWER(result.text)
            #expect(wer < 0.2, "\(backend) WER \(wer) on 44.1 kHz stereo: \(result.text)")
        }
    }

    @Test("Collecting a resampled stream preserves its duration")
    func collectPreservesDuration() async throws {
        let directory = URL.temporaryDirectory.appending(path: "circe-resample-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appending(path: "jfk-44k-stereo.wav")
        try Self.makeStereo44k(at: url)

        let file = try AVAudioFile(forReading: url)
        let sourceDuration = AudioLoader.duration(of: file).seconds
        let stream = try AudioLoader.inputStream(from: file)
        let (samples, duration) = try await AudioLoader.collectPCM16kMono(from: stream)

        // Chunked conversion through one shared converter should land within a few
        // milliseconds; a converter rebuilt per chunk drifts much further than this.
        #expect(abs(duration.seconds - sourceDuration) < 0.05)
        #expect(abs(Double(samples.count) / 16_000 - sourceDuration) < 0.05)
    }
}

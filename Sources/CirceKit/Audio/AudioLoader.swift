import AVFoundation
import CoreMedia
import Foundation

/// Turns audio files into the ``CirceAnalyzerInput`` stream a ``CirceAnalyzer`` consumes.
public nonisolated enum AudioLoader {
    /// Frames per emitted buffer. 0.5 s at 16 kHz — small enough that a streaming
    /// backend sees steady progress, large enough to keep per-buffer overhead low.
    public static let defaultChunkFrames: AVAudioFrameCount = 8_000

    /// Reads `file` into a finite stream of inputs, each tagged with its start time.
    ///
    /// Audio is delivered in `file.processingFormat`; backends convert as needed.
    /// The stream is fully buffered up front, so reading it cannot fail midway.
    public static func inputStream(
        from file: AVAudioFile,
        chunkFrames: AVAudioFrameCount = defaultChunkFrames
    ) throws -> AsyncStream<CirceAnalyzerInput> {
        let inputs = try readChunks(from: file, chunkFrames: chunkFrames)
        return AsyncStream { continuation in
            for input in inputs { continuation.yield(input) }
            continuation.finish()
        }
    }

    /// Reads `file` into chunked inputs.
    public static func readChunks(
        from file: AVAudioFile,
        chunkFrames: AVAudioFrameCount = defaultChunkFrames
    ) throws -> [CirceAnalyzerInput] {
        let format = file.processingFormat
        let timescale = CMTimeScale(format.sampleRate)
        var inputs: [CirceAnalyzerInput] = []
        var framePosition: AVAudioFramePosition = 0

        file.framePosition = 0
        while framePosition < file.length {
            let remaining = AVAudioFrameCount(file.length - framePosition)
            let frames = min(chunkFrames, remaining)
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
                throw AudioDecoder.DecodeError.bufferAllocationFailed
            }
            try file.read(into: buffer, frameCount: frames)
            // A short read means the file ended earlier than `length` claimed.
            guard buffer.frameLength > 0 else { break }

            let startTime = CMTime(value: CMTimeValue(framePosition), timescale: timescale)
            inputs.append(CirceAnalyzerInput(buffer: buffer, bufferStartTime: startTime))
            framePosition += AVAudioFramePosition(buffer.frameLength)
        }
        return inputs
    }

    /// Total duration of `file`.
    public static func duration(of file: AVAudioFile) -> CMTime {
        let sampleRate = file.processingFormat.sampleRate
        guard sampleRate > 0 else { return .zero }
        return CMTime(value: CMTimeValue(file.length), timescale: CMTimeScale(sampleRate))
    }

    /// Concatenates a stream of inputs into 16 kHz mono float samples.
    ///
    /// This is how the batch backends (Core AI, whisper.cpp) consume a stream:
    /// they cannot process incrementally, so they accumulate and run once.
    public static func collectPCM16kMono(
        from inputs: AsyncStream<CirceAnalyzerInput>
    ) async throws -> (samples: [Float], duration: CMTime) {
        var samples: [Float] = []
        for await input in inputs {
            samples.append(contentsOf: try AudioDecoder.pcm16kMono(from: input.buffer))
        }
        // Duration comes from the resampled count, so it matches what the engine sees.
        let duration = CMTime(
            value: CMTimeValue(samples.count),
            timescale: CMTimeScale(AudioDecoder.targetSampleRate)
        )
        return (samples, duration)
    }
}

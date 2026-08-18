import AVFoundation
import Foundation

/// Decodes audio files into the format the batch engines expect: 16 kHz, mono, float32.
public nonisolated enum AudioDecoder {
    /// The sample rate every CirceKit backend ultimately works in.
    public static let targetSampleRate: Double = 16_000

    public enum DecodeError: LocalizedError {
        case bufferAllocationFailed
        case converterCreationFailed

        public var errorDescription: String? {
            switch self {
            case .bufferAllocationFailed: return "Could not allocate an audio buffer."
            case .converterCreationFailed: return "Could not create an audio converter."
            }
        }
    }

    /// The canonical 16 kHz mono float32 format.
    public static var canonicalFormat: AVAudioFormat {
        AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        )!
    }

    /// Reads `url` and returns 16 kHz mono float PCM samples, resampling if needed.
    public static func decodePCM16kMono(url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let sourceFormat = file.processingFormat
        guard let sourceBuffer = AVAudioPCMBuffer(
            pcmFormat: sourceFormat,
            frameCapacity: AVAudioFrameCount(file.length)
        ) else {
            throw DecodeError.bufferAllocationFailed
        }
        try file.read(into: sourceBuffer)

        // Fast path: the file is already 16 kHz mono float.
        if sourceFormat.commonFormat == .pcmFormatFloat32,
           sourceFormat.sampleRate == targetSampleRate,
           sourceFormat.channelCount == 1 {
            return floats(from: sourceBuffer)
        }

        return try resample(sourceBuffer, from: sourceFormat, to: canonicalFormat)
    }

    /// Converts a single in-memory buffer to 16 kHz mono float samples.
    public static func pcm16kMono(from buffer: AVAudioPCMBuffer) throws -> [Float] {
        let format = buffer.format
        if format.commonFormat == .pcmFormatFloat32,
           format.sampleRate == targetSampleRate,
           format.channelCount == 1 {
            return floats(from: buffer)
        }
        return try resample(buffer, from: format, to: canonicalFormat)
    }

    /// Duration of an audio file in seconds, without decoding it.
    public static func duration(of url: URL) throws -> Double {
        let file = try AVAudioFile(forReading: url)
        let sampleRate = file.fileFormat.sampleRate
        return sampleRate > 0 ? Double(file.length) / sampleRate : 0
    }

    private static func resample(
        _ sourceBuffer: AVAudioPCMBuffer,
        from sourceFormat: AVAudioFormat,
        to targetFormat: AVAudioFormat
    ) throws -> [Float] {
        guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
            throw DecodeError.converterCreationFailed
        }

        let ratio = targetFormat.sampleRate / sourceFormat.sampleRate
        let capacity = AVAudioFrameCount(Double(sourceBuffer.frameLength) * ratio) + 1_024
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else {
            throw DecodeError.bufferAllocationFailed
        }

        // The whole input is already in memory, so feed it once then end the stream.
        var didFeed = false
        var conversionError: NSError?
        converter.convert(to: outputBuffer, error: &conversionError) { _, statusPointer in
            if didFeed {
                statusPointer.pointee = .endOfStream
                return nil
            }
            didFeed = true
            statusPointer.pointee = .haveData
            return sourceBuffer
        }
        if let conversionError { throw conversionError }

        return floats(from: outputBuffer)
    }

    private static func floats(from buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channelData = buffer.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: channelData[0], count: Int(buffer.frameLength)))
    }
}

import AVFoundation
import Foundation

/// Converts PCM buffers into a fixed target format, reusing one `AVAudioConverter`
/// per source format.
///
/// Needed on the Apple path, where the engine negotiates a format that rarely
/// matches the source file. The batch backends use ``AudioDecoder`` instead.
internal final class BufferConverter: @unchecked Sendable {
    private let targetFormat: AVAudioFormat
    private let lock = NSLock()
    private var converter: AVAudioConverter?
    private var sourceFormat: AVAudioFormat?

    init(targetFormat: AVAudioFormat) {
        self.targetFormat = targetFormat
    }

    /// Converts `buffer` to the target format, or returns it unchanged if it is
    /// already in that format.
    func convert(_ buffer: AVAudioPCMBuffer) throws -> AVAudioPCMBuffer {
        let inputFormat = buffer.format
        if inputFormat == targetFormat { return buffer }

        lock.lock()
        defer { lock.unlock() }

        // Rebuild the converter only when the source format actually changes.
        if converter == nil || sourceFormat != inputFormat {
            guard let newConverter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
                throw AudioDecoder.DecodeError.converterCreationFailed
            }
            // Sacrifice the quality of the first samples to avoid timestamp drift
            // from the source, which the analyzer would reject as disordered audio.
            newConverter.primeMethod = .none
            converter = newConverter
            sourceFormat = inputFormat
        }
        guard let converter else {
            throw AudioDecoder.DecodeError.converterCreationFailed
        }

        let ratio = targetFormat.sampleRate / inputFormat.sampleRate
        let capacity = AVAudioFrameCount((Double(buffer.frameLength) * ratio).rounded(.up))
        guard let output = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else {
            throw AudioDecoder.DecodeError.bufferAllocationFailed
        }

        // This converter is reused across the whole run, so the closure must
        // report `.noDataNow` when its single buffer is spent. Reporting
        // `.endOfStream` would put the converter into a terminal state and every
        // subsequent call would silently yield zero frames.
        var bufferProcessed = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, statusPointer in
            defer { bufferProcessed = true }
            statusPointer.pointee = bufferProcessed ? .noDataNow : .haveData
            return bufferProcessed ? nil : buffer
        }
        if status == .error { throw error ?? AudioDecoder.DecodeError.converterCreationFailed }
        return output
    }
}

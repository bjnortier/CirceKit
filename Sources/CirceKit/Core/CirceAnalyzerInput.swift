import AVFoundation
import CoreMedia
import Foundation

/// A unit of audio handed to a ``CirceAnalyzer``.
///
/// Mirrors `Speech.AnalyzerInput`. `bufferStartTime` is the presentation time of
/// the buffer's first frame; `nil` means "immediately after the previous input",
/// which is what file-driven analysis produces.
public struct CirceAnalyzerInput: Sendable {
    /// The audio samples.
    public let buffer: AVAudioPCMBuffer

    /// Presentation time of the buffer's first frame, or `nil` to continue from
    /// wherever the previous input ended.
    public let bufferStartTime: CMTime?

    /// How much audio this buffer holds, derived from its frame count and sample rate.
    public let bufferDuration: CMTime

    /// The format of ``buffer``.
    public let bufferFormat: AVAudioFormat

    public init(buffer: AVAudioPCMBuffer) {
        self.init(buffer: buffer, bufferStartTime: nil)
    }

    public init(buffer: AVAudioPCMBuffer, bufferStartTime: CMTime?) {
        self.buffer = buffer
        self.bufferStartTime = bufferStartTime
        self.bufferFormat = buffer.format
        // Use the sample rate as the timescale so frame counts convert exactly.
        let timescale = CMTimeScale(buffer.format.sampleRate)
        self.bufferDuration = CMTime(value: CMTimeValue(buffer.frameLength), timescale: timescale)
    }
}

// AVAudioPCMBuffer is a reference type that AVFoundation does not mark Sendable.
// CirceKit only ever hands a buffer forward — buffers are produced by the decoder
// or the caller and never mutated after being wrapped in a CirceAnalyzerInput.
extension AVAudioPCMBuffer: @retroactive @unchecked Sendable {}

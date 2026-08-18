import AVFoundation
import CoreMedia
import Foundation

/// A result produced by a ``CirceSpeechModule``.
///
/// Mirrors `Speech.SpeechModuleResult`.
public protocol CirceSpeechModuleResult {
    /// The span of audio this result describes.
    var range: CMTimeRange { get }

    /// The point up to which results are final. A result whose ``range`` ends at
    /// or before this time will not change.
    var resultsFinalizationTime: CMTime { get }
}

extension CirceSpeechModuleResult {
    /// Whether this result is settled, or still subject to revision.
    public var isFinal: Bool {
        resultsFinalizationTime >= range.end
    }
}

/// Something a ``CirceAnalyzer`` can drive.
///
/// Mirrors `Speech.SpeechModule`. ``CirceTranscriber`` is the only conforming
/// type CirceKit ships today.
public protocol CirceSpeechModule: AnyObject, Sendable {
    associatedtype Results: AsyncSequence & Sendable where Results.Failure == any Error
    associatedtype Result: CirceSpeechModuleResult & Sendable where Result == Results.Element

    /// Results, delivered as they are produced. The sequence finishes when the
    /// analyzer driving this module finishes.
    var results: Results { get }

    /// Audio formats this module can consume directly.
    var availableCompatibleAudioFormats: [AVAudioFormat] { get async }
}

/// The plumbing a ``CirceAnalyzer`` needs from a module.
///
/// This is deliberately internal and separate from ``CirceSpeechModule``: the
/// analyzer holds `[any CirceSpeechModule]`, and an existential cannot reach a
/// protocol's associated types. Keeping the drive interface free of associated
/// types lets the analyzer downcast to it and work with any module uniformly.
internal protocol AnalyzerAttachable: AnyObject, Sendable {
    /// The format this module wants audio in, once it knows.
    var analyzerFormat: AVAudioFormat? { get async }

    /// Begin consuming `inputs`. Returns once consumption has started, not once
    /// it has finished.
    func attach(inputs: AsyncStream<CirceAnalyzerInput>) async throws

    /// Finish the run: emit any remaining results, then end the results sequence.
    func finishAndFinalize() async throws

    /// Abandon the run without finalizing.
    func cancelNow() async
}

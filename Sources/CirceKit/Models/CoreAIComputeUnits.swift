import CoreAISpeech
import Foundation

/// The compute unit a Core AI graph is specialized for.
///
/// Mirrors `CoreAISpeech.SpeechComputeUnit` so callers need not import
/// `CoreAISpeech`, the same way ``CirceTranscriber/TranscriptionOption`` mirrors
/// Apple's.
public enum CoreAIComputeUnit: String, Sendable, Equatable, Hashable, CaseIterable, Codable {
    /// The runtime picks, probing the Neural Engine first and falling back where
    /// a graph will not lower to it.
    case automatic
    case cpu
    case gpu
    case neuralEngine

    /// A short label for pickers and result tables.
    public var displayName: String {
        switch self {
        case .automatic: "Automatic"
        case .cpu: "CPU"
        case .gpu: "GPU"
        case .neuralEngine: "Neural Engine"
        }
    }

    var coreAIValue: SpeechComputeUnit {
        switch self {
        case .automatic: .automatic
        case .cpu: .cpu
        case .gpu: .gpu
        case .neuralEngine: .neuralEngine
        }
    }
}

/// Which compute unit each stage of a Core AI speech bundle is specialized for.
///
/// The two stages want opposite things: the encoder is one large pass per window
/// and is throughput-bound, while the per-token graphs are tiny and launch-bound
/// and present a shape the accelerator backends re-specialize on every step.
///
/// ``default`` carries `CoreAISpeech`'s own policy, **which was measured on
/// Apple silicon Mac**. iOS has a much smaller GPU against a comparable Neural
/// Engine, so it is a starting point there and not a conclusion — which is why
/// this is a parameter rather than a constant.
public struct CoreAIComputeUnits: Sendable, Equatable, Hashable, Codable, RawRepresentable {
    /// The encoder: one pass per 30 s window.
    public var encoder: CoreAIComputeUnit
    /// The per-token graphs — Whisper's decoder step, Parakeet's `decoder_step`
    /// and `joint`.
    public var decoder: CoreAIComputeUnit

    public init(encoder: CoreAIComputeUnit = .automatic, decoder: CoreAIComputeUnit = .cpu) {
        self.encoder = encoder
        self.decoder = decoder
    }

    /// The measured Mac policy, and `CoreAISpeech`'s own default.
    public static let `default` = CoreAIComputeUnits()

    /// Everything `.automatic` — what the runtime did before any of this was
    /// measured, kept as an A/B baseline.
    public static let unspecialized = CoreAIComputeUnits(encoder: .automatic, decoder: .automatic)

    var coreAIValue: SpeechComputeUnits {
        SpeechComputeUnits(encoder: encoder.coreAIValue, decoder: decoder.coreAIValue)
    }

    /// A short label for logs and result tables, e.g. `"enc automatic / dec cpu"`.
    public var summary: String {
        "enc \(encoder.rawValue) / dec \(decoder.rawValue)"
    }
}

// MARK: - RawRepresentable

/// A string encoding, so the policy can be stored in `UserDefaults` — and
/// therefore in SwiftUI's `@AppStorage`.
///
/// Persistence is not incidental here: the comparison worth making is the *cold*
/// load, which needs the app relaunched between runs, so a policy that did not
/// survive a launch could not be measured at all.
extension CoreAIComputeUnits {
    public init?(rawValue: String) {
        let parts = rawValue.split(separator: "/", maxSplits: 1)
        guard parts.count == 2,
              let encoder = CoreAIComputeUnit(rawValue: String(parts[0])),
              let decoder = CoreAIComputeUnit(rawValue: String(parts[1]))
        else { return nil }
        self.init(encoder: encoder, decoder: decoder)
    }

    public var rawValue: String { "\(encoder.rawValue)/\(decoder.rawValue)" }
}

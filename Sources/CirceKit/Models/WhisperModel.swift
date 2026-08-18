import Foundation

/// A whisper.cpp ggml model, identified by its published filename.
///
/// Every case is downloadable from the canonical `ggerganov/whisper.cpp`
/// HuggingFace repository via ``CirceModelStore``.
public enum WhisperModel: String, Sendable, CaseIterable, Equatable, Hashable {
    case tiny
    case tinyEN
    case base
    case baseEN
    case small
    case smallEN
    case medium
    case mediumEN
    case largeV3
    case largeV3Turbo
    case largeV3TurboQ5_0

    /// The ggml filename, e.g. `ggml-tiny.en.bin`.
    public var filename: String { "ggml-\(modelName).bin" }

    /// The model identifier as whisper.cpp names it, e.g. `tiny.en`.
    public var modelName: String {
        switch self {
        case .tiny: return "tiny"
        case .tinyEN: return "tiny.en"
        case .base: return "base"
        case .baseEN: return "base.en"
        case .small: return "small"
        case .smallEN: return "small.en"
        case .medium: return "medium"
        case .mediumEN: return "medium.en"
        case .largeV3: return "large-v3"
        case .largeV3Turbo: return "large-v3-turbo"
        case .largeV3TurboQ5_0: return "large-v3-turbo-q5_0"
        }
    }

    /// Where the model is downloaded from.
    public var remoteURL: URL {
        URL(string: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/\(filename)")!
    }

    /// English-only models reject a non-English `language` parameter.
    public var isEnglishOnly: Bool {
        switch self {
        case .tinyEN, .baseEN, .smallEN, .mediumEN: return true
        default: return false
        }
    }

    /// Approximate download size, for progress reporting and UI.
    public var approximateBytes: Int {
        switch self {
        case .tiny, .tinyEN: return 78_000_000
        case .base, .baseEN: return 148_000_000
        case .small, .smallEN: return 488_000_000
        case .medium, .mediumEN: return 1_530_000_000
        case .largeV3: return 3_100_000_000
        case .largeV3Turbo: return 1_620_000_000
        case .largeV3TurboQ5_0: return 574_000_000
        }
    }
}

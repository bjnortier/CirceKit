import Foundation

/// Errors raised by CirceKit itself. Backend engines surface their own errors
/// unchanged where they are more informative.
public enum CirceError: LocalizedError {
    /// A module was handed to a ``CirceAnalyzer`` that it does not know how to drive.
    case unsupportedModule(String)

    /// The backend is not usable on this platform.
    case unsupportedPlatform(String)

    /// A model is required but not present, and cannot be fetched automatically.
    case modelUnavailable(String)

    /// The locale is not supported by the selected backend.
    case localeNotSupported(Locale)

    /// Audio could not be converted into the format the backend requires.
    case audioConversionFailed(String)

    /// A run was started on an analyzer that is already running, or finalized
    /// before it was started.
    case invalidState(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedModule(let name):
            return "CirceAnalyzer cannot drive a module of type \(name)."
        case .unsupportedPlatform(let detail):
            return "Not supported on this platform: \(detail)."
        case .modelUnavailable(let detail):
            return "Model unavailable: \(detail)."
        case .localeNotSupported(let locale):
            return "Locale \(locale.identifier(.bcp47)) is not supported by this backend."
        case .audioConversionFailed(let detail):
            return "Audio conversion failed: \(detail)."
        case .invalidState(let detail):
            return "Invalid analyzer state: \(detail)."
        }
    }
}

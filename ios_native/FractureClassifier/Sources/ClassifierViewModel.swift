//
//  ClassifierViewModel.swift
//  CoreML + Vision inference driver. Runs entirely on-device (ANE-accelerated).
//

import SwiftUI
import CoreML
import Vision
import UIKit

struct ClassificationItem: Identifiable, Hashable {
    var id: String { label }
    let label: String
    let confidence: Double      // 0...1
}

struct ClassificationResult: Identifiable {
    let id = UUID()
    let top: ClassificationItem
    let ranked: [ClassificationItem]     // sorted high→low, length 10
    let inferenceMs: Double
}

@MainActor
final class ClassifierViewModel: ObservableObject {
    @Published var selectedImage: UIImage?
    @Published var result: ClassificationResult?
    @Published var errorMessage: String?
    @Published var isRunning: Bool = false

    private lazy var visionModel: VNCoreMLModel? = {
        do {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .all     // CPU + GPU + Neural Engine
            let core = try FractureClassifier(configuration: cfg).model
            return try VNCoreMLModel(for: core)
        } catch {
            assertionFailure("Failed to load CoreML model: \(error)")
            return nil
        }
    }()

    func set(image: UIImage) {
        // Normalize orientation so Vision sees an upright image
        selectedImage = image.fixingOrientation()
        result = nil
        errorMessage = nil
    }

    func classify() {
        guard let image = selectedImage else { return }
        guard let visionModel else {
            errorMessage = "err_model_load".localized
            return
        }
        guard let cgImage = image.cgImage else {
            errorMessage = "err_image_decode".localized
            return
        }

        isRunning = true
        errorMessage = nil
        result = nil

        let start = Date()
        let request = VNCoreMLRequest(model: visionModel) { [weak self] req, err in
            DispatchQueue.main.async {
                guard let self else { return }
                self.isRunning = false
                if let err {
                    self.errorMessage = err.localizedDescription
                    return
                }
                guard let obs = req.results as? [VNClassificationObservation], !obs.isEmpty else {
                    self.errorMessage = "err_no_result".localized
                    return
                }
                let ranked = obs.map {
                    ClassificationItem(label: $0.identifier, confidence: Double($0.confidence))
                }.sorted { $0.confidence > $1.confidence }
                let top = ranked.first!
                let dt = Date().timeIntervalSince(start) * 1000
                self.result = ClassificationResult(top: top, ranked: ranked, inferenceMs: dt)
            }
        }
        request.imageCropAndScaleOption = .centerCrop    // square-crop preserves aspect

        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                DispatchQueue.main.async {
                    self.isRunning = false
                    self.errorMessage = error.localizedDescription
                }
            }
        }
    }
}

// MARK: - UIImage orientation fix

extension UIImage {
    /// Rasterizes the image so the CGImage is upright (orientation = .up).
    /// Vision honors `.imageOrientation` but Vision -> CoreML pipelines are
    /// more reliable with a normalized buffer.
    func fixingOrientation() -> UIImage {
        guard imageOrientation != .up else { return self }
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext() ?? self
    }
}

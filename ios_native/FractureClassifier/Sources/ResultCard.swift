//
//  ResultCard.swift
//  Renders Top-1 prediction + confidence + full ranked bar chart.
//

import SwiftUI

struct ResultCard: View {
    let result: ClassificationResult

    private var confidenceColor: Color {
        switch result.top.confidence {
        case 0.70...:  return .green
        case 0.40...:  return .yellow
        default:       return .orange
        }
    }

    private var confidenceBand: String {
        switch result.top.confidence {
        case 0.70...:  return "confidence_high".localized
        case 0.40...:  return "confidence_medium".localized
        default:       return "confidence_low".localized
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Top-1 banner
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Image(systemName: "checkmark.seal.fill")
                        .foregroundColor(confidenceColor)
                    Text("top1_prediction".localized)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(String(format: "%.0f ms", result.inferenceMs))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .monospacedDigit()
                }
                Text(result.top.label)
                    .font(.title2)
                    .fontWeight(.semibold)
                HStack(spacing: 6) {
                    Text(String(format: "%.1f%%", result.top.confidence * 100))
                        .font(.headline)
                        .foregroundColor(confidenceColor)
                        .monospacedDigit()
                    Text("·")
                        .foregroundColor(.secondary)
                    Text(confidenceBand)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.bottom, 4)

            Divider()

            // Per-class bars
            VStack(spacing: 8) {
                ForEach(Array(result.ranked.enumerated()), id: \.element.label) { idx, item in
                    HStack(spacing: 10) {
                        Text(item.label)
                            .font(.caption)
                            .lineLimit(1)
                            .frame(width: 130, alignment: .leading)
                        GeometryReader { geo in
                            ZStack(alignment: .leading) {
                                RoundedRectangle(cornerRadius: 4)
                                    .fill(Color(.tertiarySystemFill))
                                RoundedRectangle(cornerRadius: 4)
                                    .fill(idx == 0 ? confidenceColor : Color.accentColor)
                                    .frame(width: CGFloat(item.confidence) * geo.size.width)
                            }
                        }
                        .frame(height: 10)
                        Text(String(format: "%.1f%%", item.confidence * 100))
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .monospacedDigit()
                            .frame(width: 50, alignment: .trailing)
                    }
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color(.secondarySystemGroupedBackground))
        )
    }
}

#Preview {
    let mock = ClassificationResult(
        top: .init(label: "Greenstick fracture", confidence: 0.78),
        ranked: [
            .init(label: "Greenstick fracture",  confidence: 0.78),
            .init(label: "Avulsion fracture",    confidence: 0.10),
            .init(label: "Comminuted fracture",  confidence: 0.05),
            .init(label: "Hairline Fracture",    confidence: 0.03),
            .init(label: "Pathological fracture",confidence: 0.02),
            .init(label: "Impacted fracture",    confidence: 0.01),
            .init(label: "Oblique fracture",     confidence: 0.005),
            .init(label: "Spiral Fracture",      confidence: 0.003),
            .init(label: "Longitudinal fracture",confidence: 0.001),
            .init(label: "Fracture Dislocation", confidence: 0.001),
        ],
        inferenceMs: 32
    )
    return ResultCard(result: mock).padding()
}

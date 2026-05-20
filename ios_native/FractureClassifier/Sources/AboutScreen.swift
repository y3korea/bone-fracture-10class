//
//  AboutScreen.swift
//  Model card, author info, source links, license.
//

import SwiftUI

struct AboutScreen: View {
    var body: some View {
        NavigationView {
            List {
                Section("about_section_model".localized) {
                    LabeledValue(label: "about_model_arch", value: "MobileNetV2")
                    LabeledValue(label: "about_model_size", value: "128 × 128 RGB")
                    LabeledValue(label: "about_model_classes", value: "10")
                    LabeledValue(label: "about_model_runtime", value: "CoreML · ANE")
                    LabeledValue(label: "about_model_size_mb", value: "4.3 MB")
                }

                Section("about_section_classes".localized) {
                    ForEach(ClassifierConstants.classNames, id: \.self) { name in
                        Label(name, systemImage: "circle.dashed")
                            .font(.callout)
                    }
                }

                Section("about_section_metrics".localized) {
                    LabeledValue(label: "about_metric_dataset", value: "1,129 images")
                    LabeledValue(label: "about_metric_split",   value: "903 / 112 / 114")
                    LabeledValue(label: "about_metric_acc",     value: "35.96%")
                }

                Section("about_section_author".localized) {
                    LabeledValue(label: "about_author_name",    value: "최완석 (Wansuk Choi)")
                    LabeledValue(label: "about_author_id",      value: "10042214")
                    LabeledValue(label: "about_author_course",  value: "Medical AI · 2026")
                }

                Section("about_section_links".localized) {
                    LinkRow(icon: "globe",
                            title: "about_link_demo",
                            url: URL(string: "https://wansuk-choi-bone-fracture.vercel.app/")!)
                    LinkRow(icon: "chevron.left.slash.chevron.right",
                            title: "about_link_github",
                            url: URL(string: "https://github.com/y3korea/bone-fracture-10class")!)
                    LinkRow(icon: "doc.text",
                            title: "about_link_release",
                            url: URL(string: "https://github.com/y3korea/bone-fracture-10class/releases/tag/v1.0.0")!)
                }

                Section("about_section_legal".localized) {
                    Text("about_license_body".localized)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("about_privacy_body".localized)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Section {
                    Text("about_disclaimer_body".localized)
                        .font(.caption)
                        .foregroundColor(.orange)
                }
            }
            .navigationTitle("about_title".localized)
        }
        .navigationViewStyle(.stack)
    }
}

private struct LabeledValue: View {
    let label: LocalizedStringKey
    let value: String
    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .multilineTextAlignment(.trailing)
        }
        .font(.callout)
    }
}

private struct LinkRow: View {
    let icon: String
    let title: LocalizedStringKey
    let url: URL
    var body: some View {
        Link(destination: url) {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .frame(width: 22)
                    .foregroundColor(.accentColor)
                Text(title)
                    .foregroundColor(.primary)
                Spacer()
                Image(systemName: "arrow.up.right.square")
                    .foregroundColor(.secondary)
                    .imageScale(.small)
            }
        }
    }
}

enum ClassifierConstants {
    static let classNames: [String] = [
        "Avulsion fracture",
        "Comminuted fracture",
        "Fracture Dislocation",
        "Greenstick fracture",
        "Hairline Fracture",
        "Impacted fracture",
        "Longitudinal fracture",
        "Oblique fracture",
        "Pathological fracture",
        "Spiral Fracture",
    ]
}

#Preview {
    AboutScreen()
}

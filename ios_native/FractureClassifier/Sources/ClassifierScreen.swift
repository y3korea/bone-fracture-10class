//
//  ClassifierScreen.swift
//  Main screen: pick X-ray, run inference, show ranked predictions.
//

import SwiftUI

struct ClassifierScreen: View {
    @StateObject private var viewModel = ClassifierViewModel()
    @State private var showingImagePicker = false
    @State private var showingCamera = false

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    disclaimerBanner
                    imageCard
                    actionButtons
                    if viewModel.isRunning {
                        ProgressView("inferring".localized)
                            .padding()
                    }
                    if let result = viewModel.result {
                        ResultCard(result: result)
                            .transition(.opacity.combined(with: .move(edge: .bottom)))
                    }
                    if let error = viewModel.errorMessage {
                        errorCard(error)
                    }
                    Spacer(minLength: 16)
                }
                .padding(.horizontal)
                .padding(.bottom)
                .animation(.easeInOut, value: viewModel.result?.id)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("app_title".localized)
            .navigationBarTitleDisplayMode(.large)
            .sheet(isPresented: $showingImagePicker) {
                ImagePicker(sourceType: .photoLibrary) { image in
                    viewModel.set(image: image)
                }
            }
            .sheet(isPresented: $showingCamera) {
                ImagePicker(sourceType: .camera) { image in
                    viewModel.set(image: image)
                }
            }
        }
        .navigationViewStyle(.stack)
    }

    // MARK: - Subviews

    private var disclaimerBanner: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.orange)
                .imageScale(.medium)
                .padding(.top, 2)
            Text("disclaimer".localized)
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.leading)
            Spacer(minLength: 0)
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.orange.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.orange.opacity(0.30), lineWidth: 1)
                )
        )
    }

    private var imageCard: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 14)
                .fill(Color(.secondarySystemGroupedBackground))
            if let img = viewModel.selectedImage {
                Image(uiImage: img)
                    .resizable()
                    .scaledToFit()
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(8)
            } else {
                VStack(spacing: 10) {
                    Image(systemName: "photo.on.rectangle.angled")
                        .font(.system(size: 56, weight: .light))
                        .foregroundColor(.secondary)
                    Text("no_image_hint".localized)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 24)
                }
                .padding(.vertical, 40)
            }
        }
        .frame(minHeight: 240)
    }

    private var actionButtons: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                Button {
                    showingImagePicker = true
                } label: {
                    Label("pick_photo".localized, systemImage: "photo")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)

                Button {
                    showingCamera = true
                } label: {
                    Label("take_photo".localized, systemImage: "camera")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .disabled(!UIImagePickerController.isSourceTypeAvailable(.camera))
            }

            Button {
                viewModel.classify()
            } label: {
                Label("predict".localized, systemImage: "wand.and.stars")
                    .frame(maxWidth: .infinity)
                    .fontWeight(.semibold)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(viewModel.selectedImage == nil || viewModel.isRunning)
        }
    }

    private func errorCard(_ message: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "xmark.octagon.fill")
                .foregroundColor(.red)
            Text(message)
                .font(.callout)
                .foregroundColor(.primary)
            Spacer()
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.red.opacity(0.08))
        )
    }
}

#Preview {
    ClassifierScreen()
}

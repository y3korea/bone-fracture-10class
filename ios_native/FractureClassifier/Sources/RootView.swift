//
//  RootView.swift
//  Two-tab layout: Classifier (main) + About.
//

import SwiftUI

struct RootView: View {
    var body: some View {
        TabView {
            ClassifierScreen()
                .tabItem {
                    Label("classify".localized, systemImage: "viewfinder")
                }
            AboutScreen()
                .tabItem {
                    Label("about".localized, systemImage: "info.circle")
                }
        }
    }
}

#Preview {
    RootView()
}

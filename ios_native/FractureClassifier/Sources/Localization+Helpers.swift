//
//  Localization+Helpers.swift
//  Tiny String.localized helper so call sites stay short.
//

import Foundation

extension String {
    var localized: String {
        NSLocalizedString(self, comment: "")
    }
}

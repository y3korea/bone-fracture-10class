# Native SwiftUI iOS App — Fracture AI

웹 기반 Capacitor 래퍼(같은 폴더의 `ios/`)와 별개로 작성된 **순수 네이티브 SwiftUI 앱**입니다.

| 항목 | 값 |
|---|---|
| 언어 | Swift 5.9 |
| UI | SwiftUI |
| ML 런타임 | **CoreML + Vision** (Apple Neural Engine 가속) |
| 모델 | `FractureClassifier.mlpackage` (4.3 MB, MobileNetV2 128px) |
| 최소 iOS | **15.0** |
| 디바이스 | iPhone + iPad (universal) |
| 다국어 | 한국어 (기본) · 영어 |
| 패키지 | `com.wansukchoi.fractureclassifier` |

## 파일 구성

```
FractureClassifier/
├── project.yml                      # xcodegen 스펙 (재생성 시 사용)
├── FractureClassifier.xcodeproj     # ⭐ Xcode 에서 더블클릭
├── Sources/
│   ├── FractureClassifierApp.swift  # @main entry
│   ├── RootView.swift               # TabView (분류 / 정보)
│   ├── ClassifierScreen.swift       # 메인 분류 화면
│   ├── ClassifierViewModel.swift    # CoreML/Vision 추론 로직
│   ├── ResultCard.swift             # Top-1 + 10-class 막대 그래프
│   ├── ImagePicker.swift            # UIImagePickerController 래핑
│   ├── AboutScreen.swift            # 정보 화면 (저자, 모델 카드, 링크)
│   ├── Localization+Helpers.swift   # String.localized 헬퍼
│   └── FractureClassifier.mlpackage # CoreML 모델 (PyTorch.pth 에서 변환)
└── Resources/
    ├── ko.lproj/Localizable.strings # 한국어
    ├── en.lproj/Localizable.strings # 영어
    └── Assets.xcassets              # AppIcon (1024px) + AccentColor (light/dark)
```

## 디자인 결정 사항 (의료 AI UX)

- **상단 면책 사항** — 화면 진입 즉시 "교육·연구용, 진단 목적 사용 금지" 명시
- **신뢰도 색상 코딩** — 높음(≥70%) 녹색 · 중간(40-70%) 노랑 · 낮음(<40%) 주황 (저신뢰 자동 경고)
- **추론 시간 표시** — 사용자에게 ANE 속도 체감 (보통 < 50ms)
- **On-device only** — `Info.plist` 의 카메라/사진 권한 설명에 "외부 전송 없음" 명시
- **iPhone + iPad 모두 지원** — `TARGETED_DEVICE_FAMILY = "1,2"`
- **다크 모드** — system color (`Color(.systemGroupedBackground)`) 사용 → 자동 적응
- **SF Symbols** — 모든 아이콘 시스템 폰트 (별도 자산 불필요, 가독성↑)
- **About 화면 링크** — 라이브 웹 데모, GitHub 소스, v1.0.0 릴리즈, 라이선스, 프라이버시 정책

## 🚀 빌드 방법 (교수자용 — 약 2-3분)

### 사전 준비 (1회만)

```bash
# Xcode 가 26.5 이상 설치되어 있어야 함 (이미 설치되어 있음)
# Apple ID 계정이 Xcode 에 등록되어 있어야 함:
#   Xcode → Settings → Accounts → "+" → Apple ID 추가
```

### 1) Xcode 에서 열기

```bash
cd "최완석_app(android, or IOS)/ios_native/FractureClassifier"
open FractureClassifier.xcodeproj
```

### 2) Signing 설정 (Xcode 안에서)

1. 좌측 네비게이터 최상단 **FractureClassifier** 프로젝트 클릭
2. TARGETS → **FractureClassifier** 선택
3. **Signing & Capabilities** 탭
4. **Team** 드롭다운 → 본인 Apple ID 의 Team 선택
5. (Bundle Identifier 충돌 시) 끝에 본인 이니셜 추가 — 예: `com.wansukchoi.fractureclassifier.kbu`

### 3) iPhone 에 빌드 + 실행

1. iPhone 을 Mac 에 USB 로 연결 → 폰에서 "이 컴퓨터를 신뢰" 허용
2. Xcode 상단 디바이스 셀렉터 → 본인 iPhone 선택
3. ⌘R (또는 ▶ Run 버튼)
4. 첫 실행 시 iPhone 에서 **설정 → 일반 → VPN 및 기기 관리** → 본인 Apple ID 의 개발자 앱 신뢰

> 무료 Apple ID 의 경우 인증서가 **7일마다 만료**됩니다. 7일 후 다시 ⌘R 한 번 누르면 갱신됩니다.

### 4) (선택) 시뮬레이터로 빌드

iOS Simulator 에서 실행하려면 Xcode 가 **iOS 26.5 Simulator runtime (약 10.6 GB)** 추가 다운로드를 요구합니다.
실제 iPhone 에 설치할 경우엔 시뮬레이터 불필요 — 위 절차로 충분합니다.

## ⚠️ 모델 한계

학습 데이터가 **클래스당 평균 113 장 (총 1,129장)** 으로 적습니다.
Test accuracy 36% 수준이며, 실제 임상 적용을 위해서는:

1. 클래스당 ≥ 300장 이상 추가 데이터 필요
2. 비슷한 클래스 통합 (Spiral + Oblique + Longitudinal → "shaft fracture") 고려
3. 방사선 전문의 라벨 검증 필수

논문에서도 이 한계를 명시합니다.

## 🛠️ 프로젝트 재생성 (선택)

`project.yml` 만 수정한 뒤 `.xcodeproj` 를 재생성하려면 [xcodegen](https://github.com/yonaskolb/XcodeGen) 사용:

```bash
brew install xcodegen
cd FractureClassifier && xcodegen generate
```

## 라이선스

MIT License — 자유롭게 사용·수정·배포 가능.

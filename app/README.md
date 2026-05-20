# 🦴 Bone Fracture Classifier — 모바일/데스크탑 앱

`web/` 폴더의 웹사이트를 **온디바이스 추론(브라우저 ONNX Runtime)** + **Capacitor / Electron** 으로 패키징한 앱.

서버가 필요 없습니다 — 모델(`.onnx`)이 앱 안에 내장되어 폰/노트북 안에서 그대로 추론합니다.

```
입력 이미지 (X-ray)
    ↓
Canvas resize → 128×128 → ImageNet normalize → NCHW Float32
    ↓
ONNX Runtime Web (WASM, in-browser)
    ↓
softmax → Top-1 클래스 + 신뢰도
```

---

## 📦 빌드 결과물

| 플랫폼 | 산출물 | 설치 방법 |
|--------|--------|---------|
| 🤖 Android | `android/app/build/outputs/apk/debug/app-debug.apk` | 폰에 파일 전송 후 설치 (출처 알 수 없는 앱 허용) |
| 🍏 iOS | `ios/App/App.xcworkspace` | Xcode 에서 본인 Apple ID 로 본인 iPhone 에 직접 빌드 |
| 🖥️ macOS | `electron-dist/*.dmg` | 더블클릭 |
| 🪟 Windows | `electron-dist/*.exe` | 더블클릭 (인스톨러) |

---

## 🚀 사전 준비 (1회만)

### 공통

- **Node.js 18+** : https://nodejs.org/
- **Python 3.10+** + `torch`, `torchvision`, `onnx`, `onnxruntime` (모델 변환용)

### Android APK 빌드

- **Android Studio** : https://developer.android.com/studio
  - Android Studio 설치 후 첫 실행 시 SDK Manager → `Android SDK Platform 34+`, `Build-Tools`, `Platform-Tools` 자동 설치됨
- **JDK 17+** : Android Studio 안에 포함된 JBR 을 그대로 사용하면 됨

환경변수 (`~/.zshrc` 또는 `~/.bash_profile`):

```bash
# Android SDK
export ANDROID_HOME="$HOME/Library/Android/sdk"
export PATH="$ANDROID_HOME/platform-tools:$PATH"

# JDK 17 (Android Studio 내장 JBR)
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
```

설정 후 `java -version` 결과가 17 이상이어야 합니다.

### iOS 빌드 (Mac 전용)

- **Xcode** (App Store)
- **CocoaPods** : `sudo gem install cocoapods` 또는 `brew install cocoapods`

---

## 🛠️ 빌드 절차

### 0. 의존성 설치

```bash
cd app
npm install
```

처음 한 번만 실행하면 됩니다. `node_modules/` + `www/ort/` 가 생성됩니다.

### 1. 본인 모델을 .onnx 로 변환

학습 후 `output/run_YYYYMMDD_HHMMSS/best_model.pth` 가 만들어졌다면:

```bash
npm run export-model
# 또는: cd tools && python3 export_onnx.py
```

→ `app/www/model.onnx` + `app/www/metadata.json` 생성.

특정 run 디렉토리 사용:

```bash
cd tools && python3 export_onnx.py --run-dir ../../output/run_20260414_045350
```

### 2. 로컬에서 먼저 확인 (선택)

```bash
npm run serve
# → http://localhost:5173 에서 확인
```

브라우저에서 이미지 업로드 → 예측이 잘 되는지 점검.

### 3. Android APK 빌드

```bash
npm run android:build
# → ✅ APK : android/app/build/outputs/apk/debug/app-debug.apk
```

빌드된 APK 를 폰에 전송 (USB, 카카오톡 나에게, Google Drive 등) → 설치 시 "출처를 알 수 없는 앱" 허용.

> 처음 빌드는 5-10분 걸립니다 (Gradle 캐시 다운로드).

### 4. iOS 빌드

```bash
npm run ios:open    # Xcode 가 열림
```

Xcode 에서:
1. 좌측 트리 `App` 선택
2. **Signing & Capabilities** 탭 → Team 을 본인 Apple ID 로 변경
3. 상단 디바이스 선택을 본인 iPhone (USB 연결) 으로 변경
4. ▶️ Run 클릭

> 무료 Apple ID 의 경우 인증서가 7일마다 만료됩니다. 정식 배포는 유료 개발자 계정 필요.

### 5. macOS .dmg 빌드

```bash
npm run electron:build:mac
# → electron-dist/Fracture Classifier-1.0.0-arm64.dmg
```

> 코드 서명은 비활성화돼 있어 (`identity: null`) Apple Developer 계정 없이도 빌드됩니다.
> 처음 실행 시 macOS 가 "미확인 개발자" 경고를 띄우면 — Finder 에서 우클릭 → 열기 → 확인.

### 6. Windows .exe 빌드

```bash
npm run electron:build:win
# → electron-dist/Fracture Classifier Setup 1.0.0.exe
```

> Windows 빌드는 Windows PC 에서 실행하는 것을 권장 (Mac 에서 cross-build 도 가능하지만 wine 필요).

---

## 📁 폴더 구조

```
app/
├── README.md
├── package.json
├── capacitor.config.json
├── www/                    # 모든 빌드 타겟이 공유하는 웹 자산
│   ├── index.html
│   ├── app.js
│   ├── model.onnx          # 학생이 직접 학습한 모델로 교체
│   ├── metadata.json
│   └── ort/                # onnxruntime-web (postinstall 로 자동 복사)
├── android/                # Capacitor Android 프로젝트 (npx cap add 로 생성)
├── ios/                    # Capacitor iOS 프로젝트
├── electron/               # Electron 메인 프로세스
│   └── main.js
└── tools/
    ├── export_onnx.py      # .pth → .onnx 변환
    └── copy_ort_assets.js  # onnxruntime-web 의 wasm 을 www/ort 로 복사
```

---

## 🧪 채점/검수 (교수자용)

학생이 제출한 빌드 산출물을 노트북에서 바로 열기:

- **`.dmg` (Mac)** : 더블클릭 → 드래그하여 Applications 폴더로 이동 → 실행
- **`.exe` (Windows)** : 더블클릭 → 설치 마법사 → 실행
- **`.apk` (Android, 옵션)** : Android Studio 의 AVD 에뮬레이터에서 `adb install app-debug.apk`

학생이 소스 코드 전체를 제출한 경우 — 본인 환경에서 위 빌드 절차를 그대로 실행해 검증 가능.

---

## ❓ 자주 묻는 문제

**Q. `npm install` 이 너무 느려요.**
A. 이 폴더가 Google Drive 동기화 폴더라서 그렇습니다. `~/dev/` 같은 로컬 폴더로 옮기면 5-10배 빨라집니다.

**Q. Android 빌드에서 `JAVA_HOME` 관련 에러.**
A. JDK 17 미만이거나 환경변수가 안 잡혀 있습니다. 위 [사전 준비](#사전-준비-1회만) 참고.

**Q. 모델이 로딩되지 않아요 ("metadata.json 로드 실패").**
A. `npm run export-model` 을 먼저 실행해 `www/model.onnx` + `www/metadata.json` 을 만들었는지 확인.

**Q. iOS 빌드에서 "No Signing Certificate" 에러.**
A. Xcode → Signing & Capabilities → Team 에 본인 Apple ID 를 선택했는지 확인.

**Q. `pod install` 시 `Unicode Normalization not appropriate for ASCII-8BIT` 에러.**
A. 프로젝트 경로에 한글이 들어있고 시스템 locale 이 UTF-8 이 아닐 때 발생.
   ```bash
   cd ios/App
   LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 pod install
   ```
   매번 이 변수 지정이 귀찮으면 `~/.zshrc` 에 `export LANG=en_US.UTF-8` 추가.

**Q. 추론이 너무 느려요.**
A. TTA 체크박스를 끄세요 (TTA 는 5× 추론). 또한 첫 추론은 WASM 컴파일 때문에 1-2초 느린 게 정상입니다.

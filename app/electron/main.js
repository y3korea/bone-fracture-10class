// Electron 진입점 — Mac (.dmg) / Windows (.exe) / Linux (AppImage) 데스크탑 빌드용.
// www/index.html 을 그대로 띄운다 (모바일 앱과 동일한 코드).
const { app, BrowserWindow, Menu } = require('electron');
const path = require('path');

function createWindow() {
  const win = new BrowserWindow({
    width: 1100,
    height: 800,
    minWidth: 600,
    minHeight: 500,
    backgroundColor: '#0b1220',
    title: 'Fracture Classifier',
    webPreferences: {
      // 우리 앱은 file:// 로딩으로 ONNX/모델 파일을 자체적으로 가져온다.
      // contextIsolation/nodeIntegration 은 별도 IPC 없이 단순 웹뷰만 쓰면 충분.
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // 단순 메뉴 (Mac 표준 단축키는 유지)
  if (process.platform !== 'darwin') {
    Menu.setApplicationMenu(null);
  }

  const indexPath = path.join(__dirname, '..', 'www', 'index.html');
  win.loadFile(indexPath);

  // 개발 편의: ELECTRON_DEV=1 일 때 DevTools 자동 열기
  if (process.env.ELECTRON_DEV) win.webContents.openDevTools({ mode: 'detach' });
}

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

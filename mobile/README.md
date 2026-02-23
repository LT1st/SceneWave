# SceneWeave Mobile

SceneWeave 移动应用 - AI 智能图片重构图工具

## 支持平台

- iOS
- Android
- macOS

## 技术栈

- **框架**: Flutter 3.x
- **语言**: Dart 3.x
- **状态管理**: GetX
- **网络请求**: Dio
- **图片选择**: image_picker
- **路径管理**: path_provider
- **日志**: logger

## 功能特性

### 1. 首页 (Home)
- 图片选择（相册/相机）
- 快速操作入口
- 功能介绍展示

### 2. 构图分析 (Analyze)
- AI 智能构图评分
- 主体识别显示
- 详细评分维度
- 改进建议提供

### 3. 智能重构图 (Reframe)
- 多种输出比例选择
- 多种填充方式选择
- AI 智能扩图选项
- 实时处理进度

## 安装依赖

```bash
flutter pub get
```

## 运行应用

### iOS
```bash
flutter run -d ios
```

### Android
```bash
flutter run -d android
```

### macOS
```bash
flutter run -d macos
```

## 构建应用

### iOS Release
```bash
flutter build ios --release
```

### Android Release
```bash
flutter build apk --release
flutter build appbundle --release
```

### macOS Release
```bash
flutter build macos --release
```

## API 配置

默认 API 地址为 `http://localhost:8000`

可以通过环境变量或代码修改：

```dart
// 在 lib/core/config.dart 中修改
static const String apiBaseUrl = 'http://your-api-url';
```

或在运行时设置：

```dart
ApiService.to.setBaseUrl('http://your-api-url');
```

## 权限配置

### iOS
- 相机权限: `NSCameraUsageDescription`
- 相册权限: `NSPhotoLibraryUsageDescription`
- 网络: `NSAppTransportSecurity`

### Android
- 相机: `CAMERA`
- 存储: `READ_EXTERNAL_STORAGE`, `WRITE_EXTERNAL_STORAGE`, `READ_MEDIA_IMAGES`
- 网络: `INTERNET`, `ACCESS_NETWORK_STATE`

### macOS
- 相册: `NSPhotoLibraryUsageDescription`
- 相机: `NSCameraUsageDescription`
- 网络: `NSAppTransportSecurity`

## 项目结构

```
lib/
├── main.dart                    # 应用入口
├── app.dart                     # App 配置
├── core/                        # 核心配置
│   ├── config.dart              # 配置常量
│   └── theme.dart               # 主题配置
├── models/                      # 数据模型
│   ├── analysis_result.dart     # 分析结果模型
│   ├── subject.dart             # 主体模型
│   └── reframe_result.dart      # 重构图结果模型
├── screens/                     # 页面
│   ├── home/                    # 首页
│   ├── analyze/                 # 分析页
│   └── reframe/                 # 重构图页
├── widgets/                     # 组件
│   ├── common/                  # 通用组件
│   ├── analyze/                 # 分析相关组件
│   └── reframe/                 # 重构图相关组件
├── services/                    # 服务
│   ├── api_service.dart         # API 服务
│   ├── image_service.dart       # 图片服务
│   └── storage_service.dart     # 存储服务
└── routes/                      # 路由
    └── app_routes.dart          # 路由配置
```

## 开发规范

### 代码风格
- 使用 `flutter analyze` 检查代码
- 遵循 Dart 官方代码规范
- 所有公共 API 需要添加文档注释

### 提交规范
- feat: 新功能
- fix: 修复 bug
- docs: 文档更新
- style: 代码格式调整
- refactor: 重构
- test: 测试相关
- chore: 构建/工具相关

## 故障排查

### Flutter 版本
确保 Flutter 版本 >= 3.0.0

```bash
flutter --version
```

### 依赖问题
清理并重新获取依赖

```bash
flutter clean
flutter pub get
```

### iOS 构建
如果遇到 iOS 构建问题

```bash
cd ios
pod install
cd ..
flutter clean
flutter pub get
```

## License

MIT License

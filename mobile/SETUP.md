# SceneWeave Mobile - Flutter 移动应用

## 项目概述

SceneWeave Mobile 是 SceneWeave 项目的 Flutter 移动应用版本，支持 iOS、Android 和 macOS 三个平台。

## 快速开始

### 1. 安装 Flutter

确保已安装 Flutter 3.x 或更高版本：

```bash
flutter --version
```

### 2. 安装依赖

```bash
cd mobile
flutter pub get
```

### 3. 运行应用

选择目标设备并运行：

```bash
# iOS 模拟器
flutter run -d ios

# Android 模拟器/设备
flutter run -d android

# macOS
flutter run -d macos
```

## 功能说明

### 核心功能

1. **图片选择**
   - 从相册选择图片
   - 使用相机拍照
   - 支持多种图片格式

2. **构图分析**
   - AI 智能主体识别
   - 构图评分（0-100分）
   - 详细评分维度展示
   - 改进建议提供

3. **智能重构图**
   - 多种输出比例（1:1, 4:5, 9:16, 16:9）
   - 多种填充方式（模糊、扩展、配色等）
   - AI 智能扩图选项
   - 实时处理进度显示

## 技术架构

### 状态管理
- 使用 GetX 进行状态管理
- 响应式编程模型
- 依赖注入

### 网络请求
- Dio 进行 HTTP 请求
- 支持 FormData 上传
- 自动错误处理

### 本地存储
- SharedPreferences 存储配置
- 文件系统存储图片
- 历史记录管理

### UI 设计
- Material Design 3 风格
- 自适应响应式布局
- 深色模式支持

## 目录结构

```
lib/
├── main.dart                 # 应用入口
├── app.dart                  # 应用配置
├── core/                     # 核心配置
│   ├── config.dart           # 配置常量
│   ├── theme.dart            # 主题配置
│   └── bindings.dart         # 依赖绑定
├── models/                   # 数据模型
│   ├── analysis_result.dart  # 分析结果
│   ├── subject.dart          # 主体数据
│   └── reframe_result.dart   # 重构图结果
├── screens/                  # 页面
│   ├── home/                 # 首页
│   ├── analyze/              # 分析页
│   ├── reframe/              # 重构图页
│   └── settings/             # 设置页
├── widgets/                  # 组件
│   ├── common/               # 通用组件
│   ├── analyze/              # 分析组件
│   └── reframe/              # 重构图组件
├── services/                 # 服务
│   ├── api_service.dart      # API 服务
│   ├── image_service.dart    # 图片服务
│   └── storage_service.dart  # 存储服务
└── routes/                   # 路由
    └── app_routes.dart       # 路由配置
```

## 配置说明

### API 地址配置

默认 API 地址：`http://localhost:8000`

修改方式：
1. 在应用内：设置 -> API 配置 -> 输入新地址
2. 在代码中：修改 `lib/core/config.dart` 中的 `apiBaseUrl`

### 权限配置

#### iOS (ios/Runner/Info.plist)
- 相机权限：`NSCameraUsageDescription`
- 相册权限：`NSPhotoLibraryUsageDescription`

#### Android (android/app/src/main/AndroidManifest.xml)
- 相机：`CAMERA`
- 存储：`READ_EXTERNAL_STORAGE`, `READ_MEDIA_IMAGES`
- 网络：`INTERNET`

#### macOS (macos/Runner/Info.plist)
- 相册权限：`NSPhotoLibraryUsageDescription`

## 构建发布

### iOS

```bash
flutter build ios --release
```

生成的文件在 `build/ios/archive/`

### Android APK

```bash
flutter build apk --release
```

生成的文件在 `build/app/outputs/flutter-apk/`

### Android App Bundle

```bash
flutter build appbundle --release
```

生成的文件在 `build/app/outputs/bundle/release/`

### macOS

```bash
flutter build macos --release
```

生成的文件在 `build/macos/Build/Products/Release/`

## 常见问题

### 1. 依赖安装失败

```bash
flutter clean
flutter pub get
```

### 2. iOS 构建失败

```bash
cd ios
pod install
cd ..
```

### 3. 无法连接到 API

- 检查 API 服务是否运行
- 确认设备/模拟器能访问 localhost
- 使用实际 IP 地址代替 localhost

### 4. 图片上传失败

- 检查文件权限
- 确认图片大小在限制内（默认 10MB）
- 查看控制台错误日志

## 开发建议

1. 使用热重载提高开发效率
2. 遵循 Flutter 官方代码规范
3. 使用 `flutter analyze` 检查代码
4. 定期运行 `flutter pub upgrade` 更新依赖

## 相关资源

- [Flutter 文档](https://flutter.dev/docs)
- [GetX 文档](https://github.com/jonataslaw/getx)
- [Material Design 3](https://m3.material.io/)

## License

MIT License

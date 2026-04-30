class BackendConfig {
  const BackendConfig._();

  static const bool useMockAnalysisFlow = bool.fromEnvironment(
    'USE_MOCK_ANALYSIS_FLOW',
    defaultValue: false,
  );

  // Android emulator: http://10.0.2.2:8000
  // iOS simulator / desktop: http://127.0.0.1:8000
  // Physical device: use your computer's LAN IP, e.g. http://192.168.1.10:8000
  static const String apiBaseUrl = String.fromEnvironment(
    'ESPRESSO_API_BASE_URL',
    defaultValue: 'http://10.0.2.2:8000',
  );

  static const String analysisVideoUploadEndpoint =
      '$apiBaseUrl/api/analyse-video/upload';
  static const String analysisStartEndpoint =
      '$apiBaseUrl/api/analyse-video/start';
  static const String analysisStatusEndpoint =
      '$apiBaseUrl/api/analyse-video/status';

  static const String emulatorTestVideoPath = String.fromEnvironment(
    'CEKOPI_TEST_VIDEO_PATH',
    defaultValue:
        '/storage/emulated/0/Android/data/com.example.cekopi/files/cekopi_test.mov',
  );
}

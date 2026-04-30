import 'dart:convert';

import 'package:cekopi/core/utils/json_helpers.dart';
import 'package:http/http.dart' as http;
import 'package:cekopi/core/config/backend_config.dart';

class AnalysisPollResult {
  const AnalysisPollResult({
    required this.success,
    required this.completed,
    required this.message,
    this.errorCode,
    this.stageIndex,
    this.stageTotal,
    this.stageLabel,
    this.progress,
    this.payload,
  });

  final bool success;
  final bool completed;
  final String message;
  final String? errorCode;
  final int? stageIndex;
  final int? stageTotal;
  final String? stageLabel;
  final double? progress;
  final Map<String, dynamic>? payload;
}

class VideoUploadResult {
  const VideoUploadResult({
    required this.success,
    required this.message,
    this.errorCode,
    this.statusCode,
    this.payload,
  });

  final bool success;
  final String message;
  final String? errorCode;
  final int? statusCode;
  final Map<String, dynamic>? payload;
}

class VideoUploadService {
  VideoUploadService({http.Client? client})
    : _client = client ?? http.Client(),
      _ownsClient = client == null;

  final http.Client _client;
  final bool _ownsClient;
  static final Map<String, int> _mockPollAttemptByJob = <String, int>{};

  void close() {
    if (_ownsClient) {
      _client.close();
    }
  }

  Map<String, dynamic>? _errorPayload(Map<String, dynamic>? bodyJson) {
    return asStringKeyedMap(bodyJson?['detail']) ?? bodyJson;
  }

  Future<VideoUploadResult> uploadVideo({required String videoPath}) async {
    if (BackendConfig.useMockAnalysisFlow) {
      await Future<void>.delayed(const Duration(milliseconds: 1200));
      final mockJobId = 'mock_${DateTime.now().millisecondsSinceEpoch}';
      _mockPollAttemptByJob[mockJobId] = 0;

      return VideoUploadResult(
        success: true,
        message: 'Video uploaded. Review the placeholder ROI.',
        payload: {
          'job_id': mockJobId,
          'mode': 'mock',
          'frame_width': 1920,
          'frame_height': 1080,
          'auto_roi_ellipse': {
            'cx': 0.5,
            'cy': 0.42,
            'width': 0.54,
            'height': 0.34,
            'angle_deg': 0.0,
          },
        },
      );
    }

    if (BackendConfig.analysisVideoUploadEndpoint.isEmpty) {
      return const VideoUploadResult(
        success: false,
        errorCode: 'E-UPLOAD-CONFIG',
        message:
            'Backend URL is not configured. Update BackendConfig.apiBaseUrl first.',
      );
    }

    final uri = Uri.parse(BackendConfig.analysisVideoUploadEndpoint);
    final request = http.MultipartRequest('POST', uri)
      ..fields['source'] = 'mobile_app'
      ..files.add(await http.MultipartFile.fromPath('video', videoPath));

    try {
      final streamedResponse = await _client
          .send(request)
          .timeout(const Duration(seconds: 90));
      final response = await http.Response.fromStream(streamedResponse);
      final bodyJson = tryDecodeJsonObject(response.body);

      if (response.statusCode >= 200 && response.statusCode < 300) {
        return VideoUploadResult(
          success: true,
          message:
              bodyJson?['message']?.toString() ??
              'Video uploaded successfully.',
          statusCode: response.statusCode,
          payload: bodyJson,
        );
      }

      final errorPayload = _errorPayload(bodyJson);

      return VideoUploadResult(
        success: false,
        errorCode:
            errorPayload?['error_code']?.toString() ??
            'E-UPLOAD-HTTP-${response.statusCode}',
        message:
            errorPayload?['message']?.toString() ??
            'Upload failed (${response.statusCode}).',
        statusCode: response.statusCode,
        payload: bodyJson,
      );
    } catch (e) {
      return VideoUploadResult(
        success: false,
        errorCode: 'E-UPLOAD-NETWORK',
        message: 'Failed to upload video: $e',
      );
    }
  }

  Future<VideoUploadResult> startAnalysis({
    required String jobId,
    Map<String, dynamic>? roiEllipse,
    String? roiSource,
  }) async {
    if (BackendConfig.useMockAnalysisFlow) {
      await Future<void>.delayed(const Duration(milliseconds: 500));
      return VideoUploadResult(
        success: true,
        message: 'ROI confirmed. Starting placeholder analysis...',
        payload: {
          'job_id': jobId,
          'status': 'processing',
          'stage_index': 3,
          'stage_total': 7,
          'stage_label': 'ROI Confirmed',
          'roi_ellipse': ?roiEllipse,
          'roi_source': ?roiSource,
        },
      );
    }

    if (BackendConfig.analysisStartEndpoint.isEmpty) {
      return const VideoUploadResult(
        success: false,
        errorCode: 'E-START-CONFIG',
        message:
            'Start URL is not configured. Update BackendConfig.apiBaseUrl first.',
      );
    }

    final uri = Uri.parse(BackendConfig.analysisStartEndpoint);
    final requestBody = <String, dynamic>{
      'job_id': jobId,
      'roi_ellipse': ?roiEllipse,
      if (roiSource != null && roiSource.isNotEmpty) 'roi_source': roiSource,
    };

    try {
      final response = await _client
          .post(
            uri,
            headers: const {'Content-Type': 'application/json'},
            body: jsonEncode(requestBody),
          )
          .timeout(const Duration(seconds: 45));
      final bodyJson = tryDecodeJsonObject(response.body);

      if (response.statusCode >= 200 && response.statusCode < 300) {
        return VideoUploadResult(
          success: true,
          message:
              bodyJson?['message']?.toString() ??
              'Analysis started successfully.',
          statusCode: response.statusCode,
          payload: bodyJson,
        );
      }

      final errorPayload = _errorPayload(bodyJson);
      return VideoUploadResult(
        success: false,
        errorCode:
            errorPayload?['error_code']?.toString() ??
            'E-START-HTTP-${response.statusCode}',
        message:
            errorPayload?['message']?.toString() ??
            'Could not start analysis (${response.statusCode}).',
        statusCode: response.statusCode,
        payload: bodyJson,
      );
    } catch (e) {
      return VideoUploadResult(
        success: false,
        errorCode: 'E-START-NETWORK',
        message: 'Failed to start analysis: $e',
      );
    }
  }

  Future<AnalysisPollResult> pollAnalysisStatus({required String jobId}) async {
    if (BackendConfig.useMockAnalysisFlow) {
      await Future<void>.delayed(const Duration(milliseconds: 1000));
      final nextAttempt = (_mockPollAttemptByJob[jobId] ?? 0) + 1;
      _mockPollAttemptByJob[jobId] = nextAttempt;

      if (nextAttempt >= 4) {
        return AnalysisPollResult(
          success: true,
          completed: true,
          message: 'Analysis completed.',
          stageIndex: 7,
          stageTotal: 7,
          stageLabel: 'Completed',
          progress: 1.0,
          payload: {
            'job_id': jobId,
            'confidence': 0.92,
            'quality_score': 8.3,
            'fps': 1,
            'start_frame': 0,
            'end_frame': 11,
            'blond_frame': 7,
            'shot_time_seconds': 12.0,
            'brightness_curve': [
              0.11,
              0.16,
              0.22,
              0.31,
              0.38,
              0.46,
              0.59,
              0.71,
              0.78,
              0.85,
              0.88,
              0.9,
            ],
            'channeling_counts': [1, 1, 2, 2, 3, 2, 2, 1, 1, 1, 0, 0],
            'stream_width_curve': [
              14.8,
              15.1,
              15.6,
              16.2,
              16.0,
              15.7,
              15.0,
              14.6,
              14.2,
              13.8,
              13.5,
              13.3,
            ],
            'stream_center_offset_curve': [
              0.8,
              0.7,
              0.5,
              0.2,
              0.1,
              -0.1,
              -0.3,
              -0.6,
              -0.5,
              -0.4,
              -0.2,
              0.0,
            ],
            'highlight_regions': [
              {
                'start_frame': 0,
                'end_frame': 4,
                'x': 0.42,
                'y': 0.23,
                'width': 0.16,
                'height': 0.46,
                'label': 'Basket and stream start',
              },
              {
                'start_frame': 5,
                'end_frame': 11,
                'x': 0.45,
                'y': 0.27,
                'width': 0.12,
                'height': 0.52,
                'label': 'Main stream tracking',
              },
            ],
          },
        );
      }

      return AnalysisPollResult(
        success: true,
        completed: false,
        message: 'Processing video ($nextAttempt/4)...',
        stageIndex: (nextAttempt + 3).clamp(4, 6),
        stageTotal: 7,
        stageLabel: const {
          1: 'Frame Extraction',
          2: 'ROI Tracking',
          3: 'Feature Extraction',
          4: 'Feature Extraction',
        }[nextAttempt],
        progress: (nextAttempt + 3).clamp(4, 6) / 7,
        payload: {'job_id': jobId, 'progress': nextAttempt / 4},
      );
    }

    if (BackendConfig.analysisStatusEndpoint.isEmpty) {
      return const AnalysisPollResult(
        success: false,
        completed: false,
        errorCode: 'E-PROCESS-CONFIG',
        message:
            'Status URL is not configured. Update BackendConfig.apiBaseUrl first.',
      );
    }

    try {
      final uri = Uri.parse(
        BackendConfig.analysisStatusEndpoint,
      ).replace(queryParameters: {'job_id': jobId});
      final response = await _client
          .get(uri)
          .timeout(const Duration(seconds: 30));

      final bodyJson = tryDecodeJsonObject(response.body);

      if (response.statusCode < 200 || response.statusCode >= 300) {
        final errorPayload = _errorPayload(bodyJson);
        return AnalysisPollResult(
          success: false,
          completed: false,
          errorCode:
              errorPayload?['error_code']?.toString() ??
              'E-PROCESS-HTTP-${response.statusCode}',
          message:
              errorPayload?['message']?.toString() ??
              'Status check failed (${response.statusCode}).',
          payload: bodyJson,
        );
      }

      final status = bodyJson?['status']?.toString().toLowerCase();
      final completed = status == 'completed' || status == 'done';

      if (status == 'failed' || status == 'error') {
        return AnalysisPollResult(
          success: false,
          completed: false,
          errorCode: bodyJson?['error_code']?.toString() ?? 'E-PROCESS-FAILED',
          message:
              bodyJson?['message']?.toString() ??
              'The video could not be processed by the backend.',
          payload: bodyJson,
        );
      }

      return AnalysisPollResult(
        success: true,
        completed: completed,
        message:
            bodyJson?['message']?.toString() ??
            (completed ? 'Analysis completed.' : 'Analysis in progress...'),
        stageIndex: (bodyJson?['stage_index'] as num?)?.toInt(),
        stageTotal: (bodyJson?['stage_total'] as num?)?.toInt(),
        stageLabel: bodyJson?['stage_label']?.toString(),
        progress: (bodyJson?['progress'] as num?)?.toDouble(),
        payload: bodyJson,
      );
    } catch (e) {
      return AnalysisPollResult(
        success: false,
        completed: false,
        errorCode: 'E-PROCESS-NETWORK',
        message: 'Failed to poll analysis status: $e',
      );
    }
  }
}

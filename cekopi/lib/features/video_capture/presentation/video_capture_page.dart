import 'dart:io';

import 'package:camera/camera.dart';
import 'package:cekopi/core/config/backend_config.dart';
import 'package:cekopi/core/navigation/app_routes.dart';
import 'package:cekopi/core/theme/app_colors.dart';
import 'package:cekopi/features/video_capture/data/video_upload_service.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

class VideoCapturePage extends StatefulWidget {
  const VideoCapturePage({
    super.key,
    this.initialErrorCode,
    this.initialErrorMessage,
  });

  final String? initialErrorCode;
  final String? initialErrorMessage;

  @override
  State<VideoCapturePage> createState() => _VideoCapturePageState();
}

class _VideoCapturePageState extends State<VideoCapturePage> {
  final VideoUploadService _uploadService = VideoUploadService();

  CameraController? _cameraController;
  bool _isRecording = false;
  bool _isUploading = false;
  String? _errorMessage;
  VideoStabilizationMode _videoStabilizationMode = VideoStabilizationMode.off;

  double _scale(double value) {
    final width = MediaQuery.of(context).size.width;
    final factor = (width / 390.0).clamp(0.86, 1.18);
    return value * factor;
  }

  double _sp(double value) => _scale(value);

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _showInitialErrorIfAny();
    });
    _initializeCamera();
  }

  Future<void> _showInitialErrorIfAny() async {
    final code = widget.initialErrorCode;
    final issue = widget.initialErrorMessage;
    if (!mounted ||
        code == null ||
        code.isEmpty ||
        issue == null ||
        issue.isEmpty) {
      return;
    }

    await _showAnalysisErrorPopup(errorCode: code, issue: issue);
  }

  Future<void> _initializeCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() {
          _errorMessage = 'No camera found on this device.';
        });
        return;
      }

      final selectedCamera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      final controller = CameraController(
        selectedCamera,
        ResolutionPreset.high,
        enableAudio: true,
      );

      await controller.initialize();
      await controller.prepareForVideoRecording();
      await _configureVideoStabilization(controller);
      if (!mounted) {
        await controller.dispose();
        return;
      }

      setState(() {
        _cameraController = controller;
        _errorMessage = null;
      });
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() {
        _errorMessage = 'Failed to initialize camera: $e';
      });
    }
  }

  Future<void> _configureVideoStabilization(CameraController controller) async {
    try {
      final supportedModes =
          (await controller.getSupportedVideoStabilizationModes()).toSet();
      final preferredMode = _pickPreferredVideoStabilizationMode(
        supportedModes,
      );

      if (preferredMode != VideoStabilizationMode.off) {
        await controller.setVideoStabilizationMode(preferredMode);
      }

      _videoStabilizationMode = preferredMode;
    } on CameraException {
      _videoStabilizationMode = VideoStabilizationMode.off;
    }
  }

  VideoStabilizationMode _pickPreferredVideoStabilizationMode(
    Set<VideoStabilizationMode> supportedModes,
  ) {
    for (final mode in const [
      VideoStabilizationMode.level3,
      VideoStabilizationMode.level2,
      VideoStabilizationMode.level1,
      VideoStabilizationMode.off,
    ]) {
      if (supportedModes.contains(mode)) {
        return mode;
      }
    }

    return VideoStabilizationMode.off;
  }

  String _videoStabilizationLabel(VideoStabilizationMode mode) {
    switch (mode) {
      case VideoStabilizationMode.level3:
        return 'Level 3';
      case VideoStabilizationMode.level2:
        return 'Level 2';
      case VideoStabilizationMode.level1:
        return 'Level 1';
      case VideoStabilizationMode.off:
        return 'Off';
    }
  }

  Future<void> _startRecording() async {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }

    try {
      await controller.startVideoRecording();
      if (!mounted) {
        return;
      }
      setState(() {
        _isRecording = true;
      });
    } catch (e) {
      _showSnackBar('Could not start recording: $e');
    }
  }

  Future<void> _stopRecordingAndUpload() async {
    final controller = _cameraController;
    if (controller == null || !controller.value.isRecordingVideo) {
      return;
    }

    try {
      final video = await controller.stopVideoRecording();
      if (!mounted) {
        return;
      }

      setState(() {
        _isRecording = false;
        _isUploading = true;
      });

      final result = await _uploadService.uploadVideo(videoPath: video.path);
      if (!mounted) {
        return;
      }

      setState(() {
        _isUploading = false;
      });

      if (!result.success) {
        await _handleUploadFailure(result);
        return;
      }

      await _continueToRoiConfirmation(result: result, videoPath: video.path);
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() {
        _isRecording = false;
        _isUploading = false;
      });
      _showSnackBar('Could not stop recording: $e');
    }
  }

  Future<void> _handleUploadFailure(VideoUploadResult result) async {
    await _showAnalysisErrorPopup(
      errorCode: result.errorCode ?? 'E-UPLOAD-UNKNOWN',
      issue: result.message,
    );
  }

  Future<void> _continueToRoiConfirmation({
    required VideoUploadResult result,
    required String videoPath,
  }) async {
    final jobId = result.payload?['job_id']?.toString();
    if (jobId == null || jobId.isEmpty) {
      await _showAnalysisErrorPopup(
        errorCode: 'E-UPLOAD-NO-JOB',
        issue: 'Upload succeeded but no job ID was returned by the backend.',
      );
      return;
    }

    final controllerToDispose = _cameraController;
    _cameraController = null;
    if (controllerToDispose != null) {
      await controllerToDispose.dispose();
    }

    if (!mounted) {
      return;
    }
    Navigator.of(context).pushReplacementNamed(
      AppRoutes.roiConfirmation,
      arguments: {
        'jobId': jobId,
        'videoPath': videoPath,
        'uploadPayload': result.payload,
      },
    );
  }

  Future<void> _uploadExistingVideoForDebug() async {
    if (!kDebugMode || _isUploading || _isRecording) {
      return;
    }

    final videoPath = BackendConfig.emulatorTestVideoPath;
    if (!await File(videoPath).exists()) {
      await _showAnalysisErrorPopup(
        errorCode: 'E-TEST-VIDEO-MISSING',
        issue:
            'No test video found at $videoPath. Push a video there with adb first.',
      );
      return;
    }

    setState(() {
      _isUploading = true;
    });

    final result = await _uploadService.uploadVideo(videoPath: videoPath);
    if (!mounted) {
      return;
    }

    setState(() {
      _isUploading = false;
    });

    if (!result.success) {
      await _handleUploadFailure(result);
      return;
    }

    await _continueToRoiConfirmation(result: result, videoPath: videoPath);
  }

  Future<void> _showAnalysisErrorPopup({
    required String errorCode,
    required String issue,
  }) async {
    await showGeneralDialog<void>(
      context: context,
      barrierColor: const Color(0x66000000),
      barrierDismissible: false,
      pageBuilder: (context, _, _) {
        return SafeArea(
          child: Center(
            child: Padding(
              padding: EdgeInsets.symmetric(horizontal: _scale(28)),
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 340),
                child: Material(
                  color: AppColors.surface,
                  borderRadius: BorderRadius.circular(_scale(16)),
                  child: Padding(
                    padding: EdgeInsets.fromLTRB(
                      _scale(16),
                      _scale(18),
                      _scale(16),
                      _scale(14),
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          width: _scale(82),
                          height: _scale(82),
                          decoration: BoxDecoration(
                            color: AppColors.cream300,
                            borderRadius: BorderRadius.circular(_scale(14)),
                          ),
                        ),
                        SizedBox(height: _scale(14)),
                        Text(
                          'Oops!\nSomething went wrong',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: _sp(34),
                            fontWeight: FontWeight.w800,
                            height: 1.02,
                            color: AppColors.headingText,
                          ),
                        ),
                        SizedBox(height: _scale(8)),
                        Text(
                          '$errorCode: $issue',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            color: AppColors.bodyText,
                            fontSize: _sp(13),
                          ),
                        ),
                        SizedBox(height: _scale(16)),
                        SizedBox(
                          width: double.infinity,
                          height: _scale(42),
                          child: FilledButton(
                            style: FilledButton.styleFrom(
                              backgroundColor: AppColors.primaryAction,
                              shape: const StadiumBorder(),
                            ),
                            onPressed: () => Navigator.of(context).pop(),
                            child: const Text('Okay'),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  void dispose() {
    _uploadService.close();
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _cameraController;

    return Scaffold(
      backgroundColor: AppColors.espresso900,
      appBar: AppBar(
        backgroundColor: AppColors.espresso900,
        foregroundColor: Colors.white,
        title: Text(
          'Record Analysis Video',
          style: TextStyle(
            color: Colors.white,
            fontSize: _sp(20),
            fontWeight: FontWeight.w700,
          ),
        ),
      ),
      body: _errorMessage != null
          ? _CameraErrorView(
              message: _errorMessage!,
              onRetry: _initializeCamera,
            )
          : controller == null || !controller.value.isInitialized
          ? const Center(child: CircularProgressIndicator())
          : Stack(
              children: [
                Positioned.fill(child: CameraPreview(controller)),
                if (_isUploading)
                  Positioned.fill(
                    child: Container(
                      color: AppColors.espresso900.withValues(alpha: 0.72),
                      child: const Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            CircularProgressIndicator(color: Colors.white),
                            SizedBox(height: 12),
                            Text(
                              'Uploading video to backend...',
                              style: TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                Align(
                  alignment: Alignment.bottomCenter,
                  child: Container(
                    width: double.infinity,
                    padding: EdgeInsets.fromLTRB(
                      _scale(16),
                      _scale(16),
                      _scale(16),
                      _scale(24),
                    ),
                    color: AppColors.espresso900.withValues(alpha: 0.78),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          'Capture a clear video of the coffee sample before uploading.',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            color: AppColors.cream200,
                            fontSize: _sp(14),
                          ),
                        ),
                        SizedBox(height: _scale(8)),
                        Text(
                          'Hardware stabilization: ${_videoStabilizationLabel(_videoStabilizationMode)}',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            color: AppColors.cream200.withValues(alpha: 0.88),
                            fontSize: _sp(12),
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        SizedBox(height: _scale(14)),
                        if (_isRecording)
                          FilledButton.icon(
                            onPressed: _isUploading
                                ? null
                                : _stopRecordingAndUpload,
                            style: FilledButton.styleFrom(
                              backgroundColor: AppColors.secondaryAction,
                              minimumSize: Size(_scale(220), _scale(48)),
                            ),
                            icon: const Icon(Icons.stop_circle_outlined),
                            label: const Text('Stop and Upload'),
                          )
                        else
                          FilledButton.icon(
                            onPressed: _isUploading ? null : _startRecording,
                            style: FilledButton.styleFrom(
                              backgroundColor: AppColors.primaryAction,
                              minimumSize: Size(_scale(220), _scale(48)),
                            ),
                            icon: const Icon(Icons.videocam),
                            label: const Text('Start Recording'),
                          ),
                        if (kDebugMode) ...[
                          SizedBox(height: _scale(10)),
                          OutlinedButton.icon(
                            onPressed: _isUploading || _isRecording
                                ? null
                                : _uploadExistingVideoForDebug,
                            style: OutlinedButton.styleFrom(
                              foregroundColor: AppColors.cream200,
                              side: BorderSide(
                                color: AppColors.cream200.withValues(
                                  alpha: 0.72,
                                ),
                              ),
                              minimumSize: Size(_scale(220), _scale(44)),
                            ),
                            icon: const Icon(Icons.video_file_outlined),
                            label: const Text('Upload emulator test video'),
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
              ],
            ),
    );
  }
}

class _CameraErrorView extends StatelessWidget {
  const _CameraErrorView({required this.message, required this.onRetry});

  final String message;
  final Future<void> Function() onRetry;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline, color: Colors.white, size: 42),
            const SizedBox(height: 14),
            Text(
              message,
              style: const TextStyle(color: Colors.white),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 14),
            FilledButton(onPressed: onRetry, child: const Text('Retry')),
          ],
        ),
      ),
    );
  }
}

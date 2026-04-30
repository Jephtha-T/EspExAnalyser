import 'package:flutter/material.dart';
import 'package:cekopi/core/navigation/app_routes.dart';
import 'package:cekopi/features/video_capture/data/video_upload_service.dart';

class AnalysisProcessingPage extends StatefulWidget {
  const AnalysisProcessingPage({
    super.key,
    required this.jobId,
    this.videoPath,
    this.roiEllipse,
    this.roiSource,
  });

  final String jobId;
  final String? videoPath;
  final Map<String, dynamic>? roiEllipse;
  final String? roiSource;

  @override
  State<AnalysisProcessingPage> createState() => _AnalysisProcessingPageState();
}

class _AnalysisProcessingPageState extends State<AnalysisProcessingPage> {
  final VideoUploadService _uploadService = VideoUploadService();

  String _statusMessage = 'Preparing analysis...';
  int _stageIndex = 3;
  int _stageTotal = 7;
  String _stageLabel = 'ROI Confirmed';
  double _progress = 3 / 7;
  bool _isStarting = true;
  bool _isPolling = false;

  @override
  void initState() {
    super.initState();
    _startAndPoll();
  }

  Future<void> _startAndPoll() async {
    final startResult = await _uploadService.startAnalysis(
      jobId: widget.jobId,
      roiEllipse: widget.roiEllipse,
      roiSource: widget.roiSource,
    );
    if (!mounted) {
      return;
    }

    if (!startResult.success) {
      _returnToRecorderWithError(
        errorCode: startResult.errorCode ?? 'E-START-UNKNOWN',
        issue: startResult.message,
      );
      return;
    }

    setState(() {
      _isStarting = false;
      _isPolling = true;
      _statusMessage = startResult.message;
      _stageIndex =
          (startResult.payload?['stage_index'] as num?)?.toInt() ?? _stageIndex;
      _stageTotal =
          (startResult.payload?['stage_total'] as num?)?.toInt() ?? _stageTotal;
      _stageLabel =
          startResult.payload?['stage_label']?.toString() ?? _stageLabel;
      _progress = (_stageTotal <= 0) ? 0.0 : _stageIndex / _stageTotal;
    });

    await _pollUntilCompleted();
  }

  Future<void> _pollUntilCompleted() async {
    const maxPollAttempts = 120;

    for (var attempt = 1; attempt <= maxPollAttempts; attempt++) {
      final pollResult = await _uploadService.pollAnalysisStatus(
        jobId: widget.jobId,
      );
      if (!mounted) {
        return;
      }

      if (!pollResult.success) {
        _returnToRecorderWithError(
          errorCode: pollResult.errorCode ?? 'E-PROCESS-UNKNOWN',
          issue: pollResult.message,
        );
        return;
      }

      if (pollResult.completed) {
        final payload = Map<String, dynamic>.from(
          pollResult.payload ?? <String, dynamic>{},
        );
        if (widget.roiEllipse != null) {
          payload['roi_ellipse'] = widget.roiEllipse;
        }
        if (widget.roiSource != null && widget.roiSource!.isNotEmpty) {
          payload['roi_source'] = widget.roiSource;
        }

        Navigator.of(context).pushReplacementNamed(
          AppRoutes.analysisResult,
          arguments: {'resultPayload': payload, 'videoPath': widget.videoPath},
        );
        return;
      }

      setState(() {
        _statusMessage = pollResult.message;
        _stageIndex = pollResult.stageIndex ?? _stageIndex;
        _stageTotal = pollResult.stageTotal ?? _stageTotal;
        _stageLabel = pollResult.stageLabel ?? _stageLabel;
        _progress =
            pollResult.progress ??
            (_stageIndex / (_stageTotal <= 0 ? 1 : _stageTotal));
      });
      await Future<void>.delayed(const Duration(seconds: 1));
    }

    if (!mounted) {
      return;
    }

    _returnToRecorderWithError(
      errorCode: 'E-PROCESS-TIMEOUT',
      issue:
          'Analysis is taking too long. The backend did not finish processing this video in time.',
    );
  }

  void _returnToRecorderWithError({
    required String errorCode,
    required String issue,
  }) {
    if (!mounted) {
      return;
    }

    Navigator.of(context).pushNamedAndRemoveUntil(
      AppRoutes.videoCapture,
      (route) => false,
      arguments: {'errorCode': errorCode, 'errorMessage': issue},
    );
  }

  @override
  void dispose() {
    _uploadService.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Processing Analysis')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (_isStarting || _isPolling) const CircularProgressIndicator(),
              const SizedBox(height: 16),
              Text(
                'Stage $_stageIndex / $_stageTotal',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.titleLarge,
              ),
              const SizedBox(height: 8),
              Text(
                _stageLabel,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 16),
              LinearProgressIndicator(value: _progress.clamp(0.0, 1.0)),
              const SizedBox(height: 16),
              Text(
                _statusMessage,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.titleMedium,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

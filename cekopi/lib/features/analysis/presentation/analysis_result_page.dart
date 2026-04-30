import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'package:cekopi/core/config/backend_config.dart';
import 'package:cekopi/core/theme/app_colors.dart';
import 'package:cekopi/core/navigation/app_routes.dart';
import 'package:cekopi/features/analysis/data/coffee_library_repository.dart';
import 'package:cekopi/features/analysis/data/saved_analysis_entry.dart';

class AnalysisResultPage extends StatefulWidget {
  const AnalysisResultPage({
    super.key,
    this.resultPayload,
    this.videoPath,
    this.initialBeanName,
    this.initialBeanType,
    this.initialRoastLevel,
    this.readOnly = false,
  });

  final Map<String, dynamic>? resultPayload;
  final String? videoPath;
  final String? initialBeanName;
  final String? initialBeanType;
  final String? initialRoastLevel;
  final bool readOnly;

  @override
  State<AnalysisResultPage> createState() => _AnalysisResultPageState();
}

class _AnalysisResultPageState extends State<AnalysisResultPage> {
  final CoffeeLibraryRepository _libraryRepository = CoffeeLibraryRepository();
  final TextEditingController _beanNameController = TextEditingController();
  static const List<String> _beanTypeOptions = <String>[
    'Arabica',
    'Robusto',
    'Liberica',
  ];
  static const List<String> _roastLevelOptions = <String>[
    'Light',
    'Medium',
    'Dark',
  ];

  String? _selectedBeanType;
  String? _selectedRoastLevel;

  VideoPlayerController? _videoController;
  Timer? _frameTicker;
  bool _videoReady = false;
  bool _isSaving = false;

  late final Map<String, dynamic> _payload;
  late final List<double> _brightnessCurve;
  late final List<double> _hueCurve;
  late final List<double> _saturationCurve;
  late final List<double> _channelingCurve;
  late final List<double> _streamWidthCurve;
  late final List<double> _streamCenterOffsetCurve;
  late final List<_HighlightRegion> _highlightRegions;
  late final List<String> _replayFrameUrls;

  late final double _fps;
  late final int _startFrame;
  late final int _endFrame;
  late int _totalFrames;
  late final int? _blondFrame;
  late final double _confidence;
  late final double _qualityScore;
  late final double _shotTimeSeconds;
  late final double _replayAspectRatio;
  late final String _predictedClass;
  late final List<String> _diagnosticFlags;
  late final List<String> _qualityFlags;

  int _currentFrame = 0;

  bool get _isFrameTickerPlaying => _frameTicker != null;

  double _scale(double value) {
    final width = MediaQuery.of(context).size.width;
    final factor = (width / 390.0).clamp(0.86, 1.18);
    return value * factor;
  }

  double _sp(double value) => _scale(value);

  bool get _isPlaying {
    if (_videoReady) {
      return _videoController?.value.isPlaying ?? false;
    }
    return _isFrameTickerPlaying;
  }

  @override
  void initState() {
    super.initState();

    _payload = Map<String, dynamic>.from(
      widget.resultPayload ?? <String, dynamic>{},
    );
    _brightnessCurve = _readCurve('brightness_curve');
    _hueCurve = _readCurve('hue_curve');
    _saturationCurve = _readCurve('saturation_curve');
    _channelingCurve = _readCurve('channeling_counts');
    _streamWidthCurve = _readCurve('stream_width_curve');
    _streamCenterOffsetCurve = _readCurve('stream_center_offset_curve');
    _highlightRegions = _readHighlightRegions();
    _replayFrameUrls = _readReplayFrameUrls();

    _fps = (_readNumber('fps')?.toDouble() ?? 1.0).clamp(0.25, 120.0);
    _startFrame = _readInt('start_frame') ?? 0;
    _endFrame = _readInt('end_frame') ?? (_maxCurveLength() - 1);
    _blondFrame = _readInt('blond_frame');
    _confidence = ((_readNumber('confidence')?.toDouble() ?? 0.92) * 100).clamp(
      0.0,
      100.0,
    );
    _qualityScore = (_readNumber('quality_score')?.toDouble() ?? 8.3).clamp(
      0.0,
      10.0,
    );
    _replayAspectRatio = _readReplayAspectRatio();
    _predictedClass = _readPredictedClass();
    _diagnosticFlags = _readStringListFromPath([
      'combined_assessment',
      'top_flags',
    ]);
    _qualityFlags = _readStringListFromPath(['quality', 'flags']);

    final explicitShotTime = _readNumber('shot_time_seconds');
    if (explicitShotTime != null) {
      _shotTimeSeconds = explicitShotTime.toDouble();
    } else {
      final frameCount = (_endFrame - _startFrame + 1).clamp(0, 100000);
      _shotTimeSeconds = frameCount / _fps;
    }

    _totalFrames = _inferTotalFrames();

    _beanNameController.text =
        (widget.initialBeanName ?? _readString('bean_name') ?? '').trim();
    _selectedBeanType = _matchOption(
      source: widget.initialBeanType ?? _readString('bean_type') ?? '',
      options: _beanTypeOptions,
    );
    _selectedRoastLevel = _matchOption(
      source: widget.initialRoastLevel ?? _readString('roast_level') ?? '',
      options: _roastLevelOptions,
    );

    _initializeVideo();
  }

  @override
  void dispose() {
    _frameTicker?.cancel();
    _videoController?.removeListener(_onVideoTick);
    _videoController?.dispose();
    _beanNameController.dispose();
    super.dispose();
  }

  String? _matchOption({
    required String source,
    required List<String> options,
  }) {
    final normalized = source.trim().toLowerCase();
    if (normalized.isEmpty) {
      return null;
    }
    for (final option in options) {
      if (option.toLowerCase() == normalized) {
        return option;
      }
    }
    return null;
  }

  Future<void> _initializeVideo() async {
    final path = widget.videoPath;
    if (path == null || path.isEmpty) {
      return;
    }

    final file = File(path);
    if (!file.existsSync()) {
      return;
    }

    final controller = VideoPlayerController.file(file);
    try {
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }

      _videoController = controller;
      _videoController?.addListener(_onVideoTick);

      final videoDurationMs = controller.value.duration.inMilliseconds;
      if (videoDurationMs > 0) {
        final byDuration = ((videoDurationMs / 1000.0) * _fps).round();
        if (byDuration > _totalFrames) {
          _totalFrames = byDuration;
        }
      }

      setState(() {
        _videoReady = true;
      });
    } catch (_) {
      await controller.dispose();
    }
  }

  void _onVideoTick() {
    final controller = _videoController;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }

    final positionMs = controller.value.position.inMilliseconds;
    final frame = (positionMs / 1000.0 * _fps)
        .round()
        .clamp(0, math.max(0, _totalFrames - 1))
        .toInt();

    if (frame != _currentFrame) {
      setState(() {
        _currentFrame = frame;
      });
    }
  }

  int _maxCurveLength() {
    final lengths = [
      _brightnessCurve.length,
      _hueCurve.length,
      _saturationCurve.length,
      _channelingCurve.length,
      _streamWidthCurve.length,
      _streamCenterOffsetCurve.length,
    ];
    return lengths.fold<int>(0, math.max);
  }

  int _inferTotalFrames() {
    final byFlowWindow = (_endFrame - _startFrame + 1).clamp(0, 100000);
    final byCurves = _maxCurveLength();
    final byReplay = _replayFrameUrls.length;
    final value = math.max(2, math.max(byReplay, math.max(byFlowWindow, byCurves)));
    return value;
  }

  num? _readNumber(String key) {
    final direct = _payload[key];
    if (direct is num) {
      return direct;
    }

    final nested = _payload['timeseries'];
    if (nested is Map<String, dynamic>) {
      final nestedValue = nested[key];
      if (nestedValue is num) {
        return nestedValue;
      }
    }
    return null;
  }

  int? _readInt(String key) {
    final value = _readNumber(key);
    return value?.toInt();
  }

  String? _readString(String key) {
    final direct = _payload[key]?.toString();
    if (direct != null && direct.isNotEmpty) {
      return direct;
    }

    final nested = _payload['timeseries'];
    if (nested is Map<String, dynamic>) {
      final nestedValue = nested[key]?.toString();
      if (nestedValue != null && nestedValue.isNotEmpty) {
        return nestedValue;
      }
    }
    return null;
  }

  List<double> _readCurve(String key) {
    List<dynamic>? values;
    if (_payload[key] is List) {
      values = _payload[key] as List<dynamic>;
    } else {
      final nested = _payload['timeseries'];
      if (nested is Map<String, dynamic> && nested[key] is List) {
        values = nested[key] as List<dynamic>;
      }
    }

    if (values == null || values.isEmpty) {
      return const <double>[];
    }

    final curve = values.map((item) {
      if (item is num) {
        return item.toDouble();
      }
      return double.nan;
    }).toList();

    if (curve.every((value) => value.isNaN)) {
      return const <double>[];
    }

    return curve;
  }

  Map<String, dynamic>? _readMap(String key) {
    final value = _payload[key];
    if (value is Map<String, dynamic>) {
      return value;
    }
    if (value is Map) {
      return value.map((key, value) => MapEntry(key.toString(), value));
    }
    return null;
  }

  Object? _readPath(List<String> path) {
    Object? cursor = _payload;
    for (final segment in path) {
      if (cursor is Map<String, dynamic>) {
        cursor = cursor[segment];
      } else if (cursor is Map) {
        cursor = cursor[segment];
      } else {
        return null;
      }
    }
    return cursor;
  }

  List<String> _readStringListFromPath(List<String> path) {
    final value = _readPath(path);
    if (value is List) {
      return value
          .map((item) => item?.toString().trim() ?? '')
          .where((item) => item.isNotEmpty)
          .toList();
    }
    return const <String>[];
  }

  List<String> _readReplayFrameUrls() {
    final source = _payload['replay_frame_urls'];
    if (source is! List) {
      return const <String>[];
    }

    return source
        .map((item) => item?.toString().trim() ?? '')
        .where((item) => item.isNotEmpty)
        .map((url) {
          if (url.startsWith('http://') || url.startsWith('https://')) {
            return url;
          }
          if (url.startsWith('/')) {
            return '${BackendConfig.apiBaseUrl}$url';
          }
          return '${BackendConfig.apiBaseUrl}/$url';
        })
        .toList();
  }

  double _readReplayAspectRatio() {
    final width = _readNumber('replay_frame_width')?.toDouble();
    final height = _readNumber('replay_frame_height')?.toDouble();
    if (width != null && height != null && width > 0 && height > 0) {
      return (width / height).clamp(0.55, 2.4).toDouble();
    }
    return 0.95;
  }

  String _readPredictedClass() {
    final model = _readMap('model_prediction');
    final candidates = <Object?>[
      model?['predicted_label'],
      model?['predicted_class'],
      model?['class'],
      model?['label'],
      model?['prediction'],
      _payload['predicted_class'],
    ];
    for (final candidate in candidates) {
      final value = candidate?.toString().trim();
      if (value != null && value.isNotEmpty && value.toLowerCase() != 'null') {
        return value;
      }
    }
    return 'Not available';
  }

  List<_HighlightRegion> _readHighlightRegions() {
    final dynamic source = _payload['highlight_regions'];
    if (source is! List) {
      return const <_HighlightRegion>[];
    }

    final parsed = source
        .whereType<Map<dynamic, dynamic>>()
        .map(
          (item) => _HighlightRegion.fromMap(
            item.map((key, value) => MapEntry(key.toString(), value)),
          ),
        )
        .whereType<_HighlightRegion>()
        .toList();

    return parsed;
  }

  _HighlightRegion? _currentRegion() {
    for (final region in _highlightRegions) {
      if (region.containsFrame(_currentFrame)) {
        return region;
      }
    }
    return null;
  }

  Future<void> _togglePlayback() async {
    if (_videoReady) {
      final controller = _videoController;
      if (controller == null || !controller.value.isInitialized) {
        return;
      }

      if (controller.value.isPlaying) {
        await controller.pause();
      } else {
        await controller.play();
      }
      if (mounted) {
        setState(() {});
      }
      return;
    }

    if (_frameTicker != null) {
      _frameTicker?.cancel();
      _frameTicker = null;
      setState(() {});
      return;
    }

    final frameMillis = (1000.0 / _fps).round().clamp(20, 300);
    _frameTicker = Timer.periodic(Duration(milliseconds: frameMillis), (_) {
      if (!mounted) {
        return;
      }

      setState(() {
        if (_currentFrame >= _totalFrames - 1) {
          _currentFrame = 0;
        } else {
          _currentFrame += 1;
        }
      });
    });
    setState(() {});
  }

  Future<void> _seekToFrame(int frame) async {
    final safeFrame = frame.clamp(0, math.max(0, _totalFrames - 1)).toInt();
    if (_videoReady) {
      final controller = _videoController;
      if (controller != null && controller.value.isInitialized) {
        final targetMs = ((safeFrame / _fps) * 1000).round();
        await controller.seekTo(Duration(milliseconds: targetMs));
      }
    }

    if (!mounted) {
      return;
    }

    setState(() {
      _currentFrame = safeFrame;
    });
  }

  Future<void> _closeWithoutSaving() async {
    final shouldClose =
        await showDialog<bool>(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text('Close analysis?'),
            content: const Text(
              'This analysis will be discarded and not added to your coffee library.',
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.of(context).pop(false),
                child: const Text('Cancel'),
              ),
              FilledButton(
                onPressed: () => Navigator.of(context).pop(true),
                child: const Text('Discard'),
              ),
            ],
          ),
        ) ??
        false;

    if (!shouldClose || !mounted) {
      return;
    }

    Navigator.of(
      context,
    ).pushNamedAndRemoveUntil(AppRoutes.home, (route) => false);
  }

  Future<void> _saveAnalysis() async {
    final beanName = _beanNameController.text.trim();
    final beanType = _selectedBeanType ?? '';
    final roastLevel = _selectedRoastLevel ?? '';

    if (beanName.isEmpty || beanType.isEmpty || roastLevel.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please enter bean name, bean type, and roast level.'),
        ),
      );
      return;
    }

    setState(() {
      _isSaving = true;
    });

    final payloadToSave = <String, dynamic>{
      ..._payload,
      'bean_name': beanName,
      'bean_type': beanType,
      'roast_level': roastLevel,
      'shot_time_seconds': _shotTimeSeconds,
      'fps': _fps,
      'start_frame': _startFrame,
      'end_frame': _endFrame,
      'blond_frame': _blondFrame,
      'confidence': _confidence / 100,
      'quality_score': _qualityScore,
      'brightness_curve': _brightnessCurve,
      'hue_curve': _hueCurve,
      'saturation_curve': _saturationCurve,
      'channeling_counts': _channelingCurve,
      'stream_width_curve': _streamWidthCurve,
      'stream_center_offset_curve': _streamCenterOffsetCurve,
      'highlight_regions': _highlightRegions.map((r) => r.toMap()).toList(),
    };

    final entry = SavedAnalysisEntry(
      id: DateTime.now().microsecondsSinceEpoch.toString(),
      savedAtIso: DateTime.now().toIso8601String(),
      beanName: beanName,
      beanType: beanType,
      roastLevel: roastLevel,
      resultPayload: payloadToSave,
      videoPath: widget.videoPath,
    );

    await _libraryRepository.saveEntry(entry);

    if (!mounted) {
      return;
    }

    setState(() {
      _isSaving = false;
    });

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Analysis saved to your coffee library.')),
    );

    Navigator.of(
      context,
    ).pushNamedAndRemoveUntil(AppRoutes.coffeeLibrary, (route) => false);
  }

  Future<void> _openSaveBottomSheet() async {
    if (widget.readOnly) {
      return;
    }

    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) {
        var modalBeanType = _selectedBeanType;
        var modalRoastLevel = _selectedRoastLevel;
        final bottomInset = MediaQuery.of(context).viewInsets.bottom;
        return Padding(
          padding: EdgeInsets.fromLTRB(
            _scale(16),
            0,
            _scale(16),
            math.max(10, bottomInset + 8),
          ),
          child: Align(
            alignment: Alignment.bottomCenter,
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 420),
              child: DecoratedBox(
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  borderRadius: BorderRadius.circular(_scale(18)),
                  boxShadow: [
                    BoxShadow(
                      color: AppColors.espresso900.withValues(alpha: 0.16),
                      blurRadius: _scale(16),
                      offset: Offset(0, _scale(8)),
                    ),
                  ],
                ),
                child: SafeArea(
                  top: false,
                  child: StatefulBuilder(
                    builder: (context, setModalState) {
                      return Padding(
                        padding: EdgeInsets.fromLTRB(
                          _scale(16),
                          _scale(14),
                          _scale(16),
                          _scale(10),
                        ),
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            Text(
                              'Save Analysis',
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                fontSize: _sp(32),
                                fontWeight: FontWeight.w800,
                                color: AppColors.headingText,
                              ),
                            ),
                            SizedBox(height: _scale(14)),
                            TextField(
                              controller: _beanNameController,
                              decoration: _popupInputDecoration('Bean Name'),
                            ),
                            SizedBox(height: _scale(10)),
                            DropdownButtonFormField<String>(
                              initialValue: modalBeanType,
                              items: _beanTypeOptions
                                  .map(
                                    (type) => DropdownMenuItem<String>(
                                      value: type,
                                      child: Text(type),
                                    ),
                                  )
                                  .toList(),
                              decoration: _popupInputDecoration('Bean Type'),
                              onChanged: (value) {
                                setModalState(() {
                                  modalBeanType = value;
                                });
                              },
                            ),
                            SizedBox(height: _scale(10)),
                            DropdownButtonFormField<String>(
                              initialValue: modalRoastLevel,
                              items: _roastLevelOptions
                                  .map(
                                    (level) => DropdownMenuItem<String>(
                                      value: level,
                                      child: Text(level),
                                    ),
                                  )
                                  .toList(),
                              decoration: _popupInputDecoration('Roast Level'),
                              onChanged: (value) {
                                setModalState(() {
                                  modalRoastLevel = value;
                                });
                              },
                            ),
                            SizedBox(height: _scale(16)),
                            SizedBox(
                              height: _scale(44),
                              child: FilledButton(
                                style: FilledButton.styleFrom(
                                  backgroundColor: AppColors.primaryAction,
                                  shape: const StadiumBorder(),
                                ),
                                onPressed: _isSaving
                                    ? null
                                    : () async {
                                        FocusScope.of(context).unfocus();
                                        setState(() {
                                          _selectedBeanType = modalBeanType;
                                          _selectedRoastLevel = modalRoastLevel;
                                        });
                                        await _saveAnalysis();
                                      },
                                child: Text(
                                  _isSaving ? 'Saving...' : 'Confirm',
                                ),
                              ),
                            ),
                            SizedBox(height: _scale(6)),
                            TextButton(
                              onPressed: _isSaving
                                  ? null
                                  : () => Navigator.of(context).pop(),
                              child: Text(
                                'Cancel',
                                style: TextStyle(
                                  color: AppColors.headingText,
                                  fontWeight: FontWeight.w500,
                                  fontSize: _sp(15),
                                ),
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  InputDecoration _popupInputDecoration(String hint) {
    return InputDecoration(
      hintText: hint,
      isDense: true,
      contentPadding: EdgeInsets.symmetric(
        horizontal: _scale(12),
        vertical: _scale(12),
      ),
      hintStyle: const TextStyle(color: AppColors.mutedText),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(_scale(8)),
        borderSide: const BorderSide(color: AppColors.cream300),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(_scale(8)),
        borderSide: const BorderSide(color: AppColors.cream300),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(_scale(8)),
        borderSide: const BorderSide(
          color: AppColors.secondaryAction,
          width: 1.2,
        ),
      ),
      filled: true,
      fillColor: Colors.white,
    );
  }

  @override
  Widget build(BuildContext context) {
    final currentRegion = _currentRegion();
    final markerFraction = _blondFrame == null || _totalFrames <= 1
        ? null
        : (_blondFrame / (_totalFrames - 1)).clamp(0.0, 1.0);

    return Scaffold(
      backgroundColor: AppColors.appBackground,
      body: SafeArea(
        child: Stack(
          children: [
            Positioned.fill(
              child: SingleChildScrollView(
                padding: EdgeInsets.fromLTRB(
                  _scale(12),
                  _scale(8),
                  _scale(12),
                  _scale(100),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    _buildSummaryCard(),
                    SizedBox(height: _scale(10)),
                    _buildPlaybackCard(currentRegion),
                    SizedBox(height: _scale(10)),
                    _buildChartsCard(markerFraction),
                    SizedBox(height: _scale(10)),
                    _buildDiagnosticsCard(),
                    if (widget.readOnly) ...[
                      SizedBox(height: _scale(10)),
                      _buildReadOnlyBeanCard(),
                    ],
                  ],
                ),
              ),
            ),
            Positioned(
              left: _scale(12),
              right: _scale(12),
              bottom: _scale(10),
              child: widget.readOnly
                  ? SizedBox(
                      height: _scale(44),
                      child: ElevatedButton.icon(
                        onPressed: () => Navigator.of(context).pop(),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppColors.primaryAction,
                          foregroundColor: Colors.white,
                          shape: const StadiumBorder(),
                          elevation: 0,
                        ),
                        icon: Icon(Icons.arrow_back_rounded, size: _scale(18)),
                        label: const Text('Back'),
                      ),
                    )
                  : Row(
                      children: [
                        Expanded(
                          child: SizedBox(
                            height: _scale(44),
                            child: ElevatedButton.icon(
                              onPressed: _isSaving ? null : _closeWithoutSaving,
                              style: ElevatedButton.styleFrom(
                                backgroundColor: AppColors.secondaryAction,
                                foregroundColor: Colors.white,
                                shape: const StadiumBorder(),
                                elevation: 0,
                              ),
                              icon: Icon(Icons.close_rounded, size: _scale(18)),
                              label: const Text('Close'),
                            ),
                          ),
                        ),
                        SizedBox(width: _scale(10)),
                        Expanded(
                          child: SizedBox(
                            height: _scale(44),
                            child: ElevatedButton.icon(
                              onPressed: _isSaving
                                  ? null
                                  : _openSaveBottomSheet,
                              style: ElevatedButton.styleFrom(
                                backgroundColor: AppColors.primaryAction,
                                foregroundColor: Colors.white,
                                shape: const StadiumBorder(),
                                elevation: 0,
                              ),
                              icon: _isSaving
                                  ? const SizedBox(
                                      width: 16,
                                      height: 16,
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2,
                                        color: Colors.white,
                                      ),
                                    )
                                  : const Icon(
                                      Icons.save_alt_rounded,
                                      size: 18,
                                    ),
                              label: Text(_isSaving ? 'Saving...' : 'Save'),
                            ),
                          ),
                        ),
                      ],
                    ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSummaryCard() {
    return _panel(
      child: Padding(
        padding: EdgeInsets.all(_scale(14)),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Shot Summary',
              style: TextStyle(fontSize: _sp(22), fontWeight: FontWeight.w700),
            ),
            SizedBox(height: _scale(10)),
            Wrap(
              spacing: _scale(10),
              runSpacing: _scale(10),
              children: [
                _MetricChip(
                  label: 'Shot Time',
                  value: '${_shotTimeSeconds.toStringAsFixed(1)} s',
                ),
                _MetricChip(
                  label: 'Quality Score',
                  value: '${_qualityScore.toStringAsFixed(1)} / 10',
                ),
                _MetricChip(label: 'Predicted Class', value: _predictedClass),
                _MetricChip(
                  label: 'Confidence',
                  value: '${_confidence.toStringAsFixed(1)}%',
                ),
                _MetricChip(
                  label: 'Blonding Frame',
                  value: _blondFrame?.toString() ?? '-',
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPlaybackCard(_HighlightRegion? currentRegion) {
    return _panel(
      child: Padding(
        padding: EdgeInsets.all(_scale(14)),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Frame Replay',
              style: TextStyle(fontSize: _sp(22), fontWeight: FontWeight.w700),
            ),
            SizedBox(height: _scale(8)),
            AspectRatio(
              aspectRatio: _replayAspectRatio,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(_scale(12)),
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    _buildPlaybackBody(),
                    if (currentRegion != null)
                      CustomPaint(
                        painter: _HighlightOverlayPainter(
                          region: currentRegion,
                        ),
                      ),
                    Positioned(
                      right: 10,
                      top: 10,
                      child: DecoratedBox(
                        decoration: BoxDecoration(
                          color: const Color(0xAA111111),
                          borderRadius: BorderRadius.circular(_scale(8)),
                        ),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 8,
                            vertical: 4,
                          ),
                          child: Text(
                            'Frame ${_currentFrame + 1}/$_totalFrames',
                            style: const TextStyle(color: Colors.white),
                          ),
                        ),
                      ),
                    ),
                    if (currentRegion != null)
                      Positioned(
                        left: 10,
                        bottom: 10,
                        child: DecoratedBox(
                          decoration: BoxDecoration(
                            color: const Color(0xAA111111),
                            borderRadius: BorderRadius.circular(_scale(8)),
                          ),
                          child: Padding(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 4,
                            ),
                            child: Text(
                              currentRegion.label,
                              style: const TextStyle(color: Colors.white),
                            ),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ),
            SizedBox(height: _scale(8)),
            Row(
              children: [
                IconButton(
                  onPressed: () => _seekToFrame(_currentFrame - 1),
                  icon: const Icon(Icons.skip_previous),
                ),
                IconButton(
                  onPressed: _togglePlayback,
                  icon: Icon(
                    _isPlaying
                        ? Icons.pause_circle_filled
                        : Icons.play_circle_fill,
                  ),
                ),
                IconButton(
                  onPressed: () => _seekToFrame(_currentFrame + 1),
                  icon: const Icon(Icons.skip_next),
                ),
                const Spacer(),
                Text(
                  '${(_currentFrame / _fps).toStringAsFixed(1)} s',
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: _sp(14),
                  ),
                ),
              ],
            ),
            Slider(
              min: 0,
              max: math.max(1, _totalFrames - 1).toDouble(),
              value: _currentFrame
                  .clamp(0, math.max(1, _totalFrames - 1))
                  .toDouble(),
              onChanged: (value) => _seekToFrame(value.round()),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPlaybackBody() {
    if (_replayFrameUrls.isNotEmpty) {
      final frameIndex = _currentFrame
          .clamp(0, _replayFrameUrls.length - 1)
          .toInt();
      return Image.network(
        _replayFrameUrls[frameIndex],
        fit: BoxFit.contain,
        gaplessPlayback: true,
        errorBuilder: (context, error, stackTrace) => _buildPlaybackFallback(),
        loadingBuilder: (context, child, loadingProgress) {
          if (loadingProgress == null) {
            return child;
          }
          return ColoredBox(
            color: AppColors.espresso900,
            child: const Center(
              child: CircularProgressIndicator(color: Colors.white),
            ),
          );
        },
      );
    }

    if (_videoReady &&
        _videoController != null &&
        _videoController!.value.isInitialized) {
      return FittedBox(
        fit: BoxFit.contain,
        child: SizedBox(
          width: _videoController!.value.size.width,
          height: _videoController!.value.size.height,
          child: VideoPlayer(_videoController!),
        ),
      );
    }

    return _buildPlaybackFallback();
  }

  Widget _buildPlaybackFallback() {
    return Container(
      color: AppColors.espresso900,
      child: const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.movie_creation_outlined,
              color: Colors.white70,
              size: 44,
            ),
            SizedBox(height: 8),
            Text(
              'Replay frames unavailable for this result',
              style: TextStyle(color: Colors.white70),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildChartsCard(double? markerFraction) {
    return _panel(
      child: Padding(
        padding: EdgeInsets.all(_scale(14)),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Extraction Curves',
              style: TextStyle(fontSize: _sp(22), fontWeight: FontWeight.w700),
            ),
            SizedBox(height: _scale(10)),
            _MetricLineChart(
              title: 'Stream colour transition',
              values: _hueCurve.isNotEmpty ? _hueCurve : _brightnessCurve,
              color: const Color(0xFF8D4C24),
              markerFraction: markerFraction,
            ),
            SizedBox(height: _scale(10)),
            _MetricLineChart(
              title: 'Channeling hole count',
              values: _channelingCurve,
              color: const Color(0xFF315C76),
              markerFraction: markerFraction,
              baselineZero: true,
            ),
            if (_saturationCurve.isNotEmpty) ...[
              SizedBox(height: _scale(10)),
              _MetricLineChart(
                title: 'Stream saturation',
                values: _saturationCurve,
                color: AppColors.caramel500,
                markerFraction: markerFraction,
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildDiagnosticsCard() {
    final flags = _diagnosticFlags.isNotEmpty ? _diagnosticFlags : _qualityFlags;
    final assessment = _readMap('combined_assessment');
    final explanation = assessment?['explanation']?.toString().trim();
    final channelingStats = _readMap('channeling_stats');
    final channelingQuality = _readMap('channeling_quality');
    final averageHoles = (channelingStats?['average'] as num?)?.toDouble();
    final peakHoles = (channelingStats?['max'] as num?)?.toDouble();
    final channelScore = (channelingQuality?['quality_score'] as num?)?.toDouble();

    return _panel(
      child: Padding(
        padding: EdgeInsets.all(_scale(14)),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Diagnostics',
              style: TextStyle(fontSize: _sp(22), fontWeight: FontWeight.w700),
            ),
            SizedBox(height: _scale(10)),
            Wrap(
              spacing: _scale(10),
              runSpacing: _scale(10),
              children: [
                if (averageHoles != null)
                  _MetricChip(
                    label: 'Avg Holes',
                    value: averageHoles.toStringAsFixed(0),
                  ),
                if (peakHoles != null)
                  _MetricChip(
                    label: 'Peak Holes',
                    value: peakHoles.toStringAsFixed(0),
                  ),
                if (channelScore != null)
                  _MetricChip(
                    label: 'Channel Score',
                    value: '${(channelScore * 10).toStringAsFixed(1)} / 10',
                  ),
              ],
            ),
            if (explanation != null && explanation.isNotEmpty) ...[
              SizedBox(height: _scale(12)),
              Text(
                explanation,
                style: TextStyle(
                  color: AppColors.bodyText,
                  fontSize: _sp(14),
                  height: 1.35,
                ),
              ),
            ],
            SizedBox(height: _scale(12)),
            if (flags.isEmpty)
              Text(
                'No diagnostic flags reported.',
                style: TextStyle(color: AppColors.bodyText, fontSize: _sp(14)),
              )
            else
              Wrap(
                spacing: _scale(8),
                runSpacing: _scale(8),
                children: flags
                    .map(
                      (flag) => _FlagPill(
                        label: flag
                            .replaceAll('_', ' ')
                            .replaceAll(RegExp(r'\s+'), ' ')
                            .trim(),
                      ),
                    )
                    .toList(),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildReadOnlyBeanCard() {
    final beanName = _beanNameController.text.trim();
    final beanType = (_selectedBeanType ?? '').trim();
    final roastLevel = (_selectedRoastLevel ?? '').trim();

    return _panel(
      child: Padding(
        padding: EdgeInsets.all(_scale(14)),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Saved Coffee Metadata',
              style: TextStyle(fontSize: _sp(18), fontWeight: FontWeight.w700),
            ),
            SizedBox(height: _scale(10)),
            _ResultRow(
              label: 'Bean Name',
              value: beanName.isEmpty ? '-' : beanName,
            ),
            SizedBox(height: _scale(8)),
            _ResultRow(
              label: 'Bean Type',
              value: beanType.isEmpty ? '-' : beanType,
            ),
            SizedBox(height: _scale(8)),
            _ResultRow(
              label: 'Roast Level',
              value: roastLevel.isEmpty ? '-' : roastLevel,
            ),
          ],
        ),
      ),
    );
  }

  Widget _panel({required Widget child}) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(_scale(14)),
        boxShadow: [
          BoxShadow(
            color: AppColors.espresso900.withValues(alpha: 0.14),
            blurRadius: _scale(8),
            offset: Offset(0, _scale(3)),
          ),
        ],
      ),
      child: child,
    );
  }

}

class _MetricChip extends StatelessWidget {
  const _MetricChip({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.of(context).size.width;
    final factor = (width / 390.0).clamp(0.86, 1.18);
    return DecoratedBox(
      decoration: BoxDecoration(
        color: AppColors.cream200,
        borderRadius: BorderRadius.circular(12 * factor),
      ),
      child: Padding(
        padding: EdgeInsets.symmetric(
          horizontal: 12 * factor,
          vertical: 8 * factor,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              label,
              style: TextStyle(
                fontSize: 11 * factor,
                color: AppColors.espresso700,
              ),
            ),
            SizedBox(height: 2 * factor),
            Text(
              value,
              style: TextStyle(
                fontWeight: FontWeight.w700,
                color: AppColors.headingText,
                fontSize: 13 * factor,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _FlagPill extends StatelessWidget {
  const _FlagPill({required this.label});

  final String label;

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.of(context).size.width;
    final factor = (width / 390.0).clamp(0.86, 1.18);
    return DecoratedBox(
      decoration: BoxDecoration(
        color: AppColors.espresso900.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(8 * factor),
        border: Border.all(
          color: AppColors.espresso900.withValues(alpha: 0.18),
        ),
      ),
      child: Padding(
        padding: EdgeInsets.symmetric(
          horizontal: 10 * factor,
          vertical: 7 * factor,
        ),
        child: Text(
          label,
          style: TextStyle(
            color: AppColors.headingText,
            fontWeight: FontWeight.w600,
            fontSize: 12 * factor,
          ),
        ),
      ),
    );
  }
}

class _ResultRow extends StatelessWidget {
  const _ResultRow({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.of(context).size.width;
    final factor = (width / 390.0).clamp(0.86, 1.18);
    return Row(
      children: [
        SizedBox(
          width: 110 * factor,
          child: Text(
            label,
            style: TextStyle(
              fontWeight: FontWeight.w600,
              color: AppColors.headingText,
              fontSize: 14 * factor,
            ),
          ),
        ),
        SizedBox(width: 8 * factor),
        Expanded(
          child: Text(
            value,
            style: TextStyle(color: AppColors.bodyText, fontSize: 14 * factor),
          ),
        ),
      ],
    );
  }
}

class _MetricLineChart extends StatelessWidget {
  const _MetricLineChart({
    required this.title,
    required this.values,
    required this.color,
    this.markerFraction,
    this.baselineZero = false,
  });

  final String title;
  final List<double> values;
  final Color color;
  final double? markerFraction;
  final bool baselineZero;

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.of(context).size.width;
    final factor = (width / 390.0).clamp(0.86, 1.18);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: TextStyle(
            fontWeight: FontWeight.w600,
            color: AppColors.headingText,
            fontSize: 13 * factor,
          ),
        ),
        SizedBox(height: 6 * factor),
        SizedBox(
          height: 96 * factor,
          child: CustomPaint(
            painter: _LineChartPainter(
              values: values,
              color: color,
              markerFraction: markerFraction,
              baselineZero: baselineZero,
            ),
            child: const SizedBox.expand(),
          ),
        ),
      ],
    );
  }
}

class _LineChartPainter extends CustomPainter {
  _LineChartPainter({
    required this.values,
    required this.color,
    required this.markerFraction,
    required this.baselineZero,
  });

  final List<double> values;
  final Color color;
  final double? markerFraction;
  final bool baselineZero;

  @override
  void paint(Canvas canvas, Size size) {
    final backgroundPaint = Paint()
      ..color = AppColors.caramel400.withValues(alpha: 0.42);
    final chartRect = Offset.zero & size;
    canvas.drawRRect(
      RRect.fromRectAndRadius(chartRect, const Radius.circular(10)),
      backgroundPaint,
    );

    final finiteValues = values.where((value) => value.isFinite).toList();
    if (finiteValues.length < 2) {
      return;
    }

    var minValue = finiteValues.reduce(math.min);
    var maxValue = finiteValues.reduce(math.max);
    if (baselineZero) {
      minValue = math.min(minValue, 0.0);
      maxValue = math.max(maxValue, 0.0);
    }
    if ((maxValue - minValue).abs() < 1e-6) {
      maxValue = minValue + 1.0;
    }

    const left = 8.0;
    const right = 8.0;
    const top = 10.0;
    const bottom = 10.0;

    final plotRect = Rect.fromLTRB(
      left,
      top,
      size.width - right,
      size.height - bottom,
    );

    final gridPaint = Paint()
      ..color = Colors.transparent
      ..strokeWidth = 1;
    for (var i = 0; i <= 3; i++) {
      final dy = plotRect.top + (plotRect.height * i / 3);
      canvas.drawLine(
        Offset(plotRect.left, dy),
        Offset(plotRect.right, dy),
        gridPaint,
      );
    }

    final path = Path();
    var hasStarted = false;

    for (var i = 0; i < values.length; i++) {
      final value = values[i];
      if (!value.isFinite) {
        hasStarted = false;
        continue;
      }

      final tx = values.length <= 1 ? 0.0 : i / (values.length - 1);
      final ty = (value - minValue) / (maxValue - minValue);

      final x = plotRect.left + plotRect.width * tx;
      final y = plotRect.bottom - plotRect.height * ty;

      if (!hasStarted) {
        path.moveTo(x, y);
        hasStarted = true;
      } else {
        path.lineTo(x, y);
      }
    }

    final linePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.8
      ..strokeCap = StrokeCap.round
      ..color = color.withValues(alpha: 0.68);
    canvas.drawPath(path, linePaint);

    if (markerFraction != null) {
      final markerX =
          plotRect.left + plotRect.width * markerFraction!.clamp(0.0, 1.0);
      final markerPaint = Paint()
        ..color = AppColors.espresso900.withValues(alpha: 0.72)
        ..strokeWidth = 1.5;
      canvas.drawLine(
        Offset(markerX, plotRect.top),
        Offset(markerX, plotRect.bottom),
        markerPaint,
      );
    }

    if (baselineZero && minValue <= 0 && maxValue >= 0) {
      final baselineTy = (0 - minValue) / (maxValue - minValue);
      final baselineY = plotRect.bottom - plotRect.height * baselineTy;
      final basePaint = Paint()
        ..color = AppColors.neutral700.withValues(alpha: 0.4)
        ..strokeWidth = 1;
      canvas.drawLine(
        Offset(plotRect.left, baselineY),
        Offset(plotRect.right, baselineY),
        basePaint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _LineChartPainter oldDelegate) {
    return oldDelegate.values != values ||
        oldDelegate.color != color ||
        oldDelegate.markerFraction != markerFraction ||
        oldDelegate.baselineZero != baselineZero;
  }
}

class _HighlightRegion {
  const _HighlightRegion({
    required this.startFrame,
    required this.endFrame,
    required this.x,
    required this.y,
    required this.width,
    required this.height,
    required this.label,
  });

  final int startFrame;
  final int endFrame;
  final double x;
  final double y;
  final double width;
  final double height;
  final String label;

  static _HighlightRegion? fromMap(Map<String, dynamic> json) {
    final startFrame = (json['start_frame'] as num?)?.toInt();
    final endFrame = (json['end_frame'] as num?)?.toInt();
    final x = (json['x'] as num?)?.toDouble();
    final y = (json['y'] as num?)?.toDouble();
    final width = (json['width'] as num?)?.toDouble();
    final height = (json['height'] as num?)?.toDouble();
    final label = json['label']?.toString() ?? 'Tracked region';

    if (startFrame == null ||
        endFrame == null ||
        x == null ||
        y == null ||
        width == null ||
        height == null) {
      return null;
    }

    return _HighlightRegion(
      startFrame: startFrame,
      endFrame: endFrame,
      x: x.clamp(0.0, 1.0),
      y: y.clamp(0.0, 1.0),
      width: width.clamp(0.02, 1.0),
      height: height.clamp(0.02, 1.0),
      label: label,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'start_frame': startFrame,
      'end_frame': endFrame,
      'x': x,
      'y': y,
      'width': width,
      'height': height,
      'label': label,
    };
  }

  bool containsFrame(int frame) {
    return frame >= startFrame && frame <= endFrame;
  }

  Rect toRect(Size size) {
    final left = size.width * x;
    final top = size.height * y;
    final rectWidth = size.width * width;
    final rectHeight = size.height * height;
    return Rect.fromLTWH(left, top, rectWidth, rectHeight);
  }
}

class _HighlightOverlayPainter extends CustomPainter {
  const _HighlightOverlayPainter({required this.region});

  final _HighlightRegion region;

  @override
  void paint(Canvas canvas, Size size) {
    final rect = region.toRect(size);
    final fillPaint = Paint()
      ..color = AppColors.caramel500.withValues(alpha: 0.2);
    final strokePaint = Paint()
      ..color = AppColors.caramel500
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    canvas.drawRect(rect, fillPaint);
    canvas.drawRect(rect, strokePaint);
  }

  @override
  bool shouldRepaint(covariant _HighlightOverlayPainter oldDelegate) {
    return oldDelegate.region != region;
  }
}

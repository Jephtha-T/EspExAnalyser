import 'dart:io';

import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'package:cekopi/core/navigation/app_routes.dart';
import 'package:cekopi/core/theme/app_colors.dart';

class RoiConfirmationPage extends StatefulWidget {
  const RoiConfirmationPage({
    super.key,
    required this.jobId,
    required this.videoPath,
    this.uploadPayload,
  });

  final String jobId;
  final String videoPath;
  final Map<String, dynamic>? uploadPayload;

  @override
  State<RoiConfirmationPage> createState() => _RoiConfirmationPageState();
}

class _RoiConfirmationPageState extends State<RoiConfirmationPage> {
  VideoPlayerController? _videoController;
  String? _videoError;
  bool _isReady = false;
  bool _isManualMode = false;
  bool _isContinuing = false;

  _NormalizedEllipse? _autoEllipse;
  _NormalizedEllipse? _manualEllipse;
  Offset? _dragStart;
  Offset? _dragCurrent;

  double _scale(double value) {
    final width = MediaQuery.of(context).size.width;
    final factor = (width / 390.0).clamp(0.86, 1.18);
    return value * factor;
  }

  double _sp(double value) => _scale(value);

  @override
  void initState() {
    super.initState();
    _autoEllipse = _NormalizedEllipse.fromUploadPayload(widget.uploadPayload);
    _initializeVideo();
  }

  Future<void> _initializeVideo() async {
    final controller = VideoPlayerController.file(File(widget.videoPath));
    try {
      await controller.initialize();
      await controller.pause();
      await controller.seekTo(Duration.zero);
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() {
        _videoController = controller;
        _isReady = true;
      });
    } catch (e) {
      await controller.dispose();
      if (!mounted) {
        return;
      }
      setState(() {
        _videoError = 'Could not load video preview: $e';
      });
    }
  }

  _NormalizedEllipse? get _selectedEllipse =>
      _isManualMode ? _manualEllipse : _autoEllipse;

  void _startManualSelection() {
    setState(() {
      _isManualMode = true;
      _manualEllipse = null;
      _dragStart = null;
      _dragCurrent = null;
    });
  }

  void _redrawManualSelection() {
    setState(() {
      _manualEllipse = null;
      _dragStart = null;
      _dragCurrent = null;
    });
  }

  void _onPanStart(DragStartDetails details, Size size) {
    if (!_isManualMode || size.width <= 0 || size.height <= 0) {
      return;
    }
    setState(() {
      _dragStart = details.localPosition;
      _dragCurrent = details.localPosition;
      _manualEllipse = null;
    });
  }

  void _onPanUpdate(DragUpdateDetails details, Size size) {
    if (!_isManualMode ||
        _dragStart == null ||
        size.width <= 0 ||
        size.height <= 0) {
      return;
    }
    final x = details.localPosition.dx.clamp(0.0, size.width);
    final y = details.localPosition.dy.clamp(0.0, size.height);
    setState(() {
      _dragCurrent = Offset(x, y);
    });
  }

  void _onPanEnd(Size size) {
    if (!_isManualMode || _dragStart == null || _dragCurrent == null) {
      return;
    }

    final start = _dragStart!;
    final end = _dragCurrent!;
    final left = start.dx < end.dx ? start.dx : end.dx;
    final right = start.dx > end.dx ? start.dx : end.dx;
    final top = start.dy < end.dy ? start.dy : end.dy;
    final bottom = start.dy > end.dy ? start.dy : end.dy;

    final widthPx = right - left;
    final heightPx = bottom - top;

    if (widthPx < 16 || heightPx < 16) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Selection is too small. Drag a wider ellipse.'),
        ),
      );
      setState(() {
        _dragStart = null;
        _dragCurrent = null;
      });
      return;
    }

    setState(() {
      _manualEllipse = _NormalizedEllipse(
        cx: ((left + right) / 2.0) / size.width,
        cy: ((top + bottom) / 2.0) / size.height,
        width: widthPx / size.width,
        height: heightPx / size.height,
        angleDeg: 0,
      ).clamped();
      _dragStart = null;
      _dragCurrent = null;
    });
  }

  void _continueWithCurrentSelection() {
    final selected = _selectedEllipse;
    if (_isManualMode && selected == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Draw an ellipse first, then continue.')),
      );
      return;
    }

    if (_isContinuing) {
      return;
    }

    setState(() {
      _isContinuing = true;
    });

    Navigator.of(context).pushReplacementNamed(
      AppRoutes.analysisProcessing,
      arguments: {
        'jobId': widget.jobId,
        'videoPath': widget.videoPath,
        'roiEllipse': selected?.toMap(),
        'roiSource': _isManualMode ? 'manual' : 'auto',
      },
    );
  }

  @override
  void dispose() {
    _videoController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _videoController;

    return Scaffold(
      backgroundColor: AppColors.espresso900,
      appBar: AppBar(
        backgroundColor: AppColors.espresso900,
        foregroundColor: Colors.white,
        title: Text(
          'Confirm Portafilter ROI',
          style: TextStyle(
            color: Colors.white,
            fontSize: _sp(20),
            fontWeight: FontWeight.w700,
          ),
        ),
      ),
      body: _videoError != null
          ? _RoiErrorView(
              message: _videoError!,
              onBack: () => Navigator.of(context).pushNamedAndRemoveUntil(
                AppRoutes.videoCapture,
                (route) => false,
              ),
            )
          : !_isReady || controller == null || !controller.value.isInitialized
          ? const Center(child: CircularProgressIndicator())
          : Stack(
              children: [
                SafeArea(
                  child: Padding(
                    padding: EdgeInsets.fromLTRB(
                      _scale(14),
                      _scale(12),
                      _scale(14),
                      _scale(118),
                    ),
                    child: Center(
                      child: AspectRatio(
                        aspectRatio: controller.value.aspectRatio > 0
                            ? controller.value.aspectRatio
                            : 16 / 9,
                        child: LayoutBuilder(
                          builder: (context, constraints) {
                            final canvasSize = Size(
                              constraints.maxWidth,
                              constraints.maxHeight,
                            );

                            return ClipRRect(
                              borderRadius: BorderRadius.circular(_scale(18)),
                              child: GestureDetector(
                                onPanStart: _isManualMode
                                    ? (details) =>
                                          _onPanStart(details, canvasSize)
                                    : null,
                                onPanUpdate: _isManualMode
                                    ? (details) =>
                                          _onPanUpdate(details, canvasSize)
                                    : null,
                                onPanEnd: _isManualMode
                                    ? (_) => _onPanEnd(canvasSize)
                                    : null,
                                child: Stack(
                                  fit: StackFit.expand,
                                  children: [
                                    VideoPlayer(controller),
                                    CustomPaint(
                                      painter: _EllipseOverlayPainter(
                                        autoEllipse: _autoEllipse,
                                        manualEllipse: _manualEllipse,
                                        dragStart: _dragStart,
                                        dragCurrent: _dragCurrent,
                                        isManualMode: _isManualMode,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            );
                          },
                        ),
                      ),
                    ),
                  ),
                ),
                if (_isManualMode)
                  Positioned(
                    top: _scale(10),
                    left: _scale(14),
                    right: _scale(14),
                    child: Material(
                      color: Colors.transparent,
                      child: DecoratedBox(
                        decoration: BoxDecoration(
                          color: AppColors.caramel500,
                          borderRadius: BorderRadius.circular(_scale(14)),
                        ),
                        child: Padding(
                          padding: EdgeInsets.symmetric(
                            horizontal: _scale(14),
                            vertical: _scale(10),
                          ),
                          child: Text(
                            'Drag on the video to draw where the portafilter basket is.',
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: _sp(13),
                              fontWeight: FontWeight.w600,
                            ),
                          ),
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
                      _scale(14),
                      _scale(16),
                      _scale(20),
                    ),
                    color: AppColors.espresso900.withValues(alpha: 0.82),
                    child: Row(
                      children: [
                        Expanded(
                          child: FilledButton(
                            onPressed: _isContinuing
                                ? null
                                : (_isManualMode
                                      ? _redrawManualSelection
                                      : _startManualSelection),
                            style: FilledButton.styleFrom(
                              backgroundColor: AppColors.caramel500,
                              minimumSize: Size(0, _scale(46)),
                            ),
                            child: Text(
                              _isManualMode ? 'Redraw' : 'Select Manually',
                            ),
                          ),
                        ),
                        SizedBox(width: _scale(12)),
                        Expanded(
                          child: FilledButton(
                            onPressed: _isContinuing
                                ? null
                                : _continueWithCurrentSelection,
                            style: FilledButton.styleFrom(
                              backgroundColor: AppColors.primaryAction,
                              minimumSize: Size(0, _scale(46)),
                            ),
                            child: Text(
                              _isContinuing ? 'Loading...' : 'Continue',
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
    );
  }
}

class _RoiErrorView extends StatelessWidget {
  const _RoiErrorView({required this.message, required this.onBack});

  final String message;
  final VoidCallback onBack;

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
            FilledButton(onPressed: onBack, child: const Text('Back')),
          ],
        ),
      ),
    );
  }
}

class _EllipseOverlayPainter extends CustomPainter {
  const _EllipseOverlayPainter({
    required this.autoEllipse,
    required this.manualEllipse,
    required this.dragStart,
    required this.dragCurrent,
    required this.isManualMode,
  });

  final _NormalizedEllipse? autoEllipse;
  final _NormalizedEllipse? manualEllipse;
  final Offset? dragStart;
  final Offset? dragCurrent;
  final bool isManualMode;

  @override
  void paint(Canvas canvas, Size size) {
    final auto = autoEllipse;
    final manual = manualEllipse;

    if (!isManualMode && auto != null) {
      _drawEllipse(canvas, size, auto, strokeColor: const Color(0xFF31D2F2));
    }

    if (isManualMode && manual != null) {
      _drawEllipse(canvas, size, manual, strokeColor: const Color(0xFFFFC84A));
    }

    if (isManualMode && dragStart != null && dragCurrent != null) {
      final rect = Rect.fromPoints(dragStart!, dragCurrent!);
      final previewPaint = Paint()
        ..color = const Color(0xFFFFC84A)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5;
      canvas.drawRect(rect, previewPaint);
      canvas.drawOval(rect, previewPaint);
    }
  }

  void _drawEllipse(
    Canvas canvas,
    Size size,
    _NormalizedEllipse ellipse, {
    required Color strokeColor,
  }) {
    final rect = Rect.fromCenter(
      center: Offset(ellipse.cx * size.width, ellipse.cy * size.height),
      width: ellipse.width * size.width,
      height: ellipse.height * size.height,
    );

    final fillPaint = Paint()
      ..color = strokeColor.withValues(alpha: 0.16)
      ..style = PaintingStyle.fill;
    final strokePaint = Paint()
      ..color = strokeColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.8;

    canvas.drawOval(rect, fillPaint);
    canvas.drawOval(rect, strokePaint);
  }

  @override
  bool shouldRepaint(covariant _EllipseOverlayPainter oldDelegate) {
    return oldDelegate.autoEllipse != autoEllipse ||
        oldDelegate.manualEllipse != manualEllipse ||
        oldDelegate.dragStart != dragStart ||
        oldDelegate.dragCurrent != dragCurrent ||
        oldDelegate.isManualMode != isManualMode;
  }
}

class _NormalizedEllipse {
  const _NormalizedEllipse({
    required this.cx,
    required this.cy,
    required this.width,
    required this.height,
    required this.angleDeg,
  });

  final double cx;
  final double cy;
  final double width;
  final double height;
  final double angleDeg;

  _NormalizedEllipse clamped() {
    return _NormalizedEllipse(
      cx: cx.clamp(0.0, 1.0),
      cy: cy.clamp(0.0, 1.0),
      width: width.clamp(0.02, 1.0),
      height: height.clamp(0.02, 1.0),
      angleDeg: angleDeg,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'cx': cx,
      'cy': cy,
      'width': width,
      'height': height,
      'angle_deg': angleDeg,
    };
  }

  static _NormalizedEllipse fromUploadPayload(Map<String, dynamic>? payload) {
    final extracted = _extractRawEllipse(payload);
    if (extracted != null) {
      return extracted.clamped();
    }

    return const _NormalizedEllipse(
      cx: 0.5,
      cy: 0.45,
      width: 0.62,
      height: 0.5,
      angleDeg: 0,
    );
  }

  static _NormalizedEllipse? _extractRawEllipse(Map<String, dynamic>? payload) {
    if (payload == null) {
      return null;
    }

    final candidates = [
      payload['auto_roi_ellipse'],
      payload['auto_roi'],
      payload['portafilter_roi'],
      payload['ellipse'],
    ];

    for (final candidate in candidates) {
      final parsed = _fromDynamic(candidate, payload: payload);
      if (parsed != null) {
        return parsed;
      }
    }

    return null;
  }

  static _NormalizedEllipse? _fromDynamic(
    dynamic value, {
    required Map<String, dynamic> payload,
  }) {
    if (value is Map<String, dynamic>) {
      final cx = _toDouble(
        value['cx'] ?? value['center_x'] ?? value['centerX'] ?? value['x'],
      );
      final cy = _toDouble(
        value['cy'] ?? value['center_y'] ?? value['centerY'] ?? value['y'],
      );
      final width = _toDouble(value['width'] ?? value['w']);
      final height = _toDouble(value['height'] ?? value['h']);
      final angle = _toDouble(value['angle_deg'] ?? value['angle'] ?? 0) ?? 0;

      if (cx == null || cy == null || width == null || height == null) {
        return null;
      }

      return _normalizeIfNeeded(
        cx: cx,
        cy: cy,
        width: width,
        height: height,
        angleDeg: angle,
        payload: payload,
      );
    }

    if (value is List && value.length == 3) {
      final center = value[0];
      final size = value[1];
      final angle = _toDouble(value[2]) ?? 0;
      if (center is List &&
          center.length >= 2 &&
          size is List &&
          size.length >= 2) {
        final cx = _toDouble(center[0]);
        final cy = _toDouble(center[1]);
        final width = _toDouble(size[0]);
        final height = _toDouble(size[1]);
        if (cx == null || cy == null || width == null || height == null) {
          return null;
        }
        return _normalizeIfNeeded(
          cx: cx,
          cy: cy,
          width: width,
          height: height,
          angleDeg: angle,
          payload: payload,
        );
      }
    }

    return null;
  }

  static _NormalizedEllipse? _normalizeIfNeeded({
    required double cx,
    required double cy,
    required double width,
    required double height,
    required double angleDeg,
    required Map<String, dynamic> payload,
  }) {
    if (width <= 0 || height <= 0) {
      return null;
    }

    final valuesAreNormalized =
        cx >= 0 && cx <= 1 && cy >= 0 && cy <= 1 && width <= 1 && height <= 1;

    if (valuesAreNormalized) {
      return _NormalizedEllipse(
        cx: cx,
        cy: cy,
        width: width,
        height: height,
        angleDeg: angleDeg,
      );
    }

    final frameWidth = _toDouble(
      payload['frame_width'] ?? payload['width'] ?? payload['video_width'],
    );
    final frameHeight = _toDouble(
      payload['frame_height'] ?? payload['height'] ?? payload['video_height'],
    );

    if (frameWidth == null ||
        frameHeight == null ||
        frameWidth <= 0 ||
        frameHeight <= 0) {
      return null;
    }

    return _NormalizedEllipse(
      cx: cx / frameWidth,
      cy: cy / frameHeight,
      width: width / frameWidth,
      height: height / frameHeight,
      angleDeg: angleDeg,
    );
  }

  static double? _toDouble(dynamic value) {
    if (value is num) {
      return value.toDouble();
    }
    if (value is String) {
      return double.tryParse(value);
    }
    return null;
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) {
      return true;
    }
    return other is _NormalizedEllipse &&
        other.cx == cx &&
        other.cy == cy &&
        other.width == width &&
        other.height == height &&
        other.angleDeg == angleDeg;
  }

  @override
  int get hashCode => Object.hash(cx, cy, width, height, angleDeg);
}

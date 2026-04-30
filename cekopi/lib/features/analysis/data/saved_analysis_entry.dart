import 'package:cekopi/core/utils/json_helpers.dart';

class SavedAnalysisEntry {
  const SavedAnalysisEntry({
    required this.id,
    required this.savedAtIso,
    required this.beanName,
    required this.beanType,
    required this.roastLevel,
    required this.resultPayload,
    this.videoPath,
  });

  final String id;
  final String savedAtIso;
  final String beanName;
  final String beanType;
  final String roastLevel;
  final Map<String, dynamic> resultPayload;
  final String? videoPath;

  DateTime get savedAt {
    return DateTime.tryParse(savedAtIso) ??
        DateTime.fromMillisecondsSinceEpoch(0);
  }

  double get shotTimeSeconds {
    final direct = (resultPayload['shot_time_seconds'] as num?)?.toDouble();
    if (direct != null) {
      return direct;
    }

    final fps = ((resultPayload['fps'] as num?)?.toDouble() ?? 1.0).clamp(
      0.1,
      120.0,
    );
    final startFrame =
        (resultPayload['start_frame'] as num?)?.toDouble() ?? 0.0;
    final endFrame =
        (resultPayload['end_frame'] as num?)?.toDouble() ?? startFrame;
    final frameCount = (endFrame - startFrame + 1).clamp(0.0, 1000000.0);
    return frameCount / fps;
  }

  SavedAnalysisEntry copyWith({
    String? id,
    String? savedAtIso,
    String? beanName,
    String? beanType,
    String? roastLevel,
    Map<String, dynamic>? resultPayload,
    String? videoPath,
  }) {
    return SavedAnalysisEntry(
      id: id ?? this.id,
      savedAtIso: savedAtIso ?? this.savedAtIso,
      beanName: beanName ?? this.beanName,
      beanType: beanType ?? this.beanType,
      roastLevel: roastLevel ?? this.roastLevel,
      resultPayload: resultPayload ?? this.resultPayload,
      videoPath: videoPath ?? this.videoPath,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'saved_at': savedAtIso,
      'bean_name': beanName,
      'bean_type': beanType,
      'roast_level': roastLevel,
      'video_path': videoPath,
      'result_payload': resultPayload,
    };
  }

  static SavedAnalysisEntry fromJson(Map<String, dynamic> json) {
    return SavedAnalysisEntry(
      id: json['id']?.toString() ?? '',
      savedAtIso:
          json['saved_at']?.toString() ?? DateTime.now().toIso8601String(),
      beanName: json['bean_name']?.toString() ?? '',
      beanType: json['bean_type']?.toString() ?? '',
      roastLevel: json['roast_level']?.toString() ?? '',
      videoPath: json['video_path']?.toString(),
      resultPayload:
          asStringKeyedMap(json['result_payload']) ?? <String, dynamic>{},
    );
  }
}

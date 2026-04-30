import 'dart:convert';

import 'package:cekopi/core/utils/json_helpers.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:cekopi/features/analysis/data/saved_analysis_entry.dart';

class CoffeeLibraryRepository {
  static const String _storageKey = 'saved_analysis_entries_v1';

  Future<List<SavedAnalysisEntry>> loadEntries() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_storageKey);
    if (raw == null || raw.isEmpty) {
      return <SavedAnalysisEntry>[];
    }

    try {
      final decoded = jsonDecode(raw);
      if (decoded is! List) {
        return <SavedAnalysisEntry>[];
      }

      final entries = decoded
          .whereType<Map<dynamic, dynamic>>()
          .map(asStringKeyedMap)
          .whereType<Map<String, dynamic>>()
          .map(SavedAnalysisEntry.fromJson)
          .toList();

      entries.sort((a, b) => b.savedAt.compareTo(a.savedAt));
      return entries;
    } catch (_) {
      return <SavedAnalysisEntry>[];
    }
  }

  Future<void> saveEntry(SavedAnalysisEntry entry) async {
    final existing = await loadEntries();
    final next = <SavedAnalysisEntry>[
      entry,
      ...existing.where((e) => e.id != entry.id),
    ];
    await _persist(next);
  }

  Future<void> deleteEntryById(String id) async {
    final existing = await loadEntries();
    final next = existing.where((entry) => entry.id != id).toList();
    await _persist(next);
  }

  Future<void> deleteEntriesByIds(Iterable<String> ids) async {
    final idSet = ids.toSet();
    if (idSet.isEmpty) {
      return;
    }
    final existing = await loadEntries();
    final next = existing.where((entry) => !idSet.contains(entry.id)).toList();
    await _persist(next);
  }

  Future<void> renameBeanNameForGroup({
    required String fromBeanName,
    required String beanType,
    required String roastLevel,
    required String toBeanName,
  }) async {
    final trimmedName = toBeanName.trim();
    if (trimmedName.isEmpty) {
      return;
    }

    final existing = await loadEntries();
    final next = existing.map((entry) {
      final matchesGroup =
          entry.beanName == fromBeanName &&
          entry.beanType == beanType &&
          entry.roastLevel == roastLevel;
      if (!matchesGroup) {
        return entry;
      }
      return entry.copyWith(beanName: trimmedName);
    }).toList();

    await _persist(next);
  }

  Future<void> _persist(List<SavedAnalysisEntry> entries) async {
    final prefs = await SharedPreferences.getInstance();
    final encoded = jsonEncode(entries.map((e) => e.toJson()).toList());
    await prefs.setString(_storageKey, encoded);
  }
}

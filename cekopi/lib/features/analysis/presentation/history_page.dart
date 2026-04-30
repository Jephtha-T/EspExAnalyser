import 'package:flutter/material.dart';
import 'package:cekopi/core/navigation/app_routes.dart';
import 'package:cekopi/features/analysis/data/coffee_library_repository.dart';
import 'package:cekopi/features/analysis/data/saved_analysis_entry.dart';

class HistoryPage extends StatefulWidget {
  const HistoryPage({super.key});

  @override
  State<HistoryPage> createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  final CoffeeLibraryRepository _repository = CoffeeLibraryRepository();
  late Future<List<SavedAnalysisEntry>> _entriesFuture;

  static const List<String> _sectionOrder = <String>[
    'Today',
    'Yesterday',
    'This Week',
    'This Month',
    'Older',
  ];

  @override
  void initState() {
    super.initState();
    _entriesFuture = _repository.loadEntries();
  }

  void _refresh() {
    setState(() {
      _entriesFuture = _repository.loadEntries();
    });
  }

  Future<void> _deleteEntry(SavedAnalysisEntry entry) async {
    final shouldDelete = await showDialog<bool>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Delete history item?'),
          content: Text(
            'Remove ${entry.beanName.isEmpty ? 'this item' : entry.beanName} from history?',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(false),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.of(context).pop(true),
              child: const Text('Delete'),
            ),
          ],
        );
      },
    );

    if (shouldDelete != true) {
      return;
    }

    await _repository.deleteEntryById(entry.id);
    if (!mounted) {
      return;
    }

    _refresh();
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('History item deleted.')));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('History'),
        actions: [
          IconButton(
            onPressed: _refresh,
            icon: const Icon(Icons.refresh),
            tooltip: 'Refresh',
          ),
          IconButton(
            onPressed: () => Navigator.of(
              context,
            ).pushNamedAndRemoveUntil(AppRoutes.home, (route) => false),
            icon: const Icon(Icons.close),
            tooltip: 'Close',
          ),
        ],
      ),
      body: FutureBuilder<List<SavedAnalysisEntry>>(
        future: _entriesFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }

          final entries = snapshot.data ?? <SavedAnalysisEntry>[];
          if (entries.isEmpty) {
            return const _EmptyHistoryView();
          }

          final sections = _groupByRecency(entries);
          final ordered = _sectionOrder
              .where((section) => (sections[section]?.isNotEmpty ?? false))
              .toList();

          return ListView.builder(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
            itemCount: ordered.length,
            itemBuilder: (context, index) {
              final sectionName = ordered[index];
              final sectionEntries =
                  sections[sectionName] ?? <SavedAnalysisEntry>[];
              return Padding(
                padding: const EdgeInsets.only(bottom: 14),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      sectionName,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 8),
                    ...sectionEntries.map((entry) {
                      final shotTime = entry.shotTimeSeconds;
                      return Card(
                        child: ListTile(
                          leading: const Icon(Icons.history_rounded),
                          title: Text(
                            entry.beanName.isEmpty
                                ? 'Unnamed Bean'
                                : entry.beanName,
                          ),
                          subtitle: Text(
                            '${entry.beanType} • ${entry.roastLevel} • ${shotTime.toStringAsFixed(1)} s',
                          ),
                          trailing: IconButton(
                            icon: const Icon(Icons.delete_outline),
                            tooltip: 'Delete',
                            onPressed: () => _deleteEntry(entry),
                          ),
                          onTap: () {
                            Navigator.of(context).pushNamed(
                              AppRoutes.analysisResult,
                              arguments: {
                                'resultPayload': entry.resultPayload,
                                'videoPath': entry.videoPath,
                                'beanName': entry.beanName,
                                'beanType': entry.beanType,
                                'roastLevel': entry.roastLevel,
                                'readOnly': true,
                              },
                            );
                          },
                        ),
                      );
                    }),
                  ],
                ),
              );
            },
          );
        },
      ),
    );
  }

  Map<String, List<SavedAnalysisEntry>> _groupByRecency(
    List<SavedAnalysisEntry> entries,
  ) {
    final sorted = [...entries]..sort((a, b) => b.savedAt.compareTo(a.savedAt));
    final map = <String, List<SavedAnalysisEntry>>{};

    for (final entry in sorted) {
      final key = _recencyLabel(entry.savedAt.toLocal());
      map.putIfAbsent(key, () => <SavedAnalysisEntry>[]).add(entry);
    }
    return map;
  }

  String _recencyLabel(DateTime dt) {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final date = DateTime(dt.year, dt.month, dt.day);
    final dayDiff = today.difference(date).inDays;

    if (dayDiff == 0) {
      return 'Today';
    }
    if (dayDiff == 1) {
      return 'Yesterday';
    }
    if (dayDiff < 7) {
      return 'This Week';
    }
    if (dayDiff < 31) {
      return 'This Month';
    }
    return 'Older';
  }
}

class _EmptyHistoryView extends StatelessWidget {
  const _EmptyHistoryView();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: const [
            Icon(Icons.history_rounded, size: 52),
            SizedBox(height: 12),
            Text(
              'No analysis history yet.',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
              textAlign: TextAlign.center,
            ),
            SizedBox(height: 6),
            Text(
              'Saved analysis sessions will appear here by recency.',
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}

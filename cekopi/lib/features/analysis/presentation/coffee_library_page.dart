import 'package:flutter/material.dart';
import 'package:cekopi/core/navigation/app_routes.dart';
import 'package:cekopi/features/analysis/data/coffee_library_repository.dart';
import 'package:cekopi/features/analysis/data/saved_analysis_entry.dart';

class CoffeeLibraryPage extends StatefulWidget {
  const CoffeeLibraryPage({super.key});

  @override
  State<CoffeeLibraryPage> createState() => _CoffeeLibraryPageState();
}

class _CoffeeGroup {
  const _CoffeeGroup({
    required this.key,
    required this.beanName,
    required this.beanType,
    required this.roastLevel,
    required this.shots,
  });

  final String key;
  final String beanName;
  final String beanType;
  final String roastLevel;
  final List<SavedAnalysisEntry> shots;
}

class _CoffeeLibraryPageState extends State<CoffeeLibraryPage> {
  final CoffeeLibraryRepository _repository = CoffeeLibraryRepository();

  late Future<List<SavedAnalysisEntry>> _entriesFuture;

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

  Future<void> _deleteShot(SavedAnalysisEntry entry) async {
    final shouldDelete = await showDialog<bool>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Delete saved shot?'),
          content: const Text('This action cannot be undone.'),
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
    ).showSnackBar(const SnackBar(content: Text('Shot deleted.')));
  }

  Future<void> _deleteGroup(_CoffeeGroup group) async {
    final shouldDelete = await showDialog<bool>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Delete coffee from library?'),
          content: Text(
            'Delete ${group.shots.length} saved shot(s) for ${group.beanName}?',
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

    await _repository.deleteEntriesByIds(group.shots.map((entry) => entry.id));
    if (!mounted) {
      return;
    }

    _refresh();
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Coffee deleted from library.')),
    );
  }

  Future<void> _renameCoffee(_CoffeeGroup group) async {
    final controller = TextEditingController(text: group.beanName);
    final nextName = await showDialog<String>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Edit coffee name'),
          content: TextField(
            controller: controller,
            autofocus: true,
            textInputAction: TextInputAction.done,
            decoration: const InputDecoration(
              labelText: 'Coffee name',
              hintText: 'Enter coffee name',
            ),
            onSubmitted: (value) => Navigator.of(context).pop(value),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.of(context).pop(controller.text),
              child: const Text('Save'),
            ),
          ],
        );
      },
    );
    controller.dispose();

    final trimmed = (nextName ?? '').trim();
    if (trimmed.isEmpty || trimmed == group.beanName) {
      return;
    }

    await _repository.renameBeanNameForGroup(
      fromBeanName: group.beanName,
      beanType: group.beanType,
      roastLevel: group.roastLevel,
      toBeanName: trimmed,
    );
    if (!mounted) {
      return;
    }

    _refresh();
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('Coffee name updated.')));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Coffee Library'),
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
            return const _EmptyLibraryView();
          }

          final groups = _groupEntries(entries);
          return ListView.separated(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
            itemCount: groups.length,
            separatorBuilder: (_, _) => const SizedBox(height: 12),
            itemBuilder: (context, index) {
              final group = groups[index];
              return Card(
                child: ExpansionTile(
                  title: Text(
                    group.beanName,
                    style: const TextStyle(fontWeight: FontWeight.w700),
                  ),
                  subtitle: Text('${group.beanType} • ${group.roastLevel}'),
                  trailing: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      IconButton(
                        onPressed: () => _renameCoffee(group),
                        icon: const Icon(Icons.edit_outlined),
                        tooltip: 'Edit coffee name',
                      ),
                      IconButton(
                        onPressed: () => _deleteGroup(group),
                        icon: const Icon(Icons.delete_outline),
                        tooltip: 'Delete coffee',
                      ),
                    ],
                  ),
                  children: group.shots.map((entry) {
                    final shotTime = entry.shotTimeSeconds;
                    final savedAt = _formatDateTime(entry.savedAt);
                    return ListTile(
                      leading: const Icon(Icons.local_cafe_outlined),
                      title: Text('Shot saved $savedAt'),
                      subtitle: Text(
                        'Shot time: ${shotTime.toStringAsFixed(1)} s',
                      ),
                      trailing: IconButton(
                        icon: const Icon(Icons.delete_outline),
                        tooltip: 'Delete shot',
                        onPressed: () => _deleteShot(entry),
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
                    );
                  }).toList(),
                ),
              );
            },
          );
        },
      ),
    );
  }

  List<_CoffeeGroup> _groupEntries(List<SavedAnalysisEntry> entries) {
    final grouped = <String, List<SavedAnalysisEntry>>{};
    final labels = <String, Map<String, String>>{};

    for (final entry in entries) {
      final key = '${entry.beanName}::${entry.beanType}::${entry.roastLevel}';
      grouped.putIfAbsent(key, () => <SavedAnalysisEntry>[]).add(entry);
      labels[key] = {
        'beanName': entry.beanName,
        'beanType': entry.beanType,
        'roastLevel': entry.roastLevel,
      };
    }

    final groups = grouped.entries.map((e) {
      final list = e.value..sort((a, b) => b.savedAt.compareTo(a.savedAt));
      final label = labels[e.key] ?? <String, String>{};
      return _CoffeeGroup(
        key: e.key,
        beanName: label['beanName'] ?? 'Unknown Bean',
        beanType: label['beanType'] ?? 'Unknown Type',
        roastLevel: label['roastLevel'] ?? 'Unknown Roast',
        shots: list,
      );
    }).toList();

    groups.sort((a, b) {
      final latestA = a.shots.isNotEmpty
          ? a.shots.first.savedAt
          : DateTime.fromMillisecondsSinceEpoch(0);
      final latestB = b.shots.isNotEmpty
          ? b.shots.first.savedAt
          : DateTime.fromMillisecondsSinceEpoch(0);
      return latestB.compareTo(latestA);
    });

    return groups;
  }

  String _formatDateTime(DateTime dt) {
    final local = dt.toLocal();
    final year = local.year.toString().padLeft(4, '0');
    final month = local.month.toString().padLeft(2, '0');
    final day = local.day.toString().padLeft(2, '0');
    final hour = local.hour.toString().padLeft(2, '0');
    final minute = local.minute.toString().padLeft(2, '0');
    return '$year-$month-$day $hour:$minute';
  }
}

class _EmptyLibraryView extends StatelessWidget {
  const _EmptyLibraryView();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: const [
            Icon(Icons.local_cafe_outlined, size: 52),
            SizedBox(height: 12),
            Text(
              'No saved analyses yet.',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
              textAlign: TextAlign.center,
            ),
            SizedBox(height: 6),
            Text(
              'Run an analysis, then save it with bean details to build your coffee library.',
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}

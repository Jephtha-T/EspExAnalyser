import 'dart:convert';

Map<String, dynamic>? asStringKeyedMap(Object? value) {
  if (value is Map<String, dynamic>) {
    return value;
  }
  if (value is Map) {
    return <String, dynamic>{
      for (final entry in value.entries) entry.key.toString(): entry.value,
    };
  }
  return null;
}

Map<String, dynamic>? tryDecodeJsonObject(String body) {
  if (body.trim().isEmpty) {
    return null;
  }

  try {
    return asStringKeyedMap(jsonDecode(body));
  } catch (_) {
    return null;
  }
}

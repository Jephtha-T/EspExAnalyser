import 'package:flutter_test/flutter_test.dart';

import 'package:cekopi/app.dart';
import 'package:cekopi/features/splash/presentation/splash_screen.dart';

void main() {
  testWidgets('App boots and shows splash branding', (WidgetTester tester) async {
    await tester.pumpWidget(const CekopiApp());

    expect(find.byType(SplashScreen), findsOneWidget);
  });
}

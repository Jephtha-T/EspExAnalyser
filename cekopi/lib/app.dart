import 'package:flutter/material.dart';
import 'package:cekopi/core/navigation/app_router.dart';
import 'package:cekopi/core/navigation/app_routes.dart';
import 'package:cekopi/core/theme/app_colors.dart';

class CekopiApp extends StatelessWidget {
  const CekopiApp({super.key});

  @override
  Widget build(BuildContext context) {
    final colorScheme =
        ColorScheme.fromSeed(
          seedColor: AppColors.primaryAction,
          brightness: Brightness.light,
        ).copyWith(
          primary: AppColors.primaryAction,
          onPrimary: Colors.white,
          secondary: AppColors.secondaryAction,
          onSecondary: Colors.white,
          surface: AppColors.surface,
          onSurface: AppColors.bodyText,
        );
    final baseTheme = ThemeData.light(useMaterial3: true);

    return MaterialApp(
      title: 'Cekopi',
      debugShowCheckedModeBanner: false,
      initialRoute: AppRoutes.splash,
      onGenerateRoute: AppRouter.onGenerateRoute,
      scrollBehavior: const MaterialScrollBehavior().copyWith(
        physics: const BouncingScrollPhysics(),
      ),
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: colorScheme,
        scaffoldBackgroundColor: AppColors.appBackground,
        fontFamily: 'SF Pro Display',
        splashFactory: NoSplash.splashFactory,
        splashColor: Colors.transparent,
        highlightColor: Colors.transparent,
        pageTransitionsTheme: const PageTransitionsTheme(
          builders: {
            TargetPlatform.android: CupertinoPageTransitionsBuilder(),
            TargetPlatform.iOS: CupertinoPageTransitionsBuilder(),
            TargetPlatform.macOS: CupertinoPageTransitionsBuilder(),
            TargetPlatform.linux: FadeUpwardsPageTransitionsBuilder(),
            TargetPlatform.windows: FadeUpwardsPageTransitionsBuilder(),
          },
        ),
        textTheme: baseTheme.textTheme.apply(
          bodyColor: AppColors.bodyText,
          displayColor: AppColors.headingText,
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.transparent,
          foregroundColor: AppColors.headingText,
          elevation: 0,
          centerTitle: true,
          scrolledUnderElevation: 0,
          titleTextStyle: TextStyle(
            fontFamily: 'SF Pro Display',
            fontSize: 20,
            fontWeight: FontWeight.w700,
            color: AppColors.headingText,
          ),
        ),
        cardTheme: CardThemeData(
          color: AppColors.surface,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(18),
          ),
        ),
        dialogTheme: DialogThemeData(
          backgroundColor: AppColors.surface,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(18),
          ),
        ),
        bottomSheetTheme: const BottomSheetThemeData(
          backgroundColor: Colors.transparent,
          surfaceTintColor: Colors.transparent,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: AppColors.primaryAction,
            foregroundColor: Colors.white,
            textStyle: const TextStyle(
              fontFamily: 'SF Pro Display',
              fontSize: 15,
              fontWeight: FontWeight.w600,
            ),
            shape: const StadiumBorder(),
            elevation: 0,
          ),
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            backgroundColor: AppColors.primaryAction,
            foregroundColor: Colors.white,
            textStyle: const TextStyle(
              fontFamily: 'SF Pro Display',
              fontSize: 15,
              fontWeight: FontWeight.w600,
            ),
            shape: const StadiumBorder(),
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            foregroundColor: AppColors.headingText,
            side: const BorderSide(color: AppColors.caramel400),
            textStyle: const TextStyle(
              fontFamily: 'SF Pro Display',
              fontSize: 15,
              fontWeight: FontWeight.w600,
            ),
            shape: const StadiumBorder(),
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          hintStyle: const TextStyle(color: AppColors.mutedText),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10),
            borderSide: const BorderSide(color: AppColors.cream300),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10),
            borderSide: const BorderSide(color: AppColors.cream300),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(10),
            borderSide: const BorderSide(
              color: AppColors.caramel500,
              width: 1.2,
            ),
          ),
        ),
      ),
    );
  }
}

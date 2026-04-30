import 'package:flutter/material.dart';
import 'package:cekopi/core/utils/json_helpers.dart';
import 'package:cekopi/features/analysis/presentation/coffee_library_page.dart';
import 'package:cekopi/features/analysis/presentation/history_page.dart';
import 'package:cekopi/features/analysis/presentation/analysis_processing_page.dart';
import 'package:cekopi/features/analysis/presentation/analysis_result_page.dart';
import 'package:cekopi/features/analysis/presentation/roi_confirmation_page.dart';
import 'package:cekopi/features/home/presentation/home_page.dart';
import 'package:cekopi/features/splash/presentation/splash_screen.dart';
import 'package:cekopi/features/video_capture/presentation/video_capture_page.dart';

import 'app_routes.dart';

class AppRouter {
  const AppRouter._();

  static Route<dynamic> onGenerateRoute(RouteSettings settings) {
    final args = asStringKeyedMap(settings.arguments);

    switch (settings.name) {
      case AppRoutes.splash:
        return MaterialPageRoute<void>(
          builder: (_) => const SplashScreen(),
          settings: settings,
        );
      case AppRoutes.home:
        return MaterialPageRoute<void>(
          builder: (_) => const HomePage(),
          settings: settings,
        );
      case AppRoutes.videoCapture:
        final errorCode = args?['errorCode']?.toString();
        final errorMessage = args?['errorMessage']?.toString();
        return MaterialPageRoute<void>(
          builder: (_) => VideoCapturePage(
            initialErrorCode: errorCode,
            initialErrorMessage: errorMessage,
          ),
          settings: settings,
        );
      case AppRoutes.roiConfirmation:
        final jobId = args?['jobId']?.toString();
        final videoPath = args?['videoPath']?.toString();
        final uploadPayload = asStringKeyedMap(args?['uploadPayload']);
        if (jobId == null ||
            jobId.isEmpty ||
            videoPath == null ||
            videoPath.isEmpty) {
          return MaterialPageRoute<void>(
            builder: (_) => const VideoCapturePage(),
            settings: settings,
          );
        }
        return MaterialPageRoute<void>(
          builder: (_) => RoiConfirmationPage(
            jobId: jobId,
            videoPath: videoPath,
            uploadPayload: uploadPayload,
          ),
          settings: settings,
        );
      case AppRoutes.analysisProcessing:
        final jobId = args?['jobId']?.toString();
        final videoPath = args?['videoPath']?.toString();
        final roiEllipse = asStringKeyedMap(args?['roiEllipse']);
        final roiSource = args?['roiSource']?.toString();
        if (jobId == null || jobId.isEmpty) {
          return MaterialPageRoute<void>(
            builder: (_) => const VideoCapturePage(),
            settings: settings,
          );
        }
        return MaterialPageRoute<void>(
          builder: (_) => AnalysisProcessingPage(
            jobId: jobId,
            videoPath: videoPath,
            roiEllipse: roiEllipse,
            roiSource: roiSource,
          ),
          settings: settings,
        );
      case AppRoutes.analysisResult:
        final payload = asStringKeyedMap(args?['resultPayload']);
        final videoPath = args?['videoPath']?.toString();
        final beanName = args?['beanName']?.toString();
        final beanType = args?['beanType']?.toString();
        final roastLevel = args?['roastLevel']?.toString();
        final readOnly = args?['readOnly'] == true;
        return MaterialPageRoute<void>(
          builder: (_) => AnalysisResultPage(
            resultPayload: payload,
            videoPath: videoPath,
            initialBeanName: beanName,
            initialBeanType: beanType,
            initialRoastLevel: roastLevel,
            readOnly: readOnly,
          ),
          settings: settings,
        );
      case AppRoutes.coffeeLibrary:
        return MaterialPageRoute<void>(
          builder: (_) => const CoffeeLibraryPage(),
          settings: settings,
        );
      case AppRoutes.history:
        return MaterialPageRoute<void>(
          builder: (_) => const HistoryPage(),
          settings: settings,
        );
      default:
        return MaterialPageRoute<void>(
          builder: (_) => const SplashScreen(),
          settings: settings,
        );
    }
  }
}

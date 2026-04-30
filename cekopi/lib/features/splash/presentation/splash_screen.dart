import 'package:flutter/material.dart';
import 'package:cekopi/core/navigation/app_routes.dart';
import 'package:cekopi/core/theme/app_colors.dart';
import 'package:cekopi/features/shared/widgets/cekopi_brand.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  static const _totalSplashDuration = Duration(milliseconds: 2400);
  static const _fadeInDuration = Duration(milliseconds: 650);
  static const _fadeOutDuration = Duration(milliseconds: 500);

  late final AnimationController _fadeController;
  late final Animation<double> _opacity;

  @override
  void initState() {
    super.initState();
    _fadeController = AnimationController(
      vsync: this,
      duration: _fadeInDuration,
      reverseDuration: _fadeOutDuration,
    );
    _opacity = CurvedAnimation(
      parent: _fadeController,
      curve: Curves.easeOutCubic,
      reverseCurve: Curves.easeInCubic,
    );
    _runSplashFlow();
  }

  Future<void> _runSplashFlow() async {
    await _fadeController.forward();

    final holdDuration =
        _totalSplashDuration - _fadeInDuration - _fadeOutDuration;
    if (holdDuration > Duration.zero) {
      await Future<void>.delayed(holdDuration);
    }

    if (!mounted) {
      return;
    }
    await _fadeController.reverse();
    if (!mounted) {
      return;
    }

    Navigator.of(context).pushReplacementNamed(AppRoutes.home);
  }

  @override
  void dispose() {
    _fadeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _fadeController,
      builder: (context, child) {
        final logoOpacity = _opacity.value.clamp(0.0, 1.0);

        return ColoredBox(
          color: AppColors.splashBackground,
          child: SafeArea(
            child: Center(
              child: Opacity(
                opacity: logoOpacity,
                child: const CekopiBrand(
                  logoSize: 58,
                  textWidth: 150,
                  spacing: 14,
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

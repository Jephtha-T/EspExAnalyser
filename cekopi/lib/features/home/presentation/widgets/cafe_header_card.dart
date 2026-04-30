import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:cekopi/core/assets/app_assets.dart';

class CafeHeaderCard extends StatelessWidget {
  const CafeHeaderCard({super.key});

  static const _counterHeight = 82.0;

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final cardHeight = (screenWidth * 0.72).clamp(220.0, 285.0);

    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: SizedBox(
        height: cardHeight,
        child: Stack(
          children: [
            Positioned.fill(
              child: Container(color: const Color(0xFFDACCAA)),
            ),
            Positioned.fill(
              bottom: _counterHeight,
              child: _HeaderTopScene(),
            ),
            Positioned(
              left: 0,
              right: 0,
              bottom: _counterHeight,
              child: Container(height: 4, color: const Color(0xFF3A1519)),
            ),
            const Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: _CounterCabinet(),
            ),
          ],
        ),
      ),
    );
  }
}

class _HeaderTopScene extends StatelessWidget {
  const _HeaderTopScene();

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final sceneWidth = constraints.maxWidth;
        final sceneHeight = constraints.maxHeight;
        final bagWidth = sceneWidth * 0.118;
        final bagHeight = bagWidth * (75 / 53);
        final bagBottomOffset = -bagHeight * 0.2;

        return Stack(
          clipBehavior: Clip.none,
          children: [
            Positioned(
              left: 0,
              right: 0,
              top: -20,
              child: SvgPicture.asset(
                AppAssets.headerLights,
                width: sceneWidth,
                height: sceneHeight + 32,
                fit: BoxFit.fill,
              ),
            ),
            Positioned(
              left: sceneWidth * 0.03,
              top: sceneHeight * 0.14,
              child: SvgPicture.asset(
                AppAssets.headerBoard,
                width: sceneWidth * 0.48,
                fit: BoxFit.contain,
              ),
            ),
            Positioned(
              left: sceneWidth * 0.04,
              bottom: bagBottomOffset,
              child: SizedBox(
                width: bagWidth * 2.35,
                height: bagHeight,
                child: Stack(
                  clipBehavior: Clip.none,
                  children: [
                    Positioned(
                      left: bagWidth * 1.24,
                      bottom: 0,
                      child: SvgPicture.asset(
                        AppAssets.headerBag3,
                        width: bagWidth,
                        height: bagHeight,
                        fit: BoxFit.contain,
                      ),
                    ),
                    Positioned(
                      left: bagWidth * 0.62,
                      bottom: 0,
                      child: SvgPicture.asset(
                        AppAssets.headerBag2,
                        width: bagWidth,
                        height: bagHeight,
                        fit: BoxFit.contain,
                      ),
                    ),
                    Positioned(
                      left: 0,
                      bottom: 0,
                      child: SvgPicture.asset(
                        AppAssets.headerBag1,
                        width: bagWidth,
                        height: bagHeight,
                        fit: BoxFit.contain,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            Positioned(
              left: sceneWidth * 0.56,
              bottom: sceneHeight * 0.02,
              child: SvgPicture.asset(
                AppAssets.headerGrinder,
                width: sceneWidth * 0.105,
                fit: BoxFit.contain,
              ),
            ),
            Positioned(
              right: sceneWidth * 0.05,
              bottom: -sceneHeight * 0.01,
              child: SvgPicture.asset(
                AppAssets.headerCoffeeMachine,
                width: sceneWidth * 0.215,
                fit: BoxFit.contain,
              ),
            ),
          ],
        );
      },
    );
  }
}

class _CounterCabinet extends StatelessWidget {
  const _CounterCabinet();

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 82,
      child: ColoredBox(
        color: const Color(0xFF744230),
        child: Row(
          children: [
            Expanded(
              child: Container(
                margin: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: const Color(0xFFA67D52),
                  borderRadius: BorderRadius.circular(4),
                  border: Border.all(color: const Color(0xFF2A0D14), width: 2),
                ),
                child: const Center(
                  child: Text(
                    '||',
                    style: TextStyle(
                      color: Color(0xFFDDB122),
                      fontSize: 28,
                      letterSpacing: 4,
                    ),
                  ),
                ),
              ),
            ),
            SizedBox(
              width: 138,
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 10),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: const [
                    _DrawerStrip(),
                    _DrawerStrip(),
                    _DrawerStrip(),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _DrawerStrip extends StatelessWidget {
  const _DrawerStrip();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 98,
      height: 20,
      decoration: BoxDecoration(
        color: const Color(0xFFA67D52),
        borderRadius: BorderRadius.circular(3),
        border: Border.all(color: const Color(0xFF2A0D14), width: 2),
      ),
      child: const Center(
        child: SizedBox(
          width: 36,
          height: 2,
          child: DecoratedBox(
            decoration: BoxDecoration(color: Color(0xFFDDB122)),
          ),
        ),
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:cekopi/core/navigation/app_routes.dart';
import 'package:cekopi/core/theme/app_colors.dart';
import 'package:cekopi/features/home/presentation/widgets/cafe_header_card.dart';
import 'package:cekopi/features/home/presentation/widgets/menu_card.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    final horizontalPadding = (size.width * 0.055).clamp(16.0, 24.0);

    return Scaffold(
      backgroundColor: AppColors.homeBackground,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: EdgeInsets.fromLTRB(
            horizontalPadding,
            10,
            horizontalPadding,
            24,
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: const [
              CafeHeaderCard(),
              SizedBox(height: 24),
              MenuCard(
                title: 'Start Analysis',
                icon: Icons.camera_alt_rounded,
                gradient: [AppColors.espresso700, AppColors.espresso800],
                height: 148,
                titleSize: 18,
                onTapRouteName: AppRoutes.videoCapture,
              ),
              SizedBox(height: 20),
              Row(
                children: [
                  Expanded(
                    child: MenuCard(
                      title: 'History',
                      icon: Icons.history_rounded,
                      gradient: [
                        AppColors.secondaryAction,
                        AppColors.caramel400,
                      ],
                      height: 150,
                      titleSize: 14,
                      onTapRouteName: AppRoutes.history,
                    ),
                  ),
                  SizedBox(width: 16),
                  Expanded(
                    child: MenuCard(
                      title: 'Coffee Library',
                      icon: Icons.local_cafe_outlined,
                      gradient: [AppColors.neutral700, AppColors.espresso900],
                      height: 150,
                      titleSize: 12,
                      onTapRouteName: AppRoutes.coffeeLibrary,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

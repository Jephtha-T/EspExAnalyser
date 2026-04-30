import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:cekopi/core/assets/app_assets.dart';

class CekopiBrand extends StatelessWidget {
  const CekopiBrand({
    super.key,
    this.logoSize = 58,
    this.textWidth = 150,
    this.spacing = 14,
  });

  final double logoSize;
  final double textWidth;
  final double spacing;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Image.asset(
          AppAssets.logo,
          width: logoSize,
          height: logoSize,
          fit: BoxFit.contain,
        ),
        SizedBox(width: spacing),
        SvgPicture.asset(
          AppAssets.cekopiText,
          width: textWidth,
          fit: BoxFit.contain,
        ),
      ],
    );
  }
}

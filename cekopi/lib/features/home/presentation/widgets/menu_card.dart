import 'package:flutter/material.dart';

class MenuCard extends StatelessWidget {
  const MenuCard({
    super.key,
    required this.title,
    required this.icon,
    required this.gradient,
    required this.height,
    required this.titleSize,
    this.onTapRouteName,
  });

  final String title;
  final IconData icon;
  final List<Color> gradient;
  final double height;
  final double titleSize;
  final String? onTapRouteName;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final compact = constraints.maxHeight <= 150;
        final iconFrameSize = compact ? 46.0 : 50.0;
        final iconSize = compact ? 24.0 : 28.0;
        final verticalPadding = compact ? 10.0 : 12.0;
        final titleGap = compact ? 6.0 : 10.0;
        final tailGap = compact ? 2.0 : 4.0;

        return Material(
          color: Colors.transparent,
          borderRadius: BorderRadius.circular(34),
          child: InkWell(
            borderRadius: BorderRadius.circular(34),
            onTap: onTapRouteName == null
                ? null
                : () => Navigator.of(context).pushNamed(onTapRouteName!),
            child: Container(
              height: height,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(34),
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: gradient,
                ),
                boxShadow: const [
                  BoxShadow(
                    color: Color(0x33000000),
                    blurRadius: 14,
                    offset: Offset(0, 8),
                  ),
                ],
              ),
              child: Padding(
                padding: EdgeInsets.symmetric(horizontal: 16, vertical: verticalPadding),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      width: iconFrameSize,
                      height: iconFrameSize,
                      decoration: const BoxDecoration(
                        color: Color(0x29FFFFFF),
                        shape: BoxShape.circle,
                      ),
                      child: Icon(icon, color: const Color(0xFFF8F4ED), size: iconSize),
                    ),
                    const Spacer(),
                    Text(
                      title,
                      style: TextStyle(
                        color: const Color(0xFFF8F4ED),
                        fontSize: titleSize,
                        fontWeight: FontWeight.w600,
                        height: 1.0,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    SizedBox(height: titleGap),
                    Container(width: 80, height: 2, color: const Color(0x88F3D6A2)),
                    SizedBox(height: tailGap),
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

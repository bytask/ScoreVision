{pkgs}: {
  deps = [
    pkgs.which
    pkgs.libpng
    pkgs.libjpeg_turbo
    pkgs.xsimd
    pkgs.pkg-config
    pkgs.libxcrypt
    pkgs.libGLU
    pkgs.libGL
  ];
}

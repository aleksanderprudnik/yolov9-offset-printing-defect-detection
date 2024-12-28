{pkgs}: {
  deps = [
    pkgs.glibcLocales
    pkgs.which
    pkgs.libpng
    pkgs.libjpeg_turbo
    pkgs.libGLU
    pkgs.libGL
    pkgs.tk
    pkgs.tcl
    pkgs.qhull
    pkgs.gtk3
    pkgs.gobject-introspection
    pkgs.ghostscript
    pkgs.freetype
    pkgs.ffmpeg-full
    pkgs.cairo
    pkgs.xsimd
    pkgs.pkg-config
    pkgs.libxcrypt
  ];
}

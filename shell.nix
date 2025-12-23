{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python313 # Python 3.13
    uv # Python package manager
    nixfmt # Nix formatter
    just # Just
    zlib
    ffmpeg
  ];

  # Shell hook to set up environment
  shellHook = ''
    export TMPDIR=/tmp
    export UV_PYTHON="${pkgs.python313}/bin/python"
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.ffmpeg.out}/lib:$LD_LIBRARY_PATH"
  '';
}

{
  pkgs,
  lib,
  config,
  ...
}:
{
  env = {
    UV_PYTHON = toString config.languages.python.package.interpreter;
    NUMBA_DISABLE_INTEL_SVML = true;
  };

  scripts = {
    # yasf.exec = ''uv run yasf "$@"'';
  };

  enterShell = ''
    git --version
    if [ ! -L "$DEVENV_ROOT/.venv" ]; then
        ln -s "$DEVENV_STATE/venv/" "$DEVENV_ROOT/.venv"
    fi
  '';

  languages.python = {
    enable = true;

    uv = {
      enable = true;
      sync = {
        enable = true;
        groups = [
          "test"
          # "docs"
          # "profiling"
        ];
      };
    };

    libraries = [
      pkgs.zlib
    ];
  };
}

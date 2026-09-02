{
  pkgs,
  config,
  ...
}:
{
  env = {
    UV_PYTHON = toString config.languages.python.package.interpreter;
  };

  scripts = {
    polydisc.exec = ''uv run polydisc "$@"'';
    dashboard.exec = ''uv run streamlit run dashboard/app.py'';
    bench.exec = ''uv run python scripts/benchmark.py "$@"'';
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
          "dev"
          # "docs"
          # "profiling"
        ];
      };
    };

    libraries = with pkgs; [
      zlib
      libx11
      libxrender
      libGL
    ];
  };

  git-hooks.hooks = {
    isort = {
      enable = true;
      settings.profile = "black";
    };

    ruff-format = {
      enable = true;
      description = "Ruff formatter";
      package = config.git-hooks.tools.ruff;
      entry = "ruff format";
      types = [ "python" ];
      args = [ "--check" ];
      after = [ "isort" ];
    };

    ruff-check = {
      enable = true;
      description = "Ruff linter";
      package = config.git-hooks.tools.ruff;
      entry = "ruff check";
      types = [ "python" ];
      # args = [ "--fix" ];
      after = [ "ruff-format" ];
    };

    # ty = {
    #   enable = true;
    #   description = "ty type check";
    #   package = pkgs.ty;
    #   entry = "ty check .";
    #   pass_filenames = false;
    #   types = [ "python" ];
    #   after = [ "ruff-check" ];
    #   require_serial = true;
    # };
  };
}

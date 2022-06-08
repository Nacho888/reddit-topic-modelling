with open("requirements.txt", "r", encoding="utf-16") as f:
    with open("requirements_no_version.txt", "a+", encoding="utf-16") as f_out:
        pkgs = f.read()
        pkgs = pkgs.splitlines()

        for pkg in pkgs:
            if "#" not in pkg:
                f_out.write(pkg.split("=")[0] + "\n")
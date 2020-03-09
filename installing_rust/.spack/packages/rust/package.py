# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class Rust(Package):
    """The rust programming language toolchain"""

    homepage = "http://www.rust-lang.org"
    git      = "https://github.com/rust-lang/rust.git"

    version('develop', branch='master')
    version('1.41.0', tag='1.41.0')
    version('1.34.0', tag='1.34.0')
    version('1.32.0', tag='1.32.0')
    version('1.31.1', tag='1.31.1')
    version('1.31.0', tag='1.31.0')  # "Rust 2018" edition
    version('1.30.1', tag='1.30.1')

    extendable = True

    # Rust
    depends_on("llvm -clang -compiler-rt -internal_unwind -libcxx +link_dylib -lld -lldb")
    depends_on("curl")
    depends_on("git")
    depends_on("cmake", type='build')
    depends_on("binutils")
    depends_on("python@:2.8", type='build')

    # Cargo
    depends_on("openssl")

    variant('nvptx', default=False, description='Builds the nvptx64-nvidia-cuda toolchain')

    phases = ['configure', 'install']

    def configure(self, spec, prefix):
        configure_args = [
          '--prefix=%s' % prefix,
          '--llvm-root=' + spec['llvm'].prefix,
          # Workaround for "FileCheck does not exist" error
          '--disable-codegen-tests',
          # Includes Cargo in the build
          # https://github.com/rust-lang/cargo/issues/3772#issuecomment-283109482
          '--enable-extended',
          # Prevent build from writing bash completion into system path
          '--sysconfdir=%s' % join_path(prefix, 'etc/')
          ]

        if '+nvptx' in self.spec:
            configure_args.append('--target=nvptx64-nvidia-cuda')
            configure_args.append('--disable-docs') # workaround for bug in build system

        configure(*configure_args)

        # Build system defaults to searching in the same path as Spack's
        # compiler wrappers which causes the build to fail
        filter_file(
            '#ar = "ar"',
            'ar = "%s"' % join_path(spec['binutils'].prefix.bin, 'ar'),
            'config.toml')

    def install(self, spec, prefix):
        make()
        make("install")

    def setup_run_environment(self, env):
        env.prepend_path('PATH', join_path(self.prefix, 'bin'))
        env.prepend_path('MANPATH', join_path(self.prefix, 'share', 'man'))
        env.prepend_path('LIBRARY_PATH', join_path(self.prefix, 'lib'))
        env.prepend_path('CMAKE_PREFIX_PATH', join_path(self.prefix))
        env.set('RUST_ROOT', self.prefix)
        env.set('CARGO_HOME', join_path(self.prefix, 'cargo'))
        env.prepend_path('PATH', join_path(self.prefix, 'cargo', 'bin'))
